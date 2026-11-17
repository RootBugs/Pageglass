"""Tests for the sqlite-reader project (entries 006)."""

import contextlib
import hashlib
import io
import os
import shutil
import sqlite3
import tempfile
import unittest

import sqlite_reader
from sqlite_reader import (
    DatabaseHeader,
    SQLiteFormatError,
    decode_record,
    decode_varint,
    dump,
    index_scan,
    main,
    parse_create_table,
    read_header,
    schema,
    table_rows,
)


def build_db(path, statements, page_size=512, encoding=None):
    """Create a small-page SQLite fixture database."""
    if os.path.exists(path):
        os.remove(path)
    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA page_size=%d" % page_size)
        if encoding is not None:
            conn.execute("PRAGMA encoding='%s'" % encoding)
        # A write is required to materialise the file even with no
        # statements (page_size alone only sets an option).
        conn.execute("PRAGMA user_version=1")
        for stmt in statements:
            conn.execute(stmt)
        conn.commit()
    finally:
        conn.close()
    return path


@contextlib.contextmanager
def sqlitedb(path):
    """Open a fixture connection, committing and closing on exit."""
    conn = sqlite3.connect(path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="sqlite-reader-test-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def dbpath(self, name):
        return os.path.join(self.tmp, name)


class HeaderTests(TempDirTestCase):
    def test_read_header_from_fixture(self):
        path = build_db(self.dbpath("h.db"), ["CREATE TABLE t(a INTEGER)"])
        with open(path, "rb") as fh:
            header = read_header(fh.read())
        self.assertEqual(header.magic, b"SQLite format 3\x00")
        self.assertEqual(header.page_size, 512)
        self.assertEqual(header.text_encoding, "utf-8")
        self.assertEqual(header.database_size_pages, os.path.getsize(path) // 512)
        self.assertIsInstance(header, DatabaseHeader)

    def test_bad_magic_rejected(self):
        path = build_db(self.dbpath("m.db"), ["CREATE TABLE t(a)"])
        with open(path, "rb") as fh:
            data = bytearray(fh.read())
        data[0:16] = b"NOT-A-SQLITE-DB!"
        with self.assertRaises(SQLiteFormatError):
            read_header(bytes(data))

    def test_bad_page_size_rejected(self):
        path = build_db(self.dbpath("p.db"), ["CREATE TABLE t(a)"])
        with open(path, "rb") as fh:
            data = bytearray(fh.read())
        data[16:18] = (100).to_bytes(2, "big")  # not a power of two
        with self.assertRaises(SQLiteFormatError):
            read_header(bytes(data))

    def test_page_size_field_of_one_means_64k(self):
        path = build_db(self.dbpath("big.db"), ["CREATE TABLE t(a)"], page_size=65536)
        with open(path, "rb") as fh:
            header = read_header(fh.read())
        self.assertEqual(header.page_size, 65536)

    def test_truncated_file_rejected(self):
        with self.assertRaises(SQLiteFormatError):
            read_header(b"SQLite format 3\x00")


class VarintTests(unittest.TestCase):
    def test_single_byte(self):
        self.assertEqual(decode_varint(b"\x05"), (5, 1))
        self.assertEqual(decode_varint(b"\x7f"), (127, 1))

    def test_two_bytes(self):
        value, size = decode_varint(b"\x81\x00")
        self.assertEqual((value, size), (128, 2))
        value, size = decode_varint(b"\xff\x7f")
        self.assertEqual((value, size), (16383, 2))

    def test_nine_byte_all_ones(self):
        value, size = decode_varint(b"\xff" * 9)
        self.assertEqual(size, 9)
        self.assertEqual(value, 2 ** 64 - 1)

    def test_nine_byte_high_bit(self):
        # 9th byte contributes all 8 bits (SQLite bug-compatible behaviour):
        # seven 7-bit groups of 1,0,...,0 then a full byte -> (1 << 57) | 1.
        data = b"\x81" + b"\x80" * 7 + b"\x01"
        value, size = decode_varint(data)
        self.assertEqual(size, 9)
        self.assertEqual(value, (1 << 57) | 1)

    def test_truncated_rejected(self):
        with self.assertRaises(SQLiteFormatError):
            decode_varint(b"\x81")


class RecordDecodeTests(unittest.TestCase):
    def test_mixed_serial_types(self):
        import struct
        # Header: length varint + serial types [0, 1, 6, 7, 8, 9, 22, 13].
        # Serial 6 -> 8-byte int, 22 -> blob of (22-12)/2 = 5 bytes,
        # 13 -> text of (13-13)/2 = 0 bytes.
        serials = [0, 1, 6, 7, 8, 9, 22, 13]
        header_len = 1 + len(serials)
        body = bytearray()
        body.append(header_len)
        body.extend(serials)
        body.append(0x7F)          # serial 1 -> 1-byte int 127
        body.extend((-1234).to_bytes(8, "big", signed=True))  # serial 6
        body.extend(struct.pack(">d", 6.25))  # serial 7 -> float64
        # serial 8 and 9 store no bytes
        body.extend(b"hello")      # serial 22 -> blob of 5 bytes
        # serial 13 stores no bytes -> empty text
        values = decode_record(bytes(body), "utf-8")
        self.assertEqual(values, [None, 127, -1234, 6.25, 0, 1, b"hello", ""])

    def test_float_and_integers(self):
        import struct
        serials = [7, 8, 9, 4]
        header_len = 1 + len(serials)
        body = bytearray([header_len])
        body.extend(serials)
        body.extend(struct.pack(">d", 3.5))
        # serial 8 -> 0, serial 9 -> 1: both occupy no bytes
        body.extend((65535).to_bytes(4, "big"))
        self.assertEqual(decode_record(bytes(body), "utf-8"), [3.5, 0, 1, 65535])

    def test_wide_integer_serial_sizes(self):
        # Serial 5 occupies 6 bytes, serial 6 occupies 8 bytes.
        serials = [5, 6]
        body = bytearray([1 + len(serials)])
        body.extend(serials)
        body.extend((0x123456789ABC).to_bytes(6, "big"))
        body.extend((-2).to_bytes(8, "big", signed=True))
        self.assertEqual(
            decode_record(bytes(body), "utf-8"), [0x123456789ABC, -2]
        )

    def test_reserved_serial_types_rejected(self):
        body = bytes([2, 10])
        with self.assertRaises(SQLiteFormatError):
            decode_record(body, "utf-8")

    def test_blob_and_text_sizes(self):
        # Serial T >= 12: even -> blob of (T-12)/2 bytes, odd -> text of
        # (T-13)/2 bytes.
        header_len = 3
        body = bytearray([header_len, 22, 25])  # blob 5 bytes, text 6 bytes
        body.extend(b"hello")
        body.extend("héllo".encode("utf-8"))
        self.assertEqual(decode_record(bytes(body), "utf-8"), [b"hello", "héllo"])

    def test_negative_integers(self):
        serials = [1, 2, 3, 4, 5, 6]
        body = bytearray([1 + len(serials)])
        body.extend(serials)
        body.append(0xFF)                     # -1 (1 byte)
        body.extend((0xFFFF).to_bytes(2, "big"))   # -1
        body.extend((0xFFFFFF).to_bytes(3, "big")) # -1
        body.extend((0xFFFFFFFF).to_bytes(4, "big"))
        body.extend((0xFFFFFFFFFFFF).to_bytes(6, "big"))   # serial 5: 6 bytes
        body.extend((0xFFFFFFFFFFFFFFFF).to_bytes(8, "big"))  # serial 6: 8 bytes
        self.assertEqual(decode_record(bytes(body), "utf-8"), [-1] * 6)

    def test_utf16_text_decoding(self):
        for codec in ("utf-16-le", "utf-16-be"):
            text = "héllo wörld"
            encoded = text.encode(codec)
            body = bytearray([2, 13 + 2 * len(encoded)])
            body.extend(encoded)
            self.assertEqual(decode_record(bytes(body), codec), [text])


class SchemaTests(TempDirTestCase):
    def test_schema_lists_tables_and_views(self):
        path = build_db(self.dbpath("s.db"), [
            "CREATE TABLE people(id INTEGER PRIMARY KEY, name TEXT)",
            "CREATE INDEX idx_people_name ON people(name)",
            "CREATE VIEW adult AS SELECT * FROM people",
        ])
        with sqlitedb(path) as con:
            expected = con.execute(
                "SELECT type, name, tbl_name, sql FROM sqlite_master"
            ).fetchall()
        self.assertEqual(schema(path), expected)

    def test_schema_empty_database(self):
        path = build_db(self.dbpath("e.db"), [])
        self.assertEqual(schema(path), [])

    def test_missing_file_raises_oserror(self):
        with self.assertRaises(OSError):
            schema(self.dbpath("nope.db"))


class TableScanTests(TempDirTestCase):
    def fixture(self, name="t.db", rows=60):
        stmts = ["CREATE TABLE data(id INTEGER PRIMARY KEY, name TEXT, score REAL, note BLOB)"]
        path = build_db(self.dbpath(name), stmts)
        with sqlitedb(path) as con:
            for i in range(rows):
                con.execute(
                    "INSERT INTO data(name, score, note) VALUES (?,?,?)",
                    ("name-%04d" % i, i * 1.5, bytes([i % 256]) * (i % 5)),
                )
            con.commit()
        return path

    def test_table_scan_matches_sqlite(self):
        path = self.fixture(rows=200)
        with sqlitedb(path) as con:
            expected = con.execute(
                "SELECT id, name, score, note FROM data ORDER BY id"
            ).fetchall()
        got = [tuple(r) for r in table_rows(path, "data")]
        self.assertEqual(got, expected)

    def test_interior_pages_traversed(self):
        # 150 rows at page_size=512 forces a multi-level table b-tree.
        path = self.fixture(name="multi.db", rows=150)
        with sqlitedb(path) as con:
            root = con.execute(
                "SELECT rootpage FROM sqlite_master WHERE name='data'"
            ).fetchone()[0]
            count = con.execute("SELECT count(*) FROM data").fetchone()[0]
        with open(path, "rb") as fh:
            data = fh.read()
        # Page 1 is the schema table; the user table root lives elsewhere.
        root_type = data[(root - 1) * 512]
        self.assertEqual(root_type, 0x05)  # interior table b-tree root
        self.assertEqual(len(list(table_rows(path, "data"))), count)

    def test_unicode_and_null_values(self):
        path = build_db(self.dbpath("u.db"), [
            "CREATE TABLE t(id INTEGER PRIMARY KEY, s TEXT, v)",
        ])
        with sqlitedb(path) as con:
            con.execute("INSERT INTO t(s, v) VALUES (?, ?)", ("héllo wörld ✓", None))
            con.execute("INSERT INTO t(s, v) VALUES (?, ?)", ("", 42))
            con.commit()
        rows = list(table_rows(path, "t"))
        self.assertEqual(rows, [(1, "héllo wörld ✓", None), (2, "", 42)])

    def test_negative_rowids(self):
        path = build_db(self.dbpath("n.db"), [
            "CREATE TABLE neg(id INTEGER PRIMARY KEY, v TEXT)",
        ])
        with sqlitedb(path) as con:
            con.executemany("INSERT INTO neg VALUES (?, ?)",
                            [(-5, "a"), (-9000000000000000000, "b")])
            expected = con.execute("SELECT id, v FROM neg").fetchall()
        rows = list(table_rows(path, "neg"))
        self.assertEqual(rows, expected)
        self.assertEqual(rows, [(-9000000000000000000, "b"), (-5, "a")])

    def test_overflow_payload_reassembled(self):
        path = build_db(self.dbpath("o.db"), [
            "CREATE TABLE big(id INTEGER PRIMARY KEY, body TEXT)",
        ])
        long_text = ("The quick brown fox jumps over the lazy dog. " * 80)
        with sqlitedb(path) as con:
            con.execute("INSERT INTO big(body) VALUES (?)", (long_text,))
            con.execute("INSERT INTO big(body) VALUES (?)", ("short",))
            con.commit()
        rows = list(table_rows(path, "big"))
        self.assertEqual(rows, [(1, long_text), (2, "short")])

    def test_without_rowid_table_order(self):
        path = build_db(self.dbpath("w.db"), [
            "CREATE TABLE pairs(a TEXT, b TEXT, val TEXT, PRIMARY KEY (b, a)) WITHOUT ROWID",
        ])
        with sqlitedb(path) as con:
            con.executemany("INSERT INTO pairs VALUES (?, ?, ?)", [
                ("x1", "y1", "v1"), ("x2", "y1", "v2"), ("x3", "y2", "v3"),
            ])
            con.commit()
            expected = sorted(con.execute("SELECT a, b, val FROM pairs").fetchall())
        rows = sorted(tuple(r) for r in table_rows(path, "pairs"))
        self.assertEqual(rows, expected)

    def test_unknown_table_raises(self):
        path = build_db(self.dbpath("k.db"), ["CREATE TABLE t(a)"])
        with self.assertRaises(SQLiteFormatError):
            list(table_rows(path, "missing"))

    def test_empty_table(self):
        path = build_db(self.dbpath("em.db"), ["CREATE TABLE t(a TEXT)"])
        self.assertEqual(list(table_rows(path, "t")), [])


class IndexScanTests(TempDirTestCase):
    def test_index_scan_matches_sqlite_order(self):
        path = build_db(self.dbpath("i.db"), [
            "CREATE TABLE people(id INTEGER PRIMARY KEY, name TEXT, age INTEGER)",
            "CREATE INDEX idx_people_name ON people(name)",
        ])
        with sqlitedb(path) as con:
            con.executemany(
                "INSERT INTO people(name, age) VALUES (?, ?)",
                [("person-name-%04d" % (i * 7 % 200), i) for i in range(200)],
            )
            con.commit()
            expected = con.execute(
                "SELECT name, id FROM people ORDER BY name, id"
            ).fetchall()
        got = [tuple(r) for r in index_scan(path, "idx_people_name")]
        self.assertEqual(got, expected)

    def test_unknown_index_raises(self):
        path = build_db(self.dbpath("k.db"), ["CREATE TABLE t(a)"])
        with self.assertRaises(SQLiteFormatError):
            list(index_scan(path, "nope"))


class ParseCreateTableTests(unittest.TestCase):
    def test_detects_rowid_alias(self):
        info = parse_create_table(
            "CREATE TABLE people(id INTEGER PRIMARY KEY, name TEXT, age INTEGER)"
        )
        self.assertEqual(info.rowid_alias, "id")
        self.assertFalse(info.without_rowid)
        self.assertEqual(list(info.columns), ["id", "name", "age"])

    def test_detects_table_level_primary_key(self):
        info = parse_create_table(
            "CREATE TABLE pairs(a TEXT, b TEXT, val TEXT, PRIMARY KEY (b, a)) WITHOUT ROWID"
        )
        self.assertTrue(info.without_rowid)
        self.assertEqual(list(info.primary_key), ["b", "a"])
        self.assertIsNone(info.rowid_alias)

    def test_inline_pk_without_integer_is_not_rowid_alias(self):
        info = parse_create_table("CREATE TABLE t(name TEXT PRIMARY KEY, v TEXT)")
        self.assertIsNone(info.rowid_alias)
        self.assertEqual(list(info.primary_key), ["name"])

    def test_column_constraints_ignored(self):
        info = parse_create_table(
            'CREATE TABLE t("weird, col" INTEGER NOT NULL DEFAULT 5, '
            "other TEXT CHECK (length(other) > 0))"
        )
        self.assertEqual(list(info.columns), ["weird, col", "other"])


class DumpTests(TempDirTestCase):
    def test_dump_returns_all_rows(self):
        path = build_db(self.dbpath("d.db"), [
            "CREATE TABLE t(id INTEGER PRIMARY KEY, name TEXT)",
        ])
        with sqlitedb(path) as con:
            con.executemany("INSERT INTO t(name) VALUES (?)",
                            [("a",), ("b",), ("c",)])
            con.commit()
            expected = con.execute(
                "SELECT id, name FROM t ORDER BY id"
            ).fetchall()
        self.assertEqual(dump(path, "t"), expected)

    def test_dump_respects_limit(self):
        path = build_db(self.dbpath("l.db"), [
            "CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT)",
        ])
        with sqlitedb(path) as con:
            con.executemany("INSERT INTO t(v) VALUES (?)",
                            [("%d" % i,) for i in range(10)])
            con.commit()
        rows = dump(path, "t", limit=3)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0], (1, "0"))


class CommandLineTests(TempDirTestCase):
    def fixture(self):
        path = build_db(self.dbpath("cli.db"), [
            "CREATE TABLE people(id INTEGER PRIMARY KEY, name TEXT, age INTEGER)",
        ])
        with sqlitedb(path) as con:
            con.executemany("INSERT INTO people(name, age) VALUES (?, ?)", [
                ("Alice", 30), ("Bob", 40), ("Carol", 50),
            ])
            con.commit()
        return path

    def run_main(self, argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(argv)
        return code, out.getvalue()

    def test_schema_command(self):
        path = self.fixture()
        code, out = self.run_main(["schema", path])
        self.assertEqual(code, 0)
        self.assertIn("table\tpeople", out)

    def test_dump_command_tsv(self):
        path = self.fixture()
        code, out = self.run_main(["dump", path, "people"])
        self.assertEqual(code, 0)
        lines = out.rstrip("\n").split("\n")
        self.assertEqual(lines[0], "id\tname\tage")
        self.assertEqual(lines[1], "1\tAlice\t30")
        self.assertEqual(len(lines), 4)

    def test_dump_command_limit(self):
        path = self.fixture()
        code, out = self.run_main(["dump", path, "people", "--limit", "2"])
        self.assertEqual(code, 0)
        self.assertEqual(len(out.rstrip("\n").split("\n")), 3)

    def test_dump_missing_table_fails(self):
        path = self.fixture()
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code, _ = self.run_main(["dump", path, "ghost"])
        self.assertNotEqual(code, 0)
        self.assertIn("ghost", err.getvalue())

    def test_missing_file_fails(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code, _ = self.run_main(["schema", self.dbpath("absent.db")])
        self.assertNotEqual(code, 0)

    def test_no_arguments_fails(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code, _ = self.run_main([])
        self.assertNotEqual(code, 0)


class ReadOnlyTests(TempDirTestCase):
    def test_source_file_never_modified(self):
        path = build_db(self.dbpath("ro.db"), [
            "CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT)",
        ])
        with sqlitedb(path) as con:
            con.execute("INSERT INTO t(v) VALUES ('x')")
            con.commit()
        with open(path, "rb") as fh:
            before = hashlib.sha256(fh.read()).hexdigest()
        schema(path)
        list(table_rows(path, "t"))
        dump(path, "t")
        with open(path, "rb") as fh:
            after = hashlib.sha256(fh.read()).hexdigest()
        self.assertEqual(before, after)


class ModulePreambleTests(unittest.TestCase):
    def test_module_opens_read_only(self):
        # The only file open mode used by the reader must be "rb".
        source = inspect_source()
        self.assertNotIn('open(path, "r")', source)
        self.assertNotIn("open(path, 'r')", source)
        self.assertNotIn("open(path, \"w\")", source)


def inspect_source():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "sqlite_reader.py"), "r", encoding="utf-8") as fh:
        return fh.read()


if __name__ == "__main__":
    unittest.main()
