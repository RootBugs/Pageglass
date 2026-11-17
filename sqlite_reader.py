"""sqlite-reader: a read-only SQLite database file parser in pure Python.

Reads the SQLite 3 on-disk format directly from a file -- header, schema,
table b-trees, index b-trees, records and overflow pages -- without using
the sqlite3 module and without ever opening the file for writing.

Usage:
    python sqlite_reader.py schema DATABASE
    python sqlite_reader.py dump DATABASE TABLE [--limit N]
"""

from __future__ import annotations

import argparse
import re
import struct
import sys
from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple

MAGIC = b"SQLite format 3\x00"

# Page types.
PAGE_INTERIOR_INDEX = 0x02
PAGE_INTERIOR_TABLE = 0x05
PAGE_LEAF_INDEX = 0x0A
PAGE_LEAF_TABLE = 0x0D

ENCODINGS = {1: "utf-8", 2: "utf-16-le", 3: "utf-16-be"}


class SQLiteFormatError(Exception):
    """Raised when a file is not a valid or supported SQLite database."""


@dataclass(frozen=True)
class DatabaseHeader:
    """Decoded fields of the 100-byte SQLite database header."""

    magic: bytes
    page_size: int
    reserved_space: int
    file_change_counter: int
    database_size_pages: int
    freelist_trunk_page: int
    freelist_page_count: int
    schema_cookie: int
    text_encoding: str
    user_version: int
    incremental_vacuum: int
    application_id: int
    version_valid_for: int
    sqlite_version: int

    @property
    def usable_size(self) -> int:
        """Bytes on each page available for content (page minus reserved)."""
        return self.page_size - self.reserved_space


@dataclass(frozen=True)
class TableInfo:
    """Schema facts about a table needed to interpret its records."""

    name: str
    columns: Tuple[str, ...]
    primary_key: Tuple[str, ...]
    without_rowid: bool
    rowid_alias: Optional[str]
    sql: str


# ---------------------------------------------------------------------------
# Varints
# ---------------------------------------------------------------------------

def decode_varint(data: bytes, offset: int = 0) -> Tuple[int, int]:
    """Decode a SQLite big-endian varint at ``offset``.

    Returns ``(value, bytes_consumed)``.  The value is returned as the raw
    unsigned 64-bit pattern; callers interpret signedness themselves.
    """
    value = 0
    for i in range(9):
        if offset + i >= len(data):
            raise SQLiteFormatError("truncated varint at offset %d" % offset)
        byte = data[offset + i]
        if i == 8:
            value = (value << 8) | byte
            return value, 9
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, i + 1
    raise SQLiteFormatError("unterminated varint at offset %d" % offset)


def _signed64(value: int) -> int:
    """Reinterpret a 64-bit unsigned pattern as a signed integer."""
    if value >= 1 << 63:
        return value - (1 << 64)
    return value


def _signed_varint(data: bytes, offset: int = 0) -> Tuple[int, int]:
    raw, size = decode_varint(data, offset)
    return _signed64(raw), size


# ---------------------------------------------------------------------------
# Database header
# ---------------------------------------------------------------------------

def read_header(data: bytes) -> DatabaseHeader:
    """Parse the 100-byte database header from the start of ``data``."""
    if len(data) < 100:
        raise SQLiteFormatError("file too small to be a SQLite database")
    magic = data[:16]
    if magic != MAGIC:
        raise SQLiteFormatError("bad magic: not a SQLite 3 database file")
    page_size = struct.unpack(">H", data[16:18])[0]
    if page_size == 1:
        page_size = 65536
    if page_size < 512 or page_size & (page_size - 1):
        raise SQLiteFormatError("invalid page size: %d" % page_size)
    reserved = data[20]
    if reserved >= page_size:
        raise SQLiteFormatError("invalid reserved space: %d" % reserved)
    encoding_id = struct.unpack(">I", data[56:60])[0]
    encoding = ENCODINGS.get(encoding_id)
    if encoding is None:
        # Encoding 0 means "unspecified"; the default is UTF-8.
        encoding = "utf-8" if encoding_id == 0 else None
    if encoding is None:
        raise SQLiteFormatError("unsupported text encoding id: %d" % encoding_id)
    return DatabaseHeader(
        magic=magic,
        page_size=page_size,
        reserved_space=reserved,
        file_change_counter=struct.unpack(">I", data[24:28])[0],
        database_size_pages=struct.unpack(">I", data[28:32])[0],
        freelist_trunk_page=struct.unpack(">I", data[32:36])[0],
        freelist_page_count=struct.unpack(">I", data[36:40])[0],
        schema_cookie=struct.unpack(">I", data[40:44])[0],
        text_encoding=encoding,
        user_version=struct.unpack(">I", data[44:48])[0],
        incremental_vacuum=struct.unpack(">I", data[52:56])[0],
        application_id=struct.unpack(">I", data[68:72])[0],
        version_valid_for=struct.unpack(">I", data[92:96])[0],
        sqlite_version=struct.unpack(">I", data[96:100])[0],
    )


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

def _serial_size(serial_type: int) -> int:
    """Data-section bytes occupied by a value of ``serial_type``."""
    if serial_type in (0, 8, 9):
        return 0
    if 1 <= serial_type <= 6:
        # 1..4 -> 1..4 bytes, 5 -> 6 bytes, 6 -> 8 bytes (signed big-endian)
        return (1, 2, 3, 4, 6, 8)[serial_type - 1]
    if serial_type == 7:
        return 8
    if serial_type >= 12:
        return (serial_type - 12) // 2
    raise SQLiteFormatError("reserved serial type: %d" % serial_type)


def decode_record(payload: bytes, encoding: str = "utf-8") -> List[object]:
    """Decode a record payload into a list of Python values."""
    header_len, used = decode_varint(payload, 0)
    if header_len > len(payload) or header_len < used:
        raise SQLiteFormatError("record header length exceeds payload")
    values: List[object] = []
    cursor = header_len  # data section starts right after the header
    pos = used
    while pos < header_len:
        serial, size = decode_varint(payload, pos)
        pos += size
        if serial in (10, 11):
            raise SQLiteFormatError("reserved serial type: %d" % serial)
        n = _serial_size(serial)
        if cursor + n > len(payload):
            raise SQLiteFormatError("record payload truncated")
        values.append(_decode_single(serial, payload[cursor:cursor + n], encoding))
        cursor += n
    if cursor != len(payload):
        raise SQLiteFormatError(
            "record payload has %d trailing bytes" % (len(payload) - cursor))
    return values


def _decode_single(serial: int, chunk: bytes, encoding: str) -> object:
    if serial == 0:
        return None
    if 1 <= serial <= 6:
        return int.from_bytes(chunk, "big", signed=True)
    if serial == 7:
        return struct.unpack(">d", chunk)[0]
    if serial == 8:
        return 0
    if serial == 9:
        return 1
    if serial >= 12:
        if serial % 2 == 0:
            return bytes(chunk)
        return chunk.decode(encoding, errors="replace")
    raise SQLiteFormatError("reserved serial type: %d" % serial)


# ---------------------------------------------------------------------------
# Minimal SQL schema parsing
# ---------------------------------------------------------------------------

_IDENT = r'"([^"]+)"|`([^`]+)`|\[([^\]]+)\]|([A-Za-z_][A-Za-z0-9_]*)'


def _split_top_level(text: str) -> List[str]:
    """Split ``text`` on commas that are not nested in quotes or parens."""
    parts: List[str] = []
    depth = 0
    quote: Optional[str] = None
    current: List[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if quote is not None:
            current.append(ch)
            if ch == quote:
                # SQL escapes a quote by doubling it.
                if i + 1 < len(text) and text[i + 1] == quote:
                    current.append(text[i + 1])
                    i += 1
                else:
                    quote = None
        elif ch in "'\"`":
            quote = ch
            current.append(ch)
        elif ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
        i += 1
    if current:
        parts.append("".join(current))
    return [p.strip() for p in parts if p.strip()]


def _parse_column_name(fragment: str) -> str:
    m = re.match(r"\s*" + _IDENT, fragment)
    if not m:
        return fragment.strip()
    return next(g for g in m.groups() if g is not None)


def _split_body(sql: str) -> str:
    """Return the parenthesised column list of a CREATE TABLE statement."""
    start = sql.find("(")
    if start < 0:
        raise SQLiteFormatError("malformed CREATE TABLE: %r" % sql)
    depth = 0
    quote: Optional[str] = None
    for i in range(start, len(sql)):
        ch = sql[i]
        if quote is not None:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return sql[start + 1:i]
    raise SQLiteFormatError("unbalanced CREATE TABLE: %r" % sql)


def parse_create_table(sql: str) -> TableInfo:
    """Extract column names, primary key and table flavour from CREATE TABLE."""
    if not sql or not sql.lstrip().upper().startswith("CREATE TABLE"):
        raise SQLiteFormatError("not a CREATE TABLE statement: %r" % sql)
    m = re.search(r'(?i)create\s+table\s+(?:if\s+not\s+exists\s+)?("([^"]+)"|`([^`]+)`|\[([^\]]+)\]|([A-Za-z_][A-Za-z0-9_]*))', sql)
    if not m:
        raise SQLiteFormatError("cannot parse table name: %r" % sql)
    name = next(g for g in m.groups()[1:] if g is not None)
    body = _split_body(sql)
    columns: List[str] = []
    pk_cols: List[str] = []
    without_rowid = bool(re.search(r"(?i)\bwithout\s+rowid\b", sql))
    rowid_alias: Optional[str] = None
    table_pk: Optional[List[str]] = None
    for frag in _split_top_level(body):
        upper = frag.upper()
        if upper.startswith("PRIMARY KEY"):
            inner = frag[frag.find("(") + 1:frag.rfind(")")]
            table_pk = [_parse_column_name(p) for p in _split_top_level(inner)]
            continue
        if upper.startswith(("UNIQUE", "CHECK", "CONSTRAINT", "FOREIGN KEY")):
            continue
        col_name = _parse_column_name(frag)
        columns.append(col_name)
        if re.search(r"(?i)\bprimary\s+key\b", frag):
            pk_cols.append(col_name)
            if (re.search(r"(?i)\binteger\s+primary\s+key\b", frag)
                    and not without_rowid
                    and not re.search(r"(?i)\bautoincrement\b", frag)):
                rowid_alias = col_name
    primary_key = tuple(table_pk) if table_pk else tuple(pk_cols)
    if without_rowid and not primary_key:
        raise SQLiteFormatError("WITHOUT ROWID table without primary key: %r" % sql)
    return TableInfo(
        name=name,
        columns=tuple(columns),
        primary_key=primary_key,
        without_rowid=without_rowid,
        rowid_alias=rowid_alias,
        sql=sql,
    )


# ---------------------------------------------------------------------------
# Database reader
# ---------------------------------------------------------------------------

class Database:
    """A read-only SQLite database file."""

    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as fh:
            self.data = fh.read()
        self.header = read_header(self.data)
        if len(self.data) < self.header.page_size:
            raise SQLiteFormatError("file truncated: expected at least one full page")
        self.page_count = max(1, len(self.data) // self.header.page_size)
        self.encoding = self.header.text_encoding

    # -- page access -------------------------------------------------------

    def page(self, number: int) -> bytes:
        if number < 1 or number > self.page_count:
            raise SQLiteFormatError("page %d out of range (1..%d)" % (number, self.page_count))
        start = (number - 1) * self.header.page_size
        return self.data[start:start + self.header.page_size]

    def page_content(self, number: int) -> bytes:
        """Page bytes excluding the reserved trailer."""
        return self.page(number)[: self.header.usable_size]

    def _page_header_offset(self, page: bytes) -> int:
        return 100 if len(page) >= 100 and page[:16] == MAGIC else 0

    # -- record payload reassembly ----------------------------------------

    def read_payload(self, page_no: int, offset: int, payload_length: int,
                     local_length: int) -> bytes:
        """Read ``payload_length`` bytes starting at ``offset`` on ``page_no``,
        following the overflow chain when the payload does not fit locally."""
        page = self.page_content(page_no)
        local = page[offset:offset + local_length]
        if len(local) < local_length:
            raise SQLiteFormatError("cell payload runs past end of page %d" % page_no)
        payload = bytearray(local)
        if len(payload) >= payload_length:
            return bytes(payload[:payload_length])
        # Overflow pointer sits right after the local payload.
        ptr_at = offset + local_length
        if ptr_at + 4 > len(page):
            raise SQLiteFormatError("missing overflow pointer on page %d" % page_no)
        next_page = struct.unpack(">I", page[ptr_at:ptr_at + 4])[0]
        remaining = payload_length - len(payload)
        seen = set()
        while next_page:
            if next_page in seen or next_page > self.page_count:
                raise SQLiteFormatError("bad overflow chain at page %d" % next_page)
            seen.add(next_page)
            opage = self.page_content(next_page)
            chunk = opage[4:]
            take = min(len(chunk), remaining)
            payload.extend(chunk[:take])
            remaining -= take
            if remaining <= 0:
                break
            next_page = struct.unpack(">I", opage[0:4])[0]
        if remaining > 0:
            raise SQLiteFormatError("overflow chain ended early")
        return bytes(payload[:payload_length])

    # -- b-tree walking ----------------------------------------------------

    @staticmethod
    def _cell_pointers(page: bytes, header_offset: int, page_type: int) -> Tuple[int, int, List[int]]:
        """Return (first cell byte offset, header size, cell pointer list)."""
        if page_type in (PAGE_LEAF_TABLE, PAGE_LEAF_INDEX):
            header_size = 8
        elif page_type in (PAGE_INTERIOR_TABLE, PAGE_INTERIOR_INDEX):
            header_size = 12
        else:
            raise SQLiteFormatError("unknown page type 0x%02X" % page_type)
        cell_count = struct.unpack(">H", page[header_offset + 3:header_offset + 5])[0]
        array_start = header_offset + header_size
        pointers = []
        for i in range(cell_count):
            at = array_start + 2 * i
            if at + 2 > len(page):
                raise SQLiteFormatError("cell pointer array truncated")
            pointers.append(struct.unpack(">H", page[at:at + 2])[0])
        return array_start + 2 * cell_count, header_size, pointers

    def walk(self, root_page: int) -> Iterator[Tuple[Optional[int], List[object]]]:
        """Yield ``(rowid, record_values)`` for every entry in a b-tree.

        ``rowid`` is ``None`` for index b-trees; for table b-trees it is the
        signed rowid of the leaf cell.
        """
        yield from self._walk(root_page)

    def _walk(self, page_no: int) -> Iterator[Tuple[Optional[int], List[object]]]:
        page = self.page_content(page_no)
        if len(page) < 8:
            raise SQLiteFormatError("page %d too small" % page_no)
        header_offset = self._page_header_offset(page)
        page_type = page[header_offset]
        if page_type not in (PAGE_INTERIOR_INDEX, PAGE_INTERIOR_TABLE,
                             PAGE_LEAF_INDEX, PAGE_LEAF_TABLE):
            raise SQLiteFormatError(
                "page %d has invalid b-tree type 0x%02X" % (page_no, page_type))
        if page_type in (PAGE_INTERIOR_INDEX, PAGE_INTERIOR_TABLE):
            right_child = struct.unpack(">I", page[header_offset + 8:header_offset + 12])[0]
        _, _, pointers = self._cell_pointers(page, header_offset, page_type)
        for ptr in pointers:
            if ptr <= 0 or ptr >= len(page):
                raise SQLiteFormatError("cell pointer %d out of range on page %d" % (ptr, page_no))
            if page_type in (PAGE_INTERIOR_INDEX, PAGE_INTERIOR_TABLE):
                child = struct.unpack(">I", page[ptr:ptr + 4])[0]
                if page_type == PAGE_INTERIOR_TABLE:
                    yield from self._walk(child)
                else:
                    # Index interior cell: 4-byte child, then varint payload
                    # length, then the (local part of the) payload.  The entry
                    # itself belongs in-order, before the right subtree.
                    key_len, used_len = decode_varint(page, ptr + 4)
                    cell_offset = ptr + 4 + used_len
                    local = self._local_length(key_len, index=True)
                    payload = self.read_payload(page_no, cell_offset, key_len, local)
                    yield from self._walk(child)
                    yield None, decode_record(payload, self.encoding)
            else:
                if page_type == PAGE_LEAF_TABLE:
                    pay_len, used_len = decode_varint(page, ptr)
                    rowid, used_rowid = _signed_varint(page, ptr + used_len)
                    cell_offset = ptr + used_len + used_rowid
                    local = self._local_length(pay_len, index=False)
                    payload = self.read_payload(page_no, cell_offset, pay_len, local)
                    yield rowid, decode_record(payload, self.encoding)
                else:
                    pay_len, used_len = decode_varint(page, ptr)
                    cell_offset = ptr + used_len
                    local = self._local_length(pay_len, index=True)
                    payload = self.read_payload(page_no, cell_offset, pay_len, local)
                    yield None, decode_record(payload, self.encoding)
        if page_type in (PAGE_INTERIOR_INDEX, PAGE_INTERIOR_TABLE):
            yield from self._walk(right_child)

    def _local_length(self, payload_length: int, index: bool = False) -> int:
        """Bytes of a cell payload stored on the page itself.

        SQLite stores the whole payload when it fits within the space left
        after the cell header: the table-leaf bound ``U - 35``, or the index
        bound ``X = ((U - 12) * 64 // 255) - 23`` for index cells.  Otherwise
        the payload spills to an overflow chain and the local part follows
        SQLite's formula::

            M = ((U - 12) * 32 // 255) - 23
            K = M + ((P - M) % (U - 4))       # K == P when P <= M

        clipped to the applicable limit.  Verified against fixture databases
        produced by the real sqlite3 module (page sizes 512, 1024, 4096 and
        65536; see tests).
        """
        u = self.header.usable_size
        m = ((u - 12) * 32 // 255) - 23
        if payload_length <= m:
            return payload_length
        wrapped = m + ((payload_length - m) % (u - 4))
        if index:
            x = ((u - 12) * 64 // 255) - 23
            if payload_length <= x:
                return payload_length
            return m if wrapped > x else wrapped
        return m if wrapped > u - 35 else wrapped

    # -- schema ------------------------------------------------------------

    def schema(self) -> List[Tuple[str, str, str, Optional[str]]]:
        """Return rows of the sqlite_master-like schema table."""
        root = self._schema_root_page()
        if root is None:
            return []
        rows = []
        for _, values in self.walk(root):
            values = list(values)
            obj_type = values[0] if values else None
            obj_name = values[1] if len(values) > 1 else None
            tbl_name = values[2] if len(values) > 2 else None
            sql = values[4] if len(values) > 4 else None
            rows.append((obj_type, obj_name, tbl_name, sql))
        return rows

    def _schema_root_page(self) -> Optional[int]:
        # The sqlite_schema table is the first entry of the schema b-tree,
        # which lives at page 1 as a leaf/interior starting at offset 100.
        # Instead of guessing, walk page 1's b-tree directly.
        try:
            page = self.page_content(1)
        except SQLiteFormatError:
            return None
        if not page or page[100] not in (PAGE_LEAF_TABLE, PAGE_INTERIOR_TABLE):
            return None
        return 1

    def tables(self) -> List[TableInfo]:
        """Parsed info for every table (and WITHOUT ROWID 'index-style' table)."""
        out: List[TableInfo] = []
        for obj_type, _, _, sql in self.schema():
            if obj_type == "table" and sql:
                out.append(parse_create_table(sql))
        return out

    def table(self, name: str) -> TableInfo:
        for info in self.tables():
            if info.name == name:
                return info
        raise SQLiteFormatError("no such table: %s" % name)

    def _root_page_for(self, name: str) -> int:
        # sqlite_master rows carry the root page as column 3 (records are
        # type, name, tbl_name, rootpage, sql).
        root = self._schema_root_page()
        if root is None:
            raise SQLiteFormatError("cannot read schema")
        for _, values in self.walk(root):
            if len(values) > 3 and values[1] == name:
                rp = values[3]
                if isinstance(rp, int):
                    return rp
        raise SQLiteFormatError("no root page for %s" % name)

    # -- high level --------------------------------------------------------

    def table_rows(self, table: str) -> Iterator[Tuple[object, ...]]:
        """Yield rows of ``table`` in declared column order (rowid tables)."""
        info = self.table(table)
        root = self._root_page_for(table)
        if info.without_rowid:
            stored = list(info.primary_key) + [
                c for c in info.columns if c not in info.primary_key
            ]
            for _, values in self.walk(root):
                mapping = dict(zip(stored, values))
                yield tuple(mapping.get(c) for c in info.columns)
        else:
            alias_index = (
                info.columns.index(info.rowid_alias)
                if info.rowid_alias in info.columns else None
            )
            for rowid, values in self.walk(root):
                row = list(values)
                if alias_index is not None and alias_index < len(row):
                    if row[alias_index] is None:
                        row[alias_index] = rowid
                yield tuple(row)

    def index_scan(self, index: str) -> Iterator[Tuple[object, ...]]:
        """Yield every entry of ``index`` in b-tree order (key columns only,
        including the trailing rowid column for rowid-table indexes)."""
        root = self._root_page_for(index)
        for _, values in self.walk(root):
            yield tuple(values)

    def dump(self, table: str, limit: Optional[int] = None) -> List[Tuple[object, ...]]:
        rows: List[Tuple[object, ...]] = []
        for row in self.table_rows(table):
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
        return rows


# ---------------------------------------------------------------------------
# Module-level convenience API
# ---------------------------------------------------------------------------

def table_rows(path: str, table: str) -> Iterator[Tuple[object, ...]]:
    """Yield rows of ``table`` from the database file at ``path``."""
    return Database(path).table_rows(table)


def index_scan(path: str, index: str) -> Iterator[Tuple[object, ...]]:
    """Yield entries of ``index`` from the database file at ``path``."""
    return Database(path).index_scan(index)


def schema(path: str) -> List[Tuple[str, str, str, Optional[str]]]:
    """Return ``(type, name, tbl_name, sql)`` for every schema object."""
    return Database(path).schema()


def dump(path: str, table: str, limit: Optional[int] = None) -> List[Tuple[object, ...]]:
    """Return up to ``limit`` rows of ``table`` in declared column order."""
    return Database(path).dump(table, limit=limit)


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sqlite_reader",
        description="Read a SQLite database file without the sqlite3 module.",
    )
    sub = parser.add_subparsers(dest="command")
    sub.required = True

    p_schema = sub.add_parser("schema", help="list schema objects")
    p_schema.add_argument("database")

    p_dump = sub.add_parser("dump", help="dump a table as TSV")
    p_dump.add_argument("database")
    p_dump.add_argument("table")
    p_dump.add_argument("--limit", type=int, default=None)

    argv_list = list(argv) if argv is not None else None
    try:
        args = parser.parse_args(argv_list)
    except SystemExit as exc:  # argparse already printed the usage error
        return exc.code if isinstance(exc.code, int) else 2
    try:
        if args.command == "schema":
            for obj_type, name, tbl_name, sql in schema(args.database):
                print("%s\t%s\t%s\t%s" % (obj_type, name, tbl_name, sql))
        else:
            db = Database(args.database)
            info = db.table(args.table)
            print("\t".join(info.columns))
            for row in db.dump(args.table, limit=args.limit):
                print("\t".join("" if v is None else str(v) for v in row))
    except SQLiteFormatError as exc:
        print("sqlite-reader: %s" % exc, file=sys.stderr)
        return 1
    except OSError as exc:
        print("sqlite-reader: %s" % exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
