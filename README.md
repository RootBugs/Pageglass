# Pageglass — Read SQLite Files Without sqlite3

A **read-only SQLite database reader** built entirely on `struct` and raw file
bytes: it parses the 100-byte database header, walks B-tree pages, decodes
varints and record serial types, reassembles overflow payloads, and exposes the
schema, table rows and index order — without ever opening `sqlite3`.

**Why it's special:** SQLite's on-disk format is notoriously fiddly, and this
project implements the parts real readers get wrong: the 9-byte varint where
the *9th byte contributes all 8 bits* (SQLite's documented quirk), serial
types 5/6 occupying **6/8** bytes (not 5/6), the payload-overflow formula
`M = ((U-12)*32//255)-23`, `K = M + ((P-M) % (U-4))` that decides how many
payload bytes stay on the page versus spilling to an overflow chain, index
cells' stricter `((U-12)*64//255)-23` limit, page-1's 100-byte header offset,
`PRAGMA page_size=65536` stored as the field value `1`, and negative rowids
stored as unsigned 9-byte patterns. All of it is verified against fixture
databases produced by the real `sqlite3` module at page sizes 512, 1024, 4096
and 65536.

## Architecture

```
file bytes
  │
  ├─ read_header()      100-byte header → DatabaseHeader (magic, page_size
  │                     [field 1 ⇒ 65536], reserved, encoding, counters)
  ├─ decode_varint()    big-endian varint, 1–9 bytes, full 8 bits in byte 9
  ├─ decode_record()    header varint + serial types → Python values
  │                       0 null · 1–6 signed ints (1/2/3/4/6/8 bytes)
  │                       7 float64 · 8 → 0 · 9 → 1
  │                       ≥12: even blob / odd text of (t-12)//2 bytes
  ├─ parse_create_table()  column names, PRIMARY KEY, WITHOUT ROWID,
  │                       INTEGER PRIMARY KEY rowid alias
  └─ Database
        ├─ walk(root)        recursive B-tree generator: interior table
        │                    (4-byte child + key) and index cells yield
        │                    left-child-first, leaf cells yield rows
        ├─ _local_length()   on-page payload bytes vs overflow chain
        ├─ read_payload()    follows u32-BE next-page pointers (0 = end)
        ├─ schema()          walks the sqlite_master table (page 1 root)
        └─ table_rows() / index_scan() / dump()   public readers
```

Module-level helpers `schema(path)`, `table_rows(path, table)`,
`index_scan(path, index)` and `dump(path, table, limit=None)` open the file
read-only (`"rb"` only) and return plain tuples, so results compare directly
against `sqlite3` output.

## Usage example (runnable)

Save as `demo.py` next to `sqlite_reader.py`, run `python demo.py`:

```python
import os, sqlite3, sys
sys.path.insert(0, os.getcwd())
from sqlite_reader import schema, table_rows, index_scan, dump

demo = os.path.join(os.environ["TEMP"], "demo2.sqlite")
if os.path.exists(demo):
    os.remove(demo)
con = sqlite3.connect(demo)
con.execute("CREATE TABLE artist(id INTEGER PRIMARY KEY, name TEXT, country TEXT)")
con.execute("CREATE INDEX idx_artist_country ON artist(country)")
con.executemany("INSERT INTO artist(name, country) VALUES (?, ?)", [
    ("Bjork", "IS"), ("Fela Kuti", "NG"), ("Ryuichi Sakamoto", "JP"),
    ("Caetano Veloso", "BR"), ("Nusrat Fateh Ali Khan", "PK"),
])
con.commit(); con.close()

for row in schema(demo):
    print("schema:", row[:3])
for row in table_rows(demo, "artist"):
    print("row:", row)
print("index scan (country, rowid):")
for row in index_scan(demo, "idx_artist_country"):
    print("  ", row)
os.remove(demo)
```

Output:

```
schema: ('table', 'artist', 'artist')
schema: ('index', 'idx_artist_country', 'artist')
row: (1, 'Bjork', 'IS')
row: (2, 'Fela Kuti', 'NG')
row: (3, 'Ryuichi Sakamoto', 'JP')
row: (4, 'Caetano Veloso', 'BR')
row: (5, 'Nusrat Fateh Ali Khan', 'PK')
index scan (country, rowid):
   ('BR', 4)
   ('IS', 1)
   ('JP', 3)
   ('NG', 2)
   ('PK', 5)
```

The CLI reads the same code paths:

```
> python sqlite_reader.py schema demo.sqlite
table	artist	artist	CREATE TABLE artist(id INTEGER PRIMARY KEY, name TEXT, country TEXT)

> python sqlite_reader.py dump demo.sqlite artist
id	name	country
1	Bjork	IS
2	Fela Kuti	NG
3	Ryuichi Sakamoto	JP
4	Caetano Veloso	BR

> python sqlite_reader.py dump demo.sqlite artist --limit 2
id	name	country
1	Bjork	IS
2	Fela Kuti	NG
```

## Running tests

From this directory:

```
python -m unittest discover -s tests -v
```

44 tests cover header parsing (bad magic, non-power-of-two page size, 64K
page-size encoding, truncation), varints (1/2/9-byte, high-bit quirk,
truncation), record decoding (all serial types, signed widths, UTF-16
text), CREATE TABLE parsing (rowid alias, table-level PK, WITHOUT ROWID,
quoted columns), full-table scans compared row-for-row against `sqlite3`
(200 rows, multi-level B-trees, overflow payloads, negative rowids, unicode
and NULLs, WITHOUT ROWID ordering), index scans compared against
`ORDER BY`, read-only guarantees (SHA-256 before/after), and the CLI
(schema/dump/limit/error paths).

## License

MIT — see `LICENSE`.
