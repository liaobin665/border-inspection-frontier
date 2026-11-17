"""Load a public LEMIS CSV extract into a DuckDB table named `lemis`.

    python src/build_db.py path/to/lemis.csv [data/lemis.duckdb]

Every column is loaded as text; the analysis scripts cast types in SQL, so a
plain CSV export of the public LEMIS records is enough. See docs/DATA.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "lemis.csv"
    db = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "data" / "lemis.duckdb"
    if not src.exists():
        sys.exit(f"missing input: {src}")
    db.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db))
    con.execute(
        "create or replace table lemis as "
        "select * from read_csv_auto(?, all_varchar = true, sample_size = -1)",
        [str(src)],
    )
    n = con.execute("select count(*) from lemis").fetchone()[0]
    con.close()
    print(f"{n:,} rows -> {db}")


if __name__ == "__main__":
    main()
