"""Figure-level aggregates that the paper figures need but p6_deep does not store.

- band shares of declarations, refused shipments and refused animals (test window)
- cumulative concentration curve of refused animals over refused rows

Outputs results/p6_figdata.json.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results"
LEMIS = Path(os.environ.get("LEMIS_DB", str(HERE.parent / "data" / "lemis.duckdb")))
WHOLE = ("LIV", "BOD", "DEA", "TRO", "SKE", "FIG")
BASE = """
  import_export='I' and unit='NO' and try_cast(quantity as double)>0
  and purpose={p} and try_cast(sYear as int) between {y0} and {y1}
  and coalesce(nullif(corrected,''), species) is not null
  and trim(coalesce(nullif(corrected,''), species)) <> ''
  and upper(coalesce(nullif(corrected,''), species)) not like '%UNKNOWN%'
  and description in {w}
"""
BANDS = ("1-10", "11-100", "101-1000", ">1000")


def q(sql: str) -> pd.DataFrame:
    con = duckdb.connect(str(LEMIS), read_only=True)
    d = con.execute(sql).df()
    con.close()
    return d


def band_table(purpose: str, y0: int, y1: int) -> dict:
    sql = f"""
    select case when try_cast(quantity as double)<=10 then '1-10'
                when try_cast(quantity as double)<=100 then '11-100'
                when try_cast(quantity as double)<=1000 then '101-1000'
                else '>1000' end as band,
           count(*) as rows,
           sum(case when action='R' then 1 else 0 end) as refu,
           sum(case when action='R' then try_cast(quantity as double) else 0 end) as refused_qty
    from lemis
    where {BASE.format(p=repr(purpose), y0=y0, y1=y1, w=WHOLE)}
    group by 1
    """
    d = q(sql).set_index("band").reindex(list(BANDS))
    tot = d.sum()
    out = {"total": {k: float(tot[k]) for k in ("rows", "refu", "refused_qty")}}
    out["total"]["rate"] = float(tot["refu"] / max(tot["rows"], 1))
    out["bands"] = []
    for b in BANDS:
        r = d.loc[b]
        out["bands"].append(dict(
            band=b, rows=float(r["rows"]), refu=float(r["refu"]),
            refused_qty=float(r["refused_qty"]),
            rate=float(r["refu"] / max(r["rows"], 1)),
            share_rows=float(r["rows"] / tot["rows"]),
            share_refu=float(r["refu"] / tot["refu"]),
            share_animals=float(r["refused_qty"] / tot["refused_qty"])))
    return out


def concentration_curve(purpose: str, y0: int, y1: int) -> dict:
    sql = f"""
    select try_cast(quantity as double) as quantity
    from lemis
    where {BASE.format(p=repr(purpose), y0=y0, y1=y1, w=WHOLE)}
      and action='R'
    """
    x = np.sort(q(sql)["quantity"].to_numpy())[::-1]
    n, total = len(x), x.sum()
    cum = np.cumsum(x) / total
    share_rows = np.arange(1, n + 1) / n

    # log-spaced ranks keep the curve readable across four orders of magnitude
    ranks = sorted({1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 800,
                    1000, 1500, n} & set(range(1, n + 1)))
    idx = np.linspace(0, n - 1, 400).round().astype(int)
    grid = np.unique(np.concatenate([np.array(ranks, dtype=int) - 1, idx]))

    def at(k: int) -> float:
        return float(cum[k - 1]) if 1 <= k <= n else float("nan")

    return dict(
        n=n, total=float(total), max_row=float(x.max()),
        top1=at(1), top10=at(10), top100=at(100),
        share_rows_for_50pct=float(np.searchsorted(cum, 0.50) + 1) / n,
        share_rows_for_90pct=float(np.searchsorted(cum, 0.90) + 1) / n,
        curve=dict(
            share_rows=[float(share_rows[i]) for i in grid],
            share_animals=[float(cum[i]) for i in grid],
            rank=[int(i + 1) for i in grid]))


def main() -> None:
    out = {
        "T_test_bands": band_table("T", 2019, 2022),
        "P_test_bands": band_table("P", 2019, 2022),
        "T_test_concentration": concentration_curve("T", 2019, 2022),
    }
    (OUT / "p6_figdata.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    c = out["T_test_concentration"]
    print(f"refused rows {c['n']}, animals {c['total']:,.0f}")
    print(f"top1 {c['top1']:.3f} top10 {c['top10']:.3f} top100 {c['top100']:.3f}")
    print(f"rows for 50% {c['share_rows_for_50pct']:.4%} for 90% {c['share_rows_for_90pct']:.4%}")


if __name__ == "__main__":
    main()
