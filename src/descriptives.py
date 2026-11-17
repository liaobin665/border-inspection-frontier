"""Descriptives for the dual-objective paper (no model).

Outputs results/p6_descriptives.json
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
  and purpose={p} and try_cast(sYear as int) between 2010 and 2022
  and coalesce(nullif(corrected,''), species) is not null
  and trim(coalesce(nullif(corrected,''), species)) <> ''
  and upper(coalesce(nullif(corrected,''), species)) not like '%UNKNOWN%'
"""


def q(sql: str) -> pd.DataFrame:
    con = duckdb.connect(str(LEMIS), read_only=True)
    d = con.execute(sql).df()
    con.close()
    return d


def band_stats(purpose: str, years: str, whole: bool) -> dict:
    extra = f" and description in {WHOLE}" if whole else ""
    pp = "'" + purpose + "'"
    sql = f"""
    select case when try_cast(quantity as double)<=10 then '1-10'
                when try_cast(quantity as double)<=100 then '11-100'
                when try_cast(quantity as double)<=1000 then '101-1000'
                else '>1000' end as band,
           count(*) as rows,
           sum(case when action='R' then 1 else 0 end) as refu,
           sum(case when action='R' then try_cast(quantity as double) else 0 end) as refused_qty,
           median(try_cast(quantity as double)) as median_qty
    from lemis
    where {BASE.format(p=pp)} {extra}
      and try_cast(sYear as int) between {years}
    group by 1 order by 1
    """
    d = q(sql)
    d["rate"] = d["refu"] / d["rows"]
    tot = dict(rows=int(d["rows"].sum()), refu=int(d["refu"].sum()),
               refused_qty=float(d["refused_qty"].sum()))
    tot["rate"] = tot["refu"] / max(tot["rows"], 1)
    return dict(total=tot, by_band=d.to_dict("records"))


def year_table(purpose: str, whole: bool, years="2010 and 2022") -> list:
    extra = f" and description in {WHOLE}" if whole else ""
    pp = "'" + purpose + "'"
    sql = f"""
    select try_cast(sYear as int) as y, count(*) as rows,
           sum(case when action='R' then 1 else 0 end) as refu,
           sum(case when action='R' then try_cast(quantity as double) else 0 end) as refused_qty,
           sum(try_cast(quantity as double)) as declared_qty
    from lemis
    where {BASE.format(p=pp)} {extra}
      and try_cast(sYear as int) between {years}
    group by 1 order by 1
    """
    d = q(sql)
    d["rate"] = d["refu"] / d["rows"]
    return d.to_dict("records")


def main() -> None:
    out = {}
    out["T_all"] = band_stats("T", "2010 and 2022", False)
    out["T_whole"] = band_stats("T", "2010 and 2022", True)
    out["T_whole_test"] = band_stats("T", "2019 and 2022", True)
    out["P_whole_test"] = band_stats("P", "2019 and 2022", True)
    out["years_T_whole"] = year_table("T", True)
    out["years_T_all"] = year_table("T", False)

    # distribution of refused rows by size, whole-animal, test window
    pp = "'T'"
    sql = f"""
    select try_cast(quantity as double) as quantity, action
    from lemis
    where {BASE.format(p=pp)} and description in {WHOLE}
      and try_cast(sYear as int) between 2019 and 2022
    """
    d = q(sql)
    refu = d.loc[d["action"] == "R", "quantity"].to_numpy()
    out["refused_rows_distribution"] = dict(
        n=int(len(refu)),
        share_le10=float((refu <= 10).mean()),
        share_le100=float((refu <= 100).mean()),
        share_gt1000=float((refu > 1000).mean()),
        median=float(np.median(refu)),
        p90=float(np.percentile(refu, 90)),
        p99=float(np.percentile(refu, 99)),
        max=float(refu.max()))
    (OUT / "p6_descriptives.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print("written")


if __name__ == "__main__":
    main()
