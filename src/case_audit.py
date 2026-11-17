"""Appendix B: trace six purposively chosen shipments back to raw LEMIS rows.

The six cases are fixed in this file rather than picked mechanically. They
cover small/large x refused/cleared (2x2) and all six are CITES Appendix II
taxa with an IUCN Red List status, so the audit table also carries the
conservation stakes that motivate the paper. B5 is above the 10,000-animal
cap used in the main analysis, which keeps the capping note visible.

For every case the script prints the raw fields and re-derives the three
variables the paper uses (total quantity, refusal flag, refused animals), then
compares them with the analysis table produced by p5.pull().

Usage: python p6_case_audit.py
Output: results/p6_case_audit.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data_layer as p5  # noqa: E402

ROOT = HERE.parent
OUT = ROOT / "results"
CAP = 1e4
WHOLE = ("LIV", "BOD", "DEA", "TRO", "SKE", "FIG")

RAW_SQL = """
select control_number, coalesce(nullif(corrected,''), species) as sp,
       species, quantity, unit, description, country_origin, port,
       shipment_date, sYear, purpose, import_export, action, disposition,
       disposition_date, group_
from lemis
where import_export='I' and unit='NO' and try_cast(quantity as double)>0
  and purpose='T'
  and try_cast(sYear as int) between 2019 and 2022
  and coalesce(nullif(corrected,''), species) is not null
  and trim(coalesce(nullif(corrected,''), species)) <> ''
  and upper(coalesce(nullif(corrected,''), species)) not like '%UNKNOWN%'
  and description in ('LIV','BOD','DEA','TRO','SKE','FIG')
"""


def main():
    con = duckdb.connect(str(p5.LEMIS), read_only=True)
    raw = con.execute(RAW_SQL).df()
    con.close()
    raw["qty"] = raw["quantity"].astype(float)
    raw["refu"] = (raw["action"] == "R").astype(int)

    g = (raw.groupby("control_number", observed=True)
            .agg(rows=("qty", "size"), qty=("qty", "sum"),
                 refu=("refu", "max"), refused_qty=("qty", lambda s: 0.0),
                 year=("sYear", "first"), port=("port", "first"),
                 origin=("country_origin", "first"),
                 date=("shipment_date", "first")).reset_index())
    # refused quantity must be summed over refused rows only
    ref = (raw[raw["refu"] == 1].groupby("control_number", observed=True)["qty"]
           .sum().rename("refused_qty").reset_index())
    g = g.drop(columns=["refused_qty"]).merge(ref, on="control_number", how="left")
    g["refused_qty"] = g["refused_qty"].fillna(0.0)
    g["year"] = g["year"].astype(int)

    # Deliberate selection: small/large x refused/released, 2x2 design.
    # B1 small refused: 1 live Centrochelys sulcata (EN), San Diego 2019-06-02.
    # B2 small cleared: 2 rows / 2 live Testudo hermanni (VU), Miami 2020-02-18.
    # B3 small refused: 4 live Ambystoma mexicanum (CR), Houston 2019-08-07.
    # B4 large refused: 1,260 live Podocnemis unifilis (VU), LA 2020-08-23.
    # B5 large cleared: 10,397 dried Hippocampus ingens (VU), San Diego 2020-12-29.
    # B6 large refused: 10,000 live Podocnemis unifilis (VU), LA 2020-08-23.
    cases_spec = [
        ("small", "2019557007"),
        ("small", "2020709217"),
        ("small", "2019653379"),
        ("large", "2020776900"),
        ("large", "2020832787"),
        ("large", "2020777183"),
    ]
    chosen = []
    for size, cid in cases_spec:
        hit = g[g["control_number"].astype(str) == cid]
        if hit.empty:
            raise SystemExit(f"case {cid} missing from the raw data")
        qty = float(hit["qty"].iloc[0])
        is_large = 1000 <= qty
        if (size == "large") != is_large:
            raise SystemExit(f"case {cid} does not match its size class")
        chosen.append((size, cid))
    if not any(g.set_index("control_number").loc[c, "qty"] > CAP
               for _s, c in chosen):
        print("WARNING: no case above the cap was available")

    # reference: the analysis table, aggregated the same way
    df = p5.pull()
    df = df[df["y"].between(2019, 2022) & df["desc_"].isin(WHOLE)]
    ana = (df[df["control_number"].notna()]
           .groupby("control_number", observed=True)
           .agg(rows=("qty", "size"), qty=("qty", "sum"),
                refu=("refu", "max")).reset_index())
    ana["qty_capped"] = ana["qty"].clip(upper=CAP)
    ana["spec_capped"] = ana["qty_capped"] * ana["refu"]

    cases = []
    for size, cid in chosen:
        rows = raw[raw["control_number"] == cid].sort_values("sp", kind="stable")
        q = float(rows["qty"].sum())
        refu = int(rows["refu"].max())
        a = ana[ana["control_number"] == cid]
        cases.append(dict(
            size=size, control_number=str(cid), rows=int(len(rows)),
            qty_raw=q, qty_capped=min(q, CAP), refused=bool(refu),
            refused_qty=float(rows.loc[rows["refu"] == 1, "qty"].sum()),
            refused_animals_capped=min(q, CAP) if refu else 0.0,
            year=int(rows["sYear"].iloc[0]), date=str(rows["shipment_date"].iloc[0]),
            port=str(rows["port"].iloc[0]), origin=str(rows["country_origin"].iloc[0]),
            match_analysis=bool(len(a) == 1
                                and int(a["rows"].iloc[0]) == len(rows)
                                and abs(float(a["qty"].iloc[0]) - q) < 1e-6
                                and int(a["refu"].iloc[0]) == refu),
            detail=[dict(species=str(r["sp"]), quantity=str(r["quantity"]),
                         description=str(r["description"]),
                         action=str(r["action"]), disposition=str(r["disposition"]),
                         disposition_date=str(r["disposition_date"]))
                    for _, r in rows.iterrows()]))

    bad = [c["control_number"] for c in cases if not c["match_analysis"]]
    if bad:
        raise SystemExit("PREMISE FAIL: raw rows and analysis table disagree for "
                         + ", ".join(bad))

    out = dict(note="six shipments, raw LEMIS rows, whole-animal scope, test 2019-2022",
               cases=cases,
               all_match=all(c["match_analysis"] for c in cases))
    (OUT / "p6_case_audit.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote p6_case_audit.json", len(cases), "cases")

    for c in cases:
        print(f"{c['size']:5s} {c['control_number']:>12s} rows={c['rows']} "
              f"qty={c['qty_raw']:.0f} refu={c['refused']}")


if __name__ == "__main__":
    main()
