"""Sankey input table: origin country x port x shipment-size band.

No figure is drawn here. Kimi asked for the aggregation first so the column
count, the top-N lists and the "by ticket vs by animal" question can be
decided before any layout work.

Scope follows the main result exactly: LEMIS imports, unit='NO',
purpose='T', 2019-2022, quantity>0, valid species, whole-animal
description (LIV/BOD/DEA/TRO/SKE/FIG), refused rows only
(action='R'), animals capped at 1e4 per row.

Flow value  : refused animals, capped at 1e4 per row.
Flow count  : refused rows (the analysis unit; a control_number shipment
              can carry several rows).

Outputs
-------
results/p6_sankey.json
results/p6_sankey.json (the only input of the Sankey figure)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
OUT = PAPER / "results"
ROOT = PAPER.parents[1]
LEMIS = Path(os.environ.get("LEMIS_DB", str(ROOT / "data" / "lemis.duckdb")))
CODES = ROOT / "data" / "codes.csv"

WHOLE = ("LIV", "BOD", "DEA", "TRO", "SKE", "FIG")
CAP = 1e4
YEARS = (2019, 2022)
BAND_LABEL = {1: "1–10", 2: "11–100", 3: "101–1000", 4: ">1000"}

def code_names(field: str) -> dict[str, str]:
    """Country / port labels from the LEMIS code table shipped with the data."""
    if not CODES.exists():
        return {}
    d = pd.read_csv(CODES)
    d = d[d["field"].astype(str).str.lower() == field]
    return dict(zip(d["code"].astype(str), d["description"].astype(str)))


def pull() -> pd.DataFrame:
    con = duckdb.connect(str(LEMIS), read_only=True)
    q = f"""
    select country_origin as org, port, description as desc_,
           try_cast(quantity as double) as qty,
           try_cast(sYear as int) as y,
           case when action='R' then 1 else 0 end as refu
    from lemis
    where import_export='I' and unit='NO' and purpose='T'
      and try_cast(quantity as double) > 0
      and try_cast(sYear as int) between {YEARS[0]} and {YEARS[1]}
      and coalesce(nullif(corrected,''), species) is not null
      and trim(coalesce(nullif(corrected,''), species)) <> ''
      and upper(coalesce(nullif(corrected,''), species)) not like '%UNKNOWN%'
      and description in {WHOLE}
    """
    df = con.execute(q).df()
    con.close()
    df["qty"] = df["qty"].astype(float)
    df["org"] = df["org"].fillna("NA").astype(str).str.strip()
    df["port"] = df["port"].fillna("NA").astype(str).str.strip().str.upper()
    df["desc_"] = df["desc_"].astype(str)
    df["band"] = np.select([df["qty"] <= 10, df["qty"] <= 100, df["qty"] <= 1000],
                           [1, 2, 3], default=4)
    return df


def _shares(s: pd.Series) -> pd.Series:
    return s / s.sum() if s.sum() else s * np.nan


def by_key(d: pd.DataFrame, key: str, names: dict[str, str] | None = None) -> pd.DataFrame:
    g = (d.groupby(key, observed=True)
           .agg(refu_rows=("refu", "size"),
                cap_animals=("animals", "sum"),
                raw_animals=("raw_animals", "sum"))
           .reset_index())
    g["share_ticket"] = _shares(g["refu_rows"])
    g["share_animal"] = _shares(g["cap_animals"])
    g["rank_ticket"] = g["refu_rows"].rank(ascending=False, method="min").astype(int)
    g["rank_animal"] = g["cap_animals"].rank(ascending=False, method="min").astype(int)
    if names is not None:
        g["name"] = g[key].map(names).fillna(g[key])
    return g.sort_values("cap_animals", ascending=False).reset_index(drop=True)


def coverage(g: pd.DataFrame) -> dict:
    """How many units (countries/ports) carry 50/80/90% of refused animals."""
    s = g["cap_animals"].to_numpy()
    s = np.sort(s)[::-1]
    cum = np.cumsum(s) / s.sum()
    out = {}
    for thr in (0.5, 0.8, 0.9):
        out[f"n_units_{int(thr*100)}"] = int(np.searchsorted(cum, thr) + 1)
    out["n_units"] = int(len(s))
    out["top1_share_animal"] = float(s[0] / s.sum())
    out["top5_share_animal"] = float(s[:5].sum() / s.sum())
    out["top10_share_animal"] = float(s[:10].sum() / s.sum())
    return out


def spearman(a: pd.Series, b: pd.Series) -> float:
    """Rank correlation without scipy: Pearson on average ranks."""
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def top_overlap(g: pd.DataFrame, n: int = 10) -> float:
    by_a = set(g.sort_values("cap_animals", ascending=False).head(n).iloc[:, 0])
    by_t = set(g.sort_values("refu_rows", ascending=False).head(n).iloc[:, 0])
    return len(by_a & by_t) / n


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _num(x: float) -> str:
    return f"{x:,.0f}"


def main() -> None:
    df = pull()
    df["animals"] = np.where(df["refu"] == 1, np.minimum(df["qty"], CAP), 0.0)
    df["raw_animals"] = np.where(df["refu"] == 1, df["qty"], 0.0)
    ref = df[df["refu"] == 1].copy()

    cn = code_names("country")
    cn["Ctry_XX"] = "Unknown origin"
    pn = code_names("port")
    pn["XX"] = "Unknown"
    pn["NA"] = "Not recorded"
    countries = by_key(ref, "org", cn)
    ports = by_key(ref, "port", {k: f"{pn.get(k, k)} ({k})" for k in ref["port"].unique()})
    bands = by_key(ref, "band")
    bands["label"] = bands["band"].map(BAND_LABEL)

    flows = (ref.groupby(["org", "port", "band"], observed=True)
                .agg(cap_animals=("animals", "sum"), refu_rows=("refu", "size"))
                .reset_index())
    tot = float(flows["cap_animals"].sum())
    fragility = {
        "share_animals_flows_1row": float(flows.loc[flows["refu_rows"] <= 1, "cap_animals"].sum() / tot),
        "share_animals_flows_2row": float(flows.loc[flows["refu_rows"] <= 2, "cap_animals"].sum() / tot),
        "share_animals_flows_5row": float(flows.loc[flows["refu_rows"] <= 5, "cap_animals"].sum() / tot),
        "median_rows_per_flow": float(flows["refu_rows"].median()),
    }
    flows["org_name"] = flows["org"].map(cn).fillna(flows["org"])
    flows["port_name"] = flows["port"].map(pn).fillna(flows["port"])
    flows["band_label"] = flows["band"].map(BAND_LABEL)
    flows = flows.sort_values("cap_animals", ascending=False)
    top_flows = flows.head(30)

    out = {
        "scope": {
            "years": list(YEARS), "cap": CAP, "bands": BAND_LABEL,
            "rows_all": int(len(df)),
            "refused_rows": int(len(ref)),
            "refused_animals_raw": float(ref["raw_animals"].sum()),
            "refused_animals_capped": float(ref["animals"].sum()),
            "n_countries": int(ref["org"].nunique()),
            "n_ports": int(ref["port"].nunique()),
            "n_flows": int(len(flows)),
        },
        "bands": bands.to_dict("records"),
        "countries": countries.to_dict("records"),
        "ports": ports.to_dict("records"),
        "flows_top": top_flows.to_dict("records"),
        "flows_all": flows.to_dict("records"),
        "concentration": {
            "countries": coverage(countries),
            "ports": coverage(ports),
            "flows": coverage(flows.rename(columns={"org": "key"})),
        },
        "flow_fragility": fragility,
        "two_worlds": {
            "country_spearman": spearman(countries["cap_animals"], countries["refu_rows"]),
            "country_top10_overlap": top_overlap(countries, 10),
            "port_spearman": spearman(ports["cap_animals"], ports["refu_rows"]),
            "port_top10_overlap": top_overlap(ports, 10),
            "band_share_ticket": bands.set_index("label")["share_ticket"].to_dict(),
            "band_share_animal": bands.set_index("label")["share_animal"].to_dict(),
            "ticket_rank_country": countries.sort_values("refu_rows", ascending=False)
                                    .head(10)[["org", "name", "refu_rows", "share_ticket"]]
                                    .to_dict("records"),
            "animal_rank_country": countries.head(10)[
                ["org", "name", "cap_animals", "share_animal"]].to_dict("records"),
        },
    }
    ref_main = json.loads((OUT / "p6_main.json").read_text(encoding="utf-8"))
    for key, got in (("refu", out["scope"]["refused_rows"]),
                     ("refused_animals", out["scope"]["refused_animals_capped"])):
        want = ref_main[key]
        if abs(got - want) > 1e-6 * max(1.0, abs(want)):
            raise SystemExit(f"scope mismatch vs p6_main.json: {key} {got} != {want}")

    OUT.mkdir(exist_ok=True)
    (OUT / "p6_sankey.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("written", OUT / "p6_sankey.json")
    print(json.dumps(out["scope"], ensure_ascii=False))
    print("country coverage", out["concentration"]["countries"])
    print("port coverage", out["concentration"]["ports"])


if __name__ == "__main__":
    main()
