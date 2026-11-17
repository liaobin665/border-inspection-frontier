"""Metric robustness for the dual-objective feasibility run.

The raw "refused specimens" metric is dominated by a handful of extreme
declarations (one 1.075M parasitic-wasp row is 31% of refused specimens in
2019-2022). This script asks whether the dual-objective trade-off survives
when that domination is removed.

Variants
--------
raw            : q as declared
cap_X          : q_eff = min(q, X), used in both the ranking score and the metric
drop_gt_1e6    : drop rows declaring > 1e6 from the sample entirely
drop_top1      : drop the single largest refused row
live_raw       : description == 'LIV'
live_cap_1e4   : live, q capped at 10,000
live_drop_gt_1e6: live, drop rows > 1e6

For each variant: at 5/10/20% of rows, capture of refusals and of refused
specimens under rule p (probability) and rule p*q (expected refused specimens).

Outputs: results/p5b_metric_robustness.json, results/p5b_summary.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data_layer as p5  # noqa: E402

OUT = HERE.parent / "results"


def captures(score: np.ndarray, refu: np.ndarray, spec: np.ndarray) -> dict:
    order = np.argsort(-score, kind="stable")
    cr = np.cumsum(refu[order]) / max(float(refu.sum()), 1.0)
    cs = np.cumsum(spec[order]) / max(float(spec.sum()), 1.0)
    out = {}
    for k in (0.05, 0.10, 0.20):
        m = max(int(round(k * len(order))), 1)
        out[f"{int(k*100)}%"] = dict(cap_refu=float(cr[m - 1]),
                                     cap_spec=float(cs[m - 1]), rows=m)
    return out


def run_variant(d, tag: str) -> dict:
    p = d["p"].to_numpy()
    refu = d["refu"].to_numpy().astype(float)
    spec = d["spec"].to_numpy()
    return dict(
        tag=tag, rows=int(len(d)), refu=int(refu.sum()),
        refused_qty=float(spec.sum()),
        p=captures(p, refu, spec),
        pq=captures(p * d["q_eff"].to_numpy(), refu, spec))


def concentration(d: pd.DataFrame) -> dict:
    q = np.sort(d.loc[d["refu"] == 1, "q_eff"].to_numpy())[::-1]
    tot = q.sum()
    return dict(
        refused_rows=int(len(q)), refused_qty=float(tot),
        top1=float(q[:1].sum() / tot) if tot else None,
        top10=float(q[:10].sum() / tot) if tot else None,
        top100=float(q[:100].sum() / tot) if tot else None,
        gt1e4_share=float(q[q > 1e4].sum() / tot) if tot else None)


def main() -> None:
    df = p5.pull()
    df, _ = p5.build_features(df)
    model, _ = p5.fit_model(df)
    test = df[df["y"].between(*p5.TEST_YEARS)].copy()
    test["p"] = model.predict_proba(test[p5.FEATS])[:, 1]
    test["q_eff"] = test["qty"]
    test["spec"] = test["refused_qty"]

    res: dict = {}
    res["concentration_raw"] = concentration(test)

    res["raw"] = run_variant(test, "raw")
    for cap in (1e6, 1e5, 1e4, 1e3):
        d = test.copy()
        d["q_eff"] = np.minimum(d["qty"], cap)
        d["spec"] = np.where(d["refu"] == 1, d["q_eff"], 0.0)
        res[f"cap_{int(cap):d}"] = run_variant(d, f"cap_{int(cap)}")
        res[f"cap_{int(cap):d}"]["concentration"] = concentration(d)

    d = test[test["qty"] <= 1e6].copy()
    res["drop_gt_1e6"] = run_variant(d, "drop_gt_1e6")
    res["drop_gt_1e6"]["concentration"] = concentration(d)

    top = test.loc[test["refu"] == 1].nlargest(1, "qty")
    if len(top):
        d = test.drop(top.index)
        res["drop_top1_refused"] = run_variant(d, "drop_top1_refused")
        res["drop_top1_refused"]["dropped"] = dict(
            species=str(top.iloc[0]["sp"]), qty=float(top.iloc[0]["qty"]),
            year=int(top.iloc[0]["y"]))

    live = test[test["desc_"] == "LIV"].copy()
    res["live_raw"] = run_variant(live, "live_raw")
    res["live_raw"]["concentration"] = concentration(live)

    d = live.copy()
    d["q_eff"] = np.minimum(d["qty"], 1e4)
    d["spec"] = np.where(d["refu"] == 1, d["q_eff"], 0.0)
    res["live_cap_1e4"] = run_variant(d, "live_cap_1e4")
    res["live_cap_1e4"]["concentration"] = concentration(d)

    d = live[live["qty"] <= 1e6].copy()
    res["live_drop_gt_1e6"] = run_variant(d, "live_drop_gt_1e6")
    res["live_drop_gt_1e6"]["concentration"] = concentration(d)

    (OUT / "p5b_metric_robustness.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    print("wrote p5b_metric_robustness.json")


if __name__ == "__main__":
    main()
