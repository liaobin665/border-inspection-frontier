"""Experiments added for the second review round.

X1  quantity_only : rank by declared quantity q alone, next to p and p*q.
X2  cost_scenarios: inspection cost proportional to q**gamma (gamma 0/.5/1);
                    budget expressed as a share of total cost, not of rows.
X3  leave_largest : drop the largest refused rows and recompute the main table.
X4  shipment_mix  : within-shipment species mixing vs refusal (2019-2020).
X5  disposition   : what refusal records say about disposal.

Usage: python p6_revisions.py
Output: results/p6_revisions.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data_layer as p5  # noqa: E402
import analysis as deep  # noqa: E402

OUT = HERE.parent / "results"
WHOLE = deep.WHOLE
CAP = deep.CAP


def top_list(score: np.ndarray, k: float):
    o = np.argsort(-np.asarray(score, float), kind="stable")
    m = max(int(round(k * len(o))), 1)
    return o[:m]


def caps(idx, refu, spec):
    return (float(refu[idx].sum() / max(refu.sum(), 1.0)),
            float(spec[idx].sum() / max(spec.sum(), 1.0)))


def jac(a, b):
    sa, sb = set(a.tolist()), set(b.tolist())
    return len(sa & sb) / len(sa | sb)


def capture_at_cost(score, cost, budget, refu, spec):
    """Capture when spending a fixed share of total inspection cost."""
    o = np.argsort(-np.asarray(score, float), kind="stable")
    cc = np.cumsum(cost[o]) / max(cost.sum(), 1e-12)
    m = int(np.searchsorted(cc, budget, side="right"))
    m = max(m, 1)
    idx = o[:m]
    return (*caps(idx, refu, spec), float(m) / len(o))


def section_x1(p, q, refu, spec):
    k = 0.10
    ip, ipq, iq = top_list(p, k), top_list(p * q, k), top_list(q, k)
    out = {}
    for name, idx in (("p", ip), ("pq", ipq), ("q", iq)):
        c, a = caps(idx, refu, spec)
        out[name] = dict(cap_refu=c, cap_spec=a,
                         median_qty=float(np.median(q[idx])),
                         share_band4=float((q[idx] > 1000).mean()))
    out["jaccard"] = dict(p_vs_pq=jac(ip, ipq), q_vs_pq=jac(iq, ipq),
                          p_vs_q=jac(ip, iq))
    out["gap_vs_q"] = dict(spec=out["pq"]["cap_spec"] - out["q"]["cap_spec"],
                           refu=out["pq"]["cap_refu"] - out["q"]["cap_refu"])
    sweep = []
    for k in deep.KS:
        idx = top_list(q, k)
        c, a = caps(idx, refu, spec)
        sweep.append(dict(budget=k, q_cap_refu=c, q_cap_spec=a))
    out["sweep"] = sweep
    return out


def _cost_ratios(p, q, cost, refu, spec):
    """Budget needed for 80% capture under a fixed per-row cost vector."""
    res = {}
    for name, s in (("p", p), ("pq", p * q)):
        o = np.argsort(-np.asarray(s, float), kind="stable")
        cc = np.cumsum(cost[o]) / cost.sum()
        cs = np.cumsum(spec[o]) / spec.sum()
        cr = np.cumsum(refu[o]) / refu.sum()
        h_a = np.nonzero(cs >= 0.8)[0]
        h_r = np.nonzero(cr >= 0.8)[0]
        res[name] = dict(cost_for_80anim=float(cc[h_a[0]]) if len(h_a) else None,
                         cost_for_80refu=float(cc[h_r[0]]) if len(h_r) else None)
    return dict(
        ratio_80anim=(res["p"]["cost_for_80anim"] / res["pq"]["cost_for_80anim"]
                      if res["pq"]["cost_for_80anim"] else None),
        ratio_80refu=(res["pq"]["cost_for_80refu"] / res["p"]["cost_for_80refu"]
                      if res["p"]["cost_for_80refu"] else None))


def section_x2(p, q, refu, spec):
    """Inspection cost per row ~ q^gamma on the capped quantity.

    The raw quantity field holds implausible values (max ~9.4e10), so cost is
    defined on the same capped quantity as the objective; a log-cost variant
    is reported as well.
    """
    out = {}
    for gamma in (0.0, 0.5, 1.0):
        cost = np.power(np.maximum(q, 1.0), gamma)
        row = dict(gamma=gamma, top1_cost_share=float(cost.max() / cost.sum()))
        for name, s in (("p", p), ("pq", p * q)):
            o = np.argsort(-np.asarray(s, float), kind="stable")
            cc = np.cumsum(cost[o]) / cost.sum()
            cr = np.cumsum(refu[o]) / refu.sum()
            cs = np.cumsum(spec[o]) / spec.sum()
            h_a = np.nonzero(cs >= 0.8)[0]
            h_r = np.nonzero(cr >= 0.8)[0]
            row[name] = dict(cost_for_80anim=float(cc[h_a[0]]) if len(h_a) else None,
                             cost_for_80refu=float(cc[h_r[0]]) if len(h_r) else None)
            c, a, share_rows = capture_at_cost(s, cost, 0.10, refu, spec)
            row[name].update(cap_refu_at10=c, cap_spec_at10=a, row_share_at10=share_rows)
        row["ratio_80anim"] = (row["p"]["cost_for_80anim"] / row["pq"]["cost_for_80anim"]
                               if row["pq"]["cost_for_80anim"] else None)
        row["ratio_80refu"] = (row["pq"]["cost_for_80refu"] / row["p"]["cost_for_80refu"]
                               if row["p"]["cost_for_80refu"] else None)
        out[f"gamma{gamma}"] = row
    out["sweep"] = [
        dict(gamma=float(g),
             **_cost_ratios(p, q, np.power(np.maximum(q, 1.0), g), refu, spec))
        for g in np.round(np.arange(0.0, 1.0001, 0.1), 1)
    ]
    cost = 1.0 + np.log1p(q)
    row = dict(basis="1+log(1+q)", top1_cost_share=float(cost.max() / cost.sum()))
    for name, s in (("p", p), ("pq", p * q)):
        o = np.argsort(-np.asarray(s, float), kind="stable")
        cc = np.cumsum(cost[o]) / cost.sum()
        cs = np.cumsum(spec[o]) / spec.sum()
        cr = np.cumsum(refu[o]) / refu.sum()
        row[name] = dict(cost_for_80anim=float(cc[np.nonzero(cs >= 0.8)[0][0]]),
                         cost_for_80refu=float(cc[np.nonzero(cr >= 0.8)[0][0]]))
    row["ratio_80anim"] = (row["p"]["cost_for_80anim"] / row["pq"]["cost_for_80anim"])
    row["ratio_80refu"] = (row["pq"]["cost_for_80refu"] / row["p"]["cost_for_80refu"])
    out["log"] = row
    return out


def section_x3(p, q, refu, spec):
    out = {"rows": int(len(q)), "cap_spec_full": float(spec.sum())}
    for drop in (1, 10, 100):
        order = np.argsort(-np.where(refu == 1, q, 0.0), kind="stable")
        keep = np.ones(len(q), bool)
        keep[order[:drop]] = False
        ip, ipq = top_list(p[keep], 0.10), top_list((p * q)[keep], 0.10)
        c1, a1 = caps(ip, refu[keep], spec[keep])
        c2, a2 = caps(ipq, refu[keep], spec[keep])
        out[f"drop{drop}"] = dict(p_refu=c1, p_spec=a1, pq_refu=c2, pq_spec=a2,
                                  animals=float(spec[keep].sum()),
                                  jaccard=jac(ip, ipq))
    return out


def section_x4(df):
    d = df[(df["y"].isin([2019, 2020])) & df["control_number"].notna()]
    d = d[d["desc_"].isin(WHOLE)]
    g = (d.groupby("control_number", observed=True)
           .agg(rows=("refu", "size"), species=("sp", "nunique"),
                refu=("refu", "max"), qty=("qty", "sum")).reset_index())
    g["modal_share"] = d.groupby("control_number", observed=True)["sp"].agg(
        lambda s: s.value_counts(normalize=True).iloc[0]).to_numpy()
    g["mix"] = pd.cut(g["species"], bins=[0, 1, 3, 10, 1e9],
                      labels=["单物种", "2–3 种", "4–10 种", ">10 种"])
    rows = (g.groupby("mix", observed=True)
              .agg(shipments=("refu", "size"), refusal=("refu", "mean"),
                   modal_share=("modal_share", "mean"), rows=("rows", "mean"),
                   qty_median=("qty", "median")).reset_index())
    # logistic: refusal ~ modal_share + log rows + log qty, standardised
    from sklearn.linear_model import LogisticRegression
    X = np.column_stack([g["modal_share"], np.log1p(g["rows"]), np.log1p(g["qty"])])
    mu, sd = X.mean(0), X.std(0)
    lr = LogisticRegression(max_iter=2000).fit((X - mu) / sd, g["refu"])
    coef = dict(zip(["modal_share", "log_rows", "log_qty"], lr.coef_[0].tolist()))
    return dict(n_shipments=int(len(g)), refusal_rate=float(g["refu"].mean()),
                corr_modal_share_vs_refusal=float(np.corrcoef(g["modal_share"], g["refu"])[0, 1]),
                by_mix=rows.to_dict("records"),
                logit_coef=coef,
                logit_or={k: float(np.exp(v)) for k, v in coef.items()})


def section_x5(df):
    d = df[df["y"].between(*p5.TEST_YEARS)]
    seiz = float(d["seiz"].mean())
    refu = float(d["refu"].mean())
    return dict(test_seizure_rate=seiz, test_refusal_rate=refu,
                seiz_share_of_refu=float(d.loc[d["refu"] == 1, "seiz"].mean()))


def section_x6(sub, refu, spec):
    """Two data facts quoted in the text: the raw quantity tail, and how far
    an IUCN threat weighting could reach (it cannot, and we say so)."""
    import duckdb
    con = duckdb.connect(str(p5.LEMIS), read_only=True)
    q = """
    select max(try_cast(quantity as double)) from lemis
    where import_export='I' and unit='NO' and try_cast(quantity as double)>0
      and purpose='T' and try_cast(sYear as int) between 2010 and 2022
    """
    max_qty = float(con.execute(q).fetchone()[0])
    con.close()
    status = {}
    cache = Path(__file__).resolve().parent.parent / "data" / "iucn_status.csv"
    if cache.exists():
        import csv
        with cache.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                name = (row.get("species") or "").strip().lower()
                if name and row.get("status"):
                    status[name] = row["status"].lower()
    sp = sub["sp"].str.strip().str.lower()
    hit = sp.map(status).notna().to_numpy()
    return dict(max_qty_all=max_qty, iucn_cache_species=len(status),
                iucn_rows_covered=float(hit.mean()),
                iucn_animals_covered=float(spec[hit].sum() / max(spec.sum(), 1.0)))


def main():
    t0 = time.time()
    df = p5.pull()
    print("pulled", len(df), flush=True)
    df = deep.add_counts(df)
    prior = float(df.loc[df["y"].between(*p5.TRAIN_YEARS), "refu"].mean())
    df = deep.apply_prior(df, prior)
    m, info = deep.fit(df, p5.TRAIN_YEARS, p5.VALID_YEAR)
    test = df[df["y"].between(*p5.TEST_YEARS)]
    sub = test[test["desc_"].isin(WHOLE)].copy().reset_index(drop=True)
    p = m.predict_proba(sub[p5.FEATS])[:, 1]
    qty_raw = sub["qty"].to_numpy(float)
    q = np.minimum(qty_raw, CAP)
    refu = sub["refu"].to_numpy(float)
    spec = np.where(refu == 1, q, 0.0)
    out = dict(info=info, rows=int(len(sub)), refu=int(refu.sum()),
               refused_animals=float(spec.sum()))
    out["X1_quantity_only"] = section_x1(p, q, refu, spec)
    print("X1", json.dumps(out["X1_quantity_only"], ensure_ascii=False), flush=True)
    out["X2_cost"] = section_x2(p, q, refu, spec)
    print("X2 done", flush=True)
    out["X3_leave_largest"] = section_x3(p, q, refu, spec)
    print("X3", json.dumps(out["X3_leave_largest"], ensure_ascii=False), flush=True)
    out["X4_shipment_mix"] = section_x4(df)
    print("X4", json.dumps(out["X4_shipment_mix"], ensure_ascii=False), flush=True)
    out["X5_disposition"] = section_x5(df)
    out["X6_data_facts"] = section_x6(sub, refu, spec)
    (OUT / "p6_revisions.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("written", round(time.time() - t0, 1), "s", flush=True)


if __name__ == "__main__":
    main()
