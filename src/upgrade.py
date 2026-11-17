"""Computations for the upgrade round (指标治理版).

Task 1  delete_top_ci : species-clustered bootstrap of the "drop the largest
                         refused rows" check.  Drops 0/1/10/100 rows, 10%
                         budget, whole-animal scope, cap 1e4.
Task 2  shadow_cost   : budget k as a share of total inspection cost when a
                         row costs c_i = q_i**gamma.  For each k the animal
                         rule reaches some animal capture; the shadow cost is
                         the cost share the shipment rule needs to reach the
                         same capture.  Reported as a ratio (>=1 means the
                         shipment rule burns extra capacity) and as a
                         difference in cost-share points.
                         Also a fine (0.05-step) sweep of the 80% ratio, so
                         the crossing point is read off one grid everywhere.

Protocol identical to p6_deep / p6_revisions: train 2010-2017, valid 2018,
test 2019-2022, whole-animal scope, LightGBM with expanding-window historical
refusal rates, cap 1e4 on declared quantity.

Usage: python p6_upgrade.py [delete_top_ci shadow_cost]
Output: results/p6_delete_top_ci.json, results/p6_shadow_cost.json
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
CAP = deep.CAP
SEED_BOOT = 20261108


def jdump(obj, name):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print("wrote", name, flush=True)


def eval_pair(p, q, refu, spec, k=0.10):
    """Capture of both metrics under both rules at one budget share."""
    def cap(score, idx_all):
        o = np.argsort(-np.asarray(score, float), kind="stable")
        m = max(int(round(k * len(o))), 1)
        idx = o[:m]
        return (float(refu[idx].sum() / max(refu.sum(), 1.0)),
                float(spec[idx].sum() / max(spec.sum(), 1.0)), idx)
    c1, a1, i1 = cap(p, refu)
    c2, a2, i2 = cap(p * q, refu)
    s1, s2 = set(i1.tolist()), set(i2.tolist())
    return dict(p_refu=c1, p_spec=a1, pq_refu=c2, pq_spec=a2,
                diff_spec=a2 - a1, diff_refu=c1 - c2,
                jaccard=len(s1 & s2) / max(len(s1 | s2), 1))


# ------------------------------------------------------------------ task 1
def section_delete_top(p, q, refu, spec, sp, reps=500):
    order = np.argsort(-np.where(refu == 1, q, 0.0), kind="stable")
    out = dict(reps=reps, seed=SEED_BOOT, rows_full=int(len(q)),
               n_species_full=int(pd.Series(sp).nunique()), variants={})
    for drop in (0, 1, 10, 100):
        t0 = time.time()
        keep = np.ones(len(q), bool)
        keep[order[:drop]] = False
        pk, qk, rk, sk, spk = p[keep], q[keep], refu[keep], spec[keep], sp[keep]
        point = eval_pair(pk, qk, rk, sk)
        codes, uniq = pd.factorize(spk)
        idx_by = [np.nonzero(codes == i)[0] for i in range(len(uniq))]
        rng = np.random.default_rng(SEED_BOOT + drop)
        draws = []
        for _ in range(reps):
            pick = rng.integers(0, len(uniq), len(uniq))
            idx = np.concatenate([idx_by[i] for i in pick])
            r = eval_pair(pk[idx], qk[idx], rk[idx], sk[idx])
            draws.append([r["p_refu"], r["p_spec"], r["pq_refu"], r["pq_spec"],
                          r["diff_spec"], r["diff_refu"], r["jaccard"]])
        a = np.asarray(draws)
        names = ["p_refu", "p_spec", "pq_refu", "pq_spec",
                 "diff_spec", "diff_refu", "jaccard"]
        out["variants"][f"drop{drop}"] = dict(
            rows=int(keep.sum()), species=int(len(uniq)),
            refused_animals=float(sk.sum()),
            point={n: float(point[n]) for n in names},
            ci95={n: [float(np.percentile(a[:, i], 2.5)),
                      float(np.percentile(a[:, i], 97.5))]
                  for i, n in enumerate(names)},
            diff_spec_above_zero=float(np.mean(a[:, 4] > 0)),
            secs=round(time.time() - t0, 1))
        print(f"[drop {drop}] diff_spec {point['diff_spec']:.4f} "
              f"CI {out['variants'][f'drop{drop}']['ci95']['diff_spec']} "
              f"share>0 {out['variants'][f'drop{drop}']['diff_spec_above_zero']:.3f} "
              f"{out['variants'][f'drop{drop}']['secs']}s", flush=True)
    return out


# ------------------------------------------------------------------ task 2
def _share_curve(score, cost, spec, refu):
    o = np.argsort(-np.asarray(score, float), kind="stable")
    cc = np.cumsum(cost[o]) / cost.sum()
    cs = np.cumsum(spec[o]) / max(spec.sum(), 1e-12)
    cr = np.cumsum(refu[o]) / max(refu.sum(), 1e-12)
    return cc, cs, cr


def _cost_for(cc, cs, target):
    hit = np.nonzero(cs >= target)[0]
    return float(cc[hit[0]]) if len(hit) else None


def section_shadow(p, q, refu, spec):
    grid = [round(float(g), 2) for g in np.arange(0.0, 1.0001, 0.05)]
    ks = [0.05, 0.10, 0.20]
    out = dict(
        definition=("budget = share of total inspection cost when a row costs "
                    "q**gamma; target = animal capture the pq rule reaches at "
                    "budget k; value = the cost share the p rule needs to reach "
                    "the same animal capture; ratio = value / k"),
        scope="whole-animal LIV/BOD/DEA/TRO/SKE/FIG, test 2019-2022, cap 1e4",
        gamma_grid=grid, k_grid=ks, curves={f"k{k}": [] for k in ks},
        ratio80=[], by_target=[], consistency={})
    targets = (0.60, 0.70, 0.80)
    for g in grid:
        cost = np.power(np.maximum(q, 1.0), g)
        ccp, csp, crp = _share_curve(p, cost, spec, refu)
        ccq, csq, crq = _share_curve(p * q, cost, spec, refu)
        for k in ks:
            # animal capture the pq rule reaches inside budget k
            m = np.searchsorted(ccq, k, side="right")
            target = float(csq[max(m, 1) - 1]) if len(csq) else 0.0
            need = _cost_for(ccp, csp, target) if target > 0 else 0.0
            m_p = np.searchsorted(ccp, k, side="right")
            out["curves"][f"k{k}"].append(dict(
                gamma=g,
                target_anim=target,
                p_anim_at_k=float(csp[max(m_p, 1) - 1]),
                pq_anim_at_k=target,
                p_cost_for_target=need,
                ratio=(need / k) if need is not None and k > 0 else None,
                extra=(need - k) if need is not None else None))
        need80p = _cost_for(ccp, csp, 0.80)
        need80q = _cost_for(ccq, csq, 0.80)
        out["ratio80"].append(dict(
            gamma=g, cost_p=need80p, cost_pq=need80q,
            ratio=(need80p / need80q) if need80q else None))
        row = dict(gamma=g)
        for t in targets:
            cp, cq = _cost_for(ccp, csp, t), _cost_for(ccq, csq, t)
            row[f"t{int(t*100)}"] = (cp / cq) if cp and cq else None
        out["by_target"].append(row)
    # crossing of the 80% ratio through 1.0 (grid is fine and monotone)
    rs = out["ratio80"]
    gx = np.array([r["gamma"] for r in rs])
    gy = np.array([r["ratio"] for r in rs])
    cross = float(np.interp(1.0, gy[::-1], gx[::-1]))
    out["cross_80"] = dict(
        gamma=cross,
        bracket=[float(gx[gy >= 1.0].max()) if (gy >= 1.0).any() else None,
                 float(gx[gy < 1.0].min()) if (gy < 1.0).any() else None],
        monotone_decreasing=bool(np.all(np.diff(gy) < 0)))
    by_t, by_k = {}, {}
    for t in targets:
        yy = np.array([r[f"t{int(t*100)}"] for r in out["by_target"]])
        by_t[f"t{int(t*100)}"] = dict(
            gamma=float(np.interp(1.0, yy[::-1], gx[::-1])),
            monotone_decreasing=bool(np.all(np.diff(yy) < 0)))
    for k in ks:
        yy = np.array([r["ratio"] for r in out["curves"][f"k{k}"]])
        by_k[f"k{k}"] = dict(gamma=float(np.interp(1.0, yy[::-1], gx[::-1])))
    out["cross_by_target"] = by_t
    out["cross_by_budget"] = by_k
    # consistency with the 0.1 grid already published (X2 sweep)
    rev = json.loads((OUT / "p6_revisions.json").read_text(encoding="utf-8"))
    cons = {}
    for name, g in (("gamma0.0", 0.0), ("gamma0.5", 0.5), ("gamma1.0", 1.0)):
        old = rev["X2_cost"][name]["ratio_80anim"]
        new = float(np.interp(g, gx, gy))
        cons[name] = dict(old=float(old), new=new, abs_diff=abs(old - new))
    out["consistency"] = cons
    return out


# ------------------------------------------------------------------ main
def main(which):
    t0 = time.time()
    df = p5.pull()
    print("pulled", len(df), flush=True)
    df = deep.add_counts(df)
    prior = float(df.loc[df["y"].between(*p5.TRAIN_YEARS), "refu"].mean())
    df = deep.apply_prior(df, prior)
    m, info = deep.fit(df, p5.TRAIN_YEARS, p5.VALID_YEAR)
    print("fitted", info, round(time.time() - t0, 1), flush=True)
    test = df[df["y"].between(*p5.TEST_YEARS)]
    sub = test[test["desc_"].isin(deep.WHOLE)].copy().reset_index(drop=True)
    p = m.predict_proba(sub[p5.FEATS])[:, 1]
    qraw = sub["qty"].to_numpy()
    q = np.minimum(qraw, CAP)
    refu = sub["refu"].to_numpy()
    spec = np.where(refu == 1, q, 0.0)
    print(f"rows {len(sub)} refused {int(refu.sum())} animals {spec.sum():.0f}",
          flush=True)
    base = eval_pair(p, q, refu, spec)
    print("main", {k: round(v, 4) for k, v in base.items()}, flush=True)

    if "delete_top_ci" in which:
        out = section_delete_top(p, q, refu, spec, sub["sp"].to_numpy())
        out["main_reference"] = {k: float(v) for k, v in base.items()}
        jdump(out, "p6_delete_top_ci.json")

    if "shadow_cost" in which:
        out = section_shadow(p, q, refu, spec)
        out["main_reference"] = {k: round(float(v), 6) for k, v in base.items()}
        jdump(out, "p6_shadow_cost.json")
    print("ALL DONE", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["delete_top_ci", "shadow_cost"])
