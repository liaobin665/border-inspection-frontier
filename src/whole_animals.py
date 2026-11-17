"""Feasibility probe on whole-animal terms (the defensible "individual" scope).

Scope
-----
LIV (live), BOD (dead whole animal), DEA (died in shipment), TRO (trophy,
one animal per trophy), SKE (skeleton), FIG (fingerlings). Excluded: parts and
products (FEA feathers, MEA meat, EGL eggs, SHE shells, SPR shell products...)
because their declared quantity is not an animal count.

Protocol: same split as p5 (train 2010-2017, valid 2018, test 2019-2022).
Two models: all terms (transfer) and whole-animal only (scope-specific).
Rules p and p*q; quantity raw, capped at 1e4 and 1e3.

Outputs: results/p5e_whole_animals.json, results/p5e_summary.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data_layer as p5  # noqa: E402

OUT = HERE.parent / "results"
WHOLE = ("LIV", "BOD", "DEA", "TRO", "SKE", "FIG")
KS = (0.05, 0.10, 0.20)


def captures(score, refu, spec) -> dict:
    o = np.argsort(-score, kind="stable")
    cr = np.cumsum(refu[o]) / max(float(refu.sum()), 1.0)
    cs = np.cumsum(spec[o]) / max(float(spec.sum()), 1.0)
    out = {}
    for k in KS:
        m = max(int(round(k * len(o))), 1)
        out[f"{int(k*100)}%"] = dict(cap_refu=float(cr[m - 1]),
                                     cap_spec=float(cs[m - 1]), rows=m)
    return out


def evaluate(d, p) -> dict:
    refu = d["refu"].to_numpy().astype(float)
    res = dict(rows=int(len(d)), refu=int(refu.sum()))
    for cap in (None, 1e4, 1e3):
        q = d["qty"].to_numpy()
        if cap is not None:
            q = np.minimum(q, cap)
        spec = np.where(d["refu"].to_numpy() == 1, q, 0.0)
        tag = "raw" if cap is None else f"cap_{int(cap)}"
        res[tag] = dict(refused_qty=float(spec.sum()),
                        p=captures(p, refu, spec),
                        pq=captures(p * q, refu, spec))
        o_p = np.argsort(-p, kind="stable")
        o_q = np.argsort(-(p * q), kind="stable")
        m = max(int(round(0.10 * len(d))), 1)
        a, b = set(o_p[:m].tolist()), set(o_q[:m].tolist())
        res[tag]["jaccard_10"] = len(a & b) / len(a | b)
        res[tag]["p_list_in_pq"] = len(a & b) / len(a)
    return res


def concentration(d) -> dict:
    q = np.sort(d.loc[d["refu"] == 1, "qty"].to_numpy())[::-1]
    tot = float(q.sum())
    return dict(refused_rows=int(len(q)), refused_qty=tot,
                top1=float(q[:1].sum() / tot),
                top10=float(q[:10].sum() / tot),
                top100=float(q[:100].sum() / tot))


def main() -> None:
    df = p5.pull()
    df, _ = p5.build_features(df)
    test = df[df["y"].between(*p5.TEST_YEARS)].copy()
    sub = test[test["desc_"].isin(WHOLE)].copy()

    model_all, _ = p5.fit_model(df)
    p_all = model_all.predict_proba(sub[p5.FEATS])[:, 1]

    full = df[df["desc_"].isin(WHOLE)]
    model_w, _ = p5.fit_model(full)
    p_w = model_w.predict_proba(sub[p5.FEATS])[:, 1]

    refu = sub["refu"].to_numpy()
    res = dict(scope="whole animals: " + "/".join(WHOLE),
               test_rows=int(len(sub)), test_refu=int(refu.sum()),
               prevalence=float(refu.mean()),
               concentration=concentration(sub),
               all_terms=dict(auc=float(roc_auc_score(refu, p_all)),
                              ap=float(average_precision_score(refu, p_all)),
                              metrics=evaluate(sub, p_all)),
               whole_only=dict(
                   n_train=int(len(full[full.y.between(*p5.TRAIN_YEARS)])),
                   n_valid=int(len(full[full.y == p5.VALID_YEAR])),
                   auc=float(roc_auc_score(refu, p_w)),
                   ap=float(average_precision_score(refu, p_w)),
                   metrics=evaluate(sub, p_w)))
    best = "whole_only" if res["whole_only"]["auc"] >= res["all_terms"]["auc"] else "all_terms"
    res["best_model"] = best
    p_best = p_w if best == "whole_only" else p_all

    q = np.minimum(sub["qty"].to_numpy(), 1e4)
    spec = np.where(sub["refu"].to_numpy() == 1, q, 0.0)
    by_year = {}
    for y in range(p5.TEST_YEARS[0], p5.TEST_YEARS[1] + 1):
        m = (sub["y"] == y).to_numpy()
        if m.sum() == 0:
            continue
        by_year[str(y)] = dict(rows=int(m.sum()), refu=int(refu[m].sum()),
                               p=captures(p_best[m], refu[m], spec[m])["10%"],
                               pq=captures((p_best * q)[m], refu[m], spec[m])["10%"])
    res["by_year_best_cap_1e4"] = by_year

    (OUT / "p5e_whole_animals.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# 整只动物口径：双目标可行性（自动汇总）\n",
             f"- 范围：{'/'.join(WHOLE)}",
             f"- 测试期 2019–2022：{res['test_rows']:,} 行，拒收 {res['test_refu']:,}，"
             f"拒收率 {100*res['prevalence']:.2f}%",
             f"- 被拒个体集中度：前 1 行 {100*res['concentration']['top1']:.1f}%，"
             f"前 10 行 {100*res['concentration']['top10']:.1f}%，"
             f"前 100 行 {100*res['concentration']['top100']:.1f}%"]
    for name in ("all_terms", "whole_only"):
        m = res[name]
        lines.append(f"\n## {name}\n")
        lines.append(f"AUC {m['auc']:.4f}；AP {m['ap']:.4f}"
                     + (f"；训练 {m['n_train']:,} / 验证 {m['n_valid']:,}"
                        if "n_train" in m else ""))
        lines.append("\n| 口径 | 10% 票(p) | 10% 个体(p) | 10% 票(p×q) | 10% 个体(p×q) | Jaccard |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for tag in ("raw", "cap_10000", "cap_1000"):
            t = m["metrics"][tag]
            lines.append(
                f"| {tag} | {100*t['p']['10%']['cap_refu']:.2f}% | "
                f"{100*t['p']['10%']['cap_spec']:.2f}% | "
                f"{100*t['pq']['10%']['cap_refu']:.2f}% | "
                f"{100*t['pq']['10%']['cap_spec']:.2f}% | {t['jaccard_10']:.3f} |")
    lines.append(f"\n**较优模型：{best}**\n")
    lines.append("## 逐年（较优模型，10%，cap=1e4）\n")
    lines.append("| 年 | 行 | 拒收 | p 票 | p 个体 | p×q 票 | p×q 个体 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for y, v in by_year.items():
        lines.append(f"| {y} | {v['rows']:,} | {v['refu']:,} | "
                     f"{100*v['p']['cap_refu']:.1f}% | {100*v['p']['cap_spec']:.1f}% | "
                     f"{100*v['pq']['cap_refu']:.1f}% | {100*v['pq']['cap_spec']:.1f}% |")
    (OUT / "p5e_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
