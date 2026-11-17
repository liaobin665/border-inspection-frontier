"""Feasibility probe: does the dual-objective conflict survive a real risk model?

Question
--------
The earlier probe showed that a band-greedy rule (inspect small rows first)
captures refusals, while a specimen-weighted rule captures refused specimens.
A reviewer can object that this is an artefact of the coarse size bands.
This script trains one strong tabular model (LightGBM) on the same rows and
compares two decision rules that use *the same* model:

    rule A : rank by predicted refusal probability  p
    rule B : rank by expected refused specimens     p * quantity

If the conflict persists under a real model, the topic is not a banding
artefact. The model is a validity check, not the paper's contribution.

Protocol
--------
Sample : LEMIS imports, unit=NO, purpose=T, 2010-2022, quantity>0,
         species valid (drop REMAINS UNKNOWN).
Split  : train 2010-2017, valid 2018, test 2019-2022.
Target : action='R' (administrative refusal); disposition='S' as robustness.
Budget : top K% of test rows by score; K in {5,10,20,30}.

Leakage guards
--------------
* No feature uses information from year >= the row's own year.
* Historical rates are computed from years < y only (expanding, smoothed).
* The quantity cap used in the sensitivity is estimated on train only.

Outputs
-------
results/p5_dual_feasibility.json
results/p5_summary.md   (tables rendered from the JSON, no hand-typed numbers)
"""
from __future__ import annotations

import json
import time
import os
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results"
LEMIS = Path(os.environ.get("LEMIS_DB", str(HERE.parent / "data" / "lemis.duckdb")))

SEED = 7
KS = (0.05, 0.10, 0.20, 0.30)
TRAIN_YEARS = (2010, 2017)
VALID_YEAR = 2018
TEST_YEARS = (2019, 2022)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def pull() -> pd.DataFrame:
    con = duckdb.connect(str(LEMIS), read_only=True)
    q = """
    select
      coalesce(nullif(corrected,''), species)              as sp,
      country_origin                                        as org,
      port,
      description                                           as desc_,
      source,
      try_cast(quantity as double)                          as qty,
      try_cast(sYear as int)                                as y,
      month(try_cast(shipment_date as timestamp))           as month,
      dayofweek(try_cast(shipment_date as timestamp))       as dow,
      case when action='R' then 1 else 0 end                as refu,
      case when disposition='S' then 1 else 0 end           as seiz,
      group_,
      control_number
    from lemis
    where import_export='I' and unit='NO' and try_cast(quantity as double)>0
      and purpose='T'
      and try_cast(sYear as int) between 2010 and 2022
      and coalesce(nullif(corrected,''), species) is not null
      and trim(coalesce(nullif(corrected,''), species)) <> ''
      and upper(coalesce(nullif(corrected,''), species)) not like '%UNKNOWN%'
    """
    df = con.execute(q).df()
    con.close()
    df["qty"] = df["qty"].astype(float)
    df["refused_qty"] = np.where(df["refu"] == 1, df["qty"], 0.0)
    df["seized_qty"] = np.where(df["seiz"] == 1, df["qty"], 0.0)
    df["log_qty"] = np.log1p(df["qty"])
    df["band"] = np.select(
        [df["qty"] <= 10, df["qty"] <= 100, df["qty"] <= 1000],
        [1, 2, 3], default=4)
    df["month"] = df["month"].fillna(-1).astype(int)
    df["dow"] = df["dow"].fillna(-1).astype(int)
    for c in ("sp", "org", "port", "desc_", "source"):
        df[c] = df[c].fillna("NA").astype(str)
    return df


def add_rate(df: pd.DataFrame, keys: list[str], name: str,
             prior: float, alpha: float = 50.0) -> pd.DataFrame:
    """Rate from years strictly before y, smoothed toward `prior`."""
    g = (df.groupby(["y"] + keys, observed=True)
           .agg(n=("refu", "size"), r=("refu", "sum")).reset_index())
    g = g.sort_values("y", kind="stable")
    g["cn"] = g.groupby(keys, observed=True)["n"].cumsum() - g["n"]
    g["cr"] = g.groupby(keys, observed=True)["r"].cumsum() - g["r"]
    g[name] = (g["cr"] + alpha * prior) / (g["cn"] + alpha)
    out = df.merge(g[["y"] + keys + [name]], on=["y"] + keys, how="left")
    out[name] = out[name].fillna(prior)
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    train_mask = df["y"].between(*TRAIN_YEARS)
    prior = float(df.loc[train_mask, "refu"].mean())
    df = add_rate(df, ["sp"], "hist_sp", prior)
    df = add_rate(df, ["org"], "hist_org", prior)
    df = add_rate(df, ["sp", "org"], "hist_sporg", prior)
    df = add_rate(df, ["port"], "hist_port", prior)
    df = add_rate(df, ["desc_"], "hist_desc", prior)
    for c in ("sp", "org", "port", "source", "band"):
        df[c + "_c"] = df[c].astype("category")
    df["desc_c"] = df["desc_"].astype("category")
    return df, prior


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
FEATS = ["log_qty", "band_c", "sp_c", "org_c", "port_c", "desc_c", "source_c",
         "month", "dow", "hist_sp", "hist_org", "hist_sporg", "hist_port",
         "hist_desc"]
CATS = ["band_c", "sp_c", "org_c", "port_c", "desc_c", "source_c"]


def fit_model(df: pd.DataFrame) -> tuple[lgb.LGBMClassifier, dict]:
    tr = df[df["y"].between(*TRAIN_YEARS)]
    va = df[df["y"] == VALID_YEAR]
    model = lgb.LGBMClassifier(
        objective="binary", n_estimators=3000, learning_rate=0.05,
        num_leaves=127, min_child_samples=300, subsample=0.9,
        subsample_freq=1, colsample_bytree=0.9, reg_lambda=1.0,
        max_bin=255, n_jobs=8, random_state=SEED, verbose=-1)
    model.fit(
        tr[FEATS], tr["refu"], eval_set=[(va[FEATS], va["refu"])],
        eval_metric="auc", categorical_feature=CATS,
        callbacks=[lgb.early_stopping(100, verbose=False),
                   lgb.log_evaluation(200)])
    info = dict(best_iteration=int(model.best_iteration_ or model.n_estimators),
                n_train=int(len(tr)), n_valid=int(len(va)),
                valid_auc=float(model.best_score_["valid_0"]["auc"]))
    return model, info


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def order_of(score: np.ndarray) -> np.ndarray:
    return np.argsort(-np.asarray(score, dtype=float), kind="stable")


def captures(order: np.ndarray, refu: np.ndarray, spec: np.ndarray,
             ks=KS) -> dict:
    n = len(order)
    cr = np.cumsum(refu[order])
    cs = np.cumsum(spec[order])
    tot_r = max(float(refu.sum()), 1.0)
    tot_s = max(float(spec.sum()), 1.0)
    out = {}
    for k in ks:
        m = max(int(round(k * n)), 1)
        out[f"{int(k*100)}%"] = dict(
            rows=m,
            cap_refu=float(cr[m - 1] / tot_r),
            cap_spec=float(cs[m - 1] / tot_s),
            precision_refu=float(cr[m - 1] / m))
    return out


def composition(order: np.ndarray, df: pd.DataFrame, k: float) -> dict:
    m = max(int(round(k * len(df))), 1)
    idx = order[:m]
    sub = df.iloc[idx]
    return dict(
        rows=int(m),
        small_share=float((sub["band"] == 1).mean()),
        big_share=float((sub["band"] == 4).mean()),
        live_share=float((sub["desc_"] == "LIV").mean()),
        qty_sum=float(sub["qty"].sum()),
        qty_share=float(sub["qty"].sum() / df["qty"].sum()),
        refu_rate=float(sub["refu"].mean()),
        top5_species=[[str(s), int(c)] for s, c in
                      sub["sp"].value_counts().head(5).items()])


def curve(order: np.ndarray, refu: np.ndarray, spec: np.ndarray,
          grid: int = 100) -> dict:
    cr = np.cumsum(refu[order]) / max(float(refu.sum()), 1.0)
    cs = np.cumsum(spec[order]) / max(float(spec.sum()), 1.0)
    idx = (np.arange(1, grid + 1) / grid * len(order)).astype(int) - 1
    return dict(budget=[i / grid for i in range(1, grid + 1)],
                cap_refu=cr[idx].round(6).tolist(),
                cap_spec=cs[idx].round(6).tolist())


def eval_rules(df: pd.DataFrame, score: np.ndarray, seed: int = SEED) -> dict:
    refu = df["refu"].to_numpy()
    spec = df["refused_qty"].to_numpy()
    rng = np.random.default_rng(seed)
    n = len(df)
    band = df["band"].to_numpy()

    orders = {
        "model_p": order_of(score),
        "model_pq": order_of(score * df["qty"].to_numpy()),
        "qty_only": order_of(df["qty"].to_numpy()),
        "band_greedy": np.lexsort((rng.random(n), band)),
        "oracle_refu": order_of(refu.astype(float)),
        "oracle_spec": order_of(spec.astype(float)),
    }
    res = {name: captures(o, refu, spec) for name, o in orders.items()}

    # random baseline averaged over 5 permutations
    rand_acc = {}
    for k in KS:
        vals = []
        for s in range(5):
            o = np.random.default_rng(seed + s).permutation(n)
            vals.append(captures(o, refu, spec, ks=(k,))[f"{int(k*100)}%"])
        rand_acc[f"{int(k*100)}%"] = {m: float(np.mean([v[m] for v in vals]))
                                      for m in ("cap_refu", "cap_spec")}
    res["random"] = rand_acc

    res["curve"] = {
        "model_p": curve(orders["model_p"], refu, spec),
        "model_pq": curve(orders["model_pq"], refu, spec),
    }
    res["composition_10pct"] = {
        "model_p": composition(orders["model_p"], df, 0.10),
        "model_pq": composition(orders["model_pq"], df, 0.10),
        "band_greedy": composition(orders["band_greedy"], df, 0.10),
    }
    return res


def by_year(df: pd.DataFrame, score: np.ndarray) -> dict:
    out = {}
    for y in range(TEST_YEARS[0], TEST_YEARS[1] + 1):
        m = (df["y"] == y).to_numpy()
        sub = df.loc[m]
        if len(sub) == 0 or sub["refu"].sum() == 0:
            continue
        s = score[m]
        o_p = order_of(s)
        o_q = order_of(s * sub["qty"].to_numpy())
        out[str(y)] = dict(
            n=int(len(sub)), refu=int(sub["refu"].sum()),
            refused_qty=float(sub["refused_qty"].sum()),
            model_p=captures(o_p, sub["refu"].to_numpy(),
                             sub["refused_qty"].to_numpy(), ks=(0.10,))["10%"],
            model_pq=captures(o_q, sub["refu"].to_numpy(),
                              sub["refused_qty"].to_numpy(), ks=(0.10,))["10%"])
    return out


def shipment_level(df: pd.DataFrame, score: np.ndarray) -> dict:
    """Aggregate row scores to shipments (control_number valid 2016-2020)."""
    m = df["y"].isin([2019, 2020]) & df["control_number"].notna()
    sub = df.loc[m].copy()
    sub["p"] = score[m]
    sub["log1mp"] = np.log1p(-np.clip(sub["p"], 0, 1 - 1e-12))
    g = sub.groupby("control_number", observed=True).agg(
        any_refu=("refu", "max"),
        refused_qty=("refused_qty", "sum"),
        p_any=("log1mp", lambda x: 1.0 - np.exp(x.sum())),
    ).reset_index()
    # expected refused specimens per shipment = sum(p_i * q_i)
    tmp = sub.assign(pq=sub["p"] * sub["qty"]).groupby(
        "control_number", observed=True)["pq"].sum()
    g["p_qty"] = g["control_number"].map(tmp)
    refu = g["any_refu"].to_numpy().astype(float)
    spec = g["refused_qty"].to_numpy().astype(float)
    out = dict(n_shipments=int(len(g)),
               refusal_rate=float(refu.mean()))
    for name, sc in (("model_p_any", g["p_any"].to_numpy()),
                     ("model_pq", g["p_qty"].to_numpy())):
        out[name] = captures(order_of(sc), refu, spec, ks=(0.10, 0.20))
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    print("[1/6] pull", flush=True)
    df = pull()
    print(f"      rows={len(df):,} refu={int(df.refu.sum()):,} "
          f"rate={df.refu.mean():.4f}", flush=True)

    print("[2/6] features", flush=True)
    df, prior = build_features(df)
    print(f"      prior={prior:.5f}", flush=True)

    print("[3/6] train", flush=True)
    model, info = fit_model(df)
    print("      ", info, flush=True)

    test = df[df["y"].between(*TEST_YEARS)].copy()
    p = model.predict_proba(test[FEATS])[:, 1]
    test["p"] = p

    res: dict = dict(
        protocol=dict(sample_rows=int(len(df)), prior=round(prior, 6),
                      train=f"{TRAIN_YEARS[0]}-{TRAIN_YEARS[1]}",
                      valid=VALID_YEAR, test=f"{TEST_YEARS[0]}-{TEST_YEARS[1]}",
                      n_train=info["n_train"], n_valid=info["n_valid"],
                      n_test=int(len(test)), best_iter=info["best_iteration"]),
        model=dict(
            auc=float(roc_auc_score(test["refu"], p)),
            ap=float(average_precision_score(test["refu"], p)),
            brier=float(brier_score_loss(test["refu"], p)),
            prevalence=float(test["refu"].mean()),
            valid_auc=info["valid_auc"]),
        feature_gain=dict(sorted(
            zip(FEATS, model.booster_.feature_importance("gain")),
            key=lambda x: -x[1])))

    # calibration deciles
    test["dec"] = pd.qcut(test["p"], 10, labels=False, duplicates="drop")
    cal = test.groupby("dec").agg(pred=("p", "mean"), obs=("refu", "mean"),
                                  n=("refu", "size"))
    res["calibration"] = [[int(i), round(float(r.pred), 5),
                           round(float(r.obs), 5), int(r.n)]
                          for i, r in cal.iterrows()]

    print("[4/6] dual-objective rules", flush=True)
    res["rules"] = eval_rules(test, p)
    res["by_year"] = by_year(test, p)

    print("[5/6] robustness", flush=True)
    live = test[test["desc_"] == "LIV"]
    res["live_only"] = dict(
        n=int(len(live)), prevalence=float(live["refu"].mean()),
        auc=float(roc_auc_score(live["refu"], live["p"]))
        if live["refu"].nunique() > 1 else None,
        rules=eval_rules(live, live["p"].to_numpy()))

    # seizure outcome, same refusal-model scores
    se = test.copy()
    se["refu_orig"] = se["refu"]
    se["refu"] = se["seiz"]
    se["refused_qty"] = se["seized_qty"]
    res["seizure_outcome"] = eval_rules(se, se["p"].to_numpy())

    # quantity cap sensitivity (cap estimated on train only)
    cap = float(np.quantile(df.loc[df.y.between(*TRAIN_YEARS), "qty"], 0.9999))
    test["qty_cap"] = np.minimum(test["qty"], cap)
    refu = test["refu"].to_numpy()
    spec = test["refused_qty"].to_numpy()
    res["qty_cap_sensitivity"] = dict(
        cap=cap,
        raw=captures(order_of(p * test["qty"].to_numpy()), refu, spec),
        capped=captures(order_of(p * test["qty_cap"].to_numpy()), refu, spec),
        exclude_big=captures(
            order_of(p * np.where(test["qty"].to_numpy() > 1e6, 0,
                                  test["qty"].to_numpy())), refu, spec))

    # shipment-level: same model scores, aggregated to control_number
    p_all = np.full(len(df), np.nan)
    p_all[(df["y"] >= TEST_YEARS[0]).to_numpy()] = p
    res["shipment_level"] = shipment_level(df, p_all)

    # gates
    r10 = res["rules"]["model_p"]["10%"]
    r10q = res["rules"]["model_pq"]["10%"]
    band10 = res["rules"]["band_greedy"]["10%"]
    yrs = res["by_year"]
    gates = dict(
        G1_signal=bool(res["model"]["auc"] >= 0.70 and
                       res["model"]["ap"] >= 3 * res["model"]["prevalence"]),
        G2_beats_band=bool(r10["cap_refu"] >= 1.10 * band10["cap_refu"]),
        G3_conflict=bool(r10q["cap_spec"] >= 0.80 and
                         r10["cap_spec"] <= 0.10 and
                         r10["cap_refu"] >= 2 * r10q["cap_refu"]),
        G4_year_sign=bool(all(v["model_p"]["cap_refu"] > v["model_pq"]["cap_refu"]
                              and v["model_pq"]["cap_spec"] > v["model_p"]["cap_spec"]
                              for v in yrs.values())),
        G5_live_sign=bool(
            res["live_only"]["rules"]["model_p"]["10%"]["cap_refu"] >
            res["live_only"]["rules"]["model_pq"]["10%"]["cap_refu"] and
            res["live_only"]["rules"]["model_pq"]["10%"]["cap_spec"] >
            res["live_only"]["rules"]["model_p"]["10%"]["cap_spec"]),
    )
    res["gates"] = gates
    res["verdict"] = "Go" if all(gates.values()) else "Conditional/No-Go"
    res["elapsed_sec"] = round(time.time() - t0, 1)

    OUT.mkdir(exist_ok=True)
    (OUT / "p5_dual_feasibility.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    write_summary(res)
    print(json.dumps(dict(model=res["model"], gates=gates,
                          verdict=res["verdict"]), ensure_ascii=False, indent=2))
    print(f"[6/6] done in {res['elapsed_sec']}s", flush=True)


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def write_summary(res: dict) -> None:
    m, p = res["model"], res["protocol"]
    rules = res["rules"]
    lines = []
    lines.append("# 双目标模型版可行性：自动汇总\n")
    lines.append(f"- 样本 {p['sample_rows']:,} 行；训练 {p['n_train']:,}；"
                 f"验证 {p['n_valid']:,}；测试 {p['n_test']:,}")
    lines.append(f"- 测试期拒收率 {pct(m['prevalence'])}；AUC {m['auc']:.4f}；"
                 f"AP {m['ap']:.4f}（基率 {m['prevalence']:.4f}，"
                 f"{m['ap']/m['prevalence']:.1f} 倍）")
    lines.append(f"- 验证集 AUC {m['valid_auc']:.4f}；best_iter {p['best_iter']}\n")

    lines.append("## 1. 两种排序在固定预算下的捕获（测试期 2019–2022）\n")
    lines.append("| 规则 | 5% 票 | 5% 个体 | 10% 票 | 10% 个体 | 20% 票 | 20% 个体 | 30% 票 | 30% 个体 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name in ("model_p", "model_pq", "qty_only", "band_greedy",
                 "random", "oracle_refu", "oracle_spec"):
        d = rules[name]
        row = [name]
        for k in ("5%", "10%", "20%", "30%"):
            if name == "random":
                row += [pct(d[k]["cap_refu"]), pct(d[k]["cap_spec"])]
            else:
                row += [pct(d[k]["cap_refu"]), pct(d[k]["cap_spec"])]
        lines.append("| " + " | ".join(row) + " |")

    lines.append("\n## 2. 10% 预算下选中的是什么货\n")
    lines.append("| 规则 | 小行占比 | 大行占比 | 活体占比 | 申报量占比 | 拒收率 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, c in rules["composition_10pct"].items():
        lines.append(f"| {name} | {pct(c['small_share'])} | {pct(c['big_share'])} | "
                     f"{pct(c['live_share'])} | {pct(c['qty_share'])} | "
                     f"{pct(c['refu_rate'])} |")

    lines.append("\n## 3. 逐年（10% 预算）\n")
    lines.append("| 年 | 行数 | 拒收 | 模型p 票 | 模型p 个体 | 模型p×q 票 | 模型p×q 个体 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for y, v in res["by_year"].items():
        lines.append(f"| {y} | {v['n']:,} | {v['refu']:,} | "
                     f"{pct(v['model_p']['cap_refu'])} | {pct(v['model_p']['cap_spec'])} | "
                     f"{pct(v['model_pq']['cap_refu'])} | {pct(v['model_pq']['cap_spec'])} |")

    lines.append("\n## 4. 稳健性\n")
    lo = res["live_only"]
    lines.append(f"- 活体子集：{lo['n']:,} 行，AUC {lo['auc']:.4f}，"
                 f"10% 票（p vs p×q）{pct(lo['rules']['model_p']['10%']['cap_refu'])} vs "
                 f"{pct(lo['rules']['model_pq']['10%']['cap_refu'])}；"
                 f"10% 个体 {pct(lo['rules']['model_p']['10%']['cap_spec'])} vs "
                 f"{pct(lo['rules']['model_pq']['10%']['cap_spec'])}")
    sz = res["seizure_outcome"]
    lines.append(f"- 查获结局：10% 票（p vs p×q）{pct(sz['model_p']['10%']['cap_refu'])} vs "
                 f"{pct(sz['model_pq']['10%']['cap_refu'])}；10% 个体 "
                 f"{pct(sz['model_p']['10%']['cap_spec'])} vs "
                 f"{pct(sz['model_pq']['10%']['cap_spec'])}")
    cap = res["qty_cap_sensitivity"]
    lines.append(f"- 数量上限敏感性（cap={cap['cap']:.0f}）：原始 10% 个体 "
                 f"{pct(cap['raw']['10%']['cap_spec'])}；截断 "
                 f"{pct(cap['capped']['10%']['cap_spec'])}；剔除>1e6 "
                 f"{pct(cap['exclude_big']['10%']['cap_spec'])}")
    sh = res["shipment_level"]
    if sh:
        lines.append(f"- 整票口径（{sh['n_shipments']:,} 票，2019–2020）：10% 票 "
                     f"{pct(sh['model_p_any']['10%']['cap_refu'])} vs "
                     f"{pct(sh['model_pq']['10%']['cap_refu'])}；10% 个体 "
                     f"{pct(sh['model_p_any']['10%']['cap_spec'])} vs "
                     f"{pct(sh['model_pq']['10%']['cap_spec'])}")

    lines.append("\n## 5. 闸门\n")
    for k, v in res["gates"].items():
        lines.append(f"- {k}: {'PASS' if v else 'FAIL'}")
    lines.append(f"\n**判决：{res['verdict']}**")

    (OUT / "p5_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
