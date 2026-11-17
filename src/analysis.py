"""Deep checks for the dual-objective paper.

Sections
--------
rolling : expanding-window origin, test year 2013..2022
          (train 2010..y-2, validate y-1), whole-animal test rows,
          10% budget, p vs p*q, cap 1e4.
bootstrap : species-clustered bootstrap of the 2019-2022 main result.
sweep    : budget 0.01..0.50 capture curves + budget needed for 80% coverage.
lambda   : score = p * q**lam, lam in {0,.25,.5,.75,1}; 10% budget frontier.
composition : who is in each 10% list.
concentration : how concentrated refused animals are.
alt_score : simple historical-rate scorers instead of LightGBM.
purpose_p : same protocol on personal imports (purpose='P').
shipment : control_number aggregation for 2019-2020.

Usage: python p6_deep.py [section ...]
Output: results/p6_<section>.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data_layer as p5  # noqa: E402

OUT = HERE.parent / "results"
WHOLE = ("LIV", "BOD", "DEA", "TRO", "SKE", "FIG")
CAP = 1e4
KS = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)
RATE_KEYS = [(["sp"], "sp"), (["org"], "org"), (["sp", "org"], "sporg"),
             (["port"], "port"), (["desc_"], "desc")]


# ---------------------------------------------------------------- features
def add_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Cumulative (before year y) counts for each key set. No prior applied."""
    for keys, tag in RATE_KEYS:
        g = (df.groupby(["y"] + keys, observed=True)
               .agg(n=("refu", "size"), r=("refu", "sum")).reset_index())
        g = g.sort_values("y", kind="stable")
        g["cn"] = g.groupby(keys, observed=True)["n"].cumsum() - g["n"]
        g["cr"] = g.groupby(keys, observed=True)["r"].cumsum() - g["r"]
        out = df.merge(g[["y"] + keys + ["cn", "cr"]], on=["y"] + keys, how="left")
        df[f"cn_{tag}"] = out["cn"].fillna(0.0).to_numpy()
        df[f"cr_{tag}"] = out["cr"].fillna(0.0).to_numpy()
    return df


def apply_prior(df: pd.DataFrame, prior: float, alpha: float = 50.0) -> pd.DataFrame:
    for _, tag in RATE_KEYS:
        df[f"hist_{tag}"] = (df[f"cr_{tag}"] + alpha * prior) / (df[f"cn_{tag}"] + alpha)
    for c in ("sp", "org", "port", "source", "band"):
        df[c + "_c"] = df[c].astype("category")
    df["desc_c"] = df["desc_"].astype("category")
    return df


def fit(df: pd.DataFrame, train_years, valid_year, seed: int = p5.SEED):
    tr = df[df["y"].between(*train_years)]
    va = df[df["y"] == valid_year]
    import lightgbm as lgb
    m = lgb.LGBMClassifier(
        objective="binary", n_estimators=3000, learning_rate=0.05,
        num_leaves=127, min_child_samples=300, subsample=0.9,
        subsample_freq=1, colsample_bytree=0.9, reg_lambda=1.0,
        max_bin=255, n_jobs=8, random_state=seed, verbose=-1)
    m.fit(tr[p5.FEATS], tr["refu"], eval_set=[(va[p5.FEATS], va["refu"])],
          eval_metric="auc", categorical_feature=p5.CATS,
          callbacks=[lgb.early_stopping(100, verbose=False)])
    return m, dict(n_train=int(len(tr)), n_valid=int(len(va)),
                   best_iter=int(m.best_iteration_ or 3000),
                   valid_auc=float(m.best_score_["valid_0"]["auc"]))


# ---------------------------------------------------------------- metrics
def cap_at(score: np.ndarray, refu: np.ndarray, spec: np.ndarray, k: float):
    o = np.argsort(-np.asarray(score, float), kind="stable")
    cr = np.cumsum(refu[o]); cs = np.cumsum(spec[o])
    m = max(int(round(k * len(o))), 1)
    return (float(cr[m - 1] / max(refu.sum(), 1.0)),
            float(cs[m - 1] / max(spec.sum(), 1.0)), o[:m])


def jacc(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = set(a.tolist()), set(b.tolist())
    return len(sa & sb) / len(sa | sb)


def eval_pair(p: np.ndarray, q: np.ndarray, refu: np.ndarray, spec: np.ndarray,
              ks=KS):
    out = {}
    for k in ks:
        c1, s1, a1 = cap_at(p, refu, spec, k)
        c2, s2, a2 = cap_at(p * q, refu, spec, k)
        out[f"{int(k*100)}"] = dict(
            p_cap_refu=c1, p_cap_spec=s1, pq_cap_refu=c2, pq_cap_spec=s2,
            jaccard=jacc(a1, a2), p_in_pq=len(set(a1.tolist()) & set(a2.tolist())) / len(a1))
    return out


def budget_for(score: np.ndarray, refu: np.ndarray, spec: np.ndarray,
               target: float, which: str = "spec") -> float | None:
    o = np.argsort(-np.asarray(score, float), kind="stable")
    cs = np.cumsum(spec[o]) / max(spec.sum(), 1.0)
    cr = np.cumsum(refu[o]) / max(refu.sum(), 1.0)
    arr = cs if which == "spec" else cr
    hit = np.nonzero(arr >= target)[0]
    return float((hit[0] + 1) / len(o)) if len(hit) else None


# ---------------------------------------------------------------- sections
def section_rolling(df: pd.DataFrame) -> dict:
    test_whole = df[df["desc_"].isin(WHOLE)]
    out = {}
    for y in range(2013, 2023):
        t0 = time.time()
        train = (2010, y - 2)
        prior = float(df.loc[df["y"].between(*train), "refu"].mean())
        d = apply_prior(df.copy(), prior)
        m, info = fit(d, train, y - 1)
        sub = test_whole[test_whole["y"] == y].copy()
        p = m.predict_proba(sub[p5.FEATS])[:, 1]
        q = np.minimum(sub["qty"].to_numpy(), CAP)
        refu = sub["refu"].to_numpy()
        spec = np.where(refu == 1, q, 0.0)
        qv = np.minimum(sub["qty"].to_numpy(), CAP)
        r = eval_pair(p, qv, refu, spec, ks=(0.10,))
        mid = {}
        for lam in (0.25, 0.5):
            c, a, _ = cap_at(p * np.power(qv, lam), refu, spec, 0.10)
            mid[f"lam{lam}"] = dict(cap_refu=float(c), cap_spec=float(a))
        out[str(y)] = dict(info=info, rows=int(len(sub)), refu=int(refu.sum()),
                           refused_animals=float(spec.sum()),
                           auc=float(roc_auc_score(refu, p)),
                           ap=float(average_precision_score(refu, p)),
                           at10=r["10"], mid=mid, secs=round(time.time() - t0, 1))
        print(f"[rolling {y}] rows={len(sub)} refu={refu.sum()} "
              f"p={r['10']['p_cap_refu']:.3f}/{r['10']['p_cap_spec']:.3f} "
              f"pq={r['10']['pq_cap_refu']:.3f}/{r['10']['pq_cap_spec']:.3f} "
              f"{out[str(y)]['secs']}s", flush=True)
    return out


def section_bootstrap(df: pd.DataFrame, p: np.ndarray, sub: pd.DataFrame,
                      reps: int = 500) -> dict:
    q = np.minimum(sub["qty"].to_numpy(), CAP)
    refu = sub["refu"].to_numpy()
    spec = np.where(refu == 1, q, 0.0)
    sp = sub["sp"].to_numpy()
    codes, uniq = pd.factorize(sp)
    idx_by_sp = [np.nonzero(codes == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(20261108)
    draws = []
    for _ in range(reps):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([idx_by_sp[i] for i in pick])
        r = eval_pair(p[idx], q[idx], refu[idx], spec[idx], ks=(0.10,))["10"]
        draws.append([r["p_cap_refu"], r["p_cap_spec"],
                      r["pq_cap_refu"], r["pq_cap_spec"],
                      r["pq_cap_spec"] - r["p_cap_spec"],
                      r["p_cap_refu"] - r["pq_cap_refu"], r["jaccard"]])
    a = np.asarray(draws)
    names = ["p_cap_refu", "p_cap_spec", "pq_cap_refu", "pq_cap_spec",
             "diff_spec", "diff_refu", "jaccard"]
    point = eval_pair(p, q, refu, spec, ks=(0.10,))["10"]
    pv = [point["p_cap_refu"], point["p_cap_spec"], point["pq_cap_refu"],
          point["pq_cap_spec"],
          point["pq_cap_spec"] - point["p_cap_spec"],
          point["p_cap_refu"] - point["pq_cap_refu"], point["jaccard"]]
    return dict(reps=reps, n_species=int(len(uniq)),
                point=dict(zip(names, [float(x) for x in pv])),
                ci95={n: [float(np.percentile(a[:, i], 2.5)),
                          float(np.percentile(a[:, i], 97.5))]
                      for i, n in enumerate(names)})


def section_sweep(p: np.ndarray, q: np.ndarray, refu: np.ndarray,
                  spec: np.ndarray) -> dict:
    rows = []
    for k in KS:
        c1, s1, _ = cap_at(p, refu, spec, k)
        c2, s2, _ = cap_at(p * q, refu, spec, k)
        rows.append(dict(budget=k, p_cap_refu=c1, p_cap_spec=s1,
                         pq_cap_refu=c2, pq_cap_spec=s2))
    return dict(curve=rows,
                budget_80spec_p=budget_for(p, refu, spec, .8, "spec"),
                budget_80spec_pq=budget_for(p * q, refu, spec, .8, "spec"),
                budget_80refu_p=budget_for(p, refu, spec, .8, "refu"),
                budget_80refu_pq=budget_for(p * q, refu, spec, .8, "refu"),
                budget_50spec_p=budget_for(p, refu, spec, .5, "spec"),
                budget_50spec_pq=budget_for(p * q, refu, spec, .5, "spec"))


def section_lambda(p: np.ndarray, q: np.ndarray, refu: np.ndarray,
                   spec: np.ndarray) -> dict:
    out = []
    for lam in (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0):
        s = p * np.power(q, lam)
        c, a, _ = cap_at(s, refu, spec, 0.10)
        out.append(dict(lam=lam, cap_refu=c, cap_spec=a))
    return dict(lam=out)


def section_composition(p: np.ndarray, q_raw: np.ndarray, sub: pd.DataFrame,
                        refu: np.ndarray, spec: np.ndarray) -> dict:
    out = {}
    idxs = {}
    for name, s in (("p", p), ("pq", p * q_raw)):
        _, _, idx = cap_at(s, refu, spec, 0.10)
        idxs[name] = idx
        d = sub.iloc[idx]
        animal_by_sp = (pd.Series(spec[idx], index=d.index)
                          .groupby(d["sp"]).sum().sort_values(ascending=False))
        sp_counts = d.loc[d["refu"] == 1, "sp"].value_counts()
        out[name] = dict(
            rows=int(len(d)),
            share_band1=float((d["qty"] <= 10).mean()),
            share_band2=float(((d["qty"] > 10) & (d["qty"] <= 100)).mean()),
            share_band3=float(((d["qty"] > 100) & (d["qty"] <= 1000)).mean()),
            share_band4=float((d["qty"] > 1000).mean()),
            share_live_desc=float((d["desc_"] == "LIV").mean()),
            share_refu_rows=float(d["refu"].mean()),
            n_countries=int(d["org"].nunique()),
            n_species=int(d["sp"].nunique()),
            median_qty=float(d["qty"].median()),
            refused_animals=float(spec[idx].sum()),
            animal_share_top1row=float(spec[idx].max() / max(spec[idx].sum(), 1)),
            top5_refused_species=[[str(a), int(b)] for a, b in sp_counts.head(5).items()],
            top5_animal_species=[[str(a), float(b)] for a, b in
                                 animal_by_sp.head(5).items()],
            top5_group=[[str(a), int(b)] for a, b in d["group_"].value_counts().head(5).items()])
    a, b = set(idxs["p"].tolist()), set(idxs["pq"].tolist())
    refu_idx = set(np.nonzero(refu == 1)[0].tolist())
    shared = a & b
    p_only = a - b
    pq_only = b - a
    tot_animals = max(float(spec.sum()), 1.0)
    out["detail"] = dict(
        shared_rows=len(shared),
        refused_rows_p=len(a & refu_idx), refused_rows_pq=len(b & refu_idx),
        refused_rows_shared=len(a & b & refu_idx),
        share_p_refused_also_pq=len(a & b & refu_idx) / max(len(a & refu_idx), 1),
        spearman_p_vs_pq=float(pd.Series(p).corr(pd.Series(p * q_raw), method="spearman")),
        animals_shared=float(spec[list(shared)].sum()),
        animals_p_only=float(spec[list(p_only)].sum()),
        animals_pq_only=float(spec[list(pq_only)].sum()),
        animals_share_shared=float(spec[list(shared)].sum() / tot_animals),
        animals_share_p_only=float(spec[list(p_only)].sum() / tot_animals),
        animals_share_pq_only=float(spec[list(pq_only)].sum() / tot_animals))
    return out


def section_concentration(sub: pd.DataFrame) -> dict:
    q = np.sort(np.where(sub["refu"] == 1, sub["qty"].to_numpy(), 0.0))[::-1]
    tot = float(q.sum())
    nz = q[q > 0]
    asc = np.sort(nz)
    n = len(asc)
    gini = float((2 * np.sum(np.arange(1, n + 1) * asc) / (n * asc.sum())) - (n + 1) / n) \
        if n > 1 else 0.0
    out = dict(refused_rows=int(len(nz)), refused_animals=tot,
               top1=float(q[:1].sum() / tot), top10=float(q[:10].sum() / tot),
               top100=float(q[:100].sum() / tot),
               share_rows_for_50pct=float((np.nonzero(np.cumsum(q) / tot >= .5)[0][0] + 1) / len(nz)),
               gini=float(gini),
               max_row=float(q[0]) if len(nz) else 0.0)
    return out


def section_alt_score(df: pd.DataFrame, sub: pd.DataFrame) -> dict:
    out = {}
    q = np.minimum(sub["qty"].to_numpy(), CAP)
    refu = sub["refu"].to_numpy()
    spec = np.where(refu == 1, q, 0.0)
    for col in ("hist_sp", "hist_sporg", "hist_port"):
        s = sub[col].to_numpy()
        out[col] = dict(auc=float(roc_auc_score(refu, s)),
                        **eval_pair(s, np.minimum(sub["qty"].to_numpy(), CAP),
                                    refu, spec, ks=(0.10,))["10"])
    return out


def section_purpose_p() -> dict:
    import duckdb
    con = duckdb.connect(str(p5.LEMIS), read_only=True)
    q = """
    select coalesce(nullif(corrected,''), species) as sp, country_origin as org,
           port, description as desc_, source, try_cast(quantity as double) as qty,
           try_cast(sYear as int) as y,
           month(try_cast(shipment_date as timestamp)) as month,
           dayofweek(try_cast(shipment_date as timestamp)) as dow,
           case when action='R' then 1 else 0 end as refu,
           case when disposition='S' then 1 else 0 end as seiz,
           group_, control_number
    from lemis
    where import_export='I' and unit='NO' and try_cast(quantity as double)>0
      and purpose='P' and try_cast(sYear as int) between 2010 and 2022
      and coalesce(nullif(corrected,''), species) is not null
      and trim(coalesce(nullif(corrected,''), species)) <> ''
      and upper(coalesce(nullif(corrected,''), species)) not like '%UNKNOWN%'
    """
    d = con.execute(q).df(); con.close()
    d["qty"] = d["qty"].astype(float)
    d["refused_qty"] = np.where(d["refu"] == 1, d["qty"], 0.0)
    d["log_qty"] = np.log1p(d["qty"])
    d["band"] = np.select([d["qty"] <= 10, d["qty"] <= 100, d["qty"] <= 1000],
                          [1, 2, 3], default=4)
    d["month"] = d["month"].fillna(-1).astype(int)
    d["dow"] = d["dow"].fillna(-1).astype(int)
    for c in ("sp", "org", "desc_", "source"):
        d[c] = d[c].fillna("NA").astype(str)
    prior = float(d.loc[d["y"].between(*p5.TRAIN_YEARS), "refu"].mean())
    d = add_counts(d)
    d = apply_prior(d, prior)
    m, info = fit(d, p5.TRAIN_YEARS, p5.VALID_YEAR)
    test = d[d["y"].between(*p5.TEST_YEARS)]
    sub = test[test["desc_"].isin(WHOLE)].copy()
    p = m.predict_proba(sub[p5.FEATS])[:, 1]
    qv = np.minimum(sub["qty"].to_numpy(), CAP)
    refu = sub["refu"].to_numpy(); spec = np.where(refu == 1, qv, 0.0)
    by_band = (sub.groupby("band", observed=True)
                  .agg(rows=("refu", "size"), refu=("refu", "sum")).reset_index())
    by_band["rate"] = by_band["refu"] / by_band["rows"]
    return dict(info=info, test_rows=int(len(sub)), test_refu=int(refu.sum()),
                prevalence=float(refu.mean()),
                auc=float(roc_auc_score(refu, p)),
                ap=float(average_precision_score(refu, p)),
                at10=eval_pair(p, np.minimum(sub["qty"].to_numpy(), CAP), refu, spec, ks=(0.10,))["10"],
                by_band=by_band.to_dict("records"))


def section_shipment(sub: pd.DataFrame, p: np.ndarray) -> dict:
    m = sub["y"].isin([2019, 2020]) & sub["control_number"].notna()
    d = sub.loc[m].copy()
    d["p"] = p[m]
    d["log1mp"] = np.log1p(-np.clip(d["p"], 0, 1 - 1e-12))
    g = d.groupby("control_number", observed=True).agg(
        any_refu=("refu", "max"), refused_qty=("refused_qty", "sum"),
        p_any=("log1mp", lambda x: 1.0 - np.exp(x.sum()))).reset_index()
    g["p_qty"] = g["control_number"].map(
        d.assign(pq=d["p"] * d["qty"]).groupby("control_number", observed=True)["pq"].sum())
    refu = g["any_refu"].to_numpy().astype(float)
    spec = g["refused_qty"].to_numpy().astype(float)
    out = dict(n_shipments=int(len(g)), refusal_rate=float(refu.mean()),
               rows_per_shipment=float(len(d) / max(len(g), 1)))
    for name, s in (("p_any", g["p_any"].to_numpy()), ("p_qty", g["p_qty"].to_numpy())):
        out[name] = eval_pair(s, np.ones(len(g)), refu, spec, ks=(0.10,))["10"]
    return out


# ---------------------------------------------------------------- main
def main(which):
    t0 = time.time()
    df = p5.pull()
    print("pulled", len(df), flush=True)
    df = add_counts(df)
    prior = float(df.loc[df["y"].between(*p5.TRAIN_YEARS), "refu"].mean())
    df = apply_prior(df, prior)

    if "rolling" in which:
        (OUT / "p6_rolling.json").write_text(
            json.dumps(section_rolling(df), ensure_ascii=False, indent=2), encoding="utf-8")
        print("rolling done", time.time() - t0, flush=True)

    need_main = {"bootstrap", "sweep", "lambda", "composition",
                 "concentration", "alt_score", "shipment"} & set(which)
    if need_main:
        m, info = fit(df, p5.TRAIN_YEARS, p5.VALID_YEAR)
        test = df[df["y"].between(*p5.TEST_YEARS)]
        sub = test[test["desc_"].isin(WHOLE)].copy().reset_index(drop=True)
        p = m.predict_proba(sub[p5.FEATS])[:, 1]
        q = sub["qty"].to_numpy()
        refu = sub["refu"].to_numpy()
        spec = np.where(refu == 1, np.minimum(q, CAP), 0.0)
        base = dict(info=info, rows=int(len(sub)), refu=int(refu.sum()),
                    refused_animals=float(spec.sum()),
                    auc=float(roc_auc_score(refu, p)),
                    ap=float(average_precision_score(refu, p)))
        if "bootstrap" in which:
            base["bootstrap"] = section_bootstrap(df, p, sub)
            print("bootstrap done", time.time() - t0, flush=True)
        if "sweep" in which:
            base["sweep"] = section_sweep(p, np.minimum(q, CAP), refu, spec)
            print("sweep done", flush=True)
        if "lambda" in which:
            base["lambda"] = section_lambda(p, np.minimum(q, CAP), refu, spec)
            print("lambda done", flush=True)
        if "composition" in which:
            base["composition"] = section_composition(p, q, sub, refu, spec)
            print("composition done", flush=True)
        if "concentration" in which:
            base["concentration"] = section_concentration(sub)
            print("concentration done", flush=True)
        if "alt_score" in which:
            base["alt_score"] = section_alt_score(df, sub)
            print("alt done", flush=True)
        if "shipment" in which:
            base["shipment"] = section_shipment(sub, p)
            print("shipment done", flush=True)
        (OUT / "p6_main.json").write_text(
            json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")
        print("main written", flush=True)

    if "purpose_p" in which:
        (OUT / "p6_purpose_p.json").write_text(
            json.dumps(section_purpose_p(), ensure_ascii=False, indent=2), encoding="utf-8")
        print("purpose_p done", flush=True)
    print("ALL DONE", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["rolling"])
