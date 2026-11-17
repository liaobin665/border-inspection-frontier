"""Does folding the validation year into training buy anything?

Two protocols on the same features and the same test years (2019-2022):

A  current: train 2010-2017, validate 2018, early stop  -> test
B  merged : train 2010-2018, no early stopping, n_trees fixed at A's best_iter
            (the iteration count was chosen on 2018 before the merge, so the
             test years are still untouched)

If B is not clearly better, the honest simplification is to keep A and say
"tuning and evaluation are strictly separated" rather than to merge and lose
the early-stopping guard.

Also writes the ROC curve of protocol A for the concept-figure inset.

Output: results/p6_split_probe.json, results/p6_roc.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data_layer as p5  # noqa: E402
import analysis as p6  # noqa: E402

OUT = HERE.parent / "results"
WHOLE = p6.WHOLE
CAP = p6.CAP


def evaluate(scores, q, refu):
    spec = np.where(refu == 1, np.minimum(q, CAP), 0.0)
    res = p6.eval_pair(scores, np.minimum(q, CAP), refu, spec, ks=(0.10,))["10"]
    return dict(auc=float(roc_auc_score(refu, scores)),
                ap=float(average_precision_score(refu, scores)), **res)


def main() -> None:
    df = p5.pull()
    df = p6.add_counts(df)
    prior = float(df.loc[df["y"].between(*p5.TRAIN_YEARS), "refu"].mean())
    df = p6.apply_prior(df, prior)
    test = df[df["y"].between(*p5.TEST_YEARS)]
    sub = test[test["desc_"].isin(WHOLE)].reset_index(drop=True)
    q = sub["qty"].to_numpy()
    refu = sub["refu"].to_numpy()

    # A: current protocol
    mA, infoA = p6.fit(df, p5.TRAIN_YEARS, p5.VALID_YEAR)
    pA = mA.predict_proba(sub[p5.FEATS])[:, 1]
    resA = evaluate(pA, q, refu)

    # B: merged training, fixed number of trees
    import lightgbm as lgb
    trB = df[df["y"].between(2010, 2018)]
    mB = lgb.LGBMClassifier(
        objective="binary", n_estimators=int(infoA["best_iter"]), learning_rate=0.05,
        num_leaves=127, min_child_samples=300, subsample=0.9, subsample_freq=1,
        colsample_bytree=0.9, reg_lambda=1.0, max_bin=255, n_jobs=8,
        random_state=p5.SEED, verbose=-1)
    mB.fit(trB[p5.FEATS], trB["refu"], categorical_feature=p5.CATS)
    pB = mB.predict_proba(sub[p5.FEATS])[:, 1]
    resB = evaluate(pB, q, refu)

    fpr, tpr, _ = roc_curve(refu, pA)
    idx = np.unique(np.linspace(0, len(fpr) - 1, 400).astype(int))
    (OUT / "p6_roc.json").write_text(json.dumps(
        dict(auc=float(resA["auc"]), fpr=fpr[idx].round(6).tolist(),
             tpr=tpr[idx].round(6).tolist()), indent=1), encoding="utf-8")

    out = dict(
        protocol_A=dict(train="2010-2017", valid="2018", early_stop=True,
                        best_iter=int(infoA["best_iter"]), n_train=int(infoA["n_train"]),
                        result=resA),
        protocol_B=dict(train="2010-2018", valid=None,
                        n_trees=int(infoA["best_iter"]), n_train=int(len(trB)),
                        result=resB),
        delta=dict(auc=resB["auc"] - resA["auc"], ap=resB["ap"] - resA["ap"],
                   p_cap_refu=resB["p_cap_refu"] - resA["p_cap_refu"],
                   p_cap_spec=resB["p_cap_spec"] - resA["p_cap_spec"],
                   pq_cap_refu=resB["pq_cap_refu"] - resA["pq_cap_refu"],
                   pq_cap_spec=resB["pq_cap_spec"] - resA["pq_cap_spec"],
                   jaccard=resB["jaccard"] - resA["jaccard"]),
    )
    (OUT / "p6_split_probe.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
