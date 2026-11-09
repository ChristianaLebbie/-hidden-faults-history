"""Chapter 1: does history help whatever the classifier? Five models, with and without AMA history.

AMA (Age- and Missingness-Aware history framework) gives the classifier:
  * the current readings, with unmeasured sensors left missing (the model knows what was not measured),
  * the last earlier reading of every sensor and how many inspections old it is (strictly past only),
  * on data with several operating conditions, readings normalised within condition (training units only).
Baseline: current readings only (raw).

Each of five classifiers (logistic regression, random forest, histogram gradient boosting,
MLP neural network, XGBoost) is trained on both feature sets under the same grouped 5-fold CV.
Metrics: PR-AUC on all faults and, for AI4I-PMDI, PR-AUC on hidden faults only
(hidden faults vs all normal records; threshold-free).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import time
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from pfi.data import load_ai4i, load_cmapss
from pfi.schedules import ai4i_realistic, random_mix, apply_schedule, hidden_mask
from pfi.history import past_features
from pfi.methods import build_features, condition_normalise, ALL_STATS
from pfi.evaluate import cv_predict
from pfi.models import MODEL_LABELS
from pfi.runner import append, log

MODELS = ["logreg", "rf", "hgb", "mlp", "xgb"]
SEEDS = [0, 1, 2]
import os; SEEDS = SEEDS[:int(os.environ.get("NSEEDS", 3))]; ONLY = os.environ.get("ONLY")
rows = []


def score(ds, p, hidden=None):
    r = dict(PR_AUC=average_precision_score(ds.y, p), ROC_AUC=roc_auc_score(ds.y, p))
    if hidden is not None:
        keep = hidden | (ds.y == 0)
        r["PR_AUC_hidden"] = average_precision_score(ds.y[keep], p[keep])
    return r


for name in [n for n in ["AI4I-PMDI", "C-MAPSS FD001", "C-MAPSS FD002"] if not ONLY or ONLY in n]:
    if name == "AI4I-PMDI":
        ds = load_ai4i(); static = ds.static
    else:
        ds = load_cmapss(name.split()[-1])
        static = pd.get_dummies(ds.static["cond"].astype(str), prefix="cond", dtype=float)  # condition as categories
    multi = name == "C-MAPSS FD002"
    for seed in SEEDS:
        sch = ai4i_realistic(ds, 100 + seed) if name == "AI4I-PMDI" else random_mix(ds, 100 + seed)
        X = apply_schedule(ds.X, sch, ds.groups)
        H = past_features(X, ds.unit, ds.order, stats=("last", "stale"))
        hidden = hidden_mask(ds, sch) if name == "AI4I-PMDI" else None
        cache = {}

        def ama(tr):
            if not multi:
                return build_features("ama", X, H, static, tr)
            key = (len(tr), int(tr[0]), int(tr[-1]))
            if key not in cache:
                Z = condition_normalise(X, ds.static["cond"].to_numpy(), tr)
                cache.clear(); cache[key] = (Z, past_features(Z, ds.unit, ds.order, stats=("last", "stale")))
            Z, Hz = cache[key]
            return build_features("ama", Z, Hz, static, tr)

        builds = {"current": lambda tr: build_features("current", X, H, static, tr), "AMA": ama}
        for model in MODELS:
            for feats, build in builds.items():
                t0 = time.time()
                res = cv_predict(ds, build, seed, inner=False, params={"model": model})
                rows.append(dict(dataset=name, model=model, model_label=MODEL_LABELS[model], features=feats, seed=seed,
                                 seconds=time.time() - t0, **score(ds, res["p"], hidden)))
                log(name, seed, model, feats, round(rows[-1]["PR_AUC"], 3))
            append(rows, "exp13_model_comparison.csv")
log("done")
