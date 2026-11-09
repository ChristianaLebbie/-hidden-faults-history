"""Grouped cross-validation, thresholds, calibration inputs, metrics and bootstrap tests.

Outer loop: 5-fold StratifiedGroupKFold by unit (no unit in both training and test).
Inside each training fold, a 4-fold grouped inner CV produces out-of-fold probabilities for
the training records; these are the only data used to choose thresholds and fit calibrators.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

XGB = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9, eval_metric="logloss",
           verbosity=0, n_jobs=2, tree_method="hist")


def fit_predict(Xtr, ytr, Xte, seed, params=None):
    if params and "model" in params:          # another classifier (see models.py); default stays XGBoost
        from .models import make_model
        m = make_model(params["model"], seed); m.fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1], m
    m = xgb.XGBClassifier(**(params or XGB), random_state=seed)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1], m


def f1_threshold(y, p):
    """Threshold maximising F1 on (y, p); candidates are the observed probabilities."""
    order = np.argsort(-p); ys = y[order]; tp = np.cumsum(ys); k = np.arange(1, len(p) + 1)
    f1 = 2 * tp / (k + ys.sum()); i = int(np.argmax(f1))
    return float(p[order][i])


def cv_predict(ds, build, seed, inner=True, params=None, return_models=False):
    """build(train_idx) -> feature DataFrame for all records (train-only statistics allowed).

    Returns dict with out-of-fold test probabilities `p`, per-record fold id, and for each
    fold the inner out-of-fold probabilities on its training records (`inner_p`, `inner_idx`).
    """
    y, g = ds.y, ds.unit; n = len(y)
    out = dict(p=np.full(n, np.nan), fold=np.full(n, -1), inner=[], models=[])
    outer = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    for f, (tr, te) in enumerate(outer.split(np.zeros(n), y, g)):
        F = build(tr)
        p_te, model = fit_predict(F.iloc[tr], y[tr], F.iloc[te], seed, params)
        out["p"][te] = p_te; out["fold"][te] = f
        if return_models:
            out["models"].append((model, F, tr, te))
        if inner:
            ip = np.full(len(tr), np.nan)
            for a, b in StratifiedGroupKFold(4, shuffle=True, random_state=seed + 1).split(tr, y[tr], g[tr]):
                ip[b], _ = fit_predict(F.iloc[tr[a]], y[tr[a]], F.iloc[tr[b]], seed, params)
            out["inner"].append(dict(idx=tr, p=ip, test=te))
    return out


def thresholds_from_inner(res, y):
    """Per-record alarm threshold chosen on the inner out-of-fold predictions of its outer fold."""
    t = np.full(len(y), np.nan)
    for inn in res["inner"]:
        t[inn["test"]] = f1_threshold(y[inn["idx"]], inn["p"])
    return t


def metrics(y, p, t=None, extra_masks=None):
    m = dict(PR_AUC=average_precision_score(y, p), ROC_AUC=roc_auc_score(y, p), Brier=brier_score_loss(y, p))
    if t is not None:
        a = p >= t
        m.update(recall=a[y == 1].mean(), precision=y[a].mean() if a.any() else 0.0,
                 FA_per100=100 * (a & (y == 0)).sum() / (y == 0).sum())
        for name, mk in (extra_masks or {}).items():
            m[f"recall_{name}"] = a[mk].mean() if mk.any() else np.nan
    return m


def ece(y, p, bins=10, strategy="width"):
    if strategy == "width":
        e = np.linspace(0, 1, bins + 1)
    else:
        e = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    b = np.clip(np.searchsorted(e, p, side="right") - 1, 0, len(e) - 2); tot = 0.0
    for k in range(len(e) - 1):
        m = b == k
        if m.any():
            tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return tot


def bootstrap_diff(y, units, pa_list, pb_list, B=1000, seed=0, metric=average_precision_score):
    """Paired bootstrap over units, pooled over seeds: metric(b) - metric(a)."""
    uniq = np.unique(units); idx = {u: np.where(units == u)[0] for u in uniq}; r = np.random.default_rng(seed)
    obs = np.mean([metric(y, pb) - metric(y, pa) for pa, pb in zip(pa_list, pb_list)])
    D = np.empty(B)
    for b in range(B):
        ii = np.concatenate([idx[u] for u in r.choice(uniq, len(uniq))])
        D[b] = np.mean([metric(y[ii], pb[ii]) - metric(y[ii], pa[ii]) for pa, pb in zip(pa_list, pb_list)])
    lo, hi = np.percentile(D, [2.5, 97.5])
    p_two = 2 * min((D <= 0).mean(), (D >= 0).mean())
    return dict(diff=obs, lo=lo, hi=hi, p=max(p_two, 1 / B))


def holm(pvals):
    p = np.asarray(pvals, float); o = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(p) - r) * p[i])); adj[i] = run
    return adj
