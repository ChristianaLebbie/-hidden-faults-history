"""Real-data stress tests on MetroPT-3 (forward-chaining evaluation, as in exp10).

(a) shuffled / degraded history: none, true, random earlier window, history at least 1 hour old,
    at least 1 day old, mean of all earlier windows
(b) consecutive missing observations: outages with mean run of 30 min, 3 h or 12 h (30% of windows)
(c) acquisition-schedule change: trained under the uniform schedule, deployed under another
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from pfi.data import load_metropt
from pfi.schedules import random_mix, round_robin, apply_schedule, add_outages
from pfi.history import past_features, alt_history
from pfi.methods import build_features, reliability_table, horizons, ALL_STATS
from pfi.evaluate import fit_predict
from pfi.runner import append, log, PRED

ds = load_metropt(); start = ds.meta["start"]; Fl = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in ds.meta["failures"]]
mids = [Fl[k][1] + (Fl[k + 1][0] - Fl[k][1]) / 2 for k in range(3)]
BLOCKS = {"F2": (mids[0], mids[1]), "F3": (mids[1], mids[2]), "F4": (mids[2], start.max() + pd.Timedelta("1h"))}


def forward(Ftr_all, Fte_all, seed, tag=None):
    out = []; allp = np.full(len(ds.y), np.nan)
    for ev, (a, b) in BLOCKS.items():
        tr = np.where(start < a)[0]; te = np.where((start >= a) & (start < b))[0]
        p, _ = fit_predict(Ftr_all.iloc[tr], ds.y[tr], Fte_all.iloc[te], seed); yt = ds.y[te]; allp[te] = p
        days = (start.iloc[te].max() - start.iloc[te].min()).total_seconds() / 86400
        out.append(dict(event=ev, PR_AUC=average_precision_score(yt, p), ROC_AUC=roc_auc_score(yt, p),
                        FA_per_day_to_catch=((p >= p[yt == 1].max()) & (yt == 0)).sum() / days))
    if tag:
        np.save(PRED / f"{tag}.npy", allp)
    return out


rows = []
for seed in range(3):
    sch = random_mix(ds, 100 + seed); X = apply_schedule(ds.X, sch, ds.groups)
    H = past_features(X, ds.unit, ds.order, stats=ALL_STATS)
    # (a) shuffled / degraded history with the 'current + last value' model
    var = {"none": None, "true": H[[f"last_{c}" for c in X.columns]],
           "random_past": alt_history(X, ds.unit, ds.order, "random_past", seed),
           "older_1h": alt_history(X, ds.unit, ds.order, "old_past", seed, L=6),
           "older_1day": alt_history(X, ds.unit, ds.order, "old_past", seed, L=144),
           "unit_mean": alt_history(X, ds.unit, ds.order, "unit_mean", seed)}
    for v, Hv in var.items():
        F = pd.concat([X] + ([] if Hv is None else [Hv]), axis=1)
        for r in forward(F, F, seed, tag=f"metropt_shuffle_{v}_s{seed}"):
            rows.append(dict(test="shuffle", variant=v, method="cur_last", seed=seed, **r))
        log("shuffle", seed, v)
    # (b) outages
    for L in [0, 3, 18, 72]:
        Xo = add_outages(X, ds, L, 0.3 if L else 0.0, seed=1000 + seed)
        Ho = past_features(Xo, ds.unit, ds.order, stats=ALL_STATS)
        for m in ["current", "locf_ind_age", "cur_last", "gated"]:
            F = build_features(m, Xo, Ho, ds.static, np.where(start < BLOCKS["F2"][0])[0])
            for r in forward(F, F, seed):
                rows.append(dict(test="outage", variant=f"mean_run_{L}", method=m, seed=seed, **r))
        log("outage", seed, L)
    # (c) schedule change: train uniform, test other policies
    pols = {"uniform": None, "A-heavy": [0.6, 0.2, 0.2], "rare-C": [0.45, 0.45, 0.10], "round-robin": "rr"}
    for tp, pr in pols.items():
        st = round_robin(ds, 500 + seed) if pr == "rr" else random_mix(ds, 500 + seed, pr)
        Xt = apply_schedule(ds.X, st, ds.groups); Ht = past_features(Xt, ds.unit, ds.order, stats=ALL_STATS)
        for m in ["current", "locf_ind_age", "cur_last", "gated"]:
            tr0 = np.where(start < BLOCKS["F2"][0])[0]; hz = horizons(reliability_table(X, H, tr0), 0.5)  # training schedule
            Ftr = build_features(m, X, H, ds.static, tr0, hz=hz)
            Fte = build_features(m, Xt, Ht, ds.static, tr0, hz=hz)
            for r in forward(Ftr, Fte, seed):
                rows.append(dict(test="schedule", variant=f"uniform->{tp}", method=m, seed=seed, **r))
        log("schedule", seed, tp)
    append(rows, "exp12_metropt_stress.csv")
log("done")
