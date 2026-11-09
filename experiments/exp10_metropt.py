"""Real-data check on MetroPT-3 (one metro-train air compressor, Feb-Sep 2020, four reported air leaks).

Inspections are 10-minute windows. Each inspection reports one of three signal groups at random:
  A pressures (TP2, TP3, H1, DV pressure, reservoirs), B motor current and oil temperature,
  C valve and switch states.
Forward-chaining evaluation: for each of failures 2, 3 and 4, the model is trained only on data
recorded before the test period begins (which contains the earlier failures), and tested on the
period around that failure. There is no shuffling across time and no future data in training.
Metrics are threshold-free: PR-AUC in the test period and the number of false alarms per day that
must be accepted to catch the failure at all.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from pfi.data import load_metropt
from pfi.schedules import random_mix, apply_schedule
from pfi.history import past_features
from pfi.methods import build_features, reliability_table, ALL_STATS
from pfi.evaluate import fit_predict
from pfi.runner import append, log, RES

ds = load_metropt(); start = ds.meta["start"]; F = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in ds.meta["failures"]]
mids = [F[k][1] + (F[k + 1][0] - F[k][1]) / 2 for k in range(3)]
blocks = {"F2": (mids[0], mids[1]), "F3": (mids[1], mids[2]), "F4": (mids[2], start.max() + pd.Timedelta("1h"))}
METHODS = ["current", "locf_ind_age", "trend_impute", "cur_last", "cur_last_age", "cur_window", "gated"]
rows = []; rel = []
for seed in range(5):
    sch = random_mix(ds, 100 + seed); X = apply_schedule(ds.X, sch, ds.groups)
    H = past_features(X, ds.unit, ds.order, stats=ALL_STATS)
    if seed == 0:
        t = reliability_table(X, H, np.arange(len(ds.y))); t["dataset"] = "MetroPT-3"; rel.append(t)
    for ev, (a, b) in blocks.items():
        tr = np.where(start < a)[0]; te = np.where((start >= a) & (start < b))[0]
        days = (start.iloc[te].max() - start.iloc[te].min()).total_seconds() / 86400
        for m in ["complete"] + METHODS:
            Fm = pd.concat([ds.X, ds.static], axis=1) if m == "complete" else build_features(m, X, H, ds.static, tr)
            p, _ = fit_predict(Fm.iloc[tr], ds.y[tr], Fm.iloc[te], seed)
            yt = ds.y[te]; best = p[yt == 1].max()
            fa_needed = ((p >= best) & (yt == 0)).sum() / days
            rows.append(dict(event=ev, seed=seed, method=m, n_train=len(tr), train_faults=int(ds.y[tr].sum()), n_test=len(te),
                             test_faults=int(yt.sum()), PR_AUC=average_precision_score(yt, p), ROC_AUC=roc_auc_score(yt, p),
                             FA_per_day_to_catch=fa_needed, base_rate=yt.mean()))
            log(ev, seed, m, round(rows[-1]["PR_AUC"], 3), round(fa_needed, 2))
        append(rows, "exp10_metropt.csv")
pd.concat(rel).to_csv(RES / "exp10_metropt_reliability.csv", index=False)
log("done")
