"""Point 5: reconcile MetroPT Tables 3 (exp10) and 7 (exp16) by rerunning both training settings side by side."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from pfi.data import load_metropt
from pfi.schedules import random_mix, apply_schedule
from pfi.history import past_features
from pfi.methods import build_features, ALL_STATS
from pfi.evaluate import fit_predict
from pfi.runner import append, log

ds = load_metropt(); start = ds.meta["start"]; F = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in ds.meta["failures"]]
mids = [F[k][1] + (F[k + 1][0] - F[k][1]) / 2 for k in range(3)]
blocks = {"F2": (mids[0], mids[1], F[1]), "F3": (mids[1], mids[2], F[2]), "F4": (mids[2], start.max() + pd.Timedelta("1h"), F[3])}
METHODS = {"current": "Current readings only", "cur_last": "AMA without age", "cur_last_age": "AMA"}
TARGETS = [1.0, 5.0]   # false alarms per day on the calibration normals
WIN = pd.Timedelta("10min")
SEEDS = [0, 1, 2, 3, 4]
rows = []
OUTD = Path(__file__).resolve().parents[1] / "results" / "supervisor_nov"
for seed in SEEDS:
    sch = random_mix(ds, 100 + seed); X = apply_schedule(ds.X, sch, ds.groups)
    H = past_features(X, ds.unit, ds.order, stats=ALL_STATS)
    for ev, (a, b, _) in blocks.items():
        tr = np.where(start < a)[0]; te = np.where((start >= a) & (start < b))[0]
        trn = tr[ds.y[tr] == 0]; cut = start.iloc[trn].quantile(0.8)
        cal = trn[(start.iloc[trn] >= cut).to_numpy()]; fit = np.setdiff1d(tr, cal); yt = ds.y[te]
        for m in ["current", "cur_last"]:
            for setting, trainset in [("Table 3 setting (exp10): all training windows", tr),
                                      ("Table 7 setting (exp16): latest 20% of training normals held out", fit)]:
                Fm = build_features(m, X, H, ds.static, trainset)
                p, _ = fit_predict(Fm.iloc[trainset], ds.y[trainset], Fm.iloc[te], seed)
                rows.append(dict(event=ev, seed=seed, method=m, setting=setting, n_train=len(trainset),
                                 days_removed=float((start.iloc[cal].max() - start.iloc[cal].min()).total_seconds() / 86400) if trainset is fit else 0.0,
                                 PR_AUC=average_precision_score(yt, p)))
    print(seed, flush=True)
pd.DataFrame(rows).to_csv(OUTD / "p5_metropt_reconcile.csv", index=False)
