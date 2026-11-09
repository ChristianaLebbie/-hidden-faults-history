"""Control: does history help even when nothing is missing?

Separates two reasons history can help: (a) it recovers evidence the current inspection did not
measure, and (b) temporal context is informative in itself (trend, persistence, smoothing).
Compares complete current readings with and without the previous inspection's readings.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from pfi.data import load_ai4i, load_cmapss, load_metropt
from pfi.history import past_features
from pfi.methods import condition_normalise
from pfi.evaluate import fit_predict
from pfi.runner import run, append, log

rows = []
ai = load_ai4i(); Hc = past_features(ai.X, ai.unit, ai.order, stats=("last",))
for seed in range(5):
    for name, F in [("complete", pd.concat([ai.X, ai.static], axis=1)), ("complete+last", pd.concat([ai.X, Hc, ai.static], axis=1))]:
        r, _, _ = run(ai, lambda tr: F, seed, save_pred=False)
        rows.append(dict(dataset="AI4I-PMDI", variant=name, seed=seed, PR_AUC=r["PR_AUC"], ROC_AUC=r["ROC_AUC"])); log("AI4I", name, seed, round(r["PR_AUC"], 3))
for fd in ["FD001", "FD002"]:
    ds = load_cmapss(fd); cond = ds.static["cond"].to_numpy()
    for seed in range(3):
        def mk(tr, hist):
            Z = condition_normalise(ds.X, cond, tr) if ds.meta["n_conditions"] > 1 else ds.X
            parts = [Z, ds.static] + ([past_features(Z, ds.unit, ds.order, stats=("last",))] if hist else [])
            return pd.concat(parts, axis=1)
        for name, hist in [("complete", False), ("complete+last", True)]:
            r, _, _ = run(ds, lambda tr: mk(tr, hist), seed, save_pred=False)
            rows.append(dict(dataset=f"C-MAPSS {fd}", variant=name, seed=seed, PR_AUC=r["PR_AUC"], ROC_AUC=r["ROC_AUC"])); log(fd, name, seed, round(r["PR_AUC"], 3))
mp = load_metropt(); start = mp.meta["start"]; Fl = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in mp.meta["failures"]]
mids = [Fl[k][1] + (Fl[k + 1][0] - Fl[k][1]) / 2 for k in range(3)]
blocks = {"F2": (mids[0], mids[1]), "F3": (mids[1], mids[2]), "F4": (mids[2], start.max() + pd.Timedelta("1h"))}
Hm = past_features(mp.X, mp.unit, mp.order, stats=("last",))
for seed in range(5):
    for ev, (a, b) in blocks.items():
        tr = np.where(start < a)[0]; te = np.where((start >= a) & (start < b))[0]
        for name, F in [("complete", mp.X), ("complete+last", pd.concat([mp.X, Hm], axis=1))]:
            p, _ = fit_predict(F.iloc[tr], mp.y[tr], F.iloc[te], seed)
            rows.append(dict(dataset=f"MetroPT-3 {ev}", variant=name, seed=seed, PR_AUC=average_precision_score(mp.y[te], p),
                             ROC_AUC=roc_auc_score(mp.y[te], p))); log("MetroPT", ev, name, seed, round(rows[-1]["PR_AUC"], 3))
append(rows, "exp11_complete_with_history.csv"); log("done")
