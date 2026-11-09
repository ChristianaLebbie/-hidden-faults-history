"""Strict temporal validation (AI4I-PMDI): train only on records before a calendar cutoff,
test on all records after it. Grouped cross-validation keeps machines apart but ignores the
calendar; this check ensures that no decision benefits from data recorded after it was made.
History features are strictly past by construction. Thresholds come from grouped inner CV
on the training period only.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from pfi.data import load_ai4i
from pfi.schedules import ai4i_realistic, apply_schedule, hidden_mask
from pfi.history import past_features
from pfi.methods import build_features, ALL_STATS
from pfi.evaluate import fit_predict, f1_threshold, metrics
from pfi.runner import append, log

ds = load_ai4i(); rows = []
for cut_q in [0.5, 0.7]:
    cut = np.quantile(ds.order, cut_q); tr = np.where(ds.order < cut)[0]; te = np.where(ds.order >= cut)[0]
    for seed in range(5):
        sch = ai4i_realistic(ds, 100 + seed); X = apply_schedule(ds.X, sch, ds.groups)
        H = past_features(X, ds.unit, ds.order, stats=ALL_STATS); hid = hidden_mask(ds, sch)
        for m in ["current", "locf_ind_age", "trend_impute", "cur_last", "gated"]:
            F = build_features(m, X, H, ds.static, tr)
            p, _ = fit_predict(F.iloc[tr], ds.y[tr], F.iloc[te], seed)
            ip = np.full(len(tr), np.nan)
            for a, b in StratifiedGroupKFold(4, shuffle=True, random_state=seed).split(tr, ds.y[tr], ds.unit[tr]):
                ip[b], _ = fit_predict(F.iloc[tr[a]], ds.y[tr[a]], F.iloc[tr[b]], seed)
            t = f1_threshold(ds.y[tr], ip)
            r = metrics(ds.y[te], p, t, {"hidden": hid[te]})
            rows.append(dict(cutoff_quantile=cut_q, n_train=len(tr), n_test=len(te), method=m, seed=seed, **r))
            log(cut_q, seed, m, round(r["PR_AUC"], 3))
        append(rows, "exp9_forward_in_time.csv")
log("done")
