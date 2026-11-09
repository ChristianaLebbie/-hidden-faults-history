"""Study 1, Q1: which use of history helps, compared with LOCF and trend baselines (AI4I-PMDI).

Scenarios
  S0  published inspection controls
  S1  realistic controls (fault records receive a control at random)
  S2  model developed on S0, deployed on S1
  REF complete data (no missing values)
A forbidden 'future-informed' variant (uses the next observation) is run as an audit only,
to show how much a leak would inflate results.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from pfi.data import load_ai4i
from pfi.schedules import ai4i_published, ai4i_realistic, apply_schedule, hidden_mask, visible_mask
from pfi.history import past_features
from pfi.methods import build_features, ALL_STATS
from pfi.runner import run, append, log

SEEDS = [0, 1, 2, 3, 4]
METHODS = ["current", "locf", "locf_ind", "locf_ind_age", "trend_impute", "cur_last", "cur_last_age",
           "cur_lags", "cur_window", "cur_ewma", "gated", "gated_window"]

ds = load_ai4i(); rows = []
X0 = apply_schedule(ds.X, ai4i_published(ds), ds.groups); H0 = past_features(X0, ds.unit, ds.order, stats=ALL_STATS)
for seed in SEEDS:
    sch = ai4i_realistic(ds, 100 + seed); X1 = apply_schedule(ds.X, sch, ds.groups)
    H1 = past_features(X1, ds.unit, ds.order, stats=ALL_STATS)
    masks = {"hidden": hidden_mask(ds, sch), "visible": visible_mask(ds, sch)}
    for m in METHODS:
        r, _, _ = run(ds, lambda tr: build_features(m, X1, H1, ds.static, tr), seed, masks, tag=f"ai4i_S1_{m}_s{seed}")
        rows.append(dict(scenario="S1", method=m, seed=seed, **r)); log("S1", m, seed, round(r["PR_AUC"], 3))
    # S2: train on published-schedule features, test on realistic-schedule features
    for m in ["current", "locf_ind_age", "cur_last", "gated"]:
        def build(tr, m=m):
            F0 = build_features(m, X0, H0, ds.static, tr); F1 = build_features(m, X1, H1, ds.static, tr)
            F = F1.copy(); F.iloc[tr] = F0.iloc[tr].to_numpy(); return F
        r, _, _ = run(ds, build, seed, masks, tag=f"ai4i_S2_{m}_s{seed}")
        rows.append(dict(scenario="S2", method=m, seed=seed, **r)); log("S2", m, seed, round(r["PR_AUC"], 3))
    # audit only: uses the NEXT observation (future information) - not permitted in the method
    rev = past_features(X1, ds.unit, -ds.order, stats=("last",)).add_prefix("next_")
    r, _, _ = run(ds, lambda tr: pd.concat([build_features("cur_last", X1, H1, ds.static, tr), rev], axis=1), seed, masks,
                  tag=f"ai4i_S1_FUTURE_s{seed}")
    rows.append(dict(scenario="S1", method="AUDIT_future_next_obs", seed=seed, **r)); log("audit future", seed, round(r["PR_AUC"], 3))
    append(rows, "exp1_ai4i_methods.csv")

for seed in SEEDS:
    pub = ai4i_published(ds); masks = {"visible": visible_mask(ds, pub)}
    for m in ["current", "cur_last", "gated"]:
        r, _, _ = run(ds, lambda tr: build_features(m, X0, H0, ds.static, tr), seed, masks, tag=f"ai4i_S0_{m}_s{seed}")
        rows.append(dict(scenario="S0", method=m, seed=seed, **r))
    r, _, _ = run(ds, lambda tr: pd.concat([ds.X, ds.static], axis=1), seed, tag=f"ai4i_REF_s{seed}")
    rows.append(dict(scenario="REF", method="complete", seed=seed, **r)); log("S0/REF", seed)
    append(rows, "exp1_ai4i_methods.csv")
log("done")
