"""Study 1, Q3: consecutive missing observations (outages) on top of the acquisition schedule.

Each sensor of each unit goes missing for runs of consecutive inspections (geometric run
length with mean L, about 30% of inspections affected). Longer runs make the carried
history older. Also saves, for AI4I, the age of the missing causal sensors of each hidden
fault so that recall can be analysed by staleness.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from pfi.data import load_ai4i, load_cmapss
from pfi.schedules import ai4i_realistic, random_mix, apply_schedule, add_outages, hidden_mask
from pfi.history import past_features
from pfi.methods import build_features, ALL_STATS
from pfi.runner import run, append, log, PRED

CAUSAL = {"HDF": ["air", "proc", "rpm"], "PWF": ["rpm", "torque"], "OSF": ["torque", "wear"], "TWF": ["wear"]}
METHODS = ["current", "locf_ind_age", "cur_last", "cur_last_age", "cur_window", "gated"]
SETTINGS = [(0, 0.0), (2, 0.3), (5, 0.3), (10, 0.3), (20, 0.3)]
rows = []
for name, ds, mk in [("AI4I-PMDI", load_ai4i(), "ai4i"), ("C-MAPSS FD001", load_cmapss("FD001"), "cmapss")]:
    for seed in [0, 1, 2]:
        sch = ai4i_realistic(ds, 100 + seed) if mk == "ai4i" else random_mix(ds, 100 + seed)
        base = apply_schedule(ds.X, sch, ds.groups)
        for L, rate in SETTINGS:
            X = add_outages(base, ds, L, rate, seed=1000 + seed)
            H = past_features(X, ds.unit, ds.order, stats=ALL_STATS)
            masks = {"hidden": hidden_mask(ds, sch)} if mk == "ai4i" else None
            age = {}
            for c in X.columns:
                age[c] = np.where(X[c].notna(), 0, H[f"stale_{c}"].fillna(999).to_numpy())
            if mk == "ai4i":
                mode = ds.meta["mode"]; worst = np.full(len(ds.y), np.nan)
                for i in np.where(masks["hidden"])[0]:
                    worst[i] = max(age[c][i] for c in CAUSAL[mode[i]] if np.isnan(X[c].iloc[i]))
                np.save(PRED / f"ai4i_outage_L{L}_s{seed}_hidden_age.npy", worst)
            mean_age = np.mean([np.mean(age[c][X[c].isna().to_numpy() & (age[c] < 999)]) for c in X.columns])
            for m in METHODS:
                r, _, _ = run(ds, lambda tr: build_features(m, X, H, ds.static, tr), seed, masks, tag=f"{mk}_outage_L{L}_{m}_s{seed}")
                rows.append(dict(dataset=name, mean_run=L, outage_rate=rate, missing_share=float(X.isna().to_numpy().mean()),
                                 mean_age_missing=mean_age, method=m, seed=seed, **r))
                log(name, L, m, seed, round(r["PR_AUC"], 3))
            append(rows, "exp3_consecutive_missing.csv")
log("done")
