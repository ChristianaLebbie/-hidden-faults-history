"""Study 1, Q2: why does history help? Shuffled and degraded history (AI4I-PMDI and C-MAPSS FD001).

The 'current + last value' model is refitted with the history column replaced by:
  true          most recent earlier observation (the method)
  random_past   a random earlier observation of the same sensor on the same unit (destroys recency)
  old_past      the latest observation at least 10 inspections old (stale but same unit)
  unit_mean     mean of all earlier observations of the unit (unit level only)
  other_unit    history of a different unit at the same position in its life (destroys unit identity)
All variants remain strictly past and label-free.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from pfi.data import load_ai4i, load_cmapss
from pfi.schedules import ai4i_realistic, random_mix, apply_schedule, hidden_mask
from pfi.history import past_features, alt_history, other_unit
from pfi.runner import run, append, log


rows = []
for name, ds, seeds, mk in [("AI4I-PMDI", load_ai4i(), [0, 1, 2, 3, 4], "ai4i"),
                            ("C-MAPSS FD001", load_cmapss("FD001"), [0, 1, 2], "cmapss")]:
    for seed in seeds:
        sch = ai4i_realistic(ds, 100 + seed) if mk == "ai4i" else random_mix(ds, 100 + seed)
        X = apply_schedule(ds.X, sch, ds.groups)
        Ht = past_features(X, ds.unit, ds.order, stats=("last",))
        masks = {"hidden": hidden_mask(ds, sch)} if mk == "ai4i" else None
        variants = {"none": None, "true": Ht,
                    "random_past": alt_history(X, ds.unit, ds.order, "random_past", seed),
                    "old_past": alt_history(X, ds.unit, ds.order, "old_past", seed),
                    "unit_mean": alt_history(X, ds.unit, ds.order, "unit_mean", seed),
                    "other_unit": other_unit(Ht, ds.unit, ds.order, seed)}
        for vname, Hv in variants.items():
            F = pd.concat([X, ds.static] + ([] if Hv is None else [Hv]), axis=1)
            r, _, _ = run(ds, lambda tr: F, seed, masks, tag=f"{mk}_shuffle_{vname}_s{seed}")
            rows.append(dict(dataset=name, history=vname, seed=seed, **r)); log(name, vname, seed, round(r["PR_AUC"], 3))
        append(rows, "exp2_shuffled_history.csv")
log("done")
