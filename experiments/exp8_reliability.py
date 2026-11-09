"""Study 1, Q2 (descriptive): which historical information remains reliable?

For each sensor, R^2 of the last earlier reading as an estimate of the current reading, by
how many inspections old the earlier reading is. Label-free and computed on all units (it
describes the data, it is not used for model selection; the method's gate uses training folds).
C-MAPSS multi-condition subsets are analysed on raw readings and on readings z-scored within
operating condition.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from pfi.data import load_ai4i, load_cmapss
from pfi.history import past_features
from pfi.methods import condition_normalise, reliability_table
from pfi.runner import log
from pfi.schedules import ai4i_realistic, apply_schedule, random_mix

rows = []
ai = load_ai4i()
X = apply_schedule(ai.X, ai4i_realistic(ai, 100), ai.groups)
H = past_features(X, ai.unit, ai.order, stats=("last", "stale"))
t = reliability_table(X, H, np.arange(len(ai.y)))
t["dataset"] = "AI4I-PMDI"
t["readings"] = "raw"
rows.append(t)
for fd in [
    "FD001",
    "FD002",
]:  # thesis uses FD001 (one condition) and FD002 (six conditions)
    ds = load_cmapss(fd)
    X = apply_schedule(ds.X, random_mix(ds, 100), ds.groups)
    variants = [("raw", X)]
    if ds.meta["n_conditions"] > 1:
        variants.append(
            (
                "condition-normalised",
                condition_normalise(
                    X, ds.static["cond"].to_numpy(), np.arange(len(ds.y))
                ),
            )
        )
    for name, Xv in variants:
        H = past_features(Xv, ds.unit, ds.order, stats=("last", "stale"))
        t = reliability_table(Xv, H, np.arange(len(ds.y)))
        t["dataset"] = f"C-MAPSS {fd}"
        t["readings"] = name
        rows.append(t)
    log(fd)
df = pd.concat(rows)
df.to_csv(
    Path(__file__).resolve().parents[1] / "results" / "exp8_reliability.csv",
    index=False,
)
print(
    df[df.lo <= 3]
    .pivot_table(index=["dataset", "readings", "sensor"], columns="lo", values="R2")
    .round(2)
    .to_string()
)
