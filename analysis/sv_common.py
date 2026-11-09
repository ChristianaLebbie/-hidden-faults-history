"""Shared helpers for the November 2026 supervisor checks (sv_point*.py).

Everything here works on the predictions already saved by the experiments
(results/predictions/*.npz); no model is retrained.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from pfi.data import load_ai4i
from pfi.schedules import ai4i_realistic, hidden_mask, visible_mask

PRED = ROOT / "results" / "predictions"
OUT = ROOT / "results" / "supervisor_nov"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
# AMA in the thesis is the representation without the age input (cur_last).
METHODS = {"current": "Current readings only", "locf_ind": "LOCF + missingness indicators",
           "locf_ind_age": "LOCF + indicators + age", "cur_last": "AMA", "cur_last_age": "AMA + age"}
CAUSAL = {"HDF": ["air", "proc", "rpm"], "PWF": ["rpm", "torque"], "OSF": ["torque", "wear"], "TWF": ["wear"]}
FLAGS = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def ai4i():
    ds = load_ai4i()
    a = pd.read_csv(ROOT / "data" / "ai4i2020.csv")
    ds.meta["flags"] = a[FLAGS].to_numpy().astype(bool)
    ds.meta["n_modes"] = ds.meta["flags"].sum(1)
    return ds


def load_pred(tag, seed):
    z = np.load(PRED / f"{tag}_s{seed}.npz")
    inner = [(z[f"inner{i}_idx"], z[f"inner{i}_p"]) for i in range(5)]
    return z["p"], z["fold"], inner


def alarms_at(p, fold, inner, y, target):
    """Alarm rule calibrated on the inner out-of-fold scores of each outer fold's training records:
    threshold = (1 - target/100) quantile of the normal training scores. No test data are used."""
    alarm = np.zeros(len(p), bool); thr = np.zeros(len(p))
    for k, (idx, ip) in enumerate(inner):
        t = np.quantile(ip[y[idx] == 0], 1 - target / 100, method="higher")
        te = fold == k; alarm[te] = p[te] > t; thr[te] = t
    return alarm, thr


def machine_bootstrap(stat_fn, units, n_boot=2000, seed=0):
    """Resample whole machines with replacement. stat_fn(index_array) -> float.
    Returns (estimate on full data, 2.5% and 97.5% percentiles)."""
    r = np.random.default_rng(seed); uu = np.unique(units)
    by = {u: np.where(units == u)[0] for u in uu}
    full = stat_fn(np.arange(len(units)))
    vals = []
    for _ in range(n_boot):
        pick = r.choice(uu, len(uu), replace=True)
        vals.append(stat_fn(np.concatenate([by[u] for u in pick])))
    vals = np.array(vals, float)
    return full, np.nanpercentile(vals, 2.5), np.nanpercentile(vals, 97.5), np.nanmean(vals > 0) if full > 0 else np.nanmean(vals < 0)


def prauc(y, p):
    return average_precision_score(y, p) if 0 < y.sum() < len(y) else np.nan
