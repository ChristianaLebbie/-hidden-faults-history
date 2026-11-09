"""Shared helpers for experiment scripts: run a method under grouped CV, save predictions, write CSVs, log."""
import time
from pathlib import Path
import numpy as np
import pandas as pd
from .evaluate import cv_predict, thresholds_from_inner, metrics

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"; PRED = RES / "predictions"; FIG = ROOT / "figures"
for _d in (RES, PRED, FIG, RES / "logs"):
    _d.mkdir(exist_ok=True)


def log(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)


def append(rows, name):
    """Write all rows collected so far (scripts pass their cumulative list, so the file is overwritten)."""
    df = pd.DataFrame(rows); df.to_csv(RES / name, index=False); return df


def run(ds, build, seed, masks=None, tag=None, save_pred=True, inner=True):
    """Grouped 5-fold CV with inner out-of-fold predictions for thresholds and calibration.

    Returns (metrics, cv result, per-record alarm thresholds). With a tag, saves test
    probabilities, thresholds, fold ids and each fold's inner out-of-fold predictions.
    """
    t0 = time.time()
    res = cv_predict(ds, build, seed, inner=inner)
    t = thresholds_from_inner(res, ds.y) if inner else None
    m = metrics(ds.y, res["p"], t, masks); m["seconds"] = time.time() - t0
    if save_pred and tag:
        z = dict(p=res["p"], t=t if t is not None else np.full(len(ds.y), np.nan), fold=res["fold"])
        for i, inn in enumerate(res["inner"]):
            z[f"inner{i}_idx"] = inn["idx"]; z[f"inner{i}_p"] = inn["p"]
        np.savez_compressed(PRED / f"{tag}.npz", **z)
    return m, res, t
