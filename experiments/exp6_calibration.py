"""Study 2, Q1: probability calibration (post-processing of saved Study 1 predictions).

Calibrators (Platt scaling, isotonic regression) are fitted, for each outer fold, on the inner
out-of-fold predictions of that fold's training records only, then applied to its test records.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, average_precision_score
from pfi.data import load_ai4i, load_cmapss
from pfi.calibrate import calibrate_oof
from pfi.evaluate import ece
from pfi.runner import PRED, append, log

SETS = [("AI4I-PMDI", load_ai4i(), [f"ai4i_S1_{m}_s{s}" for m in ["current", "cur_last", "gated"] for s in range(5)]),
        ("C-MAPSS FD001", load_cmapss("FD001"), [f"cm_FD001_raw_{m}_s{s}" for m in ["current", "cur_last"] for s in range(3)]),
        ("C-MAPSS FD002", load_cmapss("FD002"), [f"cm_FD002_norm_{m}_s{s}" for m in ["current", "cur_last"] for s in range(3)])]
rows, rel = [], []
for name, ds, tags in SETS:
    for tag in tags:
        f = PRED / f"{tag}.npz"
        if not f.exists():
            log("missing", tag); continue
        z = np.load(f); inner = [dict(idx=z[f"inner{i}_idx"], p=z[f"inner{i}_p"]) for i in range(5)]
        method = tag.split("_s")[0].split("_", 2)[-1]; seed = int(tag.split("_s")[-1])
        for kind in ["none", "platt", "isotonic"]:
            pc, _ = calibrate_oof(ds.y, z["p"], z["fold"], inner, kind)
            rows.append(dict(dataset=name, method=method, seed=seed, calibration=kind, Brier=brier_score_loss(ds.y, pc),
                             logloss=log_loss(ds.y, np.clip(pc, 1e-6, 1 - 1e-6)), ECE10=ece(ds.y, pc, 10), ECE15=ece(ds.y, pc, 15),
                             ECE_mass10=ece(ds.y, pc, 10, "mass"), PR_AUC=average_precision_score(ds.y, pc)))
            q = np.clip((pc * 10).astype(int), 0, 9)
            for b in range(10):
                m = q == b
                if m.sum():
                    rel.append(dict(dataset=name, method=method, seed=seed, calibration=kind, bin=b, n=int(m.sum()),
                                    mean_pred=pc[m].mean(), frac_pos=ds.y[m].mean()))
        log(name, tag)
append(rows, "exp6_calibration.csv"); append(rel, "exp6_reliability_bins.csv"); log("done")
