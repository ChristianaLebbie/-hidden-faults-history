"""Chapter 1: comparison with GRU-D, a sequence model designed for missing observations (supervisor feedback, Oct 2026).

Same datasets, schedules, seeds and outer folds as exp13/exp14 (5-fold grouped CV by machine; AI4I realistic
schedule; C-MAPSS random schedule). GRU-D reads each machine's inspections in time order and outputs a fault
probability at every inspection from that inspection and earlier ones only.
Inputs: the same sensor readings (unmeasured left missing), the missingness mask, time since each sensor was
last observed (in inspections), and the same static inputs (AI4I product type; C-MAPSS operating condition).
On FD002 the readings are normalised within operating condition with training statistics, exactly as for AMA.
Inside each training fold, 20% of the training machines are held out to stop training early (by PR-AUC);
no test data are used for training or stopping.

Compared in session9_summary.py with XGBoost on current readings and with AMA (no age) from exp13/exp14.
Note: AI4I has 3 pairs of records of the same machine with the same timestamp; GRU-D reads them one after the
other in record order (AMA treats them as simultaneous). The effect on results is negligible.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import os
import time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.metrics import average_precision_score, roc_auc_score
from pfi.data import load_ai4i, load_cmapss
from pfi.schedules import ai4i_realistic, random_mix, apply_schedule, hidden_mask
from pfi.methods import condition_normalise
from pfi.grud import make_sequences, fit_predict_grud
from pfi.runner import append, log

SEEDS = [0, 1, 2][:int(os.environ.get("NSEEDS", 3))]; ONLY = os.environ.get("ONLY")
EPOCHS = int(os.environ.get("EPOCHS", 150))
rows = []
for name in [n for n in ["AI4I-PMDI", "C-MAPSS FD001", "C-MAPSS FD002"] if not ONLY or ONLY in n]:
    if name == "AI4I-PMDI":
        ds = load_ai4i(); static = ds.static.astype(float)
    else:
        ds = load_cmapss(name.split()[-1])
        static = pd.get_dummies(ds.static["cond"].astype(str), prefix="cond", dtype=float)
    multi = name == "C-MAPSS FD002"; cond = ds.static["cond"].to_numpy() if multi else None
    y = ds.y; n = len(y)
    for seed in SEEDS:
        sch = ai4i_realistic(ds, 100 + seed) if name == "AI4I-PMDI" else random_mix(ds, 100 + seed)
        X = apply_schedule(ds.X, sch, ds.groups)
        hidden = hidden_mask(ds, sch) if name == "AI4I-PMDI" else None
        p = np.full(n, np.nan); t0 = time.time(); eps = []
        outer = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        for f, (tr, te) in enumerate(outer.split(np.zeros(n), y, ds.unit)):
            Xf = condition_normalise(X, cond, tr) if multi else X
            mu = np.nanmean(Xf.to_numpy(float)[tr], 0); sd = np.nanstd(Xf.to_numpy(float)[tr], 0); sd[sd == 0] = 1
            tr_units = np.unique(ds.unit[tr]); te_units = np.unique(ds.unit[te])
            gs = GroupShuffleSplit(1, test_size=0.2, random_state=seed + f)
            a, b = next(gs.split(tr_units, groups=tr_units)); fit_u, val_u = tr_units[a], tr_units[b]
            seqs = {}
            for k, us in [("fit", fit_u), ("val", val_u), ("te", te_units)]:
                s, idxs = make_sequences(Xf, static, ds.unit, ds.order, us, mu, sd)
                yy = np.zeros(s["valid"].shape)
                for bi, i in enumerate(idxs):
                    yy[bi, :len(i)] = y[i]
                seqs[k] = (s, torch.tensor(yy, dtype=torch.float32), idxs)
            pt, ep = fit_predict_grud(seqs["fit"][0], seqs["fit"][1], seqs["val"][0], seqs["val"][1], seqs["te"][0], seed, epochs=EPOCHS)
            eps.append(ep)
            for bi, i in enumerate(seqs["te"][2]):
                p[i] = pt[bi, :len(i)]
        r = dict(dataset=name, model="grud", model_label="GRU-D", features="GRU-D", seed=seed, seconds=time.time() - t0,
                 epochs_mean=np.mean(eps), PR_AUC=average_precision_score(y, p), ROC_AUC=roc_auc_score(y, p))
        if hidden is not None:
            keep = hidden | (y == 0); r["PR_AUC_hidden"] = average_precision_score(y[keep], p[keep])
        rows.append(r); log(name, seed, "GRU-D PR-AUC", round(r["PR_AUC"], 3),
                            "| hidden", round(r.get("PR_AUC_hidden", np.nan), 3), "| minutes", round(r["seconds"] / 60, 1))
        append(rows, "exp18_grud.csv")
log("done")
