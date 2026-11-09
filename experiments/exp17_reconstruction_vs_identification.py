"""Chapter 2: reconstructing the current reading vs improving fault identification (supervisor feedback, Oct 2026).

Question: an earlier reading can be a poor ESTIMATE of a sensor's current value and still HELP the classifier
identify faults. When does each happen, and how does it depend on elapsed time (age of the reading) and on a
change of operating condition between the earlier reading and now?

For every record and every sensor that is NOT measured at the current inspection but has an earlier reading:
  carried value = last earlier reading of that sensor (strictly past, same machine)
  true value    = the sensor's actual current value (known here because the schedules are simulated
                  from complete data; never given to any model)

Three measures, reported by age bin and (C-MAPSS) by whether the operating condition changed:
  R2_reconstruction  how well the carried value estimates the true current value
                     (1 = perfect, 0 = no better than the overall mean, below 0 = worse than the mean)
  sep_carried        how well the carried value on its own separates fault from normal records
                     (2 x |ROC-AUC - 0.5|; 0 = no information, 1 = perfect separation)
  sep_true           the same for the true current value (what a complete measurement would give)
Plus, per record, the gain in PR-AUC of the XGBoost model with history (current readings + last earlier
reading, i.e. AMA without age) over current readings only, within bins of the record's mean age of carried
values and its share of carried values recorded under a different operating condition.

History is analysed raw and, for C-MAPSS, normalised within operating condition (z-score per condition).
For this descriptive analysis the normalisation statistics use all records; the model comparison uses
training records only, as in exp5 and exp13.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from pfi.data import load_ai4i, load_cmapss
from pfi.schedules import ai4i_realistic, random_mix, apply_schedule
from pfi.history import past_features
from pfi.methods import build_features, condition_normalise
from pfi.evaluate import cv_predict
from pfi.runner import RES, log

SEEDS = [0, 1, 2][:int(os.environ.get("NSEEDS", 3))]
AGE_BINS = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 10**9)]
REC_AGE_BINS = [(1, 1.5), (1.5, 2.5), (2.5, 4), (4, 10**9)]
DIFF_BINS = [(0, 0.34), (0.34, 0.67), (0.67, 1.01)]


def age_label(lo, hi):
    return f"{lo}" if lo == hi else (f"{lo}+" if hi > 1000 else f"{lo}-{hi}")


def sep(y, v):
    if len(np.unique(y)) < 2 or len(y) < 30:
        return np.nan
    return 2 * abs(roc_auc_score(y, v) - 0.5)


sensor_rows, record_rows = [], []
for name in ["AI4I-PMDI", "C-MAPSS FD001", "C-MAPSS FD002"]:
    if name == "AI4I-PMDI":
        ds = load_ai4i(); static = ds.static; cond = None
    else:
        ds = load_cmapss(name.split()[-1]); cond = ds.static["cond"].to_numpy()
        static = pd.get_dummies(ds.static["cond"].astype(str), prefix="cond", dtype=float)
    multi = cond is not None and len(np.unique(cond)) > 1
    for seed in SEEDS:
        sch = ai4i_realistic(ds, 100 + seed) if name == "AI4I-PMDI" else random_mix(ds, 100 + seed)
        X = apply_schedule(ds.X, sch, ds.groups)
        variants = {"raw": (X, ds.X)}
        if multi:
            allr = np.arange(len(ds.y))
            variants["normalised"] = (condition_normalise(X, cond, allr), condition_normalise(ds.X, cond, allr))
        # condition at the time of each sensor's last earlier reading
        if multi:
            Cm = pd.DataFrame({c: np.where(X[c].notna(), cond, np.nan) for c in X.columns})
            lastcond = past_features(Cm, ds.unit, ds.order, stats=("last",))
        rec_age = rec_diff = None
        for hv, (Xv, Tv) in variants.items():
            H = past_features(Xv, ds.unit, ds.order, stats=("last", "stale"))
            ages, diffs = [], []
            for c in X.columns:
                miss = X[c].isna().to_numpy(); last = H[f"last_{c}"].to_numpy(); stale = H[f"stale_{c}"].to_numpy()
                true = Tv[c].to_numpy(dtype=float); ok = miss & ~np.isnan(last); var = np.var(true)
                changed = (lastcond[f"last_{c}"].to_numpy() != cond) if multi else np.zeros(len(true), bool)
                ages.append(np.where(ok, stale, np.nan)); diffs.append(np.where(ok, changed.astype(float), np.nan))
                groups = [("all", np.ones(len(true), bool))] + ([("same condition", ~changed), ("condition changed", changed)] if multi else [])
                for cname, cm in groups:
                    for lo, hi in AGE_BINS:
                        m = ok & cm & (stale >= lo) & (stale <= hi)
                        if m.sum() < 30:
                            continue
                        sensor_rows.append(dict(dataset=name, seed=seed, history=hv, sensor=c, condition=cname, age=age_label(lo, hi),
                                                n=int(m.sum()), R2_reconstruction=1 - np.mean((true[m] - last[m]) ** 2) / var,
                                                sep_carried=sep(ds.y[m], last[m]), sep_true=sep(ds.y[m], true[m])))
            if hv == "raw":
                with np.errstate(all="ignore"):
                    rec_age = np.nanmean(np.vstack(ages), axis=0); rec_diff = np.nanmean(np.vstack(diffs), axis=0)
        # model comparison: current readings only vs current + last earlier reading (AMA without age)
        cache = {}

        def hist_build(tr, norm):
            if not norm:
                return build_features("cur_last", X, past_features(X, ds.unit, ds.order, stats=("last", "stale")), static, tr)
            key = (len(tr), int(tr[0]), int(tr[-1]))
            if key not in cache:
                Z = condition_normalise(X, cond, tr); cache.clear(); cache[key] = (Z, past_features(Z, ds.unit, ds.order, stats=("last", "stale")))
            Z, Hz = cache[key]; return build_features("cur_last", Z, Hz, static, tr)

        Hraw = past_features(X, ds.unit, ds.order, stats=("last", "stale"))
        preds = {"current": cv_predict(ds, lambda tr: build_features("current", X, Hraw, static, tr), seed, inner=False)["p"],
                 "history raw": cv_predict(ds, lambda tr: hist_build(tr, False), seed, inner=False)["p"]}
        if multi:
            preds["history normalised"] = cv_predict(ds, lambda tr: hist_build(tr, True), seed, inner=False)["p"]
        bins = [("record mean age", lo, hi, (rec_age >= lo) & (rec_age < hi)) for lo, hi in REC_AGE_BINS]
        if multi:
            bins += [("share of carried values from a different condition", lo, hi, (rec_diff >= lo) & (rec_diff < hi)) for lo, hi in DIFF_BINS]
        for kind, lo, hi, m in bins:
            if m.sum() < 30 or len(np.unique(ds.y[m])) < 2:
                continue
            r = dict(dataset=name, seed=seed, bin_type=kind, bin=f"{lo:g}-{hi:g}" if hi < 1000 else f"{lo:g}+", n=int(m.sum()), faults=int(ds.y[m].sum()))
            for k, p in preds.items():
                r[f"PR_AUC {k}"] = average_precision_score(ds.y[m], p[m])
            record_rows.append(r)
        r = dict(dataset=name, seed=seed, bin_type="all records", bin="all", n=len(ds.y), faults=int(ds.y.sum()))
        for k, p in preds.items():
            r[f"PR_AUC {k}"] = average_precision_score(ds.y, p)
        record_rows.append(r)
        log(name, seed, {k: round(v, 3) for k, v in r.items() if k.startswith("PR_AUC")})
        pd.DataFrame(sensor_rows).to_csv(RES / "exp17_reconstruction_by_sensor.csv", index=False)
        pd.DataFrame(record_rows).to_csv(RES / "exp17_identification_by_bin.csv", index=False)
log("done")
