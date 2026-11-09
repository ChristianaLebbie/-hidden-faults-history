"""Chapter 1, MetroPT-3 event by event: detection delay and false alarms (supervisor feedback, Oct 2026).

Same data, schedules and forward-in-time splits as exp10: for each air leak F2, F3, F4 the model is
trained only on data before the test period (which contains the earlier leaks) and tested on the period
around that leak.

Alarm threshold (no test data used)
  The most recent 20% of the NORMAL training windows (by time) are held out as a calibration set; the
  model is trained on the remaining training windows (all earlier leak windows stay in training).
  The threshold is set so that the calibration normals raise 1 false alarm per day (and 5 per day as a
  second setting). The same model and threshold are then applied to the test period.

Reported per event, method and seed
  PR_AUC                 threshold-free ranking quality in the test period
  detected               1 if any window inside the leak raises an alarm
  delay_min              minutes from the reported leak start to the end of the first alarming window
                         inside the leak (empty if not detected)
  leak_windows_flagged   share of leak windows that raise an alarm
  FA_per_day             alarms on normal test windows per day of the test period
  FA_per_day_to_catch    threshold-free: false alarms per day that must be accepted to catch the leak at all
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from pfi.data import load_metropt
from pfi.schedules import random_mix, apply_schedule
from pfi.history import past_features
from pfi.methods import build_features, ALL_STATS
from pfi.evaluate import fit_predict
from pfi.runner import append, log

ds = load_metropt(); start = ds.meta["start"]; F = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in ds.meta["failures"]]
mids = [F[k][1] + (F[k + 1][0] - F[k][1]) / 2 for k in range(3)]
blocks = {"F2": (mids[0], mids[1], F[1]), "F3": (mids[1], mids[2], F[2]), "F4": (mids[2], start.max() + pd.Timedelta("1h"), F[3])}
METHODS = {"current": "Current readings only", "cur_last": "AMA without age", "cur_last_age": "AMA"}
TARGETS = [1.0, 5.0]   # false alarms per day on the calibration normals
WIN = pd.Timedelta("10min")
SEEDS = [0, 1, 2, 3, 4]
rows = []
for seed in SEEDS:
    sch = random_mix(ds, 100 + seed); X = apply_schedule(ds.X, sch, ds.groups)
    H = past_features(X, ds.unit, ds.order, stats=ALL_STATS)
    for ev, (a, b, (la, lb)) in blocks.items():
        tr = np.where(start < a)[0]; te = np.where((start >= a) & (start < b))[0]
        trn = tr[ds.y[tr] == 0]; cut = start.iloc[trn].quantile(0.8)
        cal = trn[(start.iloc[trn] >= cut).to_numpy()]; fit = np.setdiff1d(tr, cal)
        cal_days = (start.iloc[cal].max() - start.iloc[cal].min() + WIN).total_seconds() / 86400
        te_days = (start.iloc[te].max() - start.iloc[te].min() + WIN).total_seconds() / 86400
        st = start.iloc[te].reset_index(drop=True); yt = ds.y[te]
        leak = ((st + WIN > la) & (st <= lb)).to_numpy() & (yt == 1)
        for m, label in METHODS.items():
            Fm = build_features(m, X, H, ds.static, fit)
            p_all, _ = fit_predict(Fm.iloc[fit], ds.y[fit], Fm.iloc[np.concatenate([cal, te])], seed)
            pc, pt = p_all[:len(cal)], p_all[len(cal):]
            best = pt[yt == 1].max(); fa_catch = ((pt >= best) & (yt == 0)).sum() / te_days
            for target in TARGETS:
                rate = min(1.0, target * cal_days / len(cal))           # alarms per normal window
                thr = np.quantile(pc, 1 - rate, method="higher"); al = pt > thr
                hit = np.where(al & leak)[0]
                delay = ((st.iloc[hit[0]] + WIN - la).total_seconds() / 60) if len(hit) else np.nan
                rows.append(dict(event=ev, seed=seed, method=m, label=label, FA_target_per_day=target,
                                 PR_AUC=average_precision_score(yt, pt), detected=int(len(hit) > 0), delay_min=max(delay, 0) if len(hit) else np.nan,
                                 leak_windows_flagged=al[leak].mean(), FA_per_day=(al & (yt == 0)).sum() / te_days,
                                 FA_per_day_to_catch=fa_catch, leak_start=str(la), test_days=te_days))
            r = rows[-2]
            log(ev, seed, label, "PR-AUC", round(r["PR_AUC"], 3), "| detected", r["detected"], "| delay min", r["delay_min"],
                "| FA/day", round(r["FA_per_day"], 2))
        append(rows, "exp16_metropt_events.csv")
log("done")
