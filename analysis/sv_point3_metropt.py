"""Point 3 (MetroPT): target vs realised false alarms, delay with missed leaks, recalibrated combined rule.
Same data, schedules, splits and calibration as exp16."""
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
OUTD = Path(__file__).resolve().parents[1] / "results" / "supervisor_nov"
def alarm_stats(al, yt, leak, st, la, te_days):
    hit = np.where(al & leak)[0]
    return dict(detected=int(len(hit) > 0), delay_min=max(0, (st.iloc[hit[0]] + WIN - la).total_seconds() / 60) if len(hit) else np.nan,
                leak_windows_flagged=al[leak].mean(), FA_test_per_day=(al & (yt == 0)).sum() / te_days)
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
        PC, PT = {}, {}
        for m in ["current", "cur_last"]:
            Fm = build_features(m, X, H, ds.static, fit)
            p_all, _ = fit_predict(Fm.iloc[fit], ds.y[fit], Fm.iloc[np.concatenate([cal, te])], seed)
            PC[m], PT[m] = p_all[:len(cal)], p_all[len(cal):]
        for target in TARGETS:
            rate = min(1.0, target * cal_days / len(cal))
            for m in ["current", "cur_last"]:
                thr = np.quantile(PC[m], 1 - rate, method="higher")
                rows.append(dict(event=ev, seed=seed, rule=m, FA_target_per_day=target,
                                 FA_calibration_per_day=(PC[m] > thr).sum() / cal_days,
                                 **alarm_stats(PT[m] > thr, yt, leak, st, la, te_days)))
            # combined rule: alarm if either model alarms; one shared quantile level q is recalibrated
            # on the calibration normals so that the UNION meets the same target (no test data used)
            best_q = None
            for q in np.linspace(rate, rate / 3, 400):
                t1 = np.quantile(PC["current"], 1 - q, method="higher"); t2 = np.quantile(PC["cur_last"], 1 - q, method="higher")
                if ((PC["current"] > t1) | (PC["cur_last"] > t2)).mean() <= rate:
                    best_q = q; break
            q = best_q if best_q is not None else rate / 3
            t1 = np.quantile(PC["current"], 1 - q, method="higher"); t2 = np.quantile(PC["cur_last"], 1 - q, method="higher")
            rows.append(dict(event=ev, seed=seed, rule="combined (recalibrated OR)", FA_target_per_day=target,
                             FA_calibration_per_day=((PC["current"] > t1) | (PC["cur_last"] > t2)).sum() / cal_days,
                             **alarm_stats((PT["current"] > t1) | (PT["cur_last"] > t2), yt, leak, st, la, te_days)))
            # uncalibrated OR (both thresholds at the single-model level) for contrast
            t1 = np.quantile(PC["current"], 1 - rate, method="higher"); t2 = np.quantile(PC["cur_last"], 1 - rate, method="higher")
            rows.append(dict(event=ev, seed=seed, rule="combined (not recalibrated)", FA_target_per_day=target,
                             FA_calibration_per_day=((PC["current"] > t1) | (PC["cur_last"] > t2)).sum() / cal_days,
                             **alarm_stats((PT["current"] > t1) | (PT["cur_last"] > t2), yt, leak, st, la, te_days)))
    print(seed, "done", flush=True)
pd.DataFrame(rows).to_csv(OUTD / "p3_metropt_alarms.csv", index=False)
