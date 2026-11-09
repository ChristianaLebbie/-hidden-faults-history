"""Chapter 1, hidden-fault evaluation at a common false-alarm target (supervisor feedback, Oct 2026).

Definitions (AI4I-PMDI, realistic schedule S1)
  Each inspection measures one sensor group: A = air temp, process temp, speed; B = speed, torque;
  C = torque, tool wear. Each deterministic fault type has one "revealing" group that measures all of
  its causal sensors: heat dissipation (HDF) -> A, power (PWF) -> B, overstrain (OSF) -> C, tool wear (TWF) -> C.
  Hidden fault  : a deterministic fault recorded under a group other than its revealing group.
  Visible fault : a deterministic fault recorded under its revealing group.
  Random failures (RNF) have no causal sensor, so they are neither hidden nor visible (they still count
  in "all faults"). Normal records are all records with no failure.

Common false-alarm target
  For each outer fold, the alarm threshold is set so that a given share of NORMAL records raise an alarm
  (0.5, 1 or 2 false alarms per 100 normal records). The threshold is chosen only from the inner
  out-of-fold predictions on that fold's TRAINING records (no test data). The same threshold is then used
  for hidden faults, visible faults and all faults, so their recalls are compared at the same
  false-alarm level. The realised false-alarm rate on the test records is reported as a check.

Uses the predictions saved by exp1 (results/predictions/ai4i_S1_<method>_s<seed>.npz) when they exist;
otherwise it reruns those models (same seeds, schedules and folds as exp1).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from pfi.data import load_ai4i
from pfi.schedules import ai4i_realistic, apply_schedule, hidden_mask, visible_mask
from pfi.history import past_features
from pfi.methods import build_features, ALL_STATS
from pfi.runner import run, append, log, PRED, RES

SEEDS = [0, 1, 2, 3, 4]
METHODS = {"current": "Current readings only", "locf_ind_age": "LOCF + flags + age",
           "cur_last": "AMA without age", "cur_last_age": "AMA"}
TARGETS = [0.5, 1.0, 2.0]   # false alarms per 100 normal records

ds = load_ai4i(); rows = []; counts = []
for seed in SEEDS:
    sch = ai4i_realistic(ds, 100 + seed)
    hid, vis = hidden_mask(ds, sch), visible_mask(ds, sch); rnd = ds.meta["mode"] == "RNF"; normal = ds.y == 0
    counts.append(dict(seed=seed, normal=int(normal.sum()), all_faults=int(ds.y.sum()), hidden=int(hid.sum()),
                       visible=int(vis.sum()), random_failures=int(rnd.sum()),
                       **{f"hidden_{m}": int((hid & (ds.meta["mode"] == m)).sum()) for m in ["HDF", "PWF", "OSF", "TWF"]},
                       **{f"visible_{m}": int((vis & (ds.meta["mode"] == m)).sum()) for m in ["HDF", "PWF", "OSF", "TWF"]}))
    X = H = None
    for m, label in METHODS.items():
        f = PRED / f"ai4i_S1_{m}_s{seed}.npz"
        if f.exists():
            z = np.load(f); p, fold = z["p"], z["fold"]
            inner = [(z[f"inner{i}_idx"], z[f"inner{i}_p"]) for i in range(5)]
        else:
            if X is None:
                X = apply_schedule(ds.X, sch, ds.groups); H = past_features(X, ds.unit, ds.order, stats=ALL_STATS)
            _, res, _ = run(ds, lambda tr: build_features(m, X, H, ds.static, tr), seed, tag=f"ai4i_S1_{m}_s{seed}")
            p, fold = res["p"], res["fold"]; inner = [(i["idx"], i["p"]) for i in res["inner"]]
        for target in TARGETS:
            alarm = np.zeros(len(p), bool)
            for k, (idx, ip) in enumerate(inner):
                tr_normal = ip[ds.y[idx] == 0]
                thr = np.quantile(tr_normal, 1 - target / 100, method="higher")
                te = fold == k; alarm[te] = p[te] > thr
            rows.append(dict(seed=seed, method=m, label=label, FA_target_per100=target,
                             FA_realised_per100=100 * alarm[normal].mean(),
                             recall_all=alarm[ds.y == 1].mean(), recall_hidden=alarm[hid].mean(),
                             recall_visible=alarm[vis].mean(), recall_random=alarm[rnd].mean(),
                             precision=ds.y[alarm].mean() if alarm.any() else np.nan))
        log(seed, label, "recall hidden/visible at 1 per 100:",
            round(rows[-2]["recall_hidden"], 3), round(rows[-2]["recall_visible"], 3))
    append(rows, "exp15_fixed_false_alarm.csv")
pd.DataFrame(counts).to_csv(RES / "exp15_hidden_fault_counts.csv", index=False)
log("done")
