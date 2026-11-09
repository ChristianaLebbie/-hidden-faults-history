"""Point 2: AMA against its closest historical baselines, with paired differences and 95% intervals
from a bootstrap that resamples whole machines (all records of a machine stay together)."""
import numpy as np
import pandas as pd
from sv_common import ai4i, load_pred, alarms_at, SEEDS, OUT, METHODS, machine_bootstrap, prauc
from pfi.schedules import ai4i_realistic, hidden_mask, visible_mask

ds = ai4i(); y = ds.y; units = ds.unit
TARGETS = [0.5, 1.0, 2.0]
P, A, HID, VIS = {}, {}, {}, {}
fa_rows = []
for seed in SEEDS:
    sch = ai4i_realistic(ds, 100 + seed); HID[seed], VIS[seed] = hidden_mask(ds, sch), visible_mask(ds, sch)
    for m in METHODS:
        p, fold, inner = load_pred(f"ai4i_S1_{m}", seed); P[m, seed] = p
        for tg in TARGETS:
            al, _ = alarms_at(p, fold, inner, y, tg); A[m, seed, tg] = al
            fa_rows.append(dict(seed=seed, method=METHODS[m], FA_target_per100=tg, FA_realised_per100=100 * al[y == 0].mean(),
                                recall_all=al[y == 1].mean(), recall_hidden=al[HID[seed]].mean(),
                                recall_visible=al[VIS[seed]].mean(), precision=y[al].mean()))
fa = pd.DataFrame(fa_rows); fa.to_csv(OUT / "p2_fixed_false_alarm.csv", index=False)
print(fa.groupby(["FA_target_per100", "method"]).mean(numeric_only=True).drop(columns="seed").round(3).to_string())


def stat(kind, m, idx):
    vals = []
    for s in SEEDS:
        yy = y[idx]
        if kind == "PR_AUC":
            vals.append(prauc(yy, P[m, s][idx]))
        elif kind == "PR_AUC_hidden":
            k = idx[HID[s][idx] | (y[idx] == 0)]; vals.append(prauc(y[k], P[m, s][k]))
        else:
            grp = {"recall_hidden": HID[s], "recall_visible": VIS[s], "recall_all": y == 1}[kind]
            k = idx[grp[idx]]; vals.append(A[m, s, 1.0][k].mean() if len(k) else np.nan)
    return np.nanmean(vals)


pairs = [("cur_last", "current"), ("cur_last", "locf_ind"), ("cur_last", "locf_ind_age"), ("locf_ind", "current")]
rows = []
for kind in ["PR_AUC", "PR_AUC_hidden", "recall_hidden", "recall_visible", "recall_all"]:
    for a_, b_ in pairs:
        est, lo, hi, share = machine_bootstrap(lambda idx: stat(kind, a_, idx) - stat(kind, b_, idx), units, n_boot=1000, seed=1)
        rows.append(dict(measure=kind, comparison=f"{METHODS[a_]} minus {METHODS[b_]}", difference=est, ci_low=lo, ci_high=hi,
                         share_same_sign=share, A=stat(kind, a_, np.arange(len(y))), B=stat(kind, b_, np.arange(len(y)))))
        print(rows[-1])
pd.DataFrame(rows).to_csv(OUT / "p2_paired_differences.csv", index=False)
