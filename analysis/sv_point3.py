"""Point 3 (AI4I): where the hidden-fault gain and the visible-fault loss come from, and whether the overall
benefit is stable across schedule changes (exp4)."""
import numpy as np
import pandas as pd
from sv_common import ai4i, load_pred, alarms_at, SEEDS, OUT, METHODS, ROOT
from pfi.schedules import ai4i_realistic, hidden_mask, visible_mask

ds = ai4i(); y = ds.y; mode = ds.meta["mode"]
MISSING = {"A": "torque, tool wear missing", "B": "air temp., process temp., tool wear missing",
           "C": "air temp., process temp., speed missing"}
rows, flips = [], []
for seed in SEEDS:
    sch = ai4i_realistic(ds, 100 + seed); hid, vis = hidden_mask(ds, sch), visible_mask(ds, sch)
    AL, THR, P = {}, {}, {}
    for m in ["current", "locf_ind", "cur_last"]:
        p, fold, inner = load_pred(f"ai4i_S1_{m}", seed); P[m] = p
        AL[m], THR[m] = alarms_at(p, fold, inner, y, 1.0)
    for md in ["HDF", "PWF", "OSF", "TWF"]:
        for g in ["A", "B", "C"]:
            k = (mode == md) & (sch == g)
            if k.sum() == 0:
                continue
            rows.append(dict(seed=seed, fault=md, group_measured=g, missing=MISSING[g],
                             status="visible" if ds.meta["revealing"][md] == g else "hidden", n=int(k.sum()),
                             **{f"recall_{m}": AL[m][k].mean() for m in AL}))
    # visible faults caught by current readings but missed by AMA, and the reverse
    for st, msk in [("visible", vis), ("hidden", hid)]:
        for md in ["HDF", "PWF", "OSF", "TWF"]:
            k = msk & (mode == md)
            flips.append(dict(seed=seed, status=st, fault=md, n=int(k.sum()),
                              lost_by_AMA=int((k & AL["current"] & ~AL["cur_last"]).sum()),
                              gained_by_AMA=int((k & ~AL["current"] & AL["cur_last"]).sum()),
                              lost_vs_LOCF_ind=int((k & AL["locf_ind"] & ~AL["cur_last"]).sum()),
                              gained_vs_LOCF_ind=int((k & ~AL["locf_ind"] & AL["cur_last"]).sum()),
                              thr_current=float(np.mean(THR["current"])), thr_AMA=float(np.mean(THR["cur_last"]))))
r = pd.DataFrame(rows); r.to_csv(OUT / "p3_by_type_and_group.csv", index=False)
f = pd.DataFrame(flips); f.to_csv(OUT / "p3_flips.csv", index=False)
agg = r.groupby(["fault", "group_measured", "status", "missing"]).agg(n=("n", "mean"), current=("recall_current", "mean"),
      LOCF_ind=("recall_locf_ind", "mean"), AMA=("recall_cur_last", "mean")).reset_index()
agg.to_csv(OUT / "p3_by_type_and_group_mean.csv", index=False)
print(agg.round(3).to_string(index=False))
print(f.groupby(["status", "fault"]).sum(numeric_only=True).drop(columns=["seed", "thr_current", "thr_AMA"]).to_string())

# ---- schedule changes (exp4): 5 training x 5 test policies x 3 seeds
e = pd.read_csv(ROOT / "results" / "exp4_schedule_change.csv")
pv = e.pivot_table(index=["train_policy", "test_policy", "seed"], columns="method", values=["PR_AUC", "recall_hidden"]).reset_index()
out = pd.DataFrame({"train": pv["train_policy"], "test": pv["test_policy"], "seed": pv["seed"],
                    "gain_PR_AUC_vs_current": pv[("PR_AUC", "cur_last")] - pv[("PR_AUC", "current")],
                    "gain_PR_AUC_vs_LOCF_ind_age": pv[("PR_AUC", "cur_last")] - pv[("PR_AUC", "locf_ind_age")],
                    "gain_hidden_recall_vs_current": pv[("recall_hidden", "cur_last")] - pv[("recall_hidden", "current")]})
cell = out.groupby(["train", "test"]).mean(numeric_only=True).drop(columns="seed")
cell.to_csv(OUT / "p3_schedule_stability.csv")
summ = dict(cells=len(cell), cells_gain_vs_current_positive=int((cell.gain_PR_AUC_vs_current > 0).sum()),
            gain_vs_current_min=float(cell.gain_PR_AUC_vs_current.min()), gain_vs_current_max=float(cell.gain_PR_AUC_vs_current.max()),
            gain_vs_current_mean=float(cell.gain_PR_AUC_vs_current.mean()),
            cells_gain_vs_locf_positive=int((cell.gain_PR_AUC_vs_LOCF_ind_age > 0).sum()),
            gain_vs_locf_mean=float(cell.gain_PR_AUC_vs_LOCF_ind_age.mean()), gain_vs_locf_min=float(cell.gain_PR_AUC_vs_LOCF_ind_age.min()),
            gain_vs_locf_max=float(cell.gain_PR_AUC_vs_LOCF_ind_age.max()),
            same_schedule_gain=float(cell[[a == b for a, b in cell.index]].gain_PR_AUC_vs_current.mean()),
            changed_schedule_gain=float(cell[[a != b for a, b in cell.index]].gain_PR_AUC_vs_current.mean()))
import json; json.dump(summ, open(OUT / "p3_schedule_summary.json", "w"), indent=1); print(summ)
