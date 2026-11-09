"""Point 1: data construction, record order, hidden-fault rules and the single-fault check."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sv_common import ai4i, load_pred, alarms_at, SEEDS, OUT, CAUSAL, FLAGS, prauc
from pfi.schedules import ai4i_realistic, hidden_mask, visible_mask

ds = ai4i(); a = pd.read_csv(__import__("sv_common").ROOT / "data" / "ai4i2020.csv")
p = pd.read_csv(__import__("sv_common").ROOT / "data" / "AI4I-PMDI.csv")
rep = {}

# ---- 1a. label mapping between AI4I 2020 and AI4I-PMDI
pat = a[FLAGS].apply(lambda r: "+".join(f for f in FLAGS if r[f] == 1) or "none", axis=1)
lab = pd.DataFrame({"Diagnostic": p["Diagnostic"], "AI4I flags": pat, "Machine failure": a["Machine failure"]})
tab = lab[(lab.Diagnostic != "No failure") | (lab["AI4I flags"] != "none") | (lab["Machine failure"] == 1)] \
    .groupby(["Diagnostic", "AI4I flags", "Machine failure"]).size().rename("records").reset_index()
tab.to_csv(OUT / "p1_label_mapping.csv", index=False)
rep["positives"] = int(ds.y.sum()); rep["multi_mode_records"] = int((ds.meta["n_modes"] > 1).sum())

# ---- 1b. record order: published timestamps against the original record id (UDI)
t = pd.to_datetime(p["Date"], dayfirst=True)
rows = []
for u, g in p.assign(t=t).groupby("System"):
    rho = spearmanr(g["UDI"], g["t"].astype("int64")).correlation
    gs = g.sort_values(["t", "UDI"])
    rows.append(dict(system=u, records=len(g), spearman_udi_time=rho,
                     share_udi_increasing=float((np.diff(gs["UDI"].values) > 0).mean()),
                     tied_timestamps=int(gs["t"].duplicated().sum()),
                     median_gap_hours=float(gs["t"].diff().dt.total_seconds().median() / 3600)))
order = pd.DataFrame(rows); order.to_csv(OUT / "p1_record_order.csv", index=False)
rep["order"] = dict(machines=len(order), records_per_machine_median=float(order.records.median()),
                    spearman_median=float(order.spearman_udi_time.median()),
                    spearman_min=float(order.spearman_udi_time.min()),
                    share_udi_increasing_median=float(order.share_udi_increasing.median()),
                    tied_total=int(order.tied_timestamps.sum()),
                    global_spearman=float(spearmanr(p["UDI"], t.astype("int64")).correlation),
                    median_gap_hours=float(order.median_gap_hours.median()))

# ---- 1c. hidden / visible / partly hidden and multi-mode records (seed-wise schedules)
cnt = []; single = []
for seed in SEEDS:
    sch = ai4i_realistic(ds, 100 + seed)
    hid, vis = hidden_mask(ds, sch), visible_mask(ds, sch)
    mode = ds.meta["mode"]; groups = ds.groups
    # partly hidden: at least one causal sensor of the labelled mode measured at that inspection
    part = np.array([h and any(s in groups[c] for s in CAUSAL[m]) for h, m, c in zip(hid, mode, sch)])
    # multi-mode records that are hidden for the labelled mode but visible for another of their modes
    rev = ds.meta["revealing"]; F = ds.meta["flags"]
    other_visible = np.array([h and any(F[i, FLAGS.index(o)] and o != mode[i] and rev.get(o) == sch[i] for o in rev)
                              for i, h in enumerate(hid)])
    multi = ds.meta["n_modes"] > 1
    cnt.append(dict(seed=seed, hidden=int(hid.sum()), visible=int(vis.sum()), hidden_partly=int(part.sum()),
                    hidden_fully=int((hid & ~part).sum()), hidden_multi=int((hid & multi).sum()),
                    visible_multi=int((vis & multi).sum()), hidden_but_other_mode_visible=int(other_visible.sum())))
    # single-mode check: drop every multi-mode record from evaluation (alarm thresholds unchanged)
    keep = ~multi
    for m in ["current", "locf_ind", "cur_last"]:
        pr, fold, inner = load_pred(f"ai4i_S1_{m}", seed)
        alarm, _ = alarms_at(pr, fold, inner, ds.y, 1.0)
        for subset, mask in [("all records", np.ones_like(keep)), ("single-mode only", keep)]:
            single.append(dict(seed=seed, method=m, subset=subset,
                               PR_AUC=prauc(ds.y[mask], pr[mask]),
                               PR_AUC_hidden_vs_normal=prauc(ds.y[mask & (hid | (ds.y == 0))], pr[mask & (hid | (ds.y == 0))]),
                               recall_hidden=alarm[mask & hid].mean(), recall_visible=alarm[mask & vis].mean(),
                               recall_all=alarm[mask & (ds.y == 1)].mean(), n_hidden=int((mask & hid).sum()),
                               n_visible=int((mask & vis).sum())))
pd.DataFrame(cnt).to_csv(OUT / "p1_hidden_counts.csv", index=False)
s = pd.DataFrame(single); s.to_csv(OUT / "p1_single_mode.csv", index=False)
print(tab.to_string(index=False)); print(rep)
print(pd.DataFrame(cnt).mean().round(1).to_string())
print(s.groupby(["subset", "method"]).mean(numeric_only=True).drop(columns="seed").round(3).to_string())
import json; json.dump(rep, open(OUT / "p1_summary.json", "w"), indent=1)
