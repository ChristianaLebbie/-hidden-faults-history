"""Figures for the findings report. Each figure is skipped if its results file is missing."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pfi.methods import METHOD_LABELS

ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "results"; FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
BLUE, ORANGE, AQUA, GRAY, LGRAY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#d6d5d0", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": LGRAY, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#ecebe7", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "figure.dpi": 200, "savefig.bbox": "tight", "legend.frameon": False})


def have(*names):
    return all((R / n).exists() for n in names)


def barh(ax, labels, vals, errs, colors, fmt="{:.3f}", xlim=None):
    yy = np.arange(len(labels))[::-1]
    ax.barh(yy, vals, xerr=errs, color=colors, height=0.62, error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
    ax.set_yticks(yy); ax.set_yticklabels(labels); ax.grid(axis="y", visible=False)
    if xlim:
        ax.set_xlim(*xlim)
    span = ax.get_xlim()[1] - ax.get_xlim()[0]
    for yv, v, e in zip(yy, vals, errs):
        ax.text(v + (e if np.isfinite(e) else 0) + span * 0.01, yv, fmt.format(v), va="center", fontsize=8.5, color=INK)


# F1: method comparison on AI4I S1
if have("exp1_ai4i_methods.csv"):
    d = pd.read_csv(R / "exp1_ai4i_methods.csv"); d = d[d.scenario == "S1"]
    order = ["current", "locf", "locf_ind", "locf_ind_age", "trend_impute", "cur_lags", "cur_window", "cur_ewma",
             "cur_last_age", "cur_last", "gated"]
    g = d.groupby("method").PR_AUC.agg(["mean", "std"]).loc[order]
    col = [GRAY, ORANGE, AQUA, AQUA, AQUA, BLUE, BLUE, BLUE, BLUE, BLUE, BLUE]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    barh(ax, [METHOD_LABELS[m] for m in order], g["mean"].values, g["std"].values, col, xlim=(0, 0.72))
    ax.set_xlabel("PR-AUC, realistic schedules (mean ± SD, 5 seeds)")
    fig.savefig(FIG / "f1_methods_ai4i.png"); plt.close(fig)

# F2: reliability of history by staleness
if have("exp8_reliability.csv"):
    t = pd.read_csv(R / "exp8_reliability.csv")
    fig, axs = plt.subplots(1, 3, figsize=(10.4, 3.1), sharey=True)
    lab = {"air": "air temperature", "proc": "process temperature", "rpm": "rotational speed", "torque": "torque", "wear": "tool wear"}
    sub = t[(t.dataset == "AI4I-PMDI")]
    for c, colr in zip(["air", "proc", "wear", "rpm", "torque"], [BLUE, AQUA, ORANGE, GRAY, INK2]):
        q = sub[sub.sensor == c].dropna(subset=["R2"]); axs[0].plot(range(len(q)), q.R2.clip(lower=-1.2), "-o", ms=3, color=colr, label=lab[c])
    axs[0].set_title("AI4I-PMDI", fontsize=9.5, color=INK); axs[0].legend(fontsize=7, loc="center left", bbox_to_anchor=(0, 0.42))
    for ax, (ds_, rd) in zip(axs[1:], [("C-MAPSS FD002", "raw"), ("C-MAPSS FD002", "condition-normalised")]):
        q = t[(t.dataset == ds_) & (t.readings == rd)]
        for c in q.sensor.unique():
            z = q[q.sensor == c].dropna(subset=["R2"]); ax.plot(range(len(z)), z.R2.clip(lower=-1.2), "-", lw=1, color=BLUE if rd != "raw" else ORANGE, alpha=0.6)
        ax.set_title(f"FD002, {rd} readings", fontsize=9.5, color=INK)
    for ax in axs:
        ax.set_xticks(range(7)); ax.set_xticklabels(["1", "2", "3", "4-5", "6-10", "11-20", ">20"]); ax.axhline(0, color=LGRAY, lw=0.8)
        ax.set_xlabel("Age of last reading (inspections)")
    axs[0].set_ylabel("R² of last reading for current value"); axs[0].set_ylim(-1.25, 1.05)
    fig.savefig(FIG / "f2_reliability.png"); plt.close(fig)

# F3: shuffled history
if have("exp2_shuffled_history.csv"):
    d = pd.read_csv(R / "exp2_shuffled_history.csv")
    order = ["none", "other_unit", "unit_mean", "random_past", "old_past", "true"]
    lab = {"none": "No history", "other_unit": "History from a different unit", "unit_mean": "Unit's average of past readings",
           "random_past": "Random earlier reading, same unit", "old_past": "Reading ≥10 inspections old, same unit", "true": "Most recent earlier reading"}
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 2.9), sharey=True)
    for ax, ds_ in zip(axs, ["AI4I-PMDI", "C-MAPSS FD001"]):
        g = d[d.dataset == ds_].groupby("history").PR_AUC.agg(["mean", "std"]).reindex(order)
        barh(ax, [lab[o] for o in order], g["mean"].values, g["std"].values, [GRAY, ORANGE, ORANGE, AQUA, AQUA, BLUE],
             xlim=(0, 0.75 if ds_.startswith("AI4I") else 1.08))
        ax.set_title(ds_, fontsize=9.5, color=INK); ax.set_xlabel("PR-AUC (mean ± SD)")
    fig.savefig(FIG / "f3_shuffle.png"); plt.close(fig)

# F4: consecutive missing
if have("exp3_consecutive_missing.csv"):
    d = pd.read_csv(R / "exp3_consecutive_missing.csv")
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.1))
    for ax, ds_ in zip(axs, ["AI4I-PMDI", "C-MAPSS FD001"]):
        q = d[d.dataset == ds_]
        for m, c in [("current", GRAY), ("locf_ind_age", AQUA), ("cur_last", BLUE), ("gated", ORANGE)]:
            g = q[q.method == m].groupby("mean_run").PR_AUC.agg(["mean", "std"])
            ax.errorbar(range(len(g)), g["mean"], yerr=g["std"], fmt="-o", ms=4, capsize=2, color=c, label=METHOD_LABELS[m])
        ax.set_xticks(range(len(g))); ax.set_xticklabels(["none"] + [str(int(x)) for x in g.index[1:]])
        ax.set_xlabel("Mean outage length (inspections)"); ax.set_title(ds_, fontsize=9.5, color=INK)
    axs[0].set_ylabel("PR-AUC")
    h_, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h_, l_, fontsize=7.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    fig.savefig(FIG / "f4_outages.png"); plt.close(fig)

# F5: hidden-fault recall by age
if have("stats_hidden_by_age.csv"):
    d = pd.read_csv(R / "stats_hidden_by_age.csv")
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    for m, c in [("current", GRAY), ("locf_ind_age", AQUA), ("cur_last", BLUE), ("gated", ORANGE)]:
        q = d[d.method == m]; ax.plot(range(len(q)), q["mean"], "-o", ms=4, color=c, label=METHOD_LABELS[m])
    ax.set_xticks(range(len(q))); ax.set_xticklabels(q.age_bin.astype(str)); ax.set_ylim(0, 1)
    ax.set_xlabel("Age of the missing causal reading's last value (inspections)"); ax.set_ylabel("Hidden-fault recall")
    ax.legend(fontsize=7.5); fig.savefig(FIG / "f5_hidden_by_age.png"); plt.close(fig)

# F6: operating conditions
if have("exp5_operating_conditions.csv"):
    d = pd.read_csv(R / "exp5_operating_conditions.csv")
    cases = [("raw", "current", "Current only (raw)", GRAY), ("raw", "cur_last", "Current + last (raw readings)", ORANGE),
             ("same-cond", "cur_last", "Current + last under same condition", AQUA), ("normalised", "current", "Current only (normalised)", LGRAY),
             ("normalised", "cur_last", "Current + last (normalised)", BLUE), ("normalised", "gated", "Reliability-gated (normalised)", INK2)]
    fds = ["FD001", "FD003", "FD002", "FD004"]; w = 0.13
    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    for k, (h, m, lab, c) in enumerate(cases):
        vals = [d[(d.fd == f) & (d.history == h) & (d.method == m)].PR_AUC.mean() for f in fds]
        ax.bar(np.arange(4) + (k - 2.5) * w, vals, w, color=c, label=lab)
    ref = [d[(d.fd == f) & (d.history == "complete-normalised")].PR_AUC.mean() for f in fds]
    ax.scatter(np.arange(4), ref, marker="_", s=900, color=INK, zorder=3, label="Complete data (normalised)")
    ax.set_xticks(range(4)); ax.set_xticklabels(["FD001\n1 condition", "FD003\n1 condition, 2 faults", "FD002\n6 conditions", "FD004\n6 conditions, 2 faults"])
    ax.set_ylabel("PR-AUC (mean of 3 seeds)"); ax.set_ylim(0.75, 0.97)
    ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.24))
    fig.savefig(FIG / "f6_conditions.png"); plt.close(fig)

# F7: schedule change heatmaps
if have("exp4_schedule_change.csv"):
    d = pd.read_csv(R / "exp4_schedule_change.csv"); d = d[d.dataset == "AI4I-PMDI"]
    pols = ["uniform", "A-heavy", "C-heavy", "rare-C", "round-robin"]
    fig, axs = plt.subplots(1, 2, figsize=(9.4, 3.4))
    for ax, m in zip(axs, ["current", "cur_last"]):
        M = d[d.method == m].groupby(["train_policy", "test_policy"]).PR_AUC.mean().unstack().reindex(index=pols, columns=pols)
        ax.imshow(M.values, cmap="Blues", vmin=0.2, vmax=0.8); ax.grid(False)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{M.values[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if M.values[i, j] > 0.55 else INK)
        ax.set_xticks(range(5)); ax.set_xticklabels(pols, rotation=30, ha="right"); ax.set_yticks(range(5)); ax.set_yticklabels(pols)
        ax.set_xlabel("Deployment schedule"); ax.set_title(METHOD_LABELS[m], fontsize=9.5, color=INK)
    axs[0].set_ylabel("Training schedule")
    fig.savefig(FIG / "f7_schedule_change.png"); plt.close(fig)

# F8: calibration
if have("exp6_reliability_bins.csv"):
    b = pd.read_csv(R / "exp6_reliability_bins.csv")
    fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.1), sharey=True)
    for ax, ds_ in zip(axs, ["AI4I-PMDI", "C-MAPSS FD001", "C-MAPSS FD002"]):
        q = b[(b.dataset == ds_) & (b.method.str.endswith("cur_last"))]
        for k, c in [("none", GRAY), ("platt", BLUE), ("isotonic", ORANGE)]:
            z = q[q.calibration == k].groupby("bin").apply(lambda g: pd.Series(dict(x=np.average(g.mean_pred, weights=g.n), y=np.average(g.frac_pos, weights=g.n), n=g.n.sum())), include_groups=False)
            z = z[z.n >= 20]; ax.plot(z.x, z.y, "-o", ms=3.5, color=c, label=k)
        ax.plot([0, 1], [0, 1], color=LGRAY, lw=1); ax.set_title(ds_, fontsize=9.5, color=INK); ax.set_xlabel("Predicted probability")
    axs[0].set_ylabel("Observed fault rate"); axs[0].legend(fontsize=7.5, title="calibration", title_fontsize=7.5)
    fig.savefig(FIG / "f8_calibration.png"); plt.close(fig)

# F9: referral strategies and reviewer accuracy
if have("exp7_referral.csv"):
    d = pd.read_csv(R / "exp7_referral.csv")
    strat = [("random", LGRAY), ("ambiguity", AQUA), ("below_threshold", BLUE), ("expected_cost", INK2), ("evidence_spread", ORANGE), ("upper_risk", "#e87ba4")]
    fig, axs = plt.subplots(1, 3, figsize=(9.8, 3.1), sharey=False)
    for ax, ds_ in zip(axs, ["AI4I-PMDI", "C-MAPSS FD001", "C-MAPSS FD002"]):
        q = d[(d.dataset == ds_) & (d.reviewer == "perfect")]
        for s, c in strat:
            g = q[q.strategy == s].groupby("budget").silent_miss.mean(); ax.plot(100 * g.index, 100 * g.values, "-o", ms=3, color=c, label=s.replace("_", " "))
        ax.set_title(ds_, fontsize=9.5, color=INK); ax.set_xlabel("Review budget (% of inspections)")
    axs[0].set_ylabel("Faults missed silently (%)"); axs[0].legend(fontsize=7)
    fig.savefig(FIG / "f9_referral.png"); plt.close(fig)
    q = d[(d.dataset == "AI4I-PMDI") & (d.budget == 0.10)]
    revs = ["perfect", "uniform_90_95", "uniform_70_95", "same_evidence", "reinspection"]
    fig, ax = plt.subplots(figsize=(8.6, 3.3)); w = 0.13
    base = d[(d.dataset == "AI4I-PMDI") & (d.budget == 0.0) & (d.reviewer == "perfect")].system_recall.mean()
    for k, (s, c) in enumerate(strat):
        vals = [q[(q.strategy == s) & (q.reviewer == r)].system_recall.mean() for r in revs]
        ax.bar(np.arange(5) + (k - 2.5) * w, vals, w, color=c, label=s.replace("_", " "))
    ax.axhline(base, color=INK, lw=1, ls="--"); ax.text(-0.48, base - 0.035, "no review", fontsize=7.5, ha="left")
    ax.set_xticks(range(5)); ax.set_xticklabels(["perfect", "sens 0.9", "sens 0.7", "same evidence\n(0.9 / 0.5 hidden)", "full re-inspection\n(model)"])
    ax.tick_params(axis="x", labelsize=8)
    ax.set_ylabel("System fault recall"); ax.set_ylim(0.4, 1.0); ax.legend(fontsize=7, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.savefig(FIG / "f10_reviewer.png"); plt.close(fig)

# F11: MetroPT-3
if have("exp10_metropt.csv"):
    d = pd.read_csv(R / "exp10_metropt.csv")
    order = ["complete", "current", "locf_ind_age", "trend_impute", "cur_window", "cur_last_age", "cur_last", "gated"]
    lab = {**METHOD_LABELS, "complete": "Complete data (all signals every inspection)"}
    fig, axs = plt.subplots(1, 3, figsize=(10.4, 3.3), sharey=False)
    for ax, ev in zip(axs, ["F2", "F3", "F4"]):
        g = d[d.event == ev].groupby("method").PR_AUC.agg(["mean", "std"]).reindex(order)
        barh(ax, [lab[m] for m in order] if ev == "F2" else [""] * len(order), g["mean"].values, g["std"].values,
             [INK2, GRAY, AQUA, AQUA, BLUE, BLUE, BLUE, ORANGE], xlim=(0, 1.15))
        ax.set_title(f"Air leak {ev[1]} (trained on earlier data only)", fontsize=9, color=INK); ax.set_xlabel("PR-AUC (mean ± SD, 5 schedules)")
        if ev != "F2":
            ax.tick_params(labelleft=False)
    fig.savefig(FIG / "f11_metropt.png"); plt.close(fig)
# F12: MetroPT-3 stress tests (shuffled history, outages)
if have("exp12_metropt_stress.csv"):
    d = pd.read_csv(R / "exp12_metropt_stress.csv")
    fig, axs = plt.subplots(1, 2, figsize=(10.0, 3.1))
    q = d[d.test == "shuffle"]; order = ["none", "unit_mean", "older_1day", "random_past", "older_1h", "true"]
    lab = {"none": "No history", "unit_mean": "Average of all earlier windows", "older_1day": "Reading ≥ 1 day old",
           "random_past": "Random earlier window", "older_1h": "Reading ≥ 1 hour old", "true": "Most recent earlier reading"}
    g = q.groupby("variant").PR_AUC.agg(["mean", "std"]).reindex(order)
    barh(axs[0], [lab[o] for o in order], g["mean"].values, g["std"].values, [GRAY, ORANGE, ORANGE, AQUA, AQUA, BLUE], xlim=(0, 0.95))
    axs[0].set_xlabel("PR-AUC, mean of 3 test periods × 3 schedules"); axs[0].set_title("Shuffled / degraded history", fontsize=9.5, color=INK)
    q = d[d.test == "outage"]
    for m, c in [("current", GRAY), ("locf_ind_age", AQUA), ("cur_last", BLUE), ("gated", ORANGE)]:
        g = q[q.method == m].groupby("variant").PR_AUC.mean()
        g = g.reindex([f"mean_run_{L}" for L in [0, 3, 18, 72]])
        axs[1].plot(range(4), g.values, "-o", ms=4, color=c, label=METHOD_LABELS[m])
    axs[1].set_xticks(range(4)); axs[1].set_xticklabels(["none", "30 min", "3 h", "12 h"])
    axs[1].set_xlabel("Mean outage length (30% of windows affected)"); axs[1].set_ylabel("PR-AUC")
    axs[1].legend(fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    axs[1].set_title("Consecutive missing observations", fontsize=9.5, color=INK)
    fig.savefig(FIG / "f12_metropt_stress.png"); plt.close(fig)

# F13: history when nothing is missing
if have("exp11_complete_with_history.csv"):
    d = pd.read_csv(R / "exp11_complete_with_history.csv")
    d["ds"] = d.dataset.str.replace(r"MetroPT-3 F\d", "MetroPT-3", regex=True)
    g = d.groupby(["ds", "variant"]).PR_AUC.mean().unstack()
    order = [x for x in ["AI4I-PMDI", "C-MAPSS FD001", "C-MAPSS FD002", "MetroPT-3"] if x in g.index]
    fig, ax = plt.subplots(figsize=(6.4, 2.8)); w = 0.36
    ax.bar(np.arange(len(order)) - w / 2, g.loc[order, "complete"], w, color=GRAY, label="Complete readings")
    ax.bar(np.arange(len(order)) + w / 2, g.loc[order, "complete+last"], w, color=BLUE, label="Complete readings + previous inspection")
    for i, o in enumerate(order):
        for k, v in enumerate([g.loc[o, "complete"], g.loc[o, "complete+last"]]):
            ax.text(i + (k - 0.5) * w, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order); ax.set_ylabel("PR-AUC"); ax.set_ylim(0, 1.25); ax.legend(fontsize=7.5, ncol=2, loc="upper center")
    fig.savefig(FIG / "f13_history_no_missing.png"); plt.close(fig)

print("figures written:", sorted(p.name for p in FIG.glob("*.png")))
