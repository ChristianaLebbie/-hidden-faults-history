"""Collects the headline numbers used in the findings report into results/report_numbers.json."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1] / "results"
N = {}


def ms(s):
    return [round(float(s.mean()), 4), round(float(s.std()), 4)]


if (R / "exp1_ai4i_methods.csv").exists():
    d = pd.read_csv(R / "exp1_ai4i_methods.csv")
    N["exp1"] = {f"{sc}|{m}": {c: ms(g[c]) for c in ["PR_AUC", "ROC_AUC", "recall", "precision", "FA_per100", "recall_hidden", "recall_visible"] if c in g and g[c].notna().any()}
                 for (sc, m), g in d.groupby(["scenario", "method"])}
for name in ["stats_study1_pairs", "stats_shuffle", "stats_metropt_history", "stats_hidden_by_age", "exp8_reliability"]:
    if (R / f"{name}.csv").exists():
        N[name] = pd.read_csv(R / f"{name}.csv").round(4).to_dict("records")
if (R / "exp2_shuffled_history.csv").exists():
    d = pd.read_csv(R / "exp2_shuffled_history.csv")
    N["exp2"] = {f"{ds}|{h}": {c: ms(g[c]) for c in ["PR_AUC", "recall_hidden"] if c in g and g[c].notna().any()} for (ds, h), g in d.groupby(["dataset", "history"])}
if (R / "exp3_consecutive_missing.csv").exists():
    d = pd.read_csv(R / "exp3_consecutive_missing.csv")
    N["exp3"] = {f"{ds}|{L}|{m}": dict(PR_AUC=ms(g.PR_AUC), missing=round(float(g.missing_share.mean()), 3), age=round(float(g.mean_age_missing.mean()), 2))
                 for (ds, L, m), g in d.groupby(["dataset", "mean_run", "method"])}
if (R / "exp4_schedule_change.csv").exists():
    d = pd.read_csv(R / "exp4_schedule_change.csv")
    N["exp4"] = {f"{m}|{a}|{b}": round(float(g.PR_AUC.mean()), 4) for (m, a, b), g in d.groupby(["method", "train_policy", "test_policy"])}
if (R / "exp5_operating_conditions.csv").exists():
    d = pd.read_csv(R / "exp5_operating_conditions.csv")
    N["exp5"] = {f"{fd}|{h}|{m}": ms(g.PR_AUC) for (fd, h, m), g in d.groupby(["fd", "history", "method"])}
if (R / "exp6_calibration.csv").exists():
    d = pd.read_csv(R / "exp6_calibration.csv")
    N["exp6"] = {f"{ds}|{m}|{k}": {c: round(float(g[c].mean()), 4) for c in ["Brier", "logloss", "ECE10", "ECE15", "ECE_mass10"]}
                 for (ds, m, k), g in d.groupby(["dataset", "method", "calibration"])}
if (R / "exp7_referral.csv").exists():
    d = pd.read_csv(R / "exp7_referral.csv")
    cols = [c for c in ["referred", "faults_referred", "referral_yield", "silent_miss", "hidden_referred", "hidden_silent_miss", "visible_referred",
                        "healthy_referred", "system_recall", "system_FA_per100", "system_precision"] if c in d]
    N["exp7"] = {f"{ds}|{b}|{s}|{r}": {c: round(float(g[c].mean()), 4) for c in cols}
                 for (ds, b, s, r), g in d.groupby(["dataset", "budget", "strategy", "reviewer"])}
if (R / "exp9_forward_in_time.csv").exists():
    d = pd.read_csv(R / "exp9_forward_in_time.csv")
    N["exp9"] = {f"{q}|{m}": {c: ms(g[c]) for c in ["PR_AUC", "recall_hidden"]} for (q, m), g in d.groupby(["cutoff_quantile", "method"])}
if (R / "exp10_metropt.csv").exists():
    d = pd.read_csv(R / "exp10_metropt.csv")
    N["exp10"] = {f"{e}|{m}": dict(PR_AUC=ms(g.PR_AUC), ROC_AUC=ms(g.ROC_AUC), FA_day=ms(g.FA_per_day_to_catch)) for (e, m), g in d.groupby(["event", "method"])}
    N["exp10_mean"] = {m: ms(g.groupby("seed").PR_AUC.mean()) for m, g in d.groupby("method")}
if (R / "exp11_complete_with_history.csv").exists():
    d = pd.read_csv(R / "exp11_complete_with_history.csv")
    N["exp11"] = {f"{ds}|{v}": ms(g.PR_AUC) for (ds, v), g in d.groupby(["dataset", "variant"])}
if (R / "exp12_metropt_stress.csv").exists():
    d = pd.read_csv(R / "exp12_metropt_stress.csv")
    N["exp12"] = {f"{t}|{v}|{m}": ms(g.groupby("seed").PR_AUC.mean()) for (t, v, m), g in d.groupby(["test", "variant", "method"])}
json.dump(N, open(R / "report_numbers.json", "w"), indent=1)
print("keys:", list(N))
