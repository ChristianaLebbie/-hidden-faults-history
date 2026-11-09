"""Point 4: what the evidence-kept measure says, and how it relates to the classification gains.
Offline only: it needs fault labels and the true values of the unmeasured readings."""
import json
import numpy as np, pandas as pd
from sv_common import ai4i, load_pred, SEEDS, OUT, ROOT, prauc
d = pd.read_csv(ROOT / "results" / "exp17_reconstruction_by_sensor.csv"); d = d[d.condition == "all"]
w = lambda x: pd.Series({"R2": np.average(x.R2_reconstruction, weights=x.n), "sep_true": np.average(x.sep_true, weights=x.n),
                         "sep_carried": np.average(x.sep_carried, weights=x.n),
                         "kept": (x.sep_carried * x.n).sum() / (x.sep_true * x.n).sum()})
by_sensor = d.groupby(["dataset", "history", "sensor"]).apply(w, include_groups=False).reset_index()
by_sensor.to_csv(OUT / "p4_evidence_by_sensor.csv", index=False)
by_set = d.groupby(["dataset", "history"]).apply(w, include_groups=False).reset_index()
# classification: share of the gap to complete data closed by history
ds = ai4i(); y = ds.y
ai = {m: np.mean([prauc(y, load_pred(t, s)[0]) for s in SEEDS]) for m, t in
      [("current", "ai4i_S1_current"), ("history", "ai4i_S1_cur_last"), ("complete", "ai4i_REF")]}
e = pd.read_csv(ROOT / "results" / "exp5_operating_conditions.csv").groupby(["fd", "history", "method"]).PR_AUC.mean()
cls = {("AI4I-PMDI", "raw"): (ai["current"], ai["history"], ai["complete"]),
       ("C-MAPSS FD001", "raw"): (e["FD001", "raw", "current"], e["FD001", "raw", "cur_last"], e["FD001", "complete-raw", "complete"]),
       ("C-MAPSS FD002", "raw"): (e["FD002", "raw", "current"], e["FD002", "raw", "cur_last"], e["FD002", "complete-raw", "complete"]),
       ("C-MAPSS FD002", "normalised"): (e["FD002", "normalised", "current"], e["FD002", "normalised", "cur_last"], e["FD002", "complete-normalised", "complete"])}
for k in ["current", "history", "complete"]:
    by_set[f"PR_AUC_{k}"] = [cls[(a, b)][["current", "history", "complete"].index(k)] for a, b in zip(by_set.dataset, by_set.history)]
by_set["gap_closed"] = (by_set.PR_AUC_history - by_set.PR_AUC_current) / (by_set.PR_AUC_complete - by_set.PR_AUC_current)
by_set.to_csv(OUT / "p4_evidence_vs_gain.csv", index=False)
print(by_set.round(3).to_string(index=False))
# AI4I per sensor vs the component analysis (gain from adding only that sensor group's last reading)
comp = {"air": 0.078, "proc": 0.078, "wear": 0.059, "rpm": -0.022, "torque": -0.022}
a = by_sensor[by_sensor.dataset == "AI4I-PMDI"].copy(); a["component_gain"] = a.sensor.map(comp)
print(a.round(3).to_string(index=False))
