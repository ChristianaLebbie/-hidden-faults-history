"""Point 5: FD002, separate the contribution of history from that of within-condition normalisation (2 x 2),
with 95% intervals from a bootstrap over engines."""
import numpy as np, pandas as pd
from sv_common import machine_bootstrap, prauc, OUT, PRED
from pfi.data import load_cmapss
ds = load_cmapss("FD002"); y = ds.y; units = ds.unit; S = [0, 1, 2]
P = {k: [np.load(PRED / f"cm_FD002_{k}_s{s}.npz")["p"] for s in S] for k in
     ["raw_current", "norm_current", "raw_cur_last", "norm_cur_last", "complete-raw", "complete-normalised"]}
pr = lambda k, idx: np.mean([prauc(y[idx], p[idx]) for p in P[k]])
allidx = np.arange(len(y))
cells = pd.DataFrame([dict(cell=k, PR_AUC=pr(k, allidx)) for k in P]); print(cells.round(3).to_string(index=False))
cells.to_csv(OUT / "p5_fd002_cells.csv", index=False)
contr = {"history, raw readings": ("raw_cur_last", "raw_current"), "history, normalised readings": ("norm_cur_last", "norm_current"),
         "normalisation, current readings only": ("norm_current", "raw_current"), "normalisation, with history": ("norm_cur_last", "raw_cur_last"),
         "both together": ("norm_cur_last", "raw_current")}
rows = []
for name, (a, b) in contr.items():
    est, lo, hi, share = machine_bootstrap(lambda idx: pr(a, idx) - pr(b, idx), units, n_boot=500, seed=2)
    rows.append(dict(contribution=name, difference=est, ci_low=lo, ci_high=hi)); print(rows[-1])
est, lo, hi, _ = machine_bootstrap(lambda idx: (pr("norm_cur_last", idx) - pr("norm_current", idx)) - (pr("raw_cur_last", idx) - pr("raw_current", idx)), units, n_boot=500, seed=3)
rows.append(dict(contribution="interaction (history gain normalised minus raw)", difference=est, ci_low=lo, ci_high=hi))
pd.DataFrame(rows).to_csv(OUT / "p5_fd002_contributions.csv", index=False); print(rows[-1])
