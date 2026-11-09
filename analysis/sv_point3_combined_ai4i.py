"""Point 3 (AI4I): combined current-only + AMA alarm rule, recalibrated on the inner out-of-fold training
scores so that the union meets the same 1-per-100 false-alarm target (no test data used)."""
import numpy as np, pandas as pd
from sv_common import ai4i, load_pred, SEEDS, OUT
from pfi.schedules import ai4i_realistic, hidden_mask, visible_mask
ds = ai4i(); y = ds.y; rows = []; TG = 1.0
for s in SEEDS:
    sch = ai4i_realistic(ds, 100 + s); hid, vis = hidden_mask(ds, sch), visible_mask(ds, sch)
    pc, fold, ic = load_pred("ai4i_S1_current", s); pa, fold2, ia = load_pred("ai4i_S1_cur_last", s)
    assert (fold == fold2).all()
    al = {"current": np.zeros(len(y), bool), "AMA": np.zeros(len(y), bool), "combined, recalibrated": np.zeros(len(y), bool),
          "combined, not recalibrated": np.zeros(len(y), bool)}
    for k in range(5):
        (idx, c_in), (idx2, a_in) = ic[k], ia[k]; assert (idx == idx2).all()
        nm = y[idx] == 0; te = fold == k; rate = TG / 100
        q1 = lambda v, r: np.quantile(v, 1 - r, method="higher")
        tc, ta = q1(c_in[nm], rate), q1(a_in[nm], rate)
        al["current"][te] = pc[te] > tc; al["AMA"][te] = pa[te] > ta
        al["combined, not recalibrated"][te] = (pc[te] > tc) | (pa[te] > ta)
        for r in np.linspace(rate, rate / 3, 400):
            t1, t2 = q1(c_in[nm], r), q1(a_in[nm], r)
            if ((c_in[nm] > t1) | (a_in[nm] > t2)).mean() <= rate:
                break
        al["combined, recalibrated"][te] = (pc[te] > t1) | (pa[te] > t2)
    for k, a in al.items():
        rows.append(dict(seed=s, rule=k, FA_test_per100=100 * a[y == 0].mean(), recall_all=a[y == 1].mean(),
                         recall_hidden=a[hid].mean(), recall_visible=a[vis].mean(), precision=y[a].mean()))
d = pd.DataFrame(rows); d.to_csv(OUT / "p3_combined_ai4i.csv", index=False)
print(d.groupby("rule").mean().drop(columns="seed").round(3).to_string())
