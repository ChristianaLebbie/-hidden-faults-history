"""Statistics for Study 1 from saved predictions: paired machine-level bootstrap with Holm
correction, recall by fault type and control, and recall of hidden faults by staleness."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from pfi.data import load_ai4i, load_cmapss
from pfi.schedules import ai4i_realistic
from pfi.evaluate import bootstrap_diff, holm
from pfi.runner import PRED, RES

ai = load_ai4i(); y = ai.y; u = ai.unit
P = lambda tag: np.load(PRED / f"{tag}.npz")
out = {}

# ---- 1. method comparisons on AI4I S1 (5 seeds)
pairs = [("current", "locf"), ("current", "locf_ind"), ("locf", "locf_ind"), ("current", "cur_last"),
         ("locf_ind_age", "cur_last"), ("locf_ind_age", "trend_impute"), ("cur_last", "cur_last_age"),
         ("cur_last", "cur_lags"), ("cur_last", "cur_window"), ("cur_last", "cur_ewma"), ("cur_last", "gated")]
rows = []
for a, b in pairs:
    pa = [P(f"ai4i_S1_{a}_s{s}")["p"] for s in range(5)]; pb = [P(f"ai4i_S1_{b}_s{s}")["p"] for s in range(5)]
    r = bootstrap_diff(y, u, pa, pb, B=1000)
    from sklearn.metrics import average_precision_score as _ap
    wins = int(sum(_ap(y, q) > _ap(y, p_) for p_, q in zip(pa, pb)))
    rows.append(dict(baseline=a, compared=b, seeds_improved=wins, **r))
T = pd.DataFrame(rows); T["p_holm"] = holm(T.p); T.to_csv(RES / "stats_study1_pairs.csv", index=False); out["pairs"] = T.round(4).to_dict("records")

# ---- 2. recall by fault type x control (seed-pooled), current vs locf_ind vs cur_last
recs = []
for m in ["current", "locf", "locf_ind", "cur_last", "gated"]:
    for s in range(5):
        z = P(f"ai4i_S1_{m}_s{s}"); a = z["p"] >= z["t"]; sch = ai4i_realistic(ai, 100 + s)
        for i in np.where(y == 1)[0]:
            recs.append(dict(method=m, seed=s, mode=ai.meta["mode"][i], control=sch[i], caught=bool(a[i])))
R = pd.DataFrame(recs)
tab = R.groupby(["method", "mode", "control"]).caught.agg(["mean", "size"]).reset_index()
tab.to_csv(RES / "stats_study1_mode_control.csv", index=False)

# ---- 3. shuffled history: bootstrap vs no history, AI4I and FD001
sh = []
for mk, ds, seeds in [("ai4i", ai, range(5)), ("cmapss", load_cmapss("FD001"), range(3))]:
    for v in ["true", "random_past", "old_past", "unit_mean", "other_unit"]:
        f0 = [PRED / f"{mk}_shuffle_none_s{s}.npz" for s in seeds]; f1 = [PRED / f"{mk}_shuffle_{v}_s{s}.npz" for s in seeds]
        if all(f.exists() for f in f0 + f1):
            r = bootstrap_diff(ds.y, ds.unit, [np.load(f)["p"] for f in f0], [np.load(f)["p"] for f in f1], B=500)
            sh.append(dict(dataset=mk, history=v, **r))
if sh:
    S = pd.DataFrame(sh); S["p_holm"] = holm(S.p); S.to_csv(RES / "stats_shuffle.csv", index=False)

# ---- 4. hidden-fault recall by staleness of the missing causal sensor (outage runs, AI4I)
st = []
for L in [0, 2, 5, 10, 20]:
    for s in range(3):
        fa = PRED / f"ai4i_outage_L{L}_s{s}_hidden_age.npy"
        if not fa.exists():
            continue
        age = np.load(fa)
        for m in ["current", "cur_last", "gated", "locf_ind_age"]:
            fp = PRED / f"ai4i_outage_L{L}_{m}_s{s}.npz"
            if not fp.exists():
                continue
            z = np.load(fp); a = z["p"] >= z["t"]
            for i in np.where(~np.isnan(age))[0]:
                st.append(dict(L=L, seed=s, method=m, age=age[i], caught=bool(a[i])))
if st:
    D = pd.DataFrame(st)
    D["age_bin"] = pd.cut(D.age, [0, 1, 2, 4, 8, 16, 1000], labels=["1", "2", "3-4", "5-8", "9-16", ">16"])
    D.groupby(["method", "age_bin"], observed=True).caught.agg(["mean", "size"]).reset_index().to_csv(RES / "stats_hidden_by_age.csv", index=False)

# ---- 5. MetroPT-3: day-block bootstrap of the history gain within each forward test period
from sklearn.metrics import average_precision_score as _ap
from pfi.data import load_metropt
mp = load_metropt(); st = mp.meta["start"]; Fl = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in mp.meta["failures"]]
mids = [Fl[k][1] + (Fl[k + 1][0] - Fl[k][1]) / 2 for k in range(3)]
blocks = {"F2": (mids[0], mids[1]), "F3": (mids[1], mids[2]), "F4": (mids[2], st.max() + pd.Timedelta("1h"))}
mrows = []
for v in ["true", "random_past", "older_1h", "older_1day", "unit_mean"]:
    f0 = [PRED / f"metropt_shuffle_none_s{s}.npy" for s in range(3)]; f1 = [PRED / f"metropt_shuffle_{v}_s{s}.npy" for s in range(3)]
    if not all(f.exists() for f in f0 + f1):
        continue
    A = [np.load(f) for f in f0]; B_ = [np.load(f) for f in f1]; rng = np.random.default_rng(0)
    for ev, (a, b) in blocks.items():
        te = np.where((st >= a) & (st < b))[0]; yt = mp.y[te]; day = st.iloc[te].dt.floor("D").to_numpy()
        obs = np.mean([_ap(yt, q[te]) - _ap(yt, p_[te]) for p_, q in zip(A, B_)])
        days = np.unique(day); idx = {d_: np.where(day == d_)[0] for d_ in days}; D = []
        for _ in range(1000):
            ii = np.concatenate([idx[d_] for d_ in rng.choice(days, len(days))])
            if yt[ii].sum() == 0:
                continue
            D.append(np.mean([_ap(yt[ii], q[te][ii]) - _ap(yt[ii], p_[te][ii]) for p_, q in zip(A, B_)]))
        D = np.array(D)
        mrows.append(dict(history=v, event=ev, diff=obs, lo=np.percentile(D, 2.5), hi=np.percentile(D, 97.5),
                          p=max(2 * min((D <= 0).mean(), (D >= 0).mean()), 1 / len(D))))
if mrows:
    M = pd.DataFrame(mrows); M["p_holm"] = holm(M.p); M.to_csv(RES / "stats_metropt_history.csv", index=False); print(M.round(3).to_string())

json.dump(out, open(RES / "stats_study1.json", "w"), indent=1, default=float)
print(T.round(3).to_string())
print(tab[tab.method.isin(["current", "locf_ind", "cur_last"])].pivot_table(index=["mode", "control"], columns="method", values="mean").round(2).to_string())
