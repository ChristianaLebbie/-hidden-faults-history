"""Study 2, Q2-Q3: referral strategies under the same review budget, and explicit reviewer accuracy.

System: history-informed classifier ('current + last value'), Platt-calibrated on inner
out-of-fold predictions; alarm threshold maximises F1 on the calibrated inner predictions.
For each outer test fold, the top b% of records by each strategy's score are referred.

Strategies (scores computed without labels):
  random          random order (control)
  ambiguity       closeness of the calibrated probability to the alarm threshold (logit scale)
  below_threshold highest probability among records the system would clear automatically
  expected_cost   expected cost of the automatic decision (missed fault cost 10, false alarm 1)
  evidence_spread spread of the predicted probability when the sensors not measured at this
                  inspection are filled with plausible values (counterfactual completions)
  upper_risk      among records cleared automatically, the 90th percentile of the probability
                  over plausible completions ("could this be a fault if the missing sensors were read?")
Plausible values: for a sensor whose history is reliable (training-fold reliability gate), the
last earlier reading plus a residual drawn from training records with similar staleness; otherwise
a draw from the training-fold distribution of that sensor.

Reviewer models (referral does not mean the fault is identified):
  perfect         every referred case resolved correctly (upper bound)
  uniform_90_95   sensitivity 0.90, specificity 0.95
  uniform_70_95   sensitivity 0.70, specificity 0.95
  same_evidence   reviewer sees the same partial evidence: sensitivity 0.9 on visible faults,
                  0.5 on hidden faults (AI4I); 0.7 on all faults (C-MAPSS); specificity 0.95
  reinspection    referred unit is fully re-inspected (all sensors) and judged by a model trained on
                  complete data in the same fold (data-driven, imperfect)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from pfi.data import load_ai4i, load_cmapss
from pfi.schedules import ai4i_realistic, random_mix, apply_schedule, hidden_mask, visible_mask
from pfi.history import past_features
from pfi.methods import build_features, reliability_table, horizons, condition_normalise, ALL_STATS, BUCKETS
from pfi.evaluate import fit_predict, f1_threshold
from pfi.calibrate import Platt
from pfi.runner import append, log, RES

BUDGETS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20]
M_DRAWS = 20


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def bucket(s):
    b = np.full(len(s), len(BUCKETS) - 1)
    for k, (lo, hi) in enumerate(BUCKETS):
        b[(s >= lo) & (s <= hi)] = k
    return b


def completions(model, F, X, H, tr, te, rng):
    """Probabilities for M plausible completions of the sensors not measured at each test inspection."""
    hz = horizons(reliability_table(X, H, tr), 0.5)
    draws = np.zeros((M_DRAWS, len(te)))
    pools = {}
    for c in X.columns:
        xo = X[c].to_numpy()[tr]; ho = H[f"last_{c}"].to_numpy()[tr]; so = H[f"stale_{c}"].to_numpy()[tr]
        ok = ~np.isnan(xo) & ~np.isnan(ho); bo = bucket(np.nan_to_num(so, nan=10**9))
        pools[c] = dict(marg=xo[~np.isnan(xo)], res={k: (xo - ho)[ok & (bo == k)] for k in range(len(BUCKETS))})
    Fte = F.iloc[te]
    for m in range(M_DRAWS):
        G = Fte.copy()
        for c in X.columns:
            miss = np.isnan(X[c].to_numpy()[te])
            if not miss.any():
                continue
            h = H[f"last_{c}"].to_numpy()[te]; s = H[f"stale_{c}"].to_numpy()[te]
            val = rng.choice(pools[c]["marg"], len(te))
            usable = miss & ~np.isnan(h) & (np.nan_to_num(s, nan=10**9) <= hz[c])
            if usable.any():
                bk = bucket(s[usable]); add = np.empty(usable.sum())
                for k in np.unique(bk):
                    pool = pools[c]["res"][k]; pool = pool if len(pool) else np.array([0.0])
                    add[bk == k] = rng.choice(pool, (bk == k).sum())
                val[usable] = h[usable] + add
            col = G[c].to_numpy().copy(); col[miss] = val[miss]; G[c] = col
        draws[m] = model.predict_proba(G)[:, 1]
    return draws


def run_dataset(name, ds, prep, seed):
    """prep(tr) -> (X, H, F, Fcomplete) built with training-fold statistics only."""
    y, g = ds.y, ds.unit; n = len(y); rng = np.random.default_rng(seed)
    out = dict(p=np.zeros(n), t=np.zeros(n), spread=np.zeros(n), upper=np.zeros(n), fold=np.zeros(n, int),
               rev_p=np.zeros(n), rev_t=np.zeros(n))
    for f, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(np.zeros(n), y, g)):
        X, H, F, Fcomplete = prep(tr)
        p_te, model = fit_predict(F.iloc[tr], y[tr], F.iloc[te], seed)
        ip = np.full(len(tr), np.nan); rp = np.full(len(tr), np.nan)
        for a, b in StratifiedGroupKFold(4, shuffle=True, random_state=seed + 1).split(tr, y[tr], g[tr]):
            ip[b], _ = fit_predict(F.iloc[tr[a]], y[tr[a]], F.iloc[tr[b]], seed)
            rp[b], _ = fit_predict(Fcomplete.iloc[tr[a]], y[tr[a]], Fcomplete.iloc[tr[b]], seed)
        cal = Platt().fit(ip, y[tr])
        out["p"][te] = cal(p_te); out["t"][te] = f1_threshold(y[tr], cal(ip)); out["fold"][te] = f
        D = cal(completions(model, F, X, H, tr, te, rng).ravel()).reshape(M_DRAWS, -1)
        out["spread"][te] = D.std(0); out["upper"][te] = np.quantile(D, 0.9, axis=0)
        rp_te, _ = fit_predict(Fcomplete.iloc[tr], y[tr], Fcomplete.iloc[te], seed)
        out["rev_p"][te] = rp_te; out["rev_t"][te] = f1_threshold(y[tr], rp)
    return out


def evaluate(name, seed, o, y, masks, hidden=None, visible=None):
    rows = []; rng = np.random.default_rng(seed + 99)
    p, t = o["p"], o["t"]; alarm = p >= t
    scores = {"random": rng.random(len(y)), "ambiguity": -np.abs(logit(p) - logit(t)),
              "below_threshold": np.where(alarm, -np.inf, p),
              "expected_cost": np.where(alarm, (1 - p) * 1.0, p * 10.0),
              "evidence_spread": o["spread"], "upper_risk": np.where(alarm, -np.inf, o["upper"])}
    rev_alarm = o["rev_p"] >= o["rev_t"]
    sens_same = np.where(hidden if hidden is not None else np.zeros(len(y), bool), 0.5, 0.9 if hidden is not None else 0.7)
    reviewers = {"perfect": (np.ones(len(y)), np.zeros(len(y))),
                 "uniform_90_95": (np.full(len(y), 0.9), np.full(len(y), 0.05)),
                 "uniform_70_95": (np.full(len(y), 0.7), np.full(len(y), 0.05)),
                 "same_evidence": (sens_same, np.full(len(y), 0.05)),
                 "reinspection": (rev_alarm.astype(float), rev_alarm.astype(float))}
    F_, H_ = y == 1, y == 0
    for b in BUDGETS:
        for sname, sc in scores.items():
            ref = np.zeros(len(y), bool)
            for f in np.unique(o["fold"]):
                idx = np.where(o["fold"] == f)[0]; k = int(round(b * len(idx)))
                if k:
                    ref[idx[np.argsort(-sc[idx], kind="stable")[:k]]] = True
            auto_tp = (alarm & ~ref & F_).sum(); auto_fp = (alarm & ~ref & H_).sum()
            base = dict(dataset=name, seed=seed, budget=b, strategy=sname, referred=ref.mean(),
                        faults_referred=(ref & F_).sum() / F_.sum(), referral_yield=(ref & F_).sum() / max(ref.sum(), 1),
                        silent_miss=(~alarm & ~ref & F_).sum() / F_.sum())
            if hidden is not None:
                base.update(hidden_referred=(ref & hidden).sum() / hidden.sum(), hidden_silent_miss=(~alarm & ~ref & hidden).sum() / hidden.sum(),
                            visible_referred=(ref & visible).sum() / visible.sum(), healthy_referred=(ref & H_).sum() / H_.sum())
            for rname, (sens, fpr) in reviewers.items():
                caught = auto_tp + (sens * (ref & F_)).sum(); fa = auto_fp + (fpr * (ref & H_)).sum()
                rows.append(dict(base, reviewer=rname, system_recall=caught / F_.sum(), system_FA_per100=100 * fa / H_.sum(),
                                 system_precision=caught / max(caught + fa, 1e-9)))
    return rows


rows = []
ai = load_ai4i()
for seed in [0, 1, 2]:
    sch = ai4i_realistic(ai, 100 + seed); X = apply_schedule(ai.X, sch, ai.groups); H = past_features(X, ai.unit, ai.order, stats=ALL_STATS)
    F = build_features("cur_last", X, H, ai.static); Fc = pd.concat([ai.X, ai.static], axis=1)
    o = run_dataset("AI4I-PMDI", ai, lambda tr: (X, H, F, Fc), seed)
    np.savez_compressed(RES / "predictions" / f"referral_ai4i_s{seed}.npz", **o)
    rows += evaluate("AI4I-PMDI", seed, o, ai.y, None, hidden_mask(ai, sch), visible_mask(ai, sch)); log("AI4I", seed)
    append(rows, "exp7_referral.csv")
for fd in ["FD001", "FD002"]:
    ds = load_cmapss(fd); cond = ds.static["cond"].to_numpy()
    for seed in ([0, 1, 2] if fd == "FD001" else [0, 1]):
        sch = random_mix(ds, 100 + seed); X0 = apply_schedule(ds.X, sch, ds.groups)

        def prep(tr, X0=X0, ds=ds, cond=cond):
            # multi-condition data: z-score within each condition using this fold's training units only
            X = condition_normalise(X0, cond, tr) if ds.meta["n_conditions"] > 1 else X0
            Xc = condition_normalise(ds.X, cond, tr) if ds.meta["n_conditions"] > 1 else ds.X
            H = past_features(X, ds.unit, ds.order, stats=ALL_STATS)
            return X, H, build_features("cur_last", X, H, ds.static), pd.concat([Xc, ds.static], axis=1)
        o = run_dataset(f"C-MAPSS {fd}", ds, prep, seed)
        np.savez_compressed(RES / "predictions" / f"referral_{fd}_s{seed}.npz", **o)
        rows += evaluate(f"C-MAPSS {fd}", seed, o, ds.y, None); log(fd, seed)
        append(rows, "exp7_referral.csv")
log("done")
