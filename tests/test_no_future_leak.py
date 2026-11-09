"""Checks that history features use only strictly earlier observations of the same unit.

1. Brute-force check: features recomputed record by record from an explicit list of
   earlier observations match the vectorised implementation.
2. Perturbation check: changing every observation at the same time or later (and every
   other unit's data) leaves a record's features unchanged.
Run: python tests/test_no_future_leak.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pfi.data import load_ai4i, load_cmapss
from pfi.schedules import ai4i_realistic, random_mix, apply_schedule, add_outages
from pfi.history import past_features, STATS


def brute(Xobs, unit, order, i, k=5, alpha=0.5):
    same = np.where((unit == unit[i]) & (order < order[i]))[0]
    same = same[np.lexsort((same, order[same]))]
    nb = len(same); res = {}
    for c in Xobs.columns:
        v = Xobs[c].to_numpy()[same]; pos = np.where(~np.isnan(v))[0]; ov = v[pos]
        if len(ov) == 0:
            res[c] = None; continue
        # position of each earlier observation among all inspections of the unit
        allidx = np.where(unit == unit[i])[0]; allidx = allidx[np.lexsort((allidx, order[allidx]))]
        rank = {rec: k_ for k_, rec in enumerate(allidx)}
        p_all = np.array([rank[rec] for rec in same[pos]])
        w = ov[-k:]; x = p_all[-k:].astype(float)
        slope = np.polyfit(x, w, 1)[0] if len(w) >= 2 and np.ptp(x) > 0 else np.nan
        e = pd.Series(ov).ewm(alpha=alpha).mean().iloc[-1]
        nb_i = np.searchsorted(order[allidx], order[i], side="left")
        res[c] = dict(last=ov[-1], stale=nb_i - p_all[-1], mean=w.mean(), slope=slope, ewma=e,
                      lag2=ov[-2] if len(ov) > 1 else np.nan, nobs=len(ov))
    return res


def check(ds, Xobs, n_check=300, seed=0):
    stats = ("last", "stale", "mean", "slope", "ewma", "lag2", "nobs", "extrap")
    H = past_features(Xobs, ds.unit, ds.order, stats=stats)
    r = np.random.default_rng(seed)
    for i in r.choice(len(Xobs), n_check, replace=False):
        b = brute(Xobs, ds.unit, ds.order, i)
        for c in Xobs.columns:
            if b[c] is None:
                assert np.isnan(H.loc[i, f"last_{c}"]); continue
            for s in ("last", "stale", "mean", "ewma", "lag2", "nobs"):
                assert np.isclose(H.loc[i, f"{s}_{c}"], b[c][s], equal_nan=True), (ds.name, i, c, s)
            assert np.isclose(H.loc[i, f"slope_{c}"], b[c]["slope"], equal_nan=True, atol=1e-6), (ds.name, i, c)
    # perturbation: scramble present + future of the same unit and everything of other units
    for i in r.choice(len(Xobs), 40, replace=False):
        P = Xobs.copy(); mask = (ds.unit != ds.unit[i]) | (ds.order >= ds.order[i])
        P.loc[mask] = P.loc[mask].to_numpy() * 1.37 + 5.0
        H2 = past_features(P, ds.unit, ds.order, stats=stats)
        assert np.allclose(H.loc[i].to_numpy(dtype=float), H2.loc[i].to_numpy(dtype=float), equal_nan=True), (ds.name, i)
    print(f"{ds.name}: brute-force and no-future-leak checks passed")


if __name__ == "__main__":
    a = load_ai4i()
    check(a, apply_schedule(a.X, ai4i_realistic(a, 100), a.groups))
    check(a, add_outages(apply_schedule(a.X, ai4i_realistic(a, 101), a.groups), a, 5, 0.3, 7))
    c = load_cmapss("FD002")
    check(c, apply_schedule(c.X, random_mix(c, 3), c.groups), n_check=150)
    print("all checks passed")
