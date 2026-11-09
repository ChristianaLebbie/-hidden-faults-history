"""Strictly-past history features.

For a record of unit u at time t, every feature is computed only from observations of
the same unit taken at times strictly earlier than t. Records that share a timestamp are
treated as simultaneous, so they never see each other. Labels are never an input.
`tests/test_no_future_leak.py` checks this by perturbing all present and future data.
"""
import numpy as np
import pandas as pd

STATS = ("last", "stale", "lag2", "lag3", "mean", "slope", "ewma", "extrap", "nobs")


def past_features(Xobs: pd.DataFrame, unit, order, k: int = 5, alpha: float = 0.5,
                  stats=("last", "stale")) -> pd.DataFrame:
    """Return a DataFrame with columns f"{stat}_{sensor}" for each requested stat.

    last   most recent earlier observation of the sensor (LOCF value)
    stale  inspections between that observation and now (1 = previous inspection)
    lag2/3 second and third most recent earlier observations
    mean   mean of the last k earlier observations
    slope  least-squares slope of the last k earlier observations per inspection step
    ewma   exponentially weighted mean of earlier observations (alpha)
    extrap last + slope * stale (linear trend extrapolated to now)
    nobs   number of earlier observations
    """
    unit = np.asarray(unit); order = np.asarray(order); n, d = Xobs.shape
    V = Xobs.to_numpy(dtype=float)
    out = {s: np.full((n, d), np.nan) for s in stats}
    for u in np.unique(unit):
        idx = np.where(unit == u)[0]
        idx = idx[np.lexsort((idx, order[idx]))]
        t = order[idx]
        nb = np.searchsorted(t, t, side="left")          # inspections strictly earlier than each record
        for j in range(d):
            v = V[idx, j]; op = np.where(~np.isnan(v))[0]
            if len(op) == 0:
                continue
            ot, ov = t[op], v[op]
            cnt = np.searchsorted(ot, t, side="left")      # observations strictly earlier in time
            has = cnt > 0; q = cnt - 1; r = np.where(has)[0]; qq = q[has]
            if "last" in out or "extrap" in out:
                last = ov[qq]
            if "last" in out:
                out["last"][idx[r], j] = last
            if "stale" in out or "extrap" in out:
                stale = nb[r] - op[qq]
            if "stale" in out:
                out["stale"][idx[r], j] = stale
            if "nobs" in out:
                out["nobs"][idx, j] = cnt
            for lag, name in ((1, "lag2"), (2, "lag3")):
                if name in out:
                    ok = qq - lag >= 0; out[name][idx[r[ok]], j] = ov[qq[ok] - lag]
            if any(s in out for s in ("mean", "slope", "extrap")):
                x = op.astype(float); w0 = np.maximum(0, qq - k + 1); m = (qq - w0 + 1).astype(float)
                C = lambda a: np.concatenate([[0.0], np.cumsum(a)])
                Sy, Sx, Sxx, Sxy = C(ov), C(x), C(x * x), C(x * ov)
                sy, sx = Sy[qq + 1] - Sy[w0], Sx[qq + 1] - Sx[w0]
                sxx, sxy = Sxx[qq + 1] - Sxx[w0], Sxy[qq + 1] - Sxy[w0]
                if "mean" in out:
                    out["mean"][idx[r], j] = sy / m
                if "slope" in out or "extrap" in out:
                    den = m * sxx - sx * sx
                    slope = np.where((m >= 2) & (den > 0), (m * sxy - sx * sy) / np.where(den > 0, den, 1), np.nan)
                    if "slope" in out:
                        out["slope"][idx[r], j] = slope
                    if "extrap" in out:
                        out["extrap"][idx[r], j] = last + np.nan_to_num(slope) * stale
            if "ewma" in out:
                e = pd.Series(ov).ewm(alpha=alpha, adjust=True).mean().to_numpy()
                out["ewma"][idx[r], j] = e[qq]
    cols = {}
    for s in stats:
        for j, c in enumerate(Xobs.columns):
            cols[f"{s}_{c}"] = out[s][:, j]
    return pd.DataFrame(cols, index=Xobs.index)


def locf_impute(Xobs: pd.DataFrame, H: pd.DataFrame, source: str = "last") -> pd.DataFrame:
    """Fill each missing current value with a history estimate (LOCF when source='last')."""
    out = Xobs.copy()
    for c in Xobs.columns:
        out[c] = Xobs[c].fillna(H[f"{source}_{c}"])
    return out


# --------------------------------------------- diagnostics: degraded / shuffled history
def alt_history(Xobs, unit, order, kind, seed, L=10):
    """Strictly-past but degraded history: random_past, old_past (>= L inspections old), unit_mean."""
    r = np.random.default_rng(seed); V = Xobs.to_numpy(float); out = np.full(V.shape, np.nan)
    units = np.unique(unit)
    for u in units:
        idx = np.where(unit == u)[0]; idx = idx[np.lexsort((idx, order[idx]))]; t = order[idx]
        nb = np.searchsorted(t, t, "left")
        for j in range(V.shape[1]):
            v = V[idx, j]; op = np.where(~np.isnan(v))[0]
            if not len(op):
                continue
            ov = v[op]; cnt = np.searchsorted(t[op], t, "left"); has = cnt > 0
            if kind == "random_past":
                k_ = np.floor(r.random(len(cnt)) * np.maximum(cnt, 1)).astype(int)
                out[idx[has], j] = ov[k_[has]]
            elif kind == "old_past":
                c2 = np.minimum(np.searchsorted(op, nb - L, "right"), cnt); ok = c2 > 0
                out[idx[ok], j] = ov[c2[ok] - 1]
            elif kind == "unit_mean":
                cm = np.cumsum(ov) / np.arange(1, len(ov) + 1)
                out[idx[has], j] = cm[cnt[has] - 1]
    return pd.DataFrame(out, columns=[f"last_{c}" for c in Xobs.columns], index=Xobs.index)


def other_unit(Htrue, unit, order, seed):
    """History of a different unit at the same position in its life (breaks unit identity)."""
    r = np.random.default_rng(seed); units = np.unique(unit); perm = units.copy()
    while np.any(perm == units):
        perm = r.permutation(units)
    out = Htrue.copy() * np.nan
    pos = {}
    for u in units:
        idx = np.where(unit == u)[0]; pos[u] = idx[np.lexsort((idx, order[idx]))]
    for u, v in zip(units, perm):
        a, b = pos[u], pos[v]; k = np.minimum(np.arange(len(a)), len(b) - 1)
        out.iloc[a] = Htrue.iloc[b[k]].to_numpy()
    return out
