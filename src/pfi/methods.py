"""Feature sets compared in Study 1.

Each method maps (observed data, strictly-past history, training indices) to model inputs.
Only the reliability gate and the condition normalisation use training statistics,
and they use training records only.
"""
import numpy as np
import pandas as pd
from .history import past_features, locf_impute

ALL_STATS = ("last", "stale", "lag2", "lag3", "mean", "slope", "ewma", "extrap")


def _cols(H, stat, sensors):
    return H[[f"{stat}_{c}" for c in sensors]]


def _stale_if_missing(Xobs, H):
    """Inspections since the value was observed: 0 if observed now, else staleness of the carried value."""
    S = pd.DataFrame(index=Xobs.index)
    for c in Xobs.columns:
        S[f"age_{c}"] = np.where(Xobs[c].notna(), 0.0, H[f"stale_{c}"])
    return S


def _ind(Xobs):
    return Xobs.isna().astype(float).add_prefix("miss_")


# ----------------------------------------------------------------- reliability gate
BUCKETS = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 20), (21, 10**9)]


def reliability_table(Xobs, H, rows):
    """R^2 of the last earlier observation as an estimate of the current value, by staleness bucket.
    Uses only `rows` (training records) where the sensor is observed now and has history."""
    out = []
    for c in Xobs.columns:
        x = Xobs[c].to_numpy()[rows]; h = H[f"last_{c}"].to_numpy()[rows]; s = H[f"stale_{c}"].to_numpy()[rows]
        ok = ~np.isnan(x) & ~np.isnan(h); var = np.nanvar(x[~np.isnan(x)])
        for lo, hi in BUCKETS:
            m = ok & (s >= lo) & (s <= hi)
            r2 = 1 - np.mean((x[m] - h[m]) ** 2) / var if m.sum() >= 30 else np.nan
            out.append(dict(sensor=c, lo=lo, hi=hi, n=int(m.sum()), R2=r2))
    return pd.DataFrame(out)


def horizons(table, tau):
    """Largest staleness up to which every estimable bucket has R^2 >= tau (0 = history unreliable).
    Buckets with too few training pairs to estimate R^2 are skipped rather than treated as unreliable
    (e.g. under a periodic schedule a sensor may never be one inspection old)."""
    h = {}
    for c, t in table.groupby("sensor", sort=False):
        H_ = 0
        for _, r in t.sort_values("lo").iterrows():
            if np.isnan(r.R2):
                continue
            if r.R2 >= tau:
                H_ = r.hi
            else:
                break
        h[c] = H_
    return h


def gate(H, Xobs, hz):
    G = pd.DataFrame(index=H.index)
    for c in Xobs.columns:
        keep = H[f"stale_{c}"].to_numpy() <= hz[c]
        G[f"last_{c}"] = np.where(keep, H[f"last_{c}"], np.nan)
        G[f"stale_{c}"] = np.where(keep, H[f"stale_{c}"], np.nan)
    return G


# ---------------------------------------------------------------- method registry
def build_features(method, Xobs, H, static, tr=None, tau=0.5, hz=None):
    """hz: reliability horizons learned at training time; if None they are estimated from (Xobs, H) on rows tr."""
    s = list(Xobs.columns)
    if method == "current":
        F = [Xobs]
    elif method == "locf":
        F = [locf_impute(Xobs, H, "last")]
    elif method == "locf_ind":
        F = [locf_impute(Xobs, H, "last"), _ind(Xobs)]
    elif method == "locf_ind_age":
        F = [locf_impute(Xobs, H, "last"), _ind(Xobs), _stale_if_missing(Xobs, H)]
    elif method == "cur_last":
        F = [Xobs, _cols(H, "last", s)]
    elif method in ("cur_last_age", "ama"):
        # AMA (Age- and Missingness-Aware history): current readings (missing left missing, so the model
        # knows what was not measured) + last earlier reading of every sensor + its age in inspections.
        # With several operating conditions, Xobs/H are passed already normalised within condition.
        F = [Xobs, _cols(H, "last", s), _cols(H, "stale", s)]
    elif method == "cur_lags":
        F = [Xobs, _cols(H, "last", s), _cols(H, "lag2", s), _cols(H, "lag3", s)]
    elif method == "cur_window":
        F = [Xobs, _cols(H, "last", s), _cols(H, "mean", s), _cols(H, "slope", s)]
    elif method == "cur_ewma":
        F = [Xobs, _cols(H, "ewma", s)]
    elif method == "trend_impute":
        F = [locf_impute(Xobs, H, "extrap"), _ind(Xobs), _stale_if_missing(Xobs, H)]
    elif method == "gated":
        hz = hz or horizons(reliability_table(Xobs, H, tr), tau)
        F = [Xobs, gate(H, Xobs, hz)]
    elif method == "gated_window":
        hz = hz or horizons(reliability_table(Xobs, H, tr), tau)
        G = gate(H, Xobs, hz); W = pd.DataFrame(index=H.index)
        for c in s:
            keep = hz[c] > 0
            W[f"mean_{c}"] = H[f"mean_{c}"] if keep else np.nan
            W[f"slope_{c}"] = H[f"slope_{c}"] if keep else np.nan
        F = [Xobs, G, W]
    else:
        raise ValueError(method)
    return pd.concat(F + [static], axis=1)


METHOD_LABELS = {
    "current": "Current readings only",
    "locf": "LOCF imputation",
    "locf_ind": "LOCF + missingness indicators",
    "locf_ind_age": "LOCF + indicators + age (GRU-D-style inputs)",
    "cur_last": "Current + last observed value",
    "cur_last_age": "Current + last value + age",
    "cur_lags": "Current + last three observed values (window)",
    "cur_window": "Current + last value + window mean + trend slope",
    "cur_ewma": "Current + exponentially weighted history",
    "trend_impute": "Trend-extrapolation imputation + indicators + age",
    "gated": "Reliability-gated history",
    "gated_window": "Reliability-gated history + window trend",
}


# ------------------------------------------------------------ condition handling
def condition_normalise(Xobs, cond, tr):
    """z-score each sensor within each operating condition using training records only."""
    Z = Xobs.astype(float).copy()
    for k in np.unique(cond):
        m_all = cond == k; m_tr = np.zeros(len(cond), bool); m_tr[tr] = True; m_tr &= m_all
        Xf = Xobs.astype(float); mu = Xf[m_tr].mean(); sd = Xf[m_tr].std().replace(0, 1)
        Z.loc[m_all] = (Xf[m_all] - mu) / sd
    return Z
