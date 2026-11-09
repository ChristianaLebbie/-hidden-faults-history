"""Acquisition schedules: which sensor group each inspection reports.

A schedule is an array of group names, one per record. `apply_schedule` turns the
complete values into the observations available at each inspection.
Outages remove additional readings for runs of consecutive inspections.
"""
import numpy as np
import pandas as pd


def apply_schedule(X: pd.DataFrame, schedule: np.ndarray, groups: dict) -> pd.DataFrame:
    """Keep only the sensors reported by each record's group; everything else becomes NaN."""
    out = X.copy()
    for g, keep in groups.items():
        drop = [c for c in X.columns if c not in keep]
        out.loc[schedule == g, drop] = np.nan
    return out


# ----------------------------------------------------------------- AI4I schedules
def ai4i_published(ds):
    return ds.meta["published_control"].copy()


def ai4i_realistic(ds, seed):
    """Published controls for healthy records; fault records get a control drawn uniformly at random.
    In the published benchmark every deterministic fault sits under its revealing control."""
    r = np.random.default_rng(seed); s = ds.meta["published_control"].copy(); f = ds.y == 1
    s[f] = r.choice(list(ds.groups), f.sum()); return s


def random_mix(ds, seed, probs=None):
    """Every record draws its group independently (probs over groups, default uniform)."""
    g = list(ds.groups); probs = probs or [1 / len(g)] * len(g)
    return np.random.default_rng(seed).choice(g, len(ds.y), p=probs)


def round_robin(ds, seed):
    """Periodic schedule per unit (A, B, C, A, B, C, ...) with a random phase per unit, in time order."""
    g = np.array(list(ds.groups)); r = np.random.default_rng(seed); s = np.empty(len(ds.y), dtype=object)
    for u in np.unique(ds.unit):
        idx = np.where(ds.unit == u)[0]; idx = idx[np.lexsort((idx, ds.order[idx]))]
        s[idx] = g[(np.arange(len(idx)) + r.integers(len(g))) % len(g)]
    return s.astype(str)


def hidden_mask(ds, schedule):
    """AI4I only: deterministic faults recorded under a control that misses at least one causal sensor."""
    rev = ds.meta["revealing"]; m = ds.meta["mode"]
    return np.array([(mm in rev) and (rev[mm] != c) for mm, c in zip(m, schedule)])


def visible_mask(ds, schedule):
    rev = ds.meta["revealing"]; m = ds.meta["mode"]
    return np.array([(mm in rev) and (rev[mm] == c) for mm, c in zip(m, schedule)])


# ------------------------------------------------------------------------ outages
def add_outages(Xobs: pd.DataFrame, ds, mean_len: float, rate: float, seed: int) -> pd.DataFrame:
    """Remove each sensor for runs of consecutive inspections of a unit.

    Outage runs have geometric length with mean `mean_len`; starts are placed so that about
    `rate` of inspections are inside an outage. Runs are drawn per unit and per sensor, in time order.
    """
    if mean_len <= 0 or rate <= 0:
        return Xobs
    r = np.random.default_rng(seed); out = Xobs.copy()
    p_start = rate / (mean_len * (1 - rate) + rate)  # stationary fraction of time in outage = rate
    for u in np.unique(ds.unit):
        idx = np.where(ds.unit == u)[0]; idx = idx[np.lexsort((idx, ds.order[idx]))]
        for c in out.columns:
            off = np.zeros(len(idx), bool); i = 0
            while i < len(idx):
                if r.random() < p_start:
                    L = r.geometric(1 / mean_len); off[i:i + L] = True; i += L
                else:
                    i += 1
            col = out.columns.get_loc(c); out.iloc[idx[off], col] = np.nan
    return out
