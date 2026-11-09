"""Dataset loading.

Every loader returns a `Dataset` with the complete sensor values, the fault label,
the unit (machine/engine) identifier, a time-order key and any metadata needed
for masking. Missingness is applied later by `schedules.py`, so the same complete
values serve as ground truth for every scenario.
"""
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data"


@dataclass
class Dataset:
    name: str
    X: pd.DataFrame            # complete sensor values (n x d)
    y: np.ndarray              # binary label (fault / final degradation stage)
    unit: np.ndarray           # machine or engine id (grouping for CV and history)
    order: np.ndarray          # sortable time key within unit
    static: pd.DataFrame       # always-available covariates (e.g. product type)
    groups: dict               # observation groups: name -> list of sensors
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- AI4I
AI4I_SENSORS = ["air", "proc", "rpm", "torque", "wear"]
AI4I_GROUPS = {"A": ["air", "proc", "rpm"], "B": ["rpm", "torque"], "C": ["torque", "wear"]}
# the one control whose sensors cover all causal variables of each deterministic fault type
AI4I_REVEALING = {"HDF": "A", "PWF": "B", "OSF": "C", "TWF": "C"}
_MODE = {"Heat Dissipation Failure": "HDF", "Power Failure": "PWF", "Overstrain Failure": "OSF",
         "Tool Wear Failure": "TWF", "Random Failures": "RNF", "No failure": "none"}


def load_ai4i() -> Dataset:
    a = pd.read_csv(DATA / "ai4i2020.csv")
    p = pd.read_csv(DATA / "AI4I-PMDI.csv")
    assert (a["UDI"].values == p["UDI"].values).all() and (a["Product ID"].values == p["Product ID"].values).all()
    cols = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
    X = a[cols].copy(); X.columns = AI4I_SENSORS
    # every value observed in AI4I-PMDI must equal the AI4I 2020 value
    pcols = ["Air temperature (K)", "Process temperature (K)", "Rotational speed (rpm)", "Torque (Nm)", "Tool wear (min)"]
    for s, c in zip(AI4I_SENSORS, pcols):
        m = p[c].notna(); assert np.allclose(p.loc[m, c].values, X.loc[m, s].values)
    t = pd.to_datetime(p["Date"], dayfirst=True)
    # order key: timestamp, ties broken by record id (ties are then treated as simultaneous in history.py)
    order = t.values.astype("datetime64[s]").astype(np.int64)
    static = pd.DataFrame({"type": a["Type"].map({"L": 0, "M": 1, "H": 2}).values})
    return Dataset(
        name="AI4I-PMDI", X=X, y=(p["Diagnostic"] != "No failure").astype(int).values,
        unit=p["System"].values, order=order, static=static, groups=AI4I_GROUPS,
        meta=dict(mode=p["Diagnostic"].map(_MODE).values, published_control=p["Control"].values,
                  record_id=p["UDI"].values, revealing=AI4I_REVEALING))


# ------------------------------------------------------------------------ C-MAPSS
CMAPSS_SENSORS = ["m2", "m3", "m4", "m7", "m8", "m9", "m11", "m12", "m13", "m14", "m15", "m17", "m20", "m21"]
CMAPSS_GROUPS = {"A": ["m2", "m3", "m4", "m7", "m11"],          # temperatures and pressures
                 "B": ["m8", "m9", "m13", "m14"],               # shaft speeds
                 "C": ["m12", "m15", "m17", "m20", "m21"]}      # fuel flow ratio, bypass ratio, bleed, coolant
RUL_THRESHOLD = 30


def load_cmapss(fd: str = "FD001") -> Dataset:
    cols = ["unit", "cycle", "set1", "set2", "set3"] + [f"m{i}" for i in range(1, 22)]
    d = pd.read_csv(DATA / "cmapss" / f"train_{fd}.txt", sep=r"\s+", header=None, names=cols)
    rul = d.groupby("unit")["cycle"].transform("max") - d["cycle"]
    # operating condition: the six regimes of FD002/FD004 are exact clusters of the three settings
    key = pd.DataFrame({"a": d["set1"].round(0) + 0.0, "b": d["set2"].round(2) + 0.0, "c": d["set3"].round(0) + 0.0})  # +0.0 removes negative zero
    cond = key.astype(str).agg("|".join, axis=1)
    codes, uniq = pd.factorize(cond, sort=True)
    static = pd.DataFrame({"cond": codes})
    return Dataset(
        name=f"C-MAPSS {fd}", X=d[CMAPSS_SENSORS].reset_index(drop=True), y=(rul <= RUL_THRESHOLD).astype(int).values,
        unit=d["unit"].values, order=d["cycle"].values.astype(np.int64), static=static, groups=CMAPSS_GROUPS,
        meta=dict(rul=rul.values, n_conditions=len(uniq), settings=d[["set1", "set2", "set3"]].values))


# ------------------------------------------------------------------- MetroPT-3
# Air-leak failure windows reported by the dataset authors (Davari et al., 2021), as reproduced in
# public analyses of MetroPT-3 (e.g. github.com/marcospaula/MetroPT_Reliability_Agent, data/events.csv).
METROPT_FAILURES = [("2020-04-18 00:00", "2020-04-18 23:59"), ("2020-05-29 23:30", "2020-05-30 06:00"),
                    ("2020-06-05 10:00", "2020-06-07 14:30"), ("2020-07-15 14:30", "2020-07-15 19:00")]
METROPT_ANALOG = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current"]
METROPT_DIGITAL = ["COMP", "DV_eletric", "Towers", "MPG", "LPS", "Pressure_switch", "Oil_level", "Caudal_impulses"]
METROPT_GROUPS = {"A": [f"{s}_{k}" for s in ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs"] for k in ("mean", "std")],   # pressures
                  "B": [f"{s}_{k}" for s in ["Oil_temperature", "Motor_current"] for k in ("mean", "std")],               # motor and oil
                  "C": [f"{s}_mean" for s in METROPT_DIGITAL]}                                                            # valves and switches


def load_metropt(window: str = "10min", min_samples: int = 30) -> Dataset:
    """MetroPT-3 aggregated into inspection windows: mean and std of analog signals, duty fraction
    of digital signals. A window is labelled 1 if it overlaps a reported air-leak interval."""
    cache = DATA / f"metropt_{window}.csv"
    if cache.exists():
        w = pd.read_csv(cache, parse_dates=["start"])
    else:
        d = pd.read_csv(DATA / "metropt_raw" / "MetroPT3(AirCompressor).csv", parse_dates=["timestamp"])
        d["start"] = d["timestamp"].dt.floor(window)
        agg = {**{s: ["mean", "std"] for s in METROPT_ANALOG}, **{s: ["mean"] for s in METROPT_DIGITAL}, "timestamp": ["count"]}
        w = d.groupby("start").agg(agg); w.columns = [f"{a}_{b}" for a, b in w.columns]; w = w.reset_index()
        w = w[w["timestamp_count"] >= min_samples].drop(columns="timestamp_count").reset_index(drop=True)
        end = w["start"] + pd.Timedelta(window)
        y = np.zeros(len(w), int)
        for a, b in METROPT_FAILURES:
            y |= ((w["start"] <= pd.Timestamp(b)) & (end > pd.Timestamp(a))).to_numpy().astype(int)
        w["y"] = y; w.to_csv(cache, index=False)
    feats = sum(METROPT_GROUPS.values(), [])
    return Dataset(name="MetroPT-3", X=w[feats].reset_index(drop=True), y=w["y"].to_numpy(), unit=np.zeros(len(w), int),
                   order=w["start"].values.astype("datetime64[s]").astype(np.int64), static=pd.DataFrame(index=w.index),
                   groups=METROPT_GROUPS, meta=dict(start=w["start"], failures=METROPT_FAILURES))
