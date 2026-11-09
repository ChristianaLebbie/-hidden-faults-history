"""Study 1, Q5: different operating conditions (C-MAPSS FD001 and FD002).

FD001 runs under one operating condition; FD002 switches among six conditions from
cycle to cycle. Each cycle reports one of three sensor
groups at random. History is tested in three forms:
  raw          last earlier observation, whatever the condition it was taken under
  same-cond    last earlier observation taken under the same operating condition
  normalised   readings z-scored within each condition (training units only) before history
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from pfi.data import load_cmapss
from pfi.history import past_features
from pfi.methods import ALL_STATS, build_features, condition_normalise
from pfi.runner import append, log, run
from pfi.schedules import apply_schedule, random_mix

RAW = ["current", "locf_ind_age", "cur_last", "cur_window", "gated"]
NORM = ["current", "locf_ind_age", "cur_last", "cur_window", "gated", "gated@0.25"]
rows = []
for fd in [
    "FD001",
    "FD002",
]:  # thesis uses FD001 (one condition) and FD002 (six conditions)
    ds = load_cmapss(fd)
    cond = ds.static["cond"].to_numpy()
    for seed in [0, 1, 2]:
        sch = random_mix(ds, 100 + seed)
        X = apply_schedule(ds.X, sch, ds.groups)
        H = past_features(X, ds.unit, ds.order, stats=ALL_STATS)
        Hc = past_features(
            X, ds.unit * 100 + cond, ds.order, stats=ALL_STATS
        )  # same-condition history
        for m in RAW:
            r, _, _ = run(
                ds,
                lambda tr: build_features(m, X, H, ds.static, tr),
                seed,
                tag=f"cm_{fd}_raw_{m}_s{seed}",
            )
            rows.append(dict(fd=fd, history="raw", method=m, seed=seed, **r))
            log(fd, "raw", m, seed, round(r["PR_AUC"], 3))
        if ds.meta["n_conditions"] > 1:
            for m in ["cur_last", "gated"]:
                r, _, _ = run(
                    ds,
                    lambda tr: build_features(m, X, Hc, ds.static, tr),
                    seed,
                    tag=f"cm_{fd}_samecond_{m}_s{seed}",
                )
                rows.append(dict(fd=fd, history="same-cond", method=m, seed=seed, **r))
                log(fd, "same-cond", m, seed, round(r["PR_AUC"], 3))
        cache = {}

        def zh(tr):
            key = (len(tr), int(tr[0]), int(tr[-1]))
            if key not in cache:
                Z = condition_normalise(X, cond, tr)
                cache.clear()
                cache[key] = (Z, past_features(Z, ds.unit, ds.order, stats=ALL_STATS))
            return cache[key]

        for m in NORM:
            base, tau = (
                (m.split("@")[0], float(m.split("@")[1])) if "@" in m else (m, 0.5)
            )

            def build(tr, base=base, tau=tau):
                Z, Hz = zh(tr)
                return build_features(base, Z, Hz, ds.static, tr, tau=tau)

            r, _, _ = run(ds, build, seed, tag=f"cm_{fd}_norm_{m}_s{seed}")
            rows.append(dict(fd=fd, history="normalised", method=m, seed=seed, **r))
            log(fd, "norm", m, seed, round(r["PR_AUC"], 3))
        for hist, build in [
            ("complete-raw", lambda tr: pd.concat([ds.X, ds.static], axis=1)),
            (
                "complete-normalised",
                lambda tr: pd.concat(
                    [condition_normalise(ds.X, cond, tr), ds.static], axis=1
                ),
            ),
        ]:
            r, _, _ = run(ds, build, seed, tag=f"cm_{fd}_{hist}_s{seed}")
            rows.append(dict(fd=fd, history=hist, method="complete", seed=seed, **r))
            log(fd, hist, seed, round(r["PR_AUC"], 3))
        append(rows, "exp5_operating_conditions.csv")
log("done")
