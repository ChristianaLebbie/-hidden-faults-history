"""Point 1 (order check): rebuild history using the original AI4I record order (UDI) instead of the
published AI4I-PMDI timestamps, and rerun current / LOCF+indicators / AMA with the same seeds and folds."""
import time
import numpy as np
import pandas as pd
from sv_common import ai4i, alarms_at, SEEDS, OUT, prauc, load_pred
from pfi.schedules import ai4i_realistic, apply_schedule, hidden_mask, visible_mask
from pfi.history import past_features
from pfi.methods import build_features, ALL_STATS
from pfi.runner import run

ds = ai4i(); rows = []; diff = []
udi = ds.meta["record_id"].astype(np.int64)
for seed in SEEDS:
    sch = ai4i_realistic(ds, 100 + seed); hid, vis = hidden_mask(ds, sch), visible_mask(ds, sch)
    X = apply_schedule(ds.X, sch, ds.groups)
    Ht = past_features(X, ds.unit, ds.order, stats=ALL_STATS)
    Hu = past_features(X, ds.unit, udi, stats=ALL_STATS)
    lt, lu = Ht.filter(like="last_").to_numpy(), Hu.filter(like="last_").to_numpy()
    same = np.isclose(lt, lu, equal_nan=True)
    diff.append(dict(seed=seed, share_last_values_identical=float(same.mean()),
                     share_records_all_identical=float(same.all(1).mean()),
                     mean_age_time=float(np.nanmean(Ht.filter(like="stale_").to_numpy())),
                     mean_age_udi=float(np.nanmean(Hu.filter(like="stale_").to_numpy()))))
    for m in ["current", "locf_ind", "cur_last"]:
        for name, H in [("timestamp order", Ht), ("record-id order", Hu)]:
            if name == "timestamp order":
                p, fold, inner = load_pred(f"ai4i_S1_{m}", seed)
            else:
                t0 = time.time()
                _, res, _ = run(ds, lambda tr: build_features(m, X, H, ds.static, tr), seed, save_pred=False)
                p, fold = res["p"], res["fold"]; inner = [(i["idx"], i["p"]) for i in res["inner"]]
            alarm, _ = alarms_at(p, fold, inner, ds.y, 1.0)
            rows.append(dict(seed=seed, method=m, order=name, PR_AUC=prauc(ds.y, p),
                             recall_hidden=alarm[hid].mean(), recall_visible=alarm[vis].mean(),
                             recall_all=alarm[ds.y == 1].mean()))
    print(seed, "done")
d = pd.DataFrame(diff); r = pd.DataFrame(rows)
d.to_csv(OUT / "p1_order_history_diff.csv", index=False); r.to_csv(OUT / "p1_order_results.csv", index=False)
print(d.mean().round(3).to_string()); print(r.groupby(["order", "method"]).mean(numeric_only=True).drop(columns="seed").round(3).to_string())
