"""Study 1, Q4: changes in acquisition schedule between training and deployment.

Policies (all records, faults included, receive a group under the policy):
  uniform      each inspection reports A, B or C with equal probability
  A-heavy      60% A, 20% B, 20% C
  C-heavy      20% A, 20% B, 60% C
  rare-C       45% A, 45% B, 10% C (one group rarely observed, so its history goes stale)
  round-robin  A, B, C, A, B, C ... per unit (history always recent)
Models are trained with features from the training policy and tested with features from the
test policy (each unit's history comes from its own observations under that policy).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pfi.data import load_ai4i, load_cmapss
from pfi.schedules import random_mix, round_robin, apply_schedule, hidden_mask
from pfi.history import past_features
from pfi.methods import build_features, reliability_table, horizons, ALL_STATS
from pfi.runner import run, append, log

POL = {"uniform": None, "A-heavy": [0.6, 0.2, 0.2], "C-heavy": [0.2, 0.2, 0.6], "rare-C": [0.45, 0.45, 0.10], "round-robin": "rr"}
rows = []
for name, ds, seeds, pols, methods in [
        ("AI4I-PMDI", load_ai4i(), [0, 1, 2], list(POL), ["current", "locf_ind_age", "cur_last", "gated"])]:
    # (schedule change on real data is tested on MetroPT-3 in exp12_metropt_stress.py)
    for seed in seeds:
        feats = {}
        for k, pol in enumerate(pols):
            sch = round_robin(ds, 300 + seed) if POL[pol] == "rr" else random_mix(ds, 300 + 10 * k + seed, POL[pol])
            X = apply_schedule(ds.X, sch, ds.groups)
            feats[pol] = (X, past_features(X, ds.unit, ds.order, stats=ALL_STATS), sch)
        for trp in pols:
            for tep in pols:
                Xa, Ha, _ = feats[trp]; Xb, Hb, schb = feats[tep]
                masks = {"hidden": hidden_mask(ds, schb)} if name.startswith("AI4I") else None
                for m in methods:
                    def build(tr, m=m):
                        hz = horizons(reliability_table(Xa, Ha, tr), 0.5)   # learned under the training schedule only
                        Fa = build_features(m, Xa, Ha, ds.static, tr, hz=hz); Fb = build_features(m, Xb, Hb, ds.static, tr, hz=hz)
                        F = Fb.copy(); F.iloc[tr] = Fa.iloc[tr].to_numpy(); return F
                    r, _, _ = run(ds, build, seed, masks, save_pred=False)
                    rows.append(dict(dataset=name, train_policy=trp, test_policy=tep, method=m, seed=seed, **r))
                log(name, seed, trp, "->", tep, [round(x["PR_AUC"], 3) for x in rows[-len(methods):]])
            append(rows, "exp4_schedule_change.csv")
log("done")
