"""GRU-D (Che et al., 2018, "Recurrent neural networks for multivariate time series with missing values").

A recurrent network made for irregularly observed data. At each inspection it receives the current readings
(missing values filled by a learned decay from the last observed value towards the training mean), the
missingness mask and, for each sensor, the time since it was last observed. The hidden state is also decayed
by the elapsed time. It reads each machine's inspections in time order and outputs a fault probability at
every inspection, using only that inspection and earlier ones (it is causal, like AMA).

Time is counted in inspections (steps), the same unit as AMA's "age".
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

torch.set_num_threads(2)


class GRUD(nn.Module):
    def __init__(self, d, d_static, hidden=48):
        super().__init__()
        self.d = d
        self.gx = nn.Parameter(torch.zeros(d)); self.bx = nn.Parameter(torch.zeros(d))   # input decay (per sensor)
        self.gh = nn.Linear(d, hidden)                                                       # hidden-state decay
        self.cell = nn.GRUCell(2 * d + d_static, hidden)
        self.out = nn.Linear(hidden + d_static, 1)

    def forward(self, x, m, delta, xlast, static):
        """x, m, delta, xlast: [B, T, d]; static: [B, T, d_static]. Returns logits [B, T]."""
        B, T, _ = x.shape
        h = x.new_zeros(B, self.gh.out_features); outs = []
        for t in range(T):
            dt = delta[:, t]
            gam_x = torch.exp(-torch.relu(self.gx * dt + self.bx))
            gam_h = torch.exp(-torch.relu(self.gh(dt)))
            # missing values: decay from the last observed value towards the (standardised) training mean = 0
            xh = m[:, t] * x[:, t] + (1 - m[:, t]) * (gam_x * xlast[:, t])
            h = self.cell(torch.cat([xh, m[:, t], static[:, t]], 1), gam_h * h)
            outs.append(self.out(torch.cat([h, static[:, t]], 1)).squeeze(1))
        return torch.stack(outs, 1)


def make_sequences(Xobs, static, unit, order, units, mu, sd):
    """Pad each unit's inspections (time order) into tensors. Values standardised with training mu/sd.
    Returns tensors and, per unit, the record indices in sequence order."""
    V = ((Xobs.to_numpy(float) - mu) / sd); M = ~np.isnan(V); S = static.to_numpy(float)
    idxs = []
    for u in units:
        i = np.where(unit == u)[0]; idxs.append(i[np.lexsort((i, order[i]))])
    T = max(len(i) for i in idxs); B = len(idxs); d = V.shape[1]
    x = np.zeros((B, T, d)); m = np.zeros((B, T, d)); dl = np.zeros((B, T, d)); xl = np.zeros((B, T, d))
    st = np.zeros((B, T, S.shape[1])); valid = np.zeros((B, T), bool)
    for b, i in enumerate(idxs):
        n = len(i); v = V[i]; mm = M[i]
        x[b, :n] = np.nan_to_num(v); m[b, :n] = mm; st[b, :n] = S[i]; valid[b, :n] = True
        last = np.zeros(d); since = np.zeros(d)
        for t in range(n):
            dl[b, t] = since; xl[b, t] = last                     # time since last observation, last observed value (strictly earlier)
            last = np.where(mm[t], v[t], last); since = np.where(mm[t], 1.0, since + 1.0)
    f = lambda a: torch.tensor(a, dtype=torch.float32)
    return dict(x=f(x), m=f(m), delta=f(dl), xlast=f(xl), static=f(st), valid=torch.tensor(valid)), idxs


def _sub(s, i):
    return {k: v[i] for k, v in s.items()}


def fit_predict_grud(seq_tr, y_tr_seq, seq_val, y_val_seq, seq_te, seed, epochs=150, patience=15, lr=5e-3, hidden=64, batch=8):
    """Train on seq_tr with mini-batches of machines (Adam), early-stop on validation PR-AUC (checked every epoch),
    and return per-step probabilities for seq_te from the best epoch."""
    torch.manual_seed(seed); np.random.seed(seed)
    d = seq_tr["x"].shape[2]; ds = seq_tr["static"].shape[2]
    net = GRUD(d, ds, hidden); opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    vtr = seq_tr["valid"]; ytr = y_tr_seq[vtr]
    pos_w = torch.tensor(min(20.0, float((ytr == 0).sum() / max(1, (ytr == 1).sum()))))
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    args = lambda s: (s["x"], s["m"], s["delta"], s["xlast"], s["static"])
    best, best_state, bad = -1, None, 0
    for ep in range(epochs):
        net.train(); perm = torch.randperm(vtr.shape[0])
        for k in range(0, len(perm), batch):
            i = perm[k:k + batch]; sb = _sub(seq_tr, i); vb = sb["valid"]
            if vb.sum() == 0:
                continue
            opt.zero_grad(); z = net(*args(sb)); loss = lossf(z[vb], y_tr_seq[i][vb]); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
        net.eval()
        with torch.no_grad():
            pv = torch.sigmoid(net(*args(seq_val)))[seq_val["valid"]].numpy()
        yv = y_val_seq[seq_val["valid"]].numpy()
        score = average_precision_score(yv, pv) if yv.sum() > 0 else -loss.item()
        if score > best + 1e-4:
            best, bad = score, 0; best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(*args(seq_te))).numpy(), ep + 1
