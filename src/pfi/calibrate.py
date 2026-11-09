"""Probability calibration fitted on inner out-of-fold predictions of each outer training fold."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

EPS = 1e-6


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS); return np.log(p / (1 - p))


class Platt:
    def fit(self, p, y):
        self.m = LogisticRegression(C=1e6).fit(_logit(p)[:, None], y); return self

    def __call__(self, p):
        return self.m.predict_proba(_logit(p)[:, None])[:, 1]


class Isotonic:
    def fit(self, p, y):
        self.m = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p, y); return self

    def __call__(self, p):
        return np.clip(self.m.predict(p), EPS, 1 - EPS)


class Identity:
    def fit(self, p, y):
        return self

    def __call__(self, p):
        return p


CALIBRATORS = {"none": Identity, "platt": Platt, "isotonic": Isotonic}


def calibrate_oof(y, p, fold, inner, kind):
    """Return calibrated test probabilities and calibrated inner probabilities per fold.

    inner: list of dicts with 'idx' (training records of fold f, in fold order), 'p' (inner OOF probs).
    """
    pc = np.full(len(p), np.nan); inner_c = []
    for f, inn in enumerate(inner):
        cal = CALIBRATORS[kind]().fit(inn["p"], y[inn["idx"]])
        te = fold == f; pc[te] = cal(p[te]); inner_c.append(dict(idx=inn["idx"], p=cal(inn["p"])))
    return pc, inner_c
