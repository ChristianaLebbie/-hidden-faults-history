"""Classifiers compared in the model-comparison experiment (exp13).

All take the same feature table. Models that cannot handle missing values natively get
median imputation (fitted on training records only) plus a missing-value flag per column,
so every model is told which readings were not observed.
"""
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

MODEL_LABELS = {
    "logreg": "Logistic regression",
    "rf": "Random forest",
    "hgb": "Histogram gradient boosting",
    "mlp": "Neural network (MLP)",
    "xgb": "XGBoost",
}


def _impute():
    return SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)


def make_model(name, seed):
    if name == "logreg":
        return make_pipeline(_impute(), StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    if name == "rf":
        return make_pipeline(_impute(), RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=2, random_state=seed))
    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4, random_state=seed)
    if name == "mlp":
        return make_pipeline(_impute(), StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-2, max_iter=500, random_state=seed))
    if name == "xgb":
        return xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9, eval_metric="logloss",
                                 verbosity=0, n_jobs=2, tree_method="hist", random_state=seed)
    raise ValueError(name)
