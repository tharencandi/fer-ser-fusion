"""FER branch: 376-dim blendshape summary features + XGBRFClassifier.

Feature extraction and classifier hyperparameters are fixed (not tuned on the CV
folds); `random_state` follows the run seed.
"""
import numpy as np

_BROW_IDX = [1, 2, 3, 4, 5]
_EYE_IDX = list(range(9, 23))
_MOUTH_JAW_IDX = list(range(23, 50))
_CHEEK_NOSE_IDX = [6, 7, 8, 50, 51]

XGBRF_PARAMS = dict(
    n_estimators=300,
    learning_rate=1.0,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.3,
    eval_metric="mlogloss",
)


def extract_stats_ratio_np(X_list, lengths_list):
    """7 stats x 53 channels + 5 inter-group ratios = 376 features per clip."""
    eps = 1e-6
    rows = []
    for x, L in zip(X_list, lengths_list):
        x = np.asarray(x, dtype=np.float32)[:L]
        vel = np.gradient(x, axis=0).astype(np.float32)

        m = x.mean(0); s = x.std(0)
        mx = x.max(0); mn = x.min(0); rng = mx - mn
        vm = vel.mean(0); vx = vel.max(0)

        b_avg = m[_BROW_IDX].mean(); e_avg = m[_EYE_IDX].mean()
        mo_avg = m[_MOUTH_JAW_IDX].mean(); c_avg = m[_CHEEK_NOSE_IDX].mean()
        ratios = np.array([
            b_avg / (e_avg + eps), b_avg / (mo_avg + eps),
            e_avg / (mo_avg + eps), c_avg / (mo_avg + eps),
            c_avg / (b_avg + eps),
        ])
        rows.append(np.concatenate([m, s, mx, mn, rng, vm, vx, ratios]))
    return np.stack(rows)


def fer_features(aligned):
    """Summary features for every aligned clip, indexed like `aligned` (HF row idx)."""
    seqs = aligned["blendshape_features"].tolist()
    return extract_stats_ratio_np(seqs, [len(s) for s in seqs]).astype(np.float32)


def train_xgbrf(X_train, y_train, seed):
    import xgboost as xgb
    from sklearn.utils.class_weight import compute_sample_weight

    clf = xgb.XGBRFClassifier(**XGBRF_PARAMS, random_state=int(seed))
    clf.fit(X_train, y_train, sample_weight=compute_sample_weight("balanced", y=y_train))
    return clf


def xgb_margins(clf, X):
    """Raw per-class margins (output_margin=True), used as logits for fusion."""
    import xgboost as xgb
    return clf.get_booster().predict(xgb.DMatrix(X), output_margin=True).astype(np.float32)
