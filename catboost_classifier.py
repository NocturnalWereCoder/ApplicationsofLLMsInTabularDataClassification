# catboost_classifier.py
import logging
import numpy as np
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier

logger = logging.getLogger(__name__)

class CatBoostClassifierWrapper:
    """
    A CatBoost classifier with sklearn-like API. Supports constant-label
    cases and multiclass via automatic label encoding.
    """

    def __init__(self, params=None):
        # Do NOT set loss_function here; we'll set it during fit based on #classes
        default_params = {
            "iterations": 500,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "random_seed": 42,
            "thread_count": -1,
        }
        self.params = {**default_params, **(params or {})}
        self.model = None
        self.constant_label = None
        self.is_constant = False
        self.le = None
        logger.info(f"CatBoost: init with params={self.params}")

    def fit(self, X, y):
        y_arr = np.array(y)
        unique_labels = np.unique(y_arr)

        # Handle constant-label case
        if unique_labels.size == 1:
            self.constant_label = unique_labels[0]
            self.is_constant = True
            logger.info(
                f"CatBoost: only one class ({self.constant_label}) in y; using constant predictor"
            )
            return self

        self.is_constant = False
        self.le = LabelEncoder()
        y_enc = self.le.fit_transform(y_arr)
        n_classes = len(self.le.classes_)

        # Choose loss/metrics based on problem type
        params = dict(self.params)  # copy
        if n_classes > 2:
            params.update({"loss_function": "MultiClass", "eval_metric": "MultiClass"})
        else:
            params.update({"loss_function": "Logloss", "eval_metric": "Logloss"})

        self.model = CatBoostClassifier(**params)
        logger.info(
            f"CatBoost: training model on {len(X)} samples with labels {self.le.classes_}"
        )
        # Keep CatBoost quiet; use logging if you want progress
        self.model.fit(X, y_enc, verbose=False)
        return self

    def predict(self, X):
        n_samples = len(X)
        logger.info(f"CatBoost: predicting {n_samples} samples")
        if self.is_constant:
            return np.full(n_samples, self.constant_label)

        # 'Class' returns integer class indices for both binary and multiclass
        preds_enc = self.model.predict(X, prediction_type="Class")
        preds_enc = np.array(preds_enc).astype(int).ravel()
        return self.le.inverse_transform(preds_enc)
