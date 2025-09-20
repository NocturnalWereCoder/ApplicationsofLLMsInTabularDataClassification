# random_forest_classifier.py

import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

class RandomForestClassifierWrapper:
    """
    sklearn RandomForestClassifier with a simple fit/predict API,
    plus constant-label handling and label encoding for safety.
    """
    def __init__(self, params=None):
        default_params = {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "n_jobs": -1,
            "random_state": 42,
        }
        self.params = {**default_params, **(params or {})}
        self.model = None
        self.constant_label = None
        self.is_constant = False
        self.le = None  # LabelEncoder to preserve original labels

        logger.info(f"RandomForest: init with params={self.params}")

    def fit(self, X, y):
        y_arr = np.array(y)
        unique_labels = np.unique(y_arr)

        if unique_labels.size == 1:
            # Handle constant-label case (e.g., a single-class fold)
            self.constant_label = unique_labels[0]
            self.is_constant = True
            logger.info(f"RandomForest: only one class ({self.constant_label}) in y; using constant predictor")
            return self

        self.is_constant = False
        self.le = LabelEncoder()
        y_enc = self.le.fit_transform(y_arr)

        self.model = RandomForestClassifier(**self.params)
        logger.info(f"RandomForest: training on {len(X)} samples; classes={list(self.le.classes_)}")
        self.model.fit(X, y_enc)
        return self

    def predict(self, X):
        n_samples = len(X)
        logger.info(f"RandomForest: predicting {n_samples} samples")
        if self.is_constant:
            return np.full(n_samples, self.constant_label)
        preds_enc = self.model.predict(X)
        return self.le.inverse_transform(preds_enc)
