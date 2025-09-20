import logging
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

class XGBoostClassifierWrapper:
    """
    A simple XGBoost classifier wrapper with sklearn-like fit/predict API,
    with support for constant-label cases and multiclass encoding.
    """
    def __init__(self, params=None):
        default_params = {
            "eval_metric": "mlogloss",
            "n_jobs": -1
        }
        self.params = {**default_params, **(params or {})}
        self.model = None
        self.constant_label = None
        self.is_constant = False
        self.le = None  # LabelEncoder for mapping original labels

        logger.info(f"XGBoost: init with params={self.params}")

    def fit(self, X, y):
        y_arr = np.array(y)
        unique_labels = np.unique(y_arr)

        if unique_labels.size == 1:
            # Handle constant-label case
            self.constant_label = unique_labels[0]
            self.is_constant = True
            logger.info(f"XGBoost: only one class ({self.constant_label}) in y; using constant predictor")
            return self

        self.is_constant = False
        self.le = LabelEncoder()
        y_enc = self.le.fit_transform(y_arr)

        n_classes = len(self.le.classes_)
        if n_classes > 2:
            self.params.update({
                "objective": "multi:softprob",
                "num_class": n_classes
            })
        else:
            self.params.update({
                "objective": "binary:logistic"
            })

        self.model = XGBClassifier(**self.params)
        logger.info(f"XGBoost: training model on {len(X)} samples with labels {self.le.classes_}")
        self.model.fit(X, y_enc)
        return self

    def predict(self, X):
        n_samples = len(X)
        logger.info(f"XGBoost: predicting {n_samples} samples")
        if self.is_constant:
            return np.full(n_samples, self.constant_label)
        preds_enc = self.model.predict(X)
        return self.le.inverse_transform(preds_enc)
