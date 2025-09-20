# knn.py
import logging
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

logger = logging.getLogger(__name__)

def knn_predict_skl(X_tr, y_tr, X_te, k):
    """
    Train a KNeighborsClassifier and predict.
    Returns: array of predictions
    """
    logger.info(f"KNN: fitting with k={k} on {X_tr.shape[0]} samples")
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn.fit(X_tr, y_tr)
    preds = knn.predict(X_te)
    logger.info(f"KNN: prediction done for {X_te.shape[0]} samples")
    return preds
