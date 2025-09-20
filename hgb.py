import logging
from sklearn.ensemble import HistGradientBoostingClassifier

logger = logging.getLogger(__name__)

def hgb_predict_skl(
    X_tr, y_tr, X_te,
    *, learning_rate=0.06, max_iter=400, max_depth=None,
    l2_regularization=0.0, early_stopping=True
):
    """
    Train a HistGradientBoostingClassifier and predict.
    Returns: array of predictions
    """
    logger.info(
        "HGB: fitting with "
        f"learning_rate={learning_rate}, max_iter={max_iter}, "
        f"max_depth={max_depth}, l2_regularization={l2_regularization}, "
        f"early_stopping={early_stopping} on {X_tr.shape[0]} samples"
    )
    hgb = HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_depth=max_depth,
        l2_regularization=l2_regularization,
        early_stopping=early_stopping,
        random_state=42
    )
    hgb.fit(X_tr, y_tr)
    preds = hgb.predict(X_te)
    logger.info(f"HGB: prediction done for {X_te.shape[0]} samples")
    return preds
