import logging
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

def lr_predict_skl(
    X_tr, y_tr, X_te,
    *, C=1.0, max_iter=1000, penalty="l2", solver="lbfgs",
    class_weight=None
):
    """
    Train a LogisticRegression classifier and predict.
    Returns: array of predictions
    """
    logger.info(
        f"LR: fitting LogisticRegression(C={C}, max_iter={max_iter}, "
        f"penalty='{penalty}', solver='{solver}') on {X_tr.shape[0]} samples"
    )
    lr = LogisticRegression(
        C=C,
        max_iter=max_iter,
        penalty=penalty,
        solver=solver,         # 'lbfgs' is a solid default
        class_weight=class_weight,
        n_jobs=-1,             # ignored by some solvers; harmless to set
        multi_class="auto",
        random_state=42
    )
    lr.fit(X_tr, y_tr)
    preds = lr.predict(X_te)
    logger.info(f"LR: prediction done for {X_te.shape[0]} samples")
    return preds
