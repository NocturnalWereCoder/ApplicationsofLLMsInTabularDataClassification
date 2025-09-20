#!/usr/bin/env python3
# main.py

import os
import sys
import time
import json
import logging

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from config import (
    MODE, SAMPLE_FRAC, DATASETS, N_SPLITS, METHODS, K_NEIGHBORS,
)
from knn import knn_predict_skl
from rag import LangChainOllamaRAGClassifier
from hybrid import HybridClassifier
from xgboost_classifier import XGBoostClassifierWrapper
from random_context import RandomContextClassifier
from catboost_classifier import CatBoostClassifierWrapper
from random_forest_classifier import RandomForestClassifierWrapper
from quantile_binned_fewshot_tabular_llm_classifier import (
    QuantileBinnedFewShotTabularLLMClassifier
)
from stacked_score_llm_classifier import StackedScoreLLMClassifier
from meta_selector_llm_classifier import MetaSelectorLLMClassifier

from nsp_classifier import NSPClassifier
from fsp_classifier import FSPClassifier
from lr import lr_predict_skl
from hgb import hgb_predict_skl
from llm_guided_hpo_classifier import LLMGuidedHPOClassifier





# ─── Logging ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

class NoHttpFilter(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith("HTTP Request:")

logger.addFilter(NoHttpFilter())

# ─── TensorFlow GPU Memory Growth ─────────────────────────────────────
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for dev in gpus:
        tf.config.experimental.set_memory_growth(dev, True)
    logger.info(f"Enabled TF memory growth: {len(gpus)} GPU(s)")
except Exception as e:
    logger.warning(f"TF memory growth failed: {e}")

def detect_gpu_count():
    return len(tf.config.experimental.list_physical_devices('GPU')) or 1

def main():
    start_time = time.time()
    logger.info(f"MODE={MODE}, SAMPLE_FRAC={SAMPLE_FRAC}")

    gpu_count = detect_gpu_count()
    logger.info(f"Detected {gpu_count} GPU(s)")

    # prepare output file
    out_path = 'ragresults.txt'
    with open(out_path, 'w') as f:
        f.write('Dataset,Method,MeanAccuracy,Model,Hyperparameters\n')


    for ds in DATASETS:
        csvp = f'Datasets/{ds}/{ds}.csv'
        mp   = f'Datasets/{ds}/{ds}_metadata.json'
        if not (os.path.exists(csvp) and os.path.exists(mp)):
            logger.warning(f"Skipping {ds}: data files missing")
            continue

        df = pd.read_csv(csvp, header=None)
        if SAMPLE_FRAC < 1.0:
            df = df.head(int(len(df)*SAMPLE_FRAC))

        meta = json.load(open(mp))
        tgt = meta['target_column_index']
        X, y = df.drop(columns=[tgt]), df.iloc[:, tgt]

        splits = list(StratifiedKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=42
        ).split(X, y))

        for fold_idx, (tr, te) in enumerate(splits, start=1):
            X_tr, X_te = X.iloc[tr], X.iloc[te]
            y_tr, y_te = y.iloc[tr], y.iloc[te]
            gpu_id = (fold_idx-1) % gpu_count

            for method in METHODS:
                logger.info(f"[{ds}][Fold {fold_idx}] Starting {method}")
                model_obj = None  # <- reset per method
                if method == 'KNN':

                    preds = knn_predict_skl(X_tr, y_tr, X_te, K_NEIGHBORS)

                elif method == 'RAG':
                    rag = LangChainOllamaRAGClassifier(
                        k_neighbors=K_NEIGHBORS,
                        persist_dir=f"chroma_db/{ds}_f{fold_idx}/{method}",
                        cuda_device=gpu_id
                    )
                    rag.fit(X_tr, y_tr)
                    preds = rag.predict(X_te)
                    model_obj = rag


                elif method == 'Hybrid_model':
                    hyb = HybridClassifier(
                        k_neighbors=K_NEIGHBORS,
                        cuda_device=gpu_id
                    )
                    hyb.fit(X_tr, y_tr)
                    preds = hyb.predict(X_te)
                    model_obj = hyb


                elif method == 'XGB':
                    # instantiate with any desired hyper-params
                    xgb_clf = XGBoostClassifierWrapper(
                        params={
                            "n_estimators": 100,
                            "learning_rate": 0.1
                        }
                    )
                    xgb_clf.fit(X_tr, y_tr)
                    preds = xgb_clf.predict(X_te)
                    model_obj = xgb_clf
            
                elif method == 'CatBoost':
                    # You can tune these; safe defaults for CPU.
                    cat_params = {
                        "iterations": 600,
                        "learning_rate": 0.05,
                        "depth": 6,
                        "l2_leaf_reg": 3.0,
                        "random_seed": 42,
                        "thread_count": -1,
                        # OPTIONAL GPU (requires CUDA + NVIDIA GPU + drivers):
                        # Uncomment the next two lines if your environment supports CatBoost GPU.
                        # "task_type": "GPU",
                        # "devices": str(gpu_id),
                    }
                    cat_clf = CatBoostClassifierWrapper(params=cat_params)
                    cat_clf.fit(X_tr, y_tr)
                    preds = cat_clf.predict(X_te)
                    model_obj = cat_clf

                elif method == 'RandomForest':
                    rf_params = {
                        # override defaults here if you like
                        # "n_estimators": 500,
                        # "max_depth": None,
                        # "class_weight": "balanced",
                        # "random_state": 42,
                    }
                    rf = RandomForestClassifierWrapper(params=rf_params)
                    rf.fit(X_tr, y_tr)
                    preds = rf.predict(X_te)
                    model_obj = rf
                elif method == 'MetaSelectorLLM':
                    clf = MetaSelectorLLMClassifier(
                        cuda_device=gpu_id,
                        oof_folds=5,
                        region_k=31,
                        top_features=10,
                        prototypes_per_class=1,
                        engage_threshold=0.08,
                        calibrate=True,
                        llm_temperature=0.0,
                        llm_top_p=1.0,
                        schema_sample_n=100,
                        max_parse_attempts=3,
                        cv_folds=5,
                        cv_max_rows=50000,
                        random_state=42
                    )
                    clf.fit(X_tr, y_tr, metadata=meta)
                    preds = clf.predict(X_te)
                    model_obj = clf  # <-- IMPORTANT

                elif method == 'NSP':
                    clf = NSPClassifier(cuda_device=gpu_id, temperature=0.0, top_p=1.0)
                    clf.fit(X_tr, y_tr)
                    preds = clf.predict(X_te)

                elif method == 'FSP':
                    clf = FSPClassifier(
                        cuda_device=gpu_id,
                        prototypes_per_class=2,   # tweak 1–3
                        max_examples=20,          # cap total prompt size
                        temperature=0.0,
                        top_p=1.0
                    )
                    clf.fit(X_tr, y_tr)
                    preds = clf.predict(X_te)
                elif method == 'LR':
                    # uses config defaults if you added them; or rely on function defaults
                    try:
                        from config import LR_C, LR_MAX_ITER, LR_PENALTY, LR_SOLVER
                        preds = lr_predict_skl(
                            X_tr, y_tr, X_te,
                            C=LR_C, max_iter=LR_MAX_ITER,
                            penalty=LR_PENALTY, solver=LR_SOLVER
                        )
                    except ImportError:
                        preds = lr_predict_skl(X_tr, y_tr, X_te)
                elif method == 'LLMGuidedHPO':
                    clf = LLMGuidedHPOClassifier(
                        schema_sample_n=100,
                        max_parse_attempts=3,
                        cv_folds=3,
                        row_cap=50000,
                        n_iter_per_family=40,
                        random_state=42,
                        n_jobs=-1,
                        calibrate=True,
                        verbose=1
                    )
                    # pass metadata dict read from your dataset's *_metadata.json
                    clf.fit(X_tr, y_tr, metadata=meta)   # <-- IMPORTANT: give it full metadata
                    preds = clf.predict(X_te)
                    model_obj = clf


                elif method == 'HGB':
                    try:
                        from config import (
                            HGB_LEARNING_RATE, HGB_MAX_ITER, HGB_MAX_DEPTH,
                            HGB_L2, HGB_EARLY_STOPPING
                        )
                        preds = hgb_predict_skl(
                            X_tr, y_tr, X_te,
                            learning_rate=HGB_LEARNING_RATE,
                            max_iter=HGB_MAX_ITER,
                            max_depth=HGB_MAX_DEPTH,
                            l2_regularization=HGB_L2,
                            early_stopping=HGB_EARLY_STOPPING
                        )
                    except ImportError:
                        preds = hgb_predict_skl(X_tr, y_tr, X_te)

                else:
                    logger.error(f"Unknown method: {method}")
                    continue
            acc = np.mean(preds == y_te)

            # Build summary only from the classifier actually used in THIS branch
            summary = ""
            model_name = ""
            hp_json = ""

            if 'model_obj' in locals() and model_obj is not None:
                # Preferred: MetaSelectorLLM exposes selection_summary_ like:
                # 'model=hgb, hp={"learning_rate":0.06,"l2_regularization":0.0,"max_depth":null,"max_iter":400}'
                sel = getattr(model_obj, "selection_summary_", "") or ""
                if sel:
                    summary = f" | {sel}"
                    # Parse it for the CSV columns
                    try:
                        _parts = sel.split(", hp=", 1)
                        if len(_parts) == 2:
                            model_name = _parts[0].replace("model=", "").strip()
                            hp_json = _parts[1].strip()
                    except Exception:
                        pass
                else:
                    # Fallback: explicit attributes if available
                    if hasattr(model_obj, "selected_model_"):
                        model_name = getattr(model_obj, "selected_model_", "") or ""
                    if hasattr(model_obj, "selected_params_"):
                        try:
                            hp_json = json.dumps(getattr(model_obj, "selected_params_", {}), sort_keys=True)
                        except Exception:
                            hp_json = str(getattr(model_obj, "selected_params_", {}))
                    if model_name or hp_json:
                        summary = f" | model={model_name} | hp={hp_json}"

            logger.info(f"[{ds}][Fold {fold_idx}][{method}] Acc={acc:.4f}{summary}")

            with open(out_path, 'a') as f:
                f.write(f"{ds},{method},{acc:.4f},{model_name},{hp_json}\n")

    elapsed = time.time() - start_time
    logger.info(f"All benchmarks done in {elapsed:.2f}s")

if __name__ == '__main__':
    main()
