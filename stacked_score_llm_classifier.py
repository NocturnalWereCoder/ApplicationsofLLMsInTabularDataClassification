# stacked_score_llm_classifier.py
import os
import re
import json
import logging
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import f_classif

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama.llms import OllamaLLM

from config import LLM_MODEL

logger = logging.getLogger(__name__)

# ---------------- Output schema ----------------
class ClassificationWithReasoning(BaseModel):
    classification: int = Field(...)
    reasoning: str = Field(...)

# ---------------- Utilities --------------------
def _json_extract(obj: str) -> str:
    m = re.search(r"\{[\s\S]*\}", str(obj))
    return m.group(0) if m else str(obj).strip()

def _align_proba(estimator, Xz: np.ndarray, labels_order: List[int]) -> np.ndarray:
    """Return probabilities aligned to integer labels_order."""
    proba = estimator.predict_proba(Xz)
    cls = list(estimator.classes_)
    out = np.zeros((Xz.shape[0], len(labels_order)), dtype=float)
    for i, c in enumerate(labels_order):
        if c in cls:
            out[:, i] = proba[:, cls.index(c)]
        else:
            out[:, i] = 0.0
    return out

def _nearest_indices(z: np.ndarray, Zc: np.ndarray, k: int = 1) -> List[int]:
    d = np.linalg.norm(Zc - z[None, :], axis=1)
    return np.argsort(d)[:k].tolist()

@dataclass
class ClassStats:
    label: int
    prior: float
    mu: np.ndarray  # mean in standardized space
    idxs: np.ndarray  # training indices of this class (global)

class StackedScoreLLMClassifier:
    """
    StackedScore LLM — small-context, calibrated tabular ensemble + LLM adjudicator (no embeddings)
    """
    def __init__(
        self,
        k_neighbors: int = 31,
        cuda_device: int = 0,
        # ensemble weights must sum≈1
        weights: Dict[str, float] = None,
        top_features: int = 10,
        top_classes_in_prompt: int = 2,
        prototypes_per_class: int = 1,
        llm_temperature: float = 0.0,
        llm_top_p: float = 1.0,
        # LLM engage threshold: if margin >= threshold, LLM is instructed to confirm the recommendation
        engage_threshold: float = 0.12,
        calibrate: bool = True
    ):
        self.k = int(k_neighbors)
        self.top_features = int(top_features)
        self.top_k = int(top_classes_in_prompt)
        self.proto_k = int(prototypes_per_class)
        self.engage_threshold = float(engage_threshold)
        self.calibrate = bool(calibrate)

        self.weights = weights or {
            "lr": 0.35,
            "hgb": 0.35,
            "rf": 0.20,
            "knn": 0.10,
        }

        self.scaler = StandardScaler()
        self.labels_: List[int] = []
        self.cols: List[Any] = []
        self.train_X: pd.DataFrame = None
        self.train_y: pd.Series = None
        self.Xz: np.ndarray = None

        # base learners
        self.lr = LogisticRegression(max_iter=1000, multi_class="auto", solver="lbfgs")
        self.hgb = HistGradientBoostingClassifier(max_depth=None, learning_rate=0.06, max_iter=400)
        self.rf = RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=1, n_jobs=-1, random_state=42
        )
        self.knn = KNeighborsClassifier(n_neighbors=self.k, weights="distance")

        self.lr_cal = None
        self.hgb_cal = None
        self.rf_cal = None
        self.knn_cal = None

        # class stats for compact numeric evidence
        self.stats: Dict[int, ClassStats] = {}

        os.environ['OLLAMA_CUDA_DEVICE'] = str(cuda_device)
        self.llm = OllamaLLM(
            model=LLM_MODEL,
            temperature=float(llm_temperature),
            top_p=float(llm_top_p),
            in_process=True
        )

        tmpl = """
You are a careful adjudicator for numeric tabular classification.
Return ONLY valid JSON: {{ "classification": <integer>, "reasoning": "<≤1 sentence>" }}.

Guidelines:
- Trust the ensemble's "recommended_class" when its margin is large (≥ {engage_threshold}).
- Consider overriding ONLY if the runner-up clearly dominates on multiple evidence fields (e.g., both ensemble and at least two base models).

Scoreboard (only top candidates):
{scoreboard}

Prototype rows (standardized z-values for top features):
{prototypes_block}

Return ONLY JSON: {{ "classification": <integer>, "reasoning": "<≤1 sentence>" }}.
""".strip()

        self.parser = PydanticOutputParser(pydantic_object=ClassificationWithReasoning)
        # Fill a static engage_threshold into the template, but keep other fields templated by LangChain.
        self.prompt = ChatPromptTemplate.from_template(
            tmpl.replace("{engage_threshold}", f"{self.engage_threshold:.3f}")
        )
        self.chain = self.prompt | self.llm

    # ---------------------- Fit ----------------------
    def fit(self, X: pd.DataFrame, y: pd.Series):
        logger.info("StackedScoreLLM: fitting scaler, base learners, calibration, and class stats")
        self.cols = list(X.columns)
        self.train_X = X.reset_index(drop=True).copy()
        self.train_y = y.reset_index(drop=True).copy()
        self.labels_ = sorted(pd.unique(self.train_y).tolist())

        # Standardize numeric features
        self.Xz = self.scaler.fit_transform(self.train_X.values.astype(float))

        # Fit base learners on full training
        self.lr.fit(self.Xz, self.train_y.values)
        self.hgb.fit(self.Xz, self.train_y.values)
        self.rf.fit(self.Xz, self.train_y.values)
        self.knn.fit(self.Xz, self.train_y.values)

        # Optional probability calibration (lightweight)
        if self.calibrate:
            cv = 3 if len(self.train_X) >= 1000 else 5
            method = "isotonic" if len(self.train_X) >= 2000 else "sigmoid"
            self.lr_cal  = CalibratedClassifierCV(self.lr,  cv=cv, method=method).fit(self.Xz, self.train_y.values)
            self.hgb_cal = CalibratedClassifierCV(self.hgb, cv=cv, method=method).fit(self.Xz, self.train_y.values)
            self.rf_cal  = CalibratedClassifierCV(self.rf,  cv=cv, method=method).fit(self.Xz, self.train_y.values)
            self.knn_cal = CalibratedClassifierCV(self.knn, cv=cv, method=method).fit(self.Xz, self.train_y.values)
        else:
            self.lr_cal = self.hgb_cal = self.rf_cal = self.knn_cal = None

        # Class priors, centroids (z-space), and per-class indices for prototypes
        n = len(self.train_y)
        self.stats = {}
        for c in self.labels_:
            idxs = np.where(self.train_y.values == c)[0]
            prior = float(len(idxs)) / n
            mu = np.mean(self.Xz[idxs], axis=0)
            self.stats[c] = ClassStats(label=int(c), prior=prior, mu=mu, idxs=idxs)

        # Rank top features for prompt compactness (ANOVA F as in your code)
        try:
            fvals, _ = f_classif(self.train_X.values.astype(float), self.train_y.values)
            ord_idx = np.argsort(fvals)[::-1]
        except Exception:
            ord_idx = np.argsort(np.var(self.train_X.values.astype(float), axis=0))[::-1]
        self.top_feat_idx = ord_idx[: max(1, self.top_features)]

        return self

    # ------------------- Predict ---------------------
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        logger.info(f"StackedScoreLLM: predicting {len(X)} samples (thread-pooled)")
        rows = list(X.reset_index(drop=True).itertuples(index=False))
        cpu_count = multiprocessing.cpu_count() or 1
        max_workers = min(32, max(1, cpu_count - 1))

        preds = [None] * len(rows)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = { pool.submit(self._predict_one, row): i for i, row in enumerate(rows) }
            for done, fut in enumerate(as_completed(futures), start=1):
                i = futures[fut]
                preds[i] = int(fut.result())
                if done % max(1, len(rows)//10) == 0 or done == len(rows):
                    logger.info(f"StackedScoreLLM progress: {done}/{len(rows)} ({done/len(rows)*100:.1f}%)")
        return np.array(preds, dtype=int)

    # ----------------- Internals ---------------------
    def _ensemble_proba(self, Z: np.ndarray) -> np.ndarray:
        labels = self.labels_
        # get aligned probs for each learner, calibrated if available
        p_lr  = _align_proba(self.lr_cal  or self.lr,  Z, labels)
        p_hgb = _align_proba(self.hgb_cal or self.hgb, Z, labels)
        p_rf  = _align_proba(self.rf_cal  or self.rf,  Z, labels)
        p_knn = _align_proba(self.knn_cal or self.knn, Z, labels)

        w = self.weights
        p = w["lr"]*p_lr + w["hgb"]*p_hgb + w["rf"]*p_rf + w["knn"]*p_knn
        # renormalize in case of tiny numerical drift
        p = np.clip(p, 1e-12, 1.0)
        p = p / p.sum(axis=1, keepdims=True)
        return p, {"lr": p_lr, "hgb": p_hgb, "rf": p_rf, "knn": p_knn}

    def _class_evidence(self, z: np.ndarray, proba_row: np.ndarray, comps: Dict[str, np.ndarray]) -> List[Dict[str, float]]:
        rows = []
        for j, c in enumerate(self.labels_):
            st = self.stats[c]
            d_centroid = float(np.linalg.norm(z - st.mu))
            sim = -d_centroid  # higher is better
            rows.append({
                "class": int(c),
                "ens_proba": float(proba_row[j]),
                "lr_proba": float(comps["lr"][0, j]),
                "hgb_proba": float(comps["hgb"][0, j]),
                "rf_proba": float(comps["rf"][0, j]),
                "knn_proba": float(comps["knn"][0, j]),
                "centroid_sim": sim,
                "prior": float(st.prior),
            })
        return rows

    def _predict_one(self, row_tuple) -> int:
        x = np.array(row_tuple, dtype=float)
        z = self.scaler.transform(x[None, :])[0][None, :]  # shape (1, d)

        # ensemble probabilities and components
        p_ens, comps = self._ensemble_proba(z)
        p_row = p_ens[0]
        rec_idx = int(np.argmax(p_row))
        rec_class = int(self.labels_[rec_idx])

        # second best & margin
        order = np.argsort(-p_row)
        best, second = order[0], order[1] if len(order) > 1 else order[0]
        margin = float(p_row[best] - p_row[second])
        top_classes = [int(self.labels_[best])]
        if second != best:
            top_classes.append(int(self.labels_[second]))

        # evidence for all classes (we will pack only the top ones)
        ev_all = self._class_evidence(z[0], p_row, comps)

        # build compact scoreboard (only top-2)
        ev_top = [e for e in ev_all if e["class"] in top_classes]
        scoreboard = {
            "recommended_class": rec_class,
            "margin": round(margin, 6),
            "per_class": ev_top
        }

        # prototypes: nearest training row of each top class (top features only, z-values)
        protos = []
        for c in top_classes:
            idxs = self.stats[c].idxs
            Zc = self.Xz[idxs]
            jloc = _nearest_indices(z[0], Zc, k=self.proto_k)[0]
            gidx = int(idxs[jloc])
            zproto = self.Xz[gidx]
            proto = {
                "label": int(c),
                "features": { f"f{int(i)}": float(zproto[int(i)]) for i in self.top_feat_idx }
            }
            protos.append(proto)

        # top feature values for the test row (z-space) for transparency
        zvals = { f"f{int(i)}": float(z[0, int(i)]) for i in self.top_feat_idx }
        scoreboard["test_row_top_features"] = zvals

        # LLM call (always called; in high-margin cases it's instructed to confirm)
        out = self.chain.invoke({
            "scoreboard": json.dumps(scoreboard, ensure_ascii=False),
            "prototypes_block": json.dumps(protos, ensure_ascii=False),
        })

        # robust parse; default to recommendation on failure or off-manifold labels
        lab = None
        for _ in range(2):
            try:
                lab = int(self.parser.parse(_json_extract(out)).classification)
                break
            except Exception:
                m = re.search(r'"classification"\s*:\s*(-?\d+)', str(out))
                if m: lab = int(m.group(1)); break

        if (lab is None) or (lab not in self.labels_):
            lab = rec_class
        return lab
