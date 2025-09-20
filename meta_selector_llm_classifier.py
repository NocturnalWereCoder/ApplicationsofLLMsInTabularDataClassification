# meta_selector_llm_classifier.py

import os
import re
import json
import logging
from typing import Dict, List, Any, Tuple, Optional, Literal

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score

from pydantic import BaseModel, Field, validator
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama.llms import OllamaLLM

from config import LLM_MODEL

logger = logging.getLogger(__name__)

# ---------------- Utilities --------------------
def _clamp_int(x, lo, hi, default):
    try:
        xi = int(x)
        if xi < lo or xi > hi:
            return default
        return xi
    except Exception:
        return default

def _clamp_float(x, lo, hi, default):
    try:
        xf = float(x)
        if xf < lo or xf > hi:
            return default
        return xf
    except Exception:
        return default

def _maybe_none(v):
    if v is None:
        return None
    if isinstance(v, str) and v.strip().lower() in {"none", "null"}:
        return None
    return v

def _extract_json_objects(text: str) -> List[str]:
    """
    Extract ALL top-level JSON objects { ... } from an arbitrary string by bracket balancing.
    Robust against code fences and extra prose. Returns raw substrings (not parsed).
    """
    objs = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    objs.append(text[start:i+1])
                    start = -1
    return objs

def _looks_like_metadata_json(j: Dict[str, Any]) -> bool:
    """Detect the dataset-metadata blob that's being echoed by some models."""
    meta_keys = {"target_column_index", "num_features", "num_samples", "feature_indices"}
    return isinstance(j, dict) and (meta_keys & set(j.keys())) != set()

# ---------------- Selection schema ----------------
ALLOWED_MODELS = ["lr", "rf", "hgb", "knn"]

# Per-model allowed hyperparameters (with safe bounds)
HP_BOUNDS = {
    "lr": {
        "C":            ("float", 1e-3, 1e3, 1.0),
        "class_weight": ("cat",   ["balanced", None], None, None),
    },
    "rf": {
        "n_estimators":      ("int",      50, 2000, 600),
        "max_depth":         ("int|none",  1,  100, None),
        "min_samples_split": ("int",       2,   50, 2),
        "min_samples_leaf":  ("int",       1,   50, 1),
    },
    "hgb": {
        "learning_rate":     ("float", 1e-4, 1.0, 0.06),
        "max_iter":          ("int",     50, 2000, 400),
        "max_depth":         ("int|none", 1,  100, None),
        "l2_regularization": ("float",   0.0, 10.0, 0.0),
    },
    "knn": {
        "n_neighbors": ("int", 3, 100, 31),  # upper clamp adjusted by sqrt(n) later
        "weights":     ("cat", ["uniform", "distance"], None, "distance"),
    },
}

class ModelSelectionPlan(BaseModel):
    """Legacy single-choice plan (kept for compatibility/repair)."""
    model: Literal["lr", "rf", "hgb", "knn"]
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    reasoning: str = "default"

    @validator("hyperparameters", pre=True)
    def ensure_dict(cls, v):
        return v or {}

class ModelCandidate(BaseModel):
    model: Literal["lr", "rf", "hgb", "knn"]
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    why: str = ""

class ModelSelectionSet(BaseModel):
    candidates: List[ModelCandidate]


# ---------------- Classifier -------------------
class MetaSelectorLLMClassifier:
    """
    LLM-assisted Model Selection for Tabular Data

    Flow:
      1) Build dataset schema + include first ≥100 rows; include full metadata.json (attempt 1)
      2) Ask LLM: return 3–6 diverse (model, hyperparameters) candidates (UNBIASED)
      3) Parse (retry with a **minimal** schema-only prompt that forbids metadata echo)
      4) If still no valid JSON -> safe default candidate set
      5) Locally quick CV over candidates; pick best
      6) Predict with single chosen model
    """

    def __init__(
        self,
        cuda_device: int = 0,
        oof_folds: int = 5,              # accepted for compat, unused
        region_k: int = 31,              # accepted for compat, unused
        top_features: int = 10,          # accepted for compat, unused
        prototypes_per_class: int = 1,   # accepted for compat, unused
        engage_threshold: float = 0.08,  # accepted for compat, unused
        calibrate: bool = True,
        llm_temperature: float = 0.0,
        llm_top_p: float = 1.0,
        schema_sample_n: int = 100,      # enforce ≥100 rows to the LLM
        max_parse_attempts: int = 5,     # retry LLM parsing before fallback
        cv_folds: int = 3,               # local CV to pick best candidate
        cv_max_rows: int = 50000,        # cap rows for quick CV
        random_state: int = 42
    ):
        self.calibrate = bool(calibrate)
        self.schema_sample_n = int(max(100, schema_sample_n))
        self.max_parse_attempts = int(max(1, max_parse_attempts))
        self.cv_folds = int(max(2, cv_folds))
        self.cv_max_rows = int(max(2000, cv_max_rows))
        self.random_state = int(random_state)

        self.scaler = StandardScaler()

        self.labels_: List[int] = []
        self.cols: List[Any] = []
        self.train_X: pd.DataFrame = None
        self.train_y: pd.Series = None
        self.Xz: np.ndarray = None

        self.selected_model_: Optional[str] = None
        self.selected_params_: Dict[str, Any] = {}
        self.reason_: str = ""
        self.selection_summary_: str = ""
        self.metadata_json_: Optional[str] = None

        self.est_: Any = None
        self.est_cal_: Any = None

        os.environ["OLLAMA_CUDA_DEVICE"] = str(cuda_device)
        self.llm = OllamaLLM(
            model=LLM_MODEL,
            temperature=float(llm_temperature),
            top_p=float(llm_top_p),
            in_process=True
        )

    # ---------------------- Fit ----------------------
    def fit(self, X: pd.DataFrame, y: pd.Series, metadata: Optional[Dict[str, Any]] = None):
        logger.info("MetaSelectorLLM: start fit()")

        self.cols = list(X.columns)
        self.train_X = X.reset_index(drop=True).copy()
        self.train_y = y.reset_index(drop=True).copy()
        self.labels_ = sorted(pd.unique(self.train_y).tolist())

        # 1) Build schema card & pack first ≥100 rows
        schema_card = self._build_schema_card(self.train_X, self.train_y)

        try:
            self.metadata_json_ = json.dumps(metadata, ensure_ascii=False) if metadata is not None else "null"
        except Exception:
            self.metadata_json_ = "null"

        # Standardize once (used for CV and final model)
        self.Xz = self.scaler.fit_transform(self.train_X.values.astype(float))

        # 2) Ask LLM for MULTIPLE candidates; locally pick best via CV
        cand_set = self._llm_propose_candidates(schema_card, self.train_X, self.train_y, self.metadata_json_)
        candidates = self._augment_with_baselines(cand_set.candidates, len(self.train_X), self.train_y)
        best_model, best_params, leaderboard = self._evaluate_candidates_cv(candidates)

        self.selected_model_ = best_model
        self.selected_params_ = best_params
        self.reason_ = "local_cv_best_of_candidates"
        self.selection_summary_ = self._format_selection() + f" | leaderboard={json.dumps(leaderboard)}"
        logger.info(f"Selected via local CV: {self.selection_summary_}")

        # 3) Instantiate and fit chosen estimator
        self.est_ = self._make_estimator(self.selected_model_, self.selected_params_)
        self.est_.fit(self.Xz, self.train_y.values)

        # Optional probability calibration
        self.est_cal_ = None
        if self.calibrate:
            method = "isotonic" if len(self.train_X) >= 2000 else "sigmoid"
            try:
                self.est_cal_ = CalibratedClassifierCV(
                    self.est_, method=method, cv=3 if len(self.train_X) >= 1000 else 5
                )
                self.est_cal_.fit(self.Xz, self.train_y.values)
            except Exception as e:
                logger.warning(f"Calibration failed ({method}); proceeding without. Error: {e}")
                self.est_cal_ = None

        logger.info("MetaSelectorLLM: fit() complete")
        return self

    # ------------------- Predict ---------------------
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self.est_ is not None, "Model not fitted yet."
        Z = self.scaler.transform(X.reset_index(drop=True).values.astype(float))
        est = self.est_cal_ if self.est_cal_ is not None else self.est_
        if hasattr(est, "predict"):
            return est.predict(Z)
        if hasattr(est, "predict_proba"):
            proba = self._align_proba(est, Z, self.labels_)
            return np.array([self.labels_[int(np.argmax(p))] for p in proba], dtype=int)
        raise RuntimeError("Selected estimator does not support prediction.")

    # ------------------- LLM step --------------------
    def _llm_propose_candidates(
        self,
        schema_card_json: str,
        X: pd.DataFrame,
        y: pd.Series,
        metadata_json: Optional[str],
    ) -> ModelSelectionSet:
        """
        Ask the LLM for 3–6 diverse candidate (model, hyperparameters) pairs (UNBIASED).
        Robustly parse/repair outputs; if parsing fails after retries, return a default set.
        """
        n_show = self.schema_sample_n
        joint = X.copy()
        joint["__target__"] = y.values
        head_rows: List[Dict[str, Any]] = joint.head(n_show).astype(object).to_dict(orient="records")

        model_catalog = {
            "lr": {
                "desc": "LogisticRegression (lbfgs, multi_class=auto)",
                "hyperparameters": {
                    "C": "float in [1e-3, 1e3] (default 1.0)",
                    "class_weight": "one of ['balanced', null] (optional)"
                }
            },
            "rf": {
                "desc": "RandomForestClassifier",
                "hyperparameters": {
                    "n_estimators": "int in [50, 2000] (trees)",
                    "max_depth": "int in [1,100] or null for unlimited",
                    "min_samples_split": "int in [2, 50]",
                    "min_samples_leaf": "int in [1, 50]"
                }
            },
            "hgb": {
                "desc": "HistGradientBoostingClassifier",
                "hyperparameters": {
                    "learning_rate": "float in [1e-4, 1.0]",
                    "max_iter": "int in [50, 2000]",
                    "max_depth": "int in [1,100] or null",
                    "l2_regularization": "float in [0.0, 10.0]"
                }
            },
            "knn": {
                "desc": "KNeighborsClassifier",
                "hyperparameters": {
                    "n_neighbors": "int in [3, min(100, round(3*sqrt(n_rows)))]",
                    "weights": "one of ['uniform','distance']"
                }
            }
        }

        parser = PydanticOutputParser(pydantic_object=ModelSelectionSet)

        example = {
            "candidates": [
                {"model": "hgb", "hyperparameters": {"learning_rate": 0.06, "max_iter": 400, "max_depth": None, "l2_regularization": 0.0}, "why": "nonlinear robust"},
                {"model": "rf",  "hyperparameters": {"n_estimators": 600, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1}, "why": "bagging trees"},
                {"model": "lr",  "hyperparameters": {"C": 1.0, "class_weight": None}, "why": "linear baseline"}
            ]
        }

        # Full-context prompt (attempt 1)
        full_prompt = ChatPromptTemplate.from_template("""
You are a JSON planner. Output ONLY one JSON object with top-level key "candidates". No prose, no markdown, no code fences.

STRICTLY FORBIDDEN: Do NOT output any keys named
["target_column_index","num_features","num_samples","feature_indices"] or any dataset metadata structure.

JSON Schema (required):
{schema}

Hard constraints:
- Produce 4–5 diverse candidates. Allowed models: ["lr","rf","hgb","knn"].
- Every candidate MUST have exactly: "model", "hyperparameters", "why".
- "hyperparameters" keys MUST come from model_catalog for that model; clamp numeric values to bounds.
- No extra fields. No comments. No trailing commas. Use null literally. No NaN/Infinity.
- Keep each "why" ≤ 12 words.

Selection policy (use the dataset summary below):
- Always include at least one of EACH family: "hgb", "rf", and "knn". Include exactly one linear "lr" baseline.
- If n_rows ≥ 50_000 (large dataset)(You might be better off selecting RF or KNN in this case), include HIGH-CAPACITY configs:
  • KNN: weights="distance", n_neighbors in [80,100] (use 97 if unsure).
  • RF: n_estimators in [1200,1600] (use 1400), max_depth=null, min_samples_split=2, min_samples_leaf=1.
  • HGB (robust): max_iter in [800,1200] (use 1000), learning_rate in [0.05,0.12] (use 0.08), max_depth=null, l2_regularization in [0.0,2.0] (use 0.0).
  • Optionally add a second HGB variant with learning_rate≈0.10 and max_depth≈16, l2_regularization≈1.0.
- If n_rows < 50_000, still ensure one of each family; pick moderate settings.
- For LR, set class_weight="balanced" ONLY if the largest class prior ≥ 0.75; otherwise null. Choose C in [0.5,3.0] (use 1.0).

Example shape (structure only; tune values per policy):
{example}

Model catalog (authoritative):
{model_catalog}

Dataset summary (read-only, includes n_rows and class_priors):
{schema_card}

First rows (≥100, features + "__target__"):
{first_rows}

Dataset metadata (verbatim for reference; DO NOT ECHO):
{metadata_json}

Return ONLY the final JSON object with "candidates".
""".strip())



        # Minimal prompt (attempts 2..N) — no metadata, schema-only, forbids echo
        minimal_prompt = ChatPromptTemplate.from_template("""
Return ONLY this JSON shape (no prose, no fences). Produce 5 candidates covering HGB, RF, KNN, a second HGB variant, and one LR baseline:

{{
  "candidates": [
    {{"model":"hgb","hyperparameters":{{"learning_rate":0.08,"max_iter":1000,"max_depth":null,"l2_regularization":0.0}},"why":"robust gradient boosting"}},
    {{"model":"rf","hyperparameters":{{"n_estimators":1400,"max_depth":null,"min_samples_split":2,"min_samples_leaf":1}},"why":"large random forest"}},
    {{"model":"knn","hyperparameters":{{"n_neighbors":97,"weights":"distance"}},"why":"high-k distance voting"}},
    {{"model":"hgb","hyperparameters":{{"learning_rate":0.10,"max_iter":800,"max_depth":16,"l2_regularization":1.0}},"why":"faster boosted trees"}},
    {{"model":"lr","hyperparameters":{{"C":1.0,"class_weight":null}},"why":"linear baseline"}}
  ]
}}

Constraints:
- Models ONLY from ["lr","rf","hgb","knn"].
- Hyperparameter keys ONLY from model_catalog for that model; values will be clamped to bounds.
- Forbid keys: ["target_column_index","num_features","num_samples","feature_indices"].
- No comments, no trailing commas, use null literally, no NaN/Infinity.
- Keep each "why" ≤ 12 words.
                                                          
Guidance:
- If the dataset is large (e.g., n_rows ≥ 50,000), lean toward Random Forest and KNN as stronger defaults for scalability (include both as above).

                                                          
JSON Schema (required):
{schema}

Model catalog (authoritative):
{model_catalog}
""".strip())



        def _invoke_first() -> str:
            return (full_prompt | self.llm).invoke({
                "schema": parser.pydantic_object.schema_json(indent=None),
                "example": json.dumps(example, ensure_ascii=False),
                "model_catalog": json.dumps(model_catalog, ensure_ascii=False),
                "schema_card": schema_card_json,
                "first_rows": json.dumps(head_rows, ensure_ascii=False),
                "metadata_json": metadata_json if metadata_json is not None else "null",
            })

        def _invoke_minimal() -> str:
            return (minimal_prompt | self.llm).invoke({
                "schema": parser.pydantic_object.schema_json(indent=None),
                "model_catalog": json.dumps(model_catalog, ensure_ascii=False),
            })

        # Attempt 1: full context (includes metadata)
        out = _invoke_first()

        # Robust parsing/repair with retries; attempts >=2 use minimal prompt (no metadata)
        last_err = None
        for attempt in range(1, self.max_parse_attempts + 1):
            try:
                # Consider ALL JSON blocks; ignore ones that look like metadata
                blocks = _extract_json_objects(out) or []
                parsed_blocks = []
                for raw in blocks:
                    try:
                        data = json.loads(raw)
                        if _looks_like_metadata_json(data):
                            continue
                        parsed_blocks.append(data)
                    except Exception:
                        continue

                # If nothing useful found, try whole string as JSON
                if not parsed_blocks:
                    try:
                        data = json.loads(out)
                        if not _looks_like_metadata_json(data):
                            parsed_blocks = [data]
                    except Exception:
                        pass

                # Try to coerce to ModelSelectionSet
                for data in parsed_blocks:
                    # Direct candidates dict
                    if isinstance(data, dict) and "candidates" in data and isinstance(data["candidates"], list):
                        cands = [ModelCandidate(**c) for c in data["candidates"] if isinstance(c, dict) and "model" in c]
                        if cands:
                            return ModelSelectionSet(candidates=cands)
                    # Single plan -> wrap
                    if isinstance(data, dict) and {"model", "hyperparameters"} <= set(data.keys()):
                        c = ModelCandidate(model=str(data["model"]),
                                           hyperparameters=data.get("hyperparameters", {}),
                                           why=str(data.get("reasoning", ""))[:64])
                        return ModelSelectionSet(candidates=[c])
                    # Bare list of candidates
                    if isinstance(data, list):
                        cands = []
                        for item in data:
                            if isinstance(item, dict) and "model" in item:
                                cands.append(ModelCandidate(
                                    model=str(item["model"]),
                                    hyperparameters=item.get("hyperparameters", {}),
                                    why=str(item.get("why", ""))[:64]
                                ))
                        if cands:
                            return ModelSelectionSet(candidates=cands)

                raise ValueError("no valid candidate structure found")
            except Exception as e:
                last_err = e
                short = (str(out)[:400] + "…") if out and len(str(out)) > 400 else str(out)
                logger.warning(f"LLM candidates parse failed (attempt {attempt}/{self.max_parse_attempts}): {e} | raw={short}")
                # Re-ask with the minimal, schema-only prompt after the first failure
                out = _invoke_first()
        else:
            out = _invoke_minimal()

        logger.error("LLM candidates failed after retries; using safe default set including HGB.")
        return ModelSelectionSet(candidates=[
            ModelCandidate(model="hgb", hyperparameters={"learning_rate": 0.06, "max_iter": 600, "max_depth": None, "l2_regularization": 0.0}, why="robust default"),
            ModelCandidate(model="rf",  hyperparameters={"n_estimators": 600, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1}, why="deeper rf"),
            ModelCandidate(model="rf",  hyperparameters={"n_estimators": 1000, "max_depth": 20, "min_samples_split": 2, "min_samples_leaf": 1}, why="more trees"),
            ModelCandidate(model="lr",  hyperparameters={"C": 1.0, "class_weight": None}, why="linear baseline"),
            ModelCandidate(model="knn", hyperparameters={"n_neighbors": 31, "weights": "distance"}, why="distance votes"),
        ])

    # ------------- Candidate augmentation + local CV selection -------------
    def _augment_with_baselines(self, cands: List[ModelCandidate], n_rows: int, y: pd.Series) -> List[ModelCandidate]:
        """Ensure strong baselines, sanitize hyperparameters, and de-duplicate."""
        def key(c: ModelCandidate) -> Tuple[str, str]:
            try:
                hp = json.dumps(self._sanitize_params(c.model, c.hyperparameters, n_rows), sort_keys=True)
            except Exception:
                hp = json.dumps({}, sort_keys=True)
            return (c.model, hp)

        y_series = pd.Series(y).reset_index(drop=True)
        max_prior = float(y_series.value_counts(normalize=True).max())

        baselines = [
            ModelCandidate(model="rf",  hyperparameters={"n_estimators": 600, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1}, why="deeper rf"),
            ModelCandidate(model="rf",  hyperparameters={"n_estimators": 1000, "max_depth": 20, "min_samples_split": 2, "min_samples_leaf": 1}, why="more trees"),
            ModelCandidate(model="hgb", hyperparameters={"learning_rate": 0.06, "max_iter": 600, "max_depth": None, "l2_regularization": 0.0}, why="longer hgb"),
            ModelCandidate(model="hgb", hyperparameters={"learning_rate": 0.1,  "max_iter": 400, "max_depth": 16, "l2_regularization": 1.0}, why="faster hgb"),
            ModelCandidate(model="lr",  hyperparameters={"C": 1.0, "class_weight": ("balanced" if max_prior >= 0.7 else None)}, why="linear check"),
            ModelCandidate(model="knn", hyperparameters={"n_neighbors": min(51, max(5, int(np.sqrt(max(n_rows, 1)) * 2))), "weights": "distance"}, why="scale k"),
        ]

        pool = list(cands) + baselines
        seen = set()
        uniq: List[ModelCandidate] = []
        for c in pool:
            k = key(c)
            if k in seen:
                continue
            seen.add(k)
            c.hyperparameters = self._sanitize_params(c.model, c.hyperparameters, n_rows)
            uniq.append(c)
        return uniq

    def _evaluate_candidates_cv(self, candidates: List[ModelCandidate]) -> Tuple[str, Dict[str, Any], List[Tuple[str, float]]]:
        """Cross-validate each candidate quickly and pick the best by mean accuracy."""
        y_series = self.train_y.reset_index(drop=True)
        if len(y_series) > self.cv_max_rows:
            frac = self.cv_max_rows / float(len(y_series))
            rng = np.random.RandomState(self.random_state)
            idx_parts = []
            for _, idxs in y_series.groupby(y_series).groups.items():
                idxs = np.array(list(idxs))
                k = max(1, int(round(len(idxs) * frac)))
                idx_parts.append(rng.choice(idxs, size=k, replace=False))
            idx = np.concatenate(idx_parts)
            Xcv = self.Xz[idx]
            ycv = y_series.values[idx]
        else:
            Xcv = self.Xz
            ycv = y_series.values

        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        scores: List[Tuple[str, float]] = []
        best = ("", -1.0, {})

        for c in candidates:
            try:
                est = self._make_estimator(c.model, c.hyperparameters)
                sc = cross_val_score(est, Xcv, ycv, cv=skf, scoring="accuracy", n_jobs=-1).mean()
                scores.append((f"{c.model}:{json.dumps(c.hyperparameters, sort_keys=True)}", float(sc)))
                if sc > best[1]:
                    best = (c.model, sc, c.hyperparameters)
            except Exception as e:
                logger.warning(f"CV failed for candidate {c.model} {c.hyperparameters}: {e}")

        scores.sort(key=lambda x: x[1], reverse=True)
        leaderboard = [(s.split(":", 1)[0], float(v)) for s, v in scores[:6]]
        if best[0] == "":
            return "hgb", {"learning_rate": 0.06, "max_iter": 600, "max_depth": None, "l2_regularization": 0.0}, leaderboard
        return best[0], best[2], leaderboard

    # ------------------- Helpers ---------------------
    def _sanitize_params(self, model: str, hp: Dict[str, Any], n_rows: int) -> Dict[str, Any]:
        """Keep only allowed keys, clamp to safe ranges, set sensible defaults."""
        if model not in HP_BOUNDS:
            return {}
        spec = HP_BOUNDS[model]
        out: Dict[str, Any] = {}
        for k, v in (hp or {}).items():
            if k not in spec:
                continue
            kind, a, b, default = spec[k]
            if kind == "cat":
                allowed = a
                out[k] = v if v in allowed else default
            elif kind == "int":
                out[k] = _clamp_int(v, a, b, default)
            elif kind == "float":
                out[k] = _clamp_float(v, a, b, default)
            elif kind == "int|none":
                if v in (None, "null", "None"):
                    out[k] = None
                else:
                    out[k] = _clamp_int(v, a, b, default)
            else:
                pass

        # Fill any missing with defaults
        for k, (kind, a, b, default) in spec.items():
            if k not in out:
                out[k] = default

        # Dataset-size specific clamp for KNN
        if model == "knn":
            max_k = max(3, min(100, int(np.sqrt(max(n_rows, 1)) * 3)))
            out["n_neighbors"] = _clamp_int(out.get("n_neighbors", 31), 3, max_k, min(31, max_k))
            out["weights"] = out.get("weights", "distance")

        return out

    def _format_selection(self) -> str:
        """Compact, one-line description of the chosen model + hyperparameters."""
        if self.selected_model_ is None:
            return "model=<unfit>, hp={}"
        return f"model={self.selected_model_}, hp={json.dumps(self.selected_params_, sort_keys=True)}"

    def _make_estimator(self, model: str, params: Dict[str, Any]):
        """Instantiate sklearn estimator by name + sanitized params."""
        if model == "lr":
            return LogisticRegression(
                C=float(params.get("C", 1.0)),
                penalty="l2",
                solver="lbfgs",
                max_iter=1000,
                multi_class="auto",
                n_jobs=None,
                class_weight=_maybe_none(params.get("class_weight", None)),
            )
        if model == "rf":
            return RandomForestClassifier(
                n_estimators=int(params.get("n_estimators", 600)),
                max_depth=_maybe_none(params.get("max_depth", None)),
                min_samples_split=int(params.get("min_samples_split", 2)),
                min_samples_leaf=int(params.get("min_samples_leaf", 1)),
                n_jobs=-1,
                random_state=self.random_state
            )
        if model == "hgb":
            return HistGradientBoostingClassifier(
                max_depth=_maybe_none(params.get("max_depth", None)),
                learning_rate=float(params.get("learning_rate", 0.06)),
                max_iter=int(params.get("max_iter", 400)),
                l2_regularization=float(params.get("l2_regularization", 0.0)),
                random_state=self.random_state
            )
        if model == "knn":
            return KNeighborsClassifier(
                n_neighbors=int(params.get("n_neighbors", 31)),
                weights=str(params.get("weights", "distance")),
            )
        raise ValueError(f"Unsupported model: {model}")

    def _align_proba(self, estimator, Xz: np.ndarray, labels_order: List[int]) -> np.ndarray:
        """Return probabilities aligned to integer labels_order (utility if needed)."""
        proba = estimator.predict_proba(Xz)
        cls = list(estimator.classes_)
        out = np.zeros((Xz.shape[0], len(labels_order)), dtype=float)
        for i, c in enumerate(labels_order):
            if c in cls:
                out[:, i] = proba[:, cls.index(c)]
            else:
                out[:, i] = 0.0
        out = np.clip(out, 1e-12, 1.0)
        out = out / out.sum(axis=1, keepdims=True)
        return out

    def _build_schema_card(self, X: pd.DataFrame, y: pd.Series) -> str:
        """Dataset summary: shape, dtypes, missing %, unique counts, class priors."""
        n, d = X.shape
        info = {
            "n_rows": int(n),
            "n_cols": int(d),
            "labels": [int(c) for c in sorted(pd.unique(y))],
            "class_priors": { int(c): float((y == c).mean()) for c in sorted(pd.unique(y)) }
        }
        cols = []
        for j, col in enumerate(X.columns):
            s = X[col]
            miss = float(s.isna().mean()) if hasattr(s, "isna") else 0.0
            try:
                uniq = int(s.nunique(dropna=True))
            except Exception:
                uniq = None
            cols.append({
                "idx": int(j),
                "name": str(col),
                "dtype": str(s.dtype),
                "unique": uniq,
                "missing_pct": round(miss, 6)
            })
        info["columns"] = cols
        return json.dumps(info, ensure_ascii=False)


