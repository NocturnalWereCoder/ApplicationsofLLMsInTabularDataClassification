# llm_guided_hpo_classifier.py
"""
LLM-guided Hyperparameter Optimization for tabular classification.

How this preserves LLM centrality (for research validity):
- The LLM returns a JSON "plan" that specifies:
  * which model families to search (subset of {"hgb","rf","lr","knn"})
  * which metric to optimize (e.g., accuracy, balanced_accuracy, roc_auc)
  * per-family search spaces: typed ranges like {"type":"loguniform","low":1e-3,"high":0.2}
- We strictly convert THE LLM'S PLAN to concrete param grids and execute HPO.
- If the plan is malformed, we retry with a minimal prompt; if still bad,
  we fall back to a tiny default *LLM-ish* plan, and we label that in logs.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Literal

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

# Prefer HalvingRandomSearchCV; fall back to RandomizedSearchCV
try:
    from sklearn.experimental import enable_halving_search_cv  # noqa: F401
    from sklearn.model_selection import HalvingRandomSearchCV as _SearchCV
    _SEARCH_KIND = "halving_random"
except Exception:
    from sklearn.model_selection import RandomizedSearchCV as _SearchCV
    _SEARCH_KIND = "random"

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

from config import LLM_MODEL

logger = logging.getLogger(__name__)

# -------------------- utility helpers --------------------

def _looks_like_metadata_json(d: Any) -> bool:
    if not isinstance(d, dict):
        return False
    meta_keys = {"target_column_index", "num_features", "num_samples", "feature_indices"}
    return bool(meta_keys & set(d.keys()))

def _extract_json_blocks(text: str) -> List[str]:
    """Extract top-level {...} JSON blocks from arbitrary text via brace balance."""
    blocks, depth, start = [], 0, -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    blocks.append(text[start:i+1])
                    start = -1
    return blocks

def _json_load_first_plan(text: str) -> Optional[Dict[str, Any]]:
    """Find the first JSON object that looks like a plan (has 'families' or 'space')."""
    for raw in _extract_json_blocks(text) or []:
        try:
            data = json.loads(raw)
            if _looks_like_metadata_json(data):
                continue
            if isinstance(data, dict) and ("families" in data or "space" in data or "models" in data):
                return data
        except Exception:
            continue
    # Try whole string
    try:
        data = json.loads(text)
        if isinstance(data, dict) and ("families" in data or "space" in data or "models" in data):
            return data if not _looks_like_metadata_json(data) else None
    except Exception:
        pass
    return None

def _safe_choice(vals: List[Any], k: int) -> List[Any]:
    vals = list(dict.fromkeys(vals))  # dedup, stable
    if not vals:
        return []
    if k >= len(vals):
        return vals
    rng = np.random.RandomState(42)
    idx = rng.choice(len(vals), size=k, replace=False)
    return [vals[i] for i in idx]

# -------------------- main class --------------------

class LLMGuidedHPOClassifier:
    """
    End-to-end LLM-guided HPO over {HGB, RF, LR, KNN} for tabular classification.

    After fit():
        selected_model_: str
        selected_params_: Dict[str, Any]
        selection_summary_: str
        leaderboard_: List[Tuple[str, float]]
        llm_plan_used_: bool  # whether we used an LLM-produced plan (vs safe fallback)
        llm_metric_: str      # metric chosen/confirmed by LLM (or our fallback)
    """

    def __init__(
        self,
        cuda_device: int = 0,
        schema_sample_n: int = 100,
        max_parse_attempts: int = 3,
        cv_folds: int = 3,
        row_cap: int = 50_000,                   # cap rows for CV speed
        n_iter_per_family: int = 40,             # HPO breadth per family
        random_state: int = 42,
        n_jobs: int = -1,
        calibrate: bool = True,
        verbose: int = 0,
    ):
        self.schema_sample_n = max(100, int(schema_sample_n))
        self.max_parse_attempts = max(1, int(max_parse_attempts))
        self.cv_folds = max(2, int(cv_folds))
        self.row_cap = max(2000, int(row_cap))
        self.n_iter_per_family = max(10, int(n_iter_per_family))
        self.random_state = int(random_state)
        self.n_jobs = int(n_jobs)
        self.calibrate = bool(calibrate)
        self.verbose = int(verbose)

        # Will fill on fit
        self.selected_model_: Optional[str] = None
        self.selected_params_: Dict[str, Any] = {}
        self.selection_summary_: str = ""
        self.leaderboard_: List[Tuple[str, float]] = []
        self.llm_plan_used_: bool = False
        self.llm_metric_: str = ""

        self.best_est_ = None
        self.best_calibrated_ = None

        os.environ["OLLAMA_CUDA_DEVICE"] = str(cuda_device)
        self.llm = OllamaLLM(
            model=LLM_MODEL,
            temperature=0.0,  # deterministic structure
            top_p=1.0,
            in_process=True
        )

    # -------------------- public API --------------------

    def fit(self, X: pd.DataFrame, y: pd.Series, metadata: Optional[Dict[str, Any]] = None):
        X = X.reset_index(drop=True).copy()
        y = y.reset_index(drop=True).copy()

        # Compact schema
        schema = self._build_schema_card(X, y)

        # First ≥100 rows including target
        n_show = self.schema_sample_n
        joint = X.copy()
        joint["__target__"] = y.values
        head_rows: List[Dict[str, Any]] = joint.head(n_show).astype(object).to_dict(orient="records")

        # Metadata verbatim
        try:
            metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata is not None else "null"
        except Exception:
            metadata_json = "null"

        # Ask LLM for a plan
        plan = self._ask_llm_for_plan(schema, head_rows, metadata_json)

        # If still None, construct tiny default plan (logged as fallback)
        if plan is None:
            self.llm_plan_used_ = False
            plan = self._default_plan(y)
            logger.error("LLM plan failed after retries; using safe default LLM-like plan.")
        else:
            self.llm_plan_used_ = True

        # Normalize + sanitize plan
        families, metric = self._normalize_plan(plan, y)

        # Downsample for CV speed (stratified)
        Xcv, ycv = self._maybe_downsample(X, y, self.row_cap)

        # CV splitter
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        # Run searches per family
        best_score = -np.inf
        best_tag = None
        best_search = None
        leaderboard: List[Tuple[str, float]] = []

        for fam in families:
            tag, search = self._build_search(fam, metric, skf)
            if search is None:
                continue
            try:
                if self.verbose:
                    logger.info(f"[LLM-Guided HPO] {_SEARCH_KIND} search for {tag} (metric={metric})")
                search.fit(Xcv, ycv)
                score = float(search.best_score_)
                leaderboard.append((tag, score))
                if self.verbose:
                    logger.info(f"[LLM-Guided HPO] {tag} best={score:.6f} params={search.best_params_}")
                if score > best_score:
                    best_score, best_tag, best_search = score, tag, search
            except Exception as e:
                logger.warning(f"[LLM-Guided HPO] search failed for {tag}: {e}")

        leaderboard.sort(key=lambda t: t[1], reverse=True)
        self.leaderboard_ = leaderboard
        self.llm_metric_ = metric

        # Safety fallback if everything failed
        if best_search is None:
            logger.error("[LLM-Guided HPO] All searches failed; falling back to default HGB.")
            self.selected_model_ = "hgb"
            self.selected_params_ = {
                "max_depth": None, "learning_rate": 0.06, "max_iter": 600, "l2_regularization": 0.0,
                "early_stopping": True, "validation_fraction": 0.1, "n_iter_no_change": 20,
                "random_state": self.random_state,
            }
            self.best_est_ = self._make_estimator(self.selected_model_, self.selected_params_)
            self.best_est_.fit(X.values, y.values)
            self._maybe_calibrate(X.values, y.values)
            self.selection_summary_ = self._format_selection(extra=f"metric={self.llm_metric_}|llm_plan={self.llm_plan_used_}")
            return self

        # Decode winning params
        model_key, flat_params = self._decode_best(best_tag, best_search.best_params_)
        self.selected_model_, self.selected_params_ = model_key, flat_params

        # Fit on full data
        self.best_est_ = self._make_estimator(model_key, flat_params)
        self.best_est_.fit(X.values, y.values)

        # Optional calibration
        self._maybe_calibrate(X.values, y.values)

        self.selection_summary_ = self._format_selection(
            extra=f"metric={self.llm_metric_}|llm_plan={self.llm_plan_used_}|leaderboard={json.dumps(self.leaderboard_[:6])}"
        )
        if self.verbose:
            logger.info(f"[LLM-Guided HPO] Selected {self.selection_summary_}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self.best_est_ is not None, "LLMGuidedHPOClassifier not fitted."
        Z = X.reset_index(drop=True).values
        est = self.best_calibrated_ if self.best_calibrated_ is not None else self.best_est_
        return est.predict(Z)

    # -------------------- LLM plan --------------------

    def _ask_llm_for_plan(self, schema_card: str, head_rows: List[Dict[str, Any]], metadata_json: str) -> Optional[Dict[str, Any]]:
        """
        Ask the LLM for a plan: { metric, families: [ {model, space, n_candidates?}, ... ] }
        Robust to metadata echo & prose; retries with a minimal prompt.
        """
        # Full prompt (attempt 1)
        full = ChatPromptTemplate.from_template("""
You design a JSON-only HPO plan. Output ONE JSON object with keys: "metric", "families".
STRICTLY FORBIDDEN: Do NOT output dataset metadata keys ["target_column_index","num_features","num_samples","feature_indices"].

Schema (must follow):
{{
  "metric": "accuracy" | "balanced_accuracy" | "roc_auc",
  "families": [
    {{
      "model": "hgb" | "rf" | "lr" | "knn",
      "space": {{
        // examples:
        // "learning_rate": {{"type":"loguniform","low":1e-3,"high":0.2}}
        // "max_iter": {{"type":"int","low":200,"high":1200,"step":100}}
        // "max_depth": {{"type":"choice","values":[null,8,12,16,24]}}
      }},
      "n_candidates": 40   // optional hint for breadth per family
    }}
  ]
}}

Guidelines:
- Choose 2–4 families likely to perform well for this dataset.
- Prefer "balanced_accuracy" if the dataset seems imbalanced; else "accuracy".
- Keep spaces reasonable (HGB, RF, LR, KNN examples below).
- Use null literally for null depth.
- No prose or code fences: return ONLY the JSON object.

Dataset summary (read-only):
{schema_card}

First rows (≥100, with "__target__"):
{first_rows}

Dataset metadata (verbatim; DO NOT ECHO):
{metadata_json}

Examples of valid space specs:
HGB: {{"learning_rate":{{"type":"loguniform","low":1e-3,"high":0.2}},"max_iter":{{"type":"int","low":200,"high":1200,"step":100}},"max_depth":{{"type":"choice","values":[null,8,12,16,24]}},"l2_regularization":{{"type":"choice","values":[0.0,0.1,1.0]}}}}
RF:  {{"n_estimators":{{"type":"int","low":300,"high":1500,"step":300}},"max_depth":{{"type":"choice","values":[null,10,20,40]}},"min_samples_leaf":{{"type":"choice","values":[1,2,4]}},"min_samples_split":{{"type":"choice","values":[2,5,10]}},"class_weight":{{"type":"choice","values":[null,"balanced"]}}}}
LR:  {{"C":{{"type":"loguniform","low":1e-3,"high":1e3}},"class_weight":{{"type":"choice","values":[null,"balanced"]}}}}
KNN: {{"n_neighbors":{{"type":"int","low":5,"high":61,"step":2}},"weights":{{"type":"choice","values":["distance","uniform"]}}}}
""".strip())

        # Minimal prompt (attempts 2+)
        minimal = ChatPromptTemplate.from_template("""
Return ONLY this JSON shape (no prose):
{{
  "metric": "accuracy" | "balanced_accuracy" | "roc_auc",
  "families": [
    {{"model":"hgb","space":{{}}}},
    {{"model":"rf","space":{{}}}}
  ]
}}
Populate "space" using the allowed spec types: "int" (low,high,step), "loguniform" (low,high), "choice" (values).
FORBID these keys anywhere: ["target_column_index","num_features","num_samples","feature_indices"].
""".strip())

        def _invoke_full() -> str:
            return (full | self.llm).invoke({
                "schema_card": schema_card,
                "first_rows": json.dumps(head_rows, ensure_ascii=False),
                "metadata_json": metadata_json,
            })

        def _invoke_min() -> str:
            return (minimal | self.llm).invoke({})

        out = _invoke_full()
        for attempt in range(1, self.max_parse_attempts + 1):
            plan = _json_load_first_plan(out)
            if isinstance(plan, dict) and "families" in plan:
                return plan
            # retry w/ minimal prompt
            out = _invoke_min()
        return None

    # -------------------- plan normalization --------------------

    def _default_plan(self, y: pd.Series) -> Dict[str, Any]:
        # heuristic metric
        metric = self._default_metric(y)
        return {
            "metric": metric,
            "families": [
                {"model": "hgb", "space": {
                    "learning_rate": {"type":"loguniform","low":1e-3,"high":0.2},
                    "max_iter": {"type":"int","low":200,"high":1200,"step":100},
                    "max_depth": {"type":"choice","values":[None,8,12,16,24]},
                    "l2_regularization": {"type":"choice","values":[0.0,0.1,1.0]}
                }, "n_candidates": 40},
                {"model": "rf", "space": {
                    "n_estimators": {"type":"int","low":300,"high":1500,"step":300},
                    "max_depth": {"type":"choice","values":[None,10,20,40]},
                    "min_samples_leaf": {"type":"choice","values":[1,2,4]},
                    "min_samples_split": {"type":"choice","values":[2,5,10]},
                    "class_weight": {"type":"choice","values":[None,"balanced"]}
                }, "n_candidates": 40},
                {"model": "lr", "space": {
                    "C": {"type":"loguniform","low":1e-3,"high":1e3},
                    "class_weight": {"type":"choice","values":[None,"balanced"]}
                }, "n_candidates": 30},
                {"model": "knn", "space": {
                    "n_neighbors": {"type":"int","low":5,"high":61,"step":2},
                    "weights": {"type":"choice","values":["distance","uniform"]}
                }, "n_candidates": 30},
            ]
        }

    def _default_metric(self, y: pd.Series) -> str:
        yv = pd.Series(y)
        if len(yv.unique()) == 2 and float(yv.value_counts(normalize=True).max()) >= 0.7:
            return "balanced_accuracy"
        return "accuracy"

    def _normalize_plan(self, plan: Dict[str, Any], y: pd.Series) -> Tuple[List[Dict[str, Any]], str]:
        # sanitize metric
        metric = str(plan.get("metric") or "").lower()
        if metric not in {"accuracy", "balanced_accuracy", "roc_auc"}:
            metric = self._default_metric(y)

        families_in = plan.get("families", [])
        families: List[Dict[str, Any]] = []
        seen = set()
        for fam in families_in:
            m = str(fam.get("model", "")).lower()
            if m not in {"hgb","rf","lr","knn"}:
                continue
            # dedup family tags
            if m in seen:
                continue
            seen.add(m)
            sp = fam.get("space", {}) or {}
            n_cand = int(fam.get("n_candidates", self.n_iter_per_family))
            families.append({"model": m, "space": sp, "n_candidates": n_cand})
        if not families:
            families = self._default_plan(y)["families"]
        return families, metric

    # -------------------- search building --------------------

    def _build_search(self, fam: Dict[str, Any], metric: str, cv) -> Tuple[str, Optional[_SearchCV]]:
        model = fam["model"]
        space = fam["space"]
        n_cand = max(10, int(fam.get("n_candidates", self.n_iter_per_family)))

        if model == "hgb":
            est = HistGradientBoostingClassifier(
                early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
                random_state=self.random_state
            )
            pdist = self._space_to_lists(space, {
                "learning_rate": ("loguniform", 1e-3, 0.2),
                "max_iter": ("int", 200, 1200, 100),
                "max_depth": ("choice", [None, 8, 12, 16, 24]),
                "l2_regularization": ("choice", [0.0, 0.1, 1.0]),
            }, n_cand)
            if not pdist: return f"{model}", None
            return f"{model}", self._make_search(est, pdist, metric, cv, n_cand)

        if model == "rf":
            est = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
            pdist = self._space_to_lists(space, {
                "n_estimators": ("int", 300, 1500, 300),
                "max_depth": ("choice", [None, 10, 20, 40]),
                "min_samples_leaf": ("choice", [1, 2, 4]),
                "min_samples_split": ("choice", [2, 5, 10]),
                "class_weight": ("choice", [None, "balanced"]),
            }, n_cand)
            if not pdist: return f"{model}", None
            return f"{model}", self._make_search(est, pdist, metric, cv, n_cand)

        if model == "lr":
            est = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    solver="lbfgs", max_iter=1000, multi_class="auto", n_jobs=None, random_state=self.random_state
                )),
            ])
            pdist = self._space_to_lists(space, {
                "clf__C": ("loguniform", 1e-3, 1e3),
                "clf__class_weight": ("choice", [None, "balanced"]),
            }, n_cand, prefix_map={"C":"clf__C","class_weight":"clf__class_weight"})
            if not pdist: return f"{model}", None
            return f"{model}", self._make_search(est, pdist, metric, cv, n_cand)

        if model == "knn":
            est = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier()),
            ])
            pdist = self._space_to_lists(space, {
                "clf__n_neighbors": ("int", 5, 61, 2),
                "clf__weights": ("choice", ["distance", "uniform"]),
            }, n_cand, prefix_map={"n_neighbors":"clf__n_neighbors","weights":"clf__weights"})
            if not pdist: return f"{model}", None
            return f"{model}", self._make_search(est, pdist, metric, cv, n_cand)

        return f"{model}", None

    def _space_to_lists(
        self,
        llm_space: Dict[str, Any],
        defaults: Dict[str, Tuple],
        n_cand: int,
        prefix_map: Optional[Dict[str,str]] = None
    ) -> Dict[str, List[Any]]:
        """
        Convert the LLM "space" spec into lists of concrete values (no SciPy dependency).
        Supports three types: int, loguniform, choice.
        Fills from defaults if LLM omits a key or gives invalid ranges.
        """
        out: Dict[str, List[Any]] = {}
        prefix_map = prefix_map or {}

        def ensure_choice(vals, fallback):
            if isinstance(vals, list) and len(vals) > 0:
                return list(vals)
            return list(fallback)

        for human_key, spec in defaults.items():
            # map human param name to estimator param name (for pipelines)
            est_key = human_key
            # If caller supplied a mapping (e.g. LR "C" -> "clf__C")
            for k, v in prefix_map.items():
                if human_key.endswith(v):  # already mapped default form
                    est_key = human_key
                elif human_key == k:
                    est_key = v
            # Now pull LLM override if present
            # Accept either est_key or original human key in the LLM space
            llm_key = None
            for candidate_key in (est_key, human_key, human_key.replace("clf__", "")):
                if candidate_key in llm_space:
                    llm_key = candidate_key
                    break

            typ = spec[0]
            if typ == "int":
                low, high, step = spec[1], spec[2], spec[3]
                if llm_key:
                    try:
                        low = int(llm_space[llm_key].get("low", low))
                        high = int(llm_space[llm_key].get("high", high))
                        step = int(llm_space[llm_key].get("step", step))
                    except Exception:
                        pass
                if high < low: low, high = high, low
                vals = list(range(low, high + 1, max(1, step)))
                out[est_key] = _safe_choice(vals, min(len(vals), max(10, n_cand // 4)))

            elif typ == "loguniform":
                low, high = spec[1], spec[2]
                if llm_key:
                    try:
                        low = float(llm_space[llm_key].get("low", low))
                        high = float(llm_space[llm_key].get("high", high))
                    except Exception:
                        pass
                if high <= low: high = low * 10.0
                num = max(8, min(20, n_cand // 3))
                vals = list(np.geomspace(low, high, num=num))
                out[est_key] = vals

            elif typ == "choice":
                choices = spec[1]
                if llm_key:
                    choices = ensure_choice(llm_space[llm_key].get("values"), choices)
                out[est_key] = choices

        return out

    def _make_search(self, estimator, param_dist: Dict[str, List[Any]], metric: str, cv, n_cand: int) -> _SearchCV:
        if _SEARCH_KIND == "halving_random":
            # n_candidates is an upper bound on initial configs; we pick based on param space size
            n_candidates = min(max(20, sum(len(v) for v in param_dist.values())), max(20, n_cand))
            return _SearchCV(
                estimator=estimator,
                param_distributions=param_dist,
                n_candidates=n_candidates,
                factor=3,
                resource="n_samples",
                min_resources="exhaust",
                random_state=self.random_state,
                scoring=metric,
                cv=cv,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
            )
        else:
            return _SearchCV(
                estimator=estimator,
                param_distributions=param_dist,
                n_iter=n_cand,
                random_state=self.random_state,
                scoring=metric,
                cv=cv,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
            )

    # -------------------- training utilities --------------------

    def _maybe_downsample(self, X: pd.DataFrame, y: pd.Series, cap: int) -> Tuple[np.ndarray, np.ndarray]:
        if len(y) <= cap:
            return X.values, y.values
        frac = cap / float(len(y))
        rng = np.random.RandomState(self.random_state)
        idx_parts = []
        ser = pd.Series(y).reset_index(drop=True)
        for _, idxs in ser.groupby(ser).groups.items():
            idxs = np.array(list(idxs))
            k = max(1, int(round(len(idxs) * frac)))
            idx_parts.append(rng.choice(idxs, size=k, replace=False))
        idx = np.concatenate(idx_parts)
        return X.values[idx], y.values[idx]

    def _decode_best(self, tag: str, best_params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        if tag == "lr" or tag.endswith(":lr"):
            flat = {}
            for k, v in best_params.items():
                flat[k.replace("clf__", "") if k.startswith("clf__") else k] = v
            # keep only clf params for final constructor
            flat = {k: v for k, v in flat.items() if k in ("C", "class_weight")}
            return "lr", flat
        if tag == "knn" or tag.endswith(":knn"):
            flat = {}
            for k, v in best_params.items():
                flat[k.replace("clf__", "") if k.startswith("clf__") else k] = v
            flat = {k: v for k, v in flat.items() if k in ("n_neighbors", "weights")}
            return "knn", flat
        if tag == "rf" or tag.endswith(":rf"):
            return "rf", dict(best_params)
        if tag == "hgb" or tag.endswith(":hgb"):
            return "hgb", dict(best_params)
        # fallback as-is
        return tag, dict(best_params)

    def _make_estimator(self, model: str, params: Dict[str, Any]):
        if model == "lr":
            return Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    C=float(params.get("C", 1.0)),
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=1000,
                    multi_class="auto",
                    n_jobs=None,
                    class_weight=params.get("class_weight", None),
                    random_state=self.random_state
                )),
            ])
        if model == "knn":
            return Pipeline([
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(
                    n_neighbors=int(params.get("n_neighbors", 31)),
                    weights=str(params.get("weights", "distance")),
                )),
            ])
        if model == "rf":
            return RandomForestClassifier(
                n_estimators=int(params.get("n_estimators", 600)),
                max_depth=params.get("max_depth", None),
                min_samples_split=int(params.get("min_samples_split", 2)),
                min_samples_leaf=int(params.get("min_samples_leaf", 1)),
                class_weight=params.get("class_weight", None),
                n_jobs=-1,
                random_state=self.random_state
            )
        if model == "hgb":
            return HistGradientBoostingClassifier(
                max_depth=params.get("max_depth", None),
                learning_rate=float(params.get("learning_rate", 0.06)),
                max_iter=int(params.get("max_iter", 400)),
                l2_regularization=float(params.get("l2_regularization", 0.0)),
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=self.random_state
            )
        raise ValueError(f"Unsupported model: {model}")

    def _maybe_calibrate(self, X: np.ndarray, y: np.ndarray):
        if not self.calibrate:
            self.best_calibrated_ = None
            return
        try:
            method = "isotonic" if len(y) >= 2000 else "sigmoid"
            self.best_calibrated_ = CalibratedClassifierCV(self.best_est_, method=method, cv=3)
            self.best_calibrated_.fit(X, y)
        except Exception as e:
            logger.warning(f"[LLM-Guided HPO] Calibration failed; continuing uncalibrated. Error: {e}")
            self.best_calibrated_ = None

    def _format_selection(self, extra: str = "") -> str:
        base = f"model={self.selected_model_}, hp={json.dumps(self.selected_params_, sort_keys=True)}"
        return base if not extra else f"{base} | {extra}"

    # -------------------- schema --------------------

    def _build_schema_card(self, X: pd.DataFrame, y: pd.Series) -> str:
        n, d = X.shape
        info = {
            "n_rows": int(n),
            "n_cols": int(d),
            "labels": [int(c) if isinstance(c, (np.integer, int)) else c for c in sorted(pd.unique(y))],
            "class_priors": { str(c): float((y==c).mean()) for c in sorted(pd.unique(y)) }
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
