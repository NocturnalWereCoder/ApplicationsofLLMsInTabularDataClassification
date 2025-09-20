# quantile_binned_fewshot_tabular_llm_classifier.py
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

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama.llms import OllamaLLM

from config import LLM_MODEL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Output schema (strict JSON)
# ---------------------------------------------------------------------
class ClassificationWithReasoning(BaseModel):
    classification: int = Field(...)
    reasoning: str = Field(...)

# ---------------------------------------------------------------------
# Utilities: detect feature kinds and bin numerics
# ---------------------------------------------------------------------
def _is_binary_series(s: pd.Series) -> bool:
    vals = pd.unique(s.dropna())
    if len(vals) <= 2:
        u = set(np.unique(vals).tolist())
        return u.issubset({0, 1, 0.0, 1.0, True, False})
    return False

def _has_small_cardinality(s: pd.Series, thr: int = 10) -> bool:
    return s.nunique(dropna=True) <= thr

@dataclass
class Binner:
    col_bins: Dict[Any, np.ndarray]   # col -> bin edges
    col_labels: Dict[Any, List[str]]  # col -> list of labels
    col_kind: Dict[Any, str]          # "bin" | "binary" | "categorical"
    cat_maps: Dict[Any, Dict[Any, str]]  # col -> {raw_value -> token}
    cols: List[Any]                   # fixed column order

    @staticmethod
    def make(X: pd.DataFrame, n_bins: int = 7, cat_thr: int = 10) -> "Binner":
        bin_names = ["vlow", "low", "mlow", "mid", "mhigh", "high", "vhigh"]
        if n_bins > len(bin_names):
            raise ValueError("n_bins cannot exceed 7 with default names.")
        col_bins, col_labels, col_kind, cat_maps = {}, {}, {}, {}
        cols = list(X.columns)

        for c in cols:
            s = X[c]
            # Treat -1 as "unknown" sentinel (common in bank/sick), but keep kind detection
            has_minus_one = (pd.api.types.is_numeric_dtype(s) and (s == -1).any())

            if _is_binary_series(s):
                col_kind[c] = "binary"

            elif pd.api.types.is_numeric_dtype(s) and not _has_small_cardinality(s, thr=cat_thr):
                # Numeric → quantile bins (robust to ties)
                if s.nunique(dropna=True) <= 1:
                    # degenerate → categorical single token
                    col_kind[c] = "categorical"
                    uniqs = list(pd.unique(s))
                    cat_maps[c] = {v: "c0" for v in uniqs}
                else:
                    values = s.dropna()
                    edges = np.quantile(values, np.linspace(0, 1, n_bins + 1))
                    # enforce strictly increasing edges
                    for i in range(1, len(edges)):
                        if edges[i] <= edges[i-1]:
                            edges[i] = np.nextafter(edges[i-1], np.inf)
                    col_kind[c] = "bin"
                    col_bins[c] = edges
                    col_labels[c] = bin_names[:n_bins]

            else:
                # Small-cardinality ints → categorical tokens
                col_kind[c] = "categorical"
                uniqs = list(pd.unique(s))
                try:
                    uniqs = sorted(uniqs)
                except Exception:
                    pass
                cat_maps[c] = {v: f"c{i}" for i, v in enumerate(uniqs)}

            # mark that -1 should become "unknown" at encode time
            if has_minus_one and c not in cat_maps and col_kind.get(c) != "binary":
                # We'll special-case -1 in encode_value below.
                pass

        return Binner(col_bins, col_labels, col_kind, cat_maps, cols)

    def encode_value(self, c, v) -> str:
        if pd.isna(v):
            return "unknown"
        # common sentinel
        try:
            if float(v) == -1.0:
                return "unknown"
        except Exception:
            pass

        kind = self.col_kind[c]
        if kind == "binary":
            try:
                return "yes" if float(v) == 1.0 else "no"
            except Exception:
                # non-numeric binary fallback
                return "yes" if str(v).lower() in {"1", "true", "yes"} else "no"

        if kind == "categorical":
            mp = self.cat_maps[c]
            return mp.get(v, "c?")

        # quantile bin
        edges = self.col_bins[c]
        labels = self.col_labels[c]
        # np.digitize over internal edges (exclude the two extremes)
        idx = int(np.clip(np.digitize([float(v)], edges[1:-1])[0], 0, len(labels) - 1))
        return labels[idx]

# ---------------------------------------------------------------------
# The LLM-only classifier (few-shot with quantile-binned features)
# ---------------------------------------------------------------------
class QuantileBinnedFewShotTabularLLMClassifier:
    """
    LLM-only tabular classifier:
    - quantile-bins numeric columns learned on training fold
    - verbalizes binaries/categoricals
    - balanced few-shot (no KNN / no vector DB)
    - majority vote over multiple LLM samples (configurable)
    """
    def __init__(
        self,
        cuda_device: int = 0,
        n_bins: int = 7,
        cat_thr: int = 10,
        prototypes_per_class: int = 3,
        max_examples: int = 20,
        samples: int = 7,
        temperature: float = 0.3,
        top_p: float = 0.95
    ):
        self.n_bins = n_bins
        self.cat_thr = cat_thr
        self.prototypes_per_class = prototypes_per_class
        self.max_examples = max_examples
        self.samples = max(1, samples)

        os.environ['OLLAMA_CUDA_DEVICE'] = str(cuda_device)
        self.llm = OllamaLLM(
            model=LLM_MODEL,
            temperature=float(temperature),
            top_p=float(top_p),
            in_process=True
        )

        template = """
You are a careful tabular classifier.
Return ONLY valid JSON like: {{ "classification": <integer>, "reasoning": "<short>" }}.

Encoding:
- binary → yes/no
- small-cardinality integers → c0,c1,...
- numeric → quantile bins among: vlow, low, mlow, mid, mhigh, high, vhigh
- -1 or missing → "unknown"

Few-shot examples (balanced across classes):
{few_shots}

Now classify this case (use same tokens; ≤1 sentence reasoning):
{test_case}

Return ONLY JSON: {{ "classification": <integer>, "reasoning": "<short>" }}.
""".strip()


        self.parser = PydanticOutputParser(pydantic_object=ClassificationWithReasoning)
        self.prompt = ChatPromptTemplate.from_template(template)
        self.chain = self.prompt | self.llm

        self._binner: Binner = None
        self._few_shot_text: str = ""
        self._label_values: List[int] = []
        self._cols: List[Any] = []

    # ---------------- Fit ----------------
    def fit(self, X: pd.DataFrame, y: pd.Series):
        logger.info("QuantileLLM: fitting binner and building few-shot context")
        self._binner = Binner.make(X, n_bins=self.n_bins, cat_thr=self.cat_thr)
        self._cols = self._binner.cols
        # preserve original label ids as ints (sorted for stability)
        self._label_values = sorted(pd.unique(y).tolist())

        # Balanced sampling per class
        per_cls = max(1, self.prototypes_per_class)
        rng = np.random.default_rng(42)

        lines: List[str] = []
        for c in self._label_values:
            idxs = np.where(y.values == c)[0]
            if len(idxs) == 0:
                continue
            if len(idxs) > per_cls:
                idxs = rng.choice(idxs, size=per_cls, replace=False)
            for j in idxs:
                row = X.iloc[j]
                serial = self._serialize_features_only(row)
                lines.append(f"- label={int(c)}; {serial}")

        # Hard cap the few-shot block
        if len(lines) > self.max_examples:
            lines = lines[: self.max_examples]

        self._few_shot_text = "\n".join(lines)
        logger.info(f"QuantileLLM: few-shot with {len(lines)} examples")
        return self

    # ---------------- Predict ----------------
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        logger.info(f"QuantileLLM: predicting {n} samples "
                    f"(samples per row = {self.samples})")

        rows = list(X.reset_index(drop=True).itertuples(index=False))
        cpu_count = multiprocessing.cpu_count() or 1
        max_workers = min(32, max(1, cpu_count - 1))

        preds = [None] * n
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = { pool.submit(self._predict_one, row): i
                        for i, row in enumerate(rows) }
            for done, fut in enumerate(as_completed(futures), start=1):
                i = futures[fut]
                preds[i] = fut.result()
                if done % max(1, n // 10) == 0 or done == n:
                    logger.info(f"QuantileLLM progress: {done}/{n} ({done/n*100:.1f}%)")
        return np.array(preds, dtype=int)

    # ---------------- Internals ----------------
    def _serialize_features_only(self, row: pd.Series) -> str:
        # stable key order f0=..., f1=..., ...
        vals = []
        for i, c in enumerate(self._cols):
            tok = self._binner.encode_value(c, row[c])
            vals.append(f"f{i}={tok}")
        return "; ".join(vals)

    def _extract_json(self, resp: str) -> str:
        m = re.search(r"\{[\s\S]*\}", str(resp))
        return m.group(0) if m else str(resp).strip()

    def _predict_one(self, row_tuple) -> int:
        # Namedtuple → Series with training columns
        row = pd.Series(row_tuple, index=self._cols)
        test_case = self._serialize_features_only(row)

        # Majority vote over multiple LLM samples
        votes: Dict[int, int] = {int(c): 0 for c in self._label_values}
        for _ in range(self.samples):
            out = self.chain.invoke({
                "few_shots": self._few_shot_text,
                "test_case": test_case
            })

            lab = None
            # try parser, then regex fallback
            try:
                parsed = self.parser.parse(self._extract_json(out))
                lab = int(parsed.classification)
            except Exception:
                mm = re.search(r'"classification"\s*:\s*(\d+)', str(out))
                if mm:
                    lab = int(mm.group(1))

            if lab in votes:
                votes[lab] += 1

        # choose class with highest votes; tie → smallest id
        best = max(votes.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        return int(best)
