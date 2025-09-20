# fsp_classifier.py
import os
import re
import json
import logging
import random
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama.llms import OllamaLLM

from config import LLM_MODEL

logger = logging.getLogger(__name__)

class ClassificationWithReasoning(BaseModel):
    classification: int = Field(...)
    reasoning: str = Field(...)

class FSPClassifier:
    """
    Few-Shot Prompt classifier.
    - Adds balanced, labeled prototypes per class into the prompt.
    - The prompt stays compact via caps (prototypes_per_class, max_examples).
    """
    def __init__(
        self,
        cuda_device=0,
        prototypes_per_class=2,
        max_examples=20,
        temperature=0.0,
        top_p=1.0,
        random_state=42
    ):
        logger.info(
            f"Initializing FSP (few-shot) cuda={cuda_device}, "
            f"prototypes_per_class={prototypes_per_class}, max_examples={max_examples}"
        )
        self.cuda = cuda_device
        self.k_per_class = prototypes_per_class
        self.max_examples = max_examples
        self.temperature = temperature
        self.top_p = top_p
        self.rng = random.Random(random_state)

        template = """
You are a classification model operating on tabular data.

Rules:
1) Return ONLY JSON (no prose, no markdown).
2) JSON must match: {{"classification": <integer>, "reasoning": "<short text>"}}.
3) The "classification" MUST be one of these integers: {label_space}.

Here are labeled examples (few-shot):
{retrieved_examples}

Classify this new row:
{test_case}
""".strip()

        self.parser = PydanticOutputParser(pydantic_object=ClassificationWithReasoning)
        self.prompt = ChatPromptTemplate.from_template(template)

        os.environ['OLLAMA_CUDA_DEVICE'] = str(cuda_device)
        self.llm = OllamaLLM(
            model=LLM_MODEL, temperature=self.temperature, top_p=self.top_p, in_process=True
        )
        self.chain = self.prompt | self.llm

    def fit(self, X, y):
        self.columns = list(X.columns)
        self.y = y.reset_index(drop=True)
        self.label_space = sorted(list(map(int, np.unique(y))))

        # Build balanced prototype set (up to max_examples)
        # Sample up to k_per_class per class, then cap globally at max_examples.
        idx_by_class = {int(c): [] for c in self.label_space}
        for i, label in enumerate(self.y):
            lb = int(label)
            if lb in idx_by_class:
                idx_by_class[lb].append(i)

        picked = []
        for lb, idxs in idx_by_class.items():
            self.rng.shuffle(idxs)
            picked.extend((lb, i) for i in idxs[:self.k_per_class])

        # If too many overall, downsample globally while keeping near-balanced mix
        if len(picked) > self.max_examples:
            self.rng.shuffle(picked)
            picked = picked[:self.max_examples]

        # Materialize the few-shot examples as a list of dicts: {feature... , "label": int}
        X_reset = X.reset_index(drop=True)
        examples = []
        for lb, i in picked:
            d = X_reset.iloc[i].to_dict()
            d["label"] = int(lb)
            examples.append(d)
        self.retrieved_examples_json = json.dumps(examples, ensure_ascii=False)
        return self

    def _extract_json(self, resp: str) -> str:
        m = re.search(r"\{.*?\}", resp, re.DOTALL)
        return m.group(0) if m else resp.strip()

    def _predict_one(self, row):
        test_dict = dict(zip(self.columns, row))
        out = self.chain.invoke({
            "label_space": json.dumps(self.label_space, ensure_ascii=False),
            "retrieved_examples": self.retrieved_examples_json,
            "test_case": json.dumps(test_dict, ensure_ascii=False),
        })

        for _ in range(3):
            try:
                parsed = self.parser.parse(self._extract_json(out))
                pred = int(parsed.classification)
                if pred in self.label_space:
                    return pred
            except Exception:
                continue
        return -1

    def predict(self, X):
        n_samples = len(X)
        logger.info(f"FSP: predicting {n_samples} samples (thread-pooled)")
        rows = list(X.reset_index(drop=True).itertuples(index=False))
        cpu_count = multiprocessing.cpu_count() or 1
        max_workers = min(32, max(1, cpu_count - 1))

        preds = [None] * n_samples
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = { pool.submit(self._predict_one, row): idx for idx, row in enumerate(rows) }
            for done_count, future in enumerate(as_completed(futures), start=1):
                idx = futures[future]
                preds[idx] = future.result()
                pct = done_count / n_samples * 100
                logger.info(f"FSP prediction progress: {pct:.1f}% ({done_count}/{n_samples})")
        return np.array(preds)
