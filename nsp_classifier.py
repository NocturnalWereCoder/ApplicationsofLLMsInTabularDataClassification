# nsp_classifier.py
import os
import re
import json
import logging
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

class NSPClassifier:
    """
    No-Shot Prompt classifier.
    - Does not include labeled examples in the prompt.
    - Constrains the model to the observed label set.
    - Returns an integer classification and reasoning (discarded by the pipeline).
    """
    def __init__(
        self,
        cuda_device=0,
        temperature=0.0,
        top_p=1.0,
        schema_sample_n=0  # keep 0 for pure no-shot; >0 would include a few *unlabeled* rows
    ):
        logger.info(f"Initializing NSP (zero-shot) cuda={cuda_device}")
        self.cuda = cuda_device
        self.temperature = temperature
        self.top_p = top_p
        self.schema_sample_n = schema_sample_n

        template = """
You are a classification model operating on tabular data.

Rules:
1) Return ONLY JSON (no prose, no markdown).
2) JSON must match: {{"classification": <integer>, "reasoning": "<short text>"}}.
3) The "classification" MUST be one of these integers: {label_space}.

Schema (column names):
{schema}

{maybe_context}

Classify this row:
{test_case}
""".strip()

        self.parser = PydanticOutputParser(pydantic_object=ClassificationWithReasoning)
        self.prompt = ChatPromptTemplate.from_template(template)

        os.environ['OLLAMA_CUDA_DEVICE'] = str(cuda_device)
        self.llm = OllamaLLM(
            model=LLM_MODEL,
            temperature=self.temperature,
            top_p=self.top_p,
            in_process=True
        )
        self.chain = self.prompt | self.llm

    def fit(self, X, y):
        self.columns = list(X.columns)
        self.label_space = sorted(list(map(int, np.unique(y))))
        # optional unlabeled context (pure NSP keeps it empty)
        self.maybe_context = ""
        if self.schema_sample_n and self.schema_sample_n > 0:
            sample = X.sample(n=min(self.schema_sample_n, len(X)), random_state=42)
            # include a few unlabeled rows just to show feature ranges/types (still zero-shot)
            self.maybe_context = f"Here are a few UNLABELED example rows (for schema context only):\n{sample.to_json(orient='records')}"
        return self

    def _extract_json(self, resp: str) -> str:
        m = re.search(r"\{.*?\}", resp, re.DOTALL)
        return m.group(0) if m else resp.strip()

    def _predict_one(self, row):
        test_dict = dict(zip(self.columns, row))
        out = self.chain.invoke({
            "label_space": json.dumps(self.label_space, ensure_ascii=False),
            "schema": json.dumps(self.columns, ensure_ascii=False),
            "maybe_context": self.maybe_context,
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
        return -1  # fallback on hard failure

    def predict(self, X):
        n_samples = len(X)
        logger.info(f"NSP: predicting {n_samples} samples (thread-pooled)")
        rows = list(X.reset_index(drop=True).itertuples(index=False))
        cpu_count = multiprocessing.cpu_count() or 1
        max_workers = min(32, max(1, cpu_count - 1))

        preds = [None] * n_samples
        from_time = 0
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = { pool.submit(self._predict_one, row): idx for idx, row in enumerate(rows) }
            for done_count, future in enumerate(as_completed(futures), start=1):
                idx = futures[future]
                preds[idx] = future.result()
                pct = done_count / n_samples * 100
                logger.info(f"NSP prediction progress: {pct:.1f}% ({done_count}/{n_samples})")
        return np.array(preds)
