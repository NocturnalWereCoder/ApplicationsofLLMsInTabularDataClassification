import logging
import numpy as np
import re
import os
import json
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_ollama.llms import OllamaLLM

from config import LLM_MODEL

logger = logging.getLogger(__name__)

class ClassificationWithReasoning(BaseModel):
    classification: int = Field(...)
    reasoning: str = Field(...)

class HybridClassifier:
    def __init__(self, k_neighbors, cuda_device):
        logger.info(f"Initializing Hybrid (KNN→LLM) k={k_neighbors}, cuda={cuda_device}")
        self.k = k_neighbors
        self.cuda = cuda_device

        template = """
        You are a classification model.

        You MUST follow these rules:
        1) Return the final answer ONLY in JSON format.
        2) Do NOT include any additional text before or after the JSON.
        3) Do NOT include markdown formatting or code fences.

        Here are some relevant training examples:

        {retrieved_examples}

        Now classify the following new data point:

        {test_case}

        The response must be valid JSON exactly like this:
        {{{{ "classification": <integer>, "reasoning": "<some text>" }}}}
        """.strip()

        self.parser = PydanticOutputParser(pydantic_object=ClassificationWithReasoning)
        self.prompt = ChatPromptTemplate.from_template(template)

        # set device for Ollama
        os.environ['OLLAMA_CUDA_DEVICE'] = str(cuda_device)
        # only completion LLM (no embedding)
        self.llm = OllamaLLM(model=LLM_MODEL, temperature=0, in_process=True)
        self.chain = self.prompt | self.llm

    def fit(self, X, y):
        logger.info(f"Hybrid: storing {len(X)} training examples for nearest‐neighbor lookup")
        self.train_X = X.reset_index(drop=True)
        self.train_y = y.reset_index(drop=True)
        return self

    def _extract_json(self, resp: str) -> str:
        m = re.search(r"\{.*?\}", resp, re.DOTALL)
        return m.group(0) if m else resp.strip()

    def _predict_one(self, row):
        # raw float KNN retrieval
        arr_test  = np.array(row, dtype=float)
        arr_train = self.train_X.values.astype(float)
        dists     = np.linalg.norm(arr_train - arr_test, axis=1)
        idxs      = dists.argsort()[:self.k]

        # build JSON list of neighbors with column names
        docs = []
        for j in idxs:
            example = self.train_X.iloc[j].to_dict()
            example['label'] = int(self.train_y.iloc[j])
            docs.append(example)
        retrieved_json = json.dumps(docs, ensure_ascii=False)

        # serialize test row
        test_dict = dict(zip(self.train_X.columns, row))
        test_json = json.dumps(test_dict, ensure_ascii=False)

        # invoke LLM chain with structured JSON examples
        out = self.chain.invoke({
            'retrieved_examples': retrieved_json,
            'test_case': test_json
        })

        # parse classification
        for _ in range(3):
            try:
                return self.parser.parse(self._extract_json(out)).classification
            except Exception:
                continue
        return -1

    def predict(self, X):
        n_samples = len(X)
        logger.info(f"Hybrid: predicting {n_samples} samples with k={self.k} (thread‐pooled)")

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
                logger.info(f"Hybrid prediction progress: {pct:.1f}% ({done_count}/{n_samples})")

        return np.array(preds)


