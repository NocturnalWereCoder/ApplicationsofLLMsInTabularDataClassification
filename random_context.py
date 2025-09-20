import logging
import numpy as np
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_ollama.llms import OllamaLLM

from config import LLM_MODEL, RANDOM_K

logger = logging.getLogger(__name__)

class ClassificationWithReasoning(BaseModel):
    classification: int = Field(...)
    reasoning: str = Field(...)

class RandomContextClassifier:
    def __init__(self, cuda_device: int):
        self.k = RANDOM_K
        logger.info(f"Initializing RandomContextClassifier k={self.k}, cuda={cuda_device}")
        os.environ['OLLAMA_CUDA_DEVICE'] = str(cuda_device)

        template = """
        You are a classification model.

        You MUST follow these rules:
        1) Return the final answer ONLY in JSON format.
        2) Do NOT include any additional text before or after the JSON.
        3) Do NOT include markdown formatting or code fences.

        Here are some randomly sampled training examples:

        {retrieved_examples}

        Now classify the following new data point:

        {test_case}

        The response must be valid JSON exactly like this:
        {{{{ "classification": <integer>, "reasoning": "<some text>" }}}}
        """.strip()

        self.parser = PydanticOutputParser(pydantic_object=ClassificationWithReasoning)
        self.prompt = ChatPromptTemplate.from_template(template)
        self.llm = OllamaLLM(model=LLM_MODEL, temperature=0, in_process=True)
        self.chain = self.prompt | self.llm

    def fit(self, X, y):
        # just store for sampling later
        self.train_X = X.reset_index(drop=True)
        self.train_y = y.reset_index(drop=True)
        logger.info(f"RandomContext: stored {len(self.train_X)} examples")
        return self

    def _predict_one(self, row):
        # pick k random indices
        idxs = np.random.choice(len(self.train_X), size=self.k, replace=False)
        docs = [
            f"Features: {', '.join(map(str, self.train_X.iloc[i]))}. Label: {self.train_y.iloc[i]}"
            for i in idxs
        ]
        q = f"Features: {', '.join(map(str, row))}"
        resp = self.chain.invoke({
            'retrieved_examples': '\n'.join(docs),
            'test_case': q
        })
        # extract and parse JSON
        import re
        m = re.search(r"\{.*?\}", resp, re.DOTALL)
        payload = m.group(0) if m else resp
        return self.parser.parse(payload).classification

    def predict(self, X):
        n_samples = len(X)
        logger.info(f"RandomContext: predicting {n_samples} samples with k={self.k}")
        rows = list(X.reset_index(drop=True).itertuples(index=False))

        start_time = time.time()
        cpu_count = os.cpu_count() or 1
        max_workers = min(32, max(1, cpu_count - 1))

        preds = [None] * n_samples
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(self._predict_one, row): idx for idx, row in enumerate(rows)}
            for done_count, future in enumerate(as_completed(futures), start=1):
                idx = futures[future]
                preds[idx] = future.result()
                pct = done_count / n_samples * 100
                elapsed = time.time() - start_time
                avg_per = elapsed / done_count
                remaining = avg_per * (n_samples - done_count)
                logger.info(
                    f"RandomContext progress: {pct:.1f}% "
                    f"({done_count}/{n_samples}) - "
                    f"Elapsed: {elapsed:.1f}s, ETA: {remaining:.1f}s"
                )

        return np.array(preds)
