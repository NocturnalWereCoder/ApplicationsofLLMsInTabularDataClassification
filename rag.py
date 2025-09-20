

# rag.py

import os
import re
import logging
import numpy as np
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed

import ollama
import chromadb

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_ollama.llms import OllamaLLM

from config import LLM_MODEL

logger = logging.getLogger(__name__)

class ClassificationWithReasoning(BaseModel):
    classification: int = Field(...)
    reasoning: str = Field(...)

class LangChainOllamaRAGClassifier:
    def __init__(self, k_neighbors, persist_dir, cuda_device):
        logger.info(f"Initializing RAG (dir={persist_dir}, cuda={cuda_device})")

        if os.path.exists(persist_dir):
            import shutil
            shutil.rmtree(persist_dir)
        os.makedirs(persist_dir, exist_ok=True)

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

        The response must be valid JSON **exactly** like this:
        {{{{ "classification": <integer>, "reasoning": "<some text>" }}}}
        """.strip()

        self.parser = PydanticOutputParser(pydantic_object=ClassificationWithReasoning)
        self.prompt = ChatPromptTemplate.from_template(template)
        os.environ['OLLAMA_CUDA_DEVICE'] = str(cuda_device)
        self.llm = OllamaLLM(model=LLM_MODEL, temperature=0, in_process=True)
        self.chain = self.prompt | self.llm

        self.vec_store = chromadb.PersistentClient(path=persist_dir)
        self.k = k_neighbors
        self.cuda = cuda_device

    def _extract_json(self, resp: str) -> str:
        m = re.search(r"\{.*?\}", resp, re.DOTALL)
        return m.group(0) if m else resp.strip()

    def fit(self, X, y):
        total = len(X)
        logger.info(f"RAG: building vector store with {total} examples")
        if 'VecDB' in self.vec_store.list_collections():
            self.vec_store.delete_collection(name='VecDB')
        coll = self.vec_store.create_collection(name='VecDB')

        for idx, row in enumerate(X.reset_index(drop=True).itertuples(index=False)):
            content = f"Features: {', '.join(map(str,row))}. Label: {y.iloc[idx]}"
            os.environ['OLLAMA_CUDA_DEVICE'] = str(self.cuda)
            emb = ollama.embeddings(model='all-minilm', prompt=content)['embedding']
            coll.add(documents=[content], embeddings=[emb], ids=[str(idx)])

            pct = (idx + 1) / total * 100
            logger.info(f"RAG DB build progress: {pct:.1f}% ({idx + 1}/{total})")

        logger.info("RAG: vector store built")
        return self

    def _predict_one(self, row):
        q = f"Features: {', '.join(map(str,row))}"
        os.environ['OLLAMA_CUDA_DEVICE'] = str(self.cuda)
        resp = (self.vec_store
                    .get_collection(name='VecDB')
                    .query(query_texts=[q], n_results=self.k, include=['documents']))
        docs = [d for sub in resp['documents'] for d in sub]

        out = self.chain.invoke({
            'retrieved_examples': '\n'.join(docs),
            'test_case': q
        })

        for _ in range(3):
            try:
                return self.parser.parse(self._extract_json(out)).classification
            except Exception:
                continue
        return -1

    def predict(self, X):
        total = len(X)
        logger.info(f"RAG: predicting {total} samples with k={self.k} (thread‐pooled)")

        rows = list(X.reset_index(drop=True).itertuples(index=False))
        cpu_count = multiprocessing.cpu_count() or 1
        max_workers = min(32, max(1, cpu_count - 1))

        preds = [None] * total
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = { pool.submit(self._predict_one, row): idx
                        for idx, row in enumerate(rows) }
            for done_count, future in enumerate(as_completed(futures), start=1):
                idx = futures[future]
                preds[idx] = future.result()
                pct = done_count / total * 100
                logger.info(f"RAG prediction progress: {pct:.1f}% ({done_count}/{total})")

        return np.array(preds)
