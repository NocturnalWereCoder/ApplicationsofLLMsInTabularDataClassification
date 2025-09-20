#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_rag_pipeline_enhanced_with_retry.py

Enhanced RAG vs RAG_no_embeddings vs KNN benchmarking script with retry handling for JSON parsing.
Configuration is done by editing the constants in the USER CONFIGURATION section below — no command-line arguments needed.
"""

# ----------------------------
# USER CONFIGURATION
# ----------------------------
# Execution mode: 'single' or 'two_pool'
MODE = 'single'
# Dataset fraction: 1.0 = 100% of rows, 0.1 = top 10% rows, etc.
SAMPLE_FRAC = 0.5
# ----------------------------
# Ensure sqlite3 compatibility
# ----------------------------
import sys
try:
    import pysqlite3
    sys.modules['sqlite3'] = pysqlite3
except ImportError:
    pass

import os
import time
import json
import re
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
import ollama
import chromadb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from concurrent.futures import ProcessPoolExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_ollama.llms import OllamaLLM

# ----------------------------
# Logging configuration
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Filter out rapid Ollama HTTP client logs
class NoHttpFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return not msg.startswith("HTTP Request:")

logger.addFilter(NoHttpFilter())

# ----------------------------
# TensorFlow GPU Memory Growth
# ----------------------------
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for dev in gpus:
        tf.config.experimental.set_memory_growth(dev, True)
    logger.info(f"Enabled TensorFlow memory growth for {len(gpus)} GPU(s)")
except Exception as e:
    logger.warning(f"Could not set TF memory growth: {e}")

def detect_gpu_count():
    return len(tf.config.experimental.list_physical_devices('GPU'))

# ----------------------------
# Pydantic model
# ----------------------------
class ClassificationWithReasoning(BaseModel):
    classification: int = Field(...)
    reasoning: str = Field(...)

# ----------------------------
# KNN via scikit-learn
# ----------------------------
def knn_predict_skl(X_tr, y_tr, X_te, k):
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn.fit(X_tr, y_tr)
    return knn.predict(X_te)

# ----------------------------
# RAG classifier with retry logic and strict prompt
# ----------------------------
class LangChainOllamaRAGClassifier:
    def __init__(self, k_neighbors, persist_dir, cuda_device):
        logger.info(f"Initializing RAG classifier (dir={persist_dir}, cuda={cuda_device})")
        if os.path.exists(persist_dir):
            import shutil; shutil.rmtree(persist_dir)
        os.makedirs(persist_dir, exist_ok=True)

        # Prompt template with escaped braces for literal JSON example
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

The response must be valid JSON like this exactly:
{{ "classification": <integer>, "reasoning": "<some text>" }}
""".strip()

        self.parser = PydanticOutputParser(pydantic_object=ClassificationWithReasoning)
        self.prompt = ChatPromptTemplate.from_template(template)
        os.environ['OLLAMA_CUDA_DEVICE'] = str(cuda_device)
        # DO NOT DELETE THIS COMMENT: 
        # self.llm = OllamaLLM(model='llama3.1', temperature=0, in_process=True)
                # self.llm = OllamaLLM(model='llama3.1:70b', temperature=0, in_process=True)

        self.llm = OllamaLLM(model='llama3.1:8b', temperature=0, in_process=True)
        self.chain = self.prompt | self.llm
        self.vec_store = chromadb.PersistentClient(path=persist_dir)
        self.k = k_neighbors
        self.cuda = cuda_device

    def extract_json(self, response: str) -> str:
        pattern = r"\{.*?\}"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(0)
        return response.strip()

    def fit(self, X, y):
        logger.info(f"Building vector store with {len(X)} examples")
        if 'VecDB' in self.vec_store.list_collections():
            self.vec_store.delete_collection(name='VecDB')
        coll = self.vec_store.create_collection(name='VecDB')
        for idx, row in X.reset_index(drop=True).iterrows():
            content = f"Features: {', '.join(map(str,row.tolist()))}. Label: {y.iloc[idx]}"
            os.environ['OLLAMA_CUDA_DEVICE'] = str(self.cuda)
            emb = ollama.embeddings(model='all-minilm', prompt=content)['embedding']
            coll.add(documents=[content], embeddings=[emb], ids=[str(idx)])
        logger.info("Vector store built successfully")
        return self

    def predict(self, X):
        logger.info(f"Predicting {len(X)} examples with RAG k={self.k}")
        preds = []
        coll = self.vec_store.get_collection(name='VecDB')
        for i, row in enumerate(X.reset_index(drop=True).itertuples(index=False)):
            q = f"Features: {', '.join(map(str,row))}"
            os.environ['OLLAMA_CUDA_DEVICE'] = str(self.cuda)
            response = coll.query(query_texts=[q], n_results=self.k, include=['documents'])
            docs = [d for sub in response['documents'] for d in sub]

            resp = self.chain.invoke({'retrieved_examples': '\n'.join(docs), 'test_case': q})

            # Retry mechanism
            retries = 0
            max_retries = 3
            js = -1
            while retries < max_retries:
                try:
                    clean_resp = self.extract_json(resp)
                    parsed = self.parser.parse(clean_resp)
                    js = parsed.classification
                    break
                except Exception as e:
                    retries += 1
                    logger.warning(f"Retry {retries}: Failed parsing example {i}: {e}")
                    if retries < max_retries:
                        logger.info(f"Retrying example {i}...")
                    else:
                        logger.error(f"Max retries reached for example {i}. Assigning -1.")
            preds.append(js)
        return np.array(preds)

# ----------------------------
# RAG without embeddings (distance-based retrieval)
# ----------------------------
class LangChainOllamaRAGNoEmbeddingsClassifier:
    def __init__(self, k_neighbors, persist_dir, cuda_device):
        logger.info(f"Initializing RAG_no_embeddings classifier (cuda={cuda_device})")
        # use the same strict prompt as the regular RAG version:
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

The response must be valid JSON like this exactly:
{{ "classification": <integer>, "reasoning": "<some text>" }}
""".strip()

        self.parser = PydanticOutputParser(pydantic_object=ClassificationWithReasoning)
        self.prompt = ChatPromptTemplate.from_template(template)
        os.environ['OLLAMA_CUDA_DEVICE'] = str(cuda_device)
        # self.llm = OllamaLLM(model='llama3.1', temperature=0, in_process=True)
        self.llm = OllamaLLM(model='llama3.1:70b', temperature=0, in_process=True)
        self.chain = self.prompt | self.llm

        self.k = k_neighbors
        # keep persist_dir arg for signature consistency, even if unused here

    def extract_json(self, response: str) -> str:
        pattern = r"\{.*?\}"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(0)
        return response.strip()

    def fit(self, X, y):
        logger.info(f"Storing {len(X)} training examples for distance-based retrieval")
        # keep raw dataframes for nearest-neighbor lookup
        self.train_X = X.reset_index(drop=True)
        self.train_y = y.reset_index(drop=True)
        return self

    def predict(self, X):
        logger.info(f"Predicting {len(X)} examples with RAG_no_embeddings k={self.k}")
        preds = []
        max_retries = 3

        for i, row in enumerate(X.reset_index(drop=True).itertuples(index=False)):
            q = f"Features: {', '.join(map(str,row))}"

            # ─── distance-based retrieval ───
            import numpy as _np
            test_arr = _np.array(row, dtype=float)
            train_arr = self.train_X.values.astype(float)
            distances = _np.linalg.norm(train_arr - test_arr, axis=1)
            nearest_idx = distances.argsort()[:self.k]
            docs = [
                f"Features: {', '.join(map(str, self.train_X.iloc[j].tolist()))}. "
                f"Label: {self.train_y.iloc[j]}"
                for j in nearest_idx
            ]

            resp = self.chain.invoke({
                'retrieved_examples': '\n'.join(docs),
                'test_case': q
            })

            # Retry logic
            retries = 0
            js = -1
            while retries < max_retries:
                try:
                    clean_resp = self.extract_json(resp)
                    parsed = self.parser.parse(clean_resp)
                    js = parsed.classification
                    break
                except Exception as e:
                    retries += 1
                    logger.warning(f"Retry {retries}: Failed parsing example {i}: {e}")
                    if retries < max_retries:
                        logger.info(f"Retrying example {i}...")
                    else:
                        logger.error(f"Max retries reached for example {i}. Assigning -1.")
            preds.append(js)

        return np.array(preds)

# ----------------------------
# Task runners for two_pool
# ----------------------------
def run_knn_task(ds, fold, X_tr, y_tr, X_te, y_te, k):
    logger.info(f"[CPU-Pool] KNN start -> {ds} Fold={fold}")
    acc = np.mean(knn_predict_skl(X_tr, y_tr, X_te, k) == y_te)
    logger.info(f"[CPU-Pool] KNN done -> {ds} Fold={fold} Acc={acc:.4f}")
    return (ds, 'KNN', fold, acc)

def run_rag_task(ds, fold, method, X_tr, y_tr, X_te, y_te, k, gpu_id):
    logger.info(f"[GPU-Pool] {method} start -> {ds} Fold={fold} on GPU {gpu_id}")
    if method == 'RAG_no_embeddings':
        cls = LangChainOllamaRAGNoEmbeddingsClassifier(k_neighbors=k, persist_dir=None, cuda_device=gpu_id)
        cls.fit(X_tr, y_tr)
    else:
        cls = LangChainOllamaRAGClassifier(k_neighbors=k,
                                           persist_dir=f"chroma_db/{ds}_f{fold}/{method}",
                                           cuda_device=gpu_id)
        cls.fit(X_tr, y_tr)
    preds = cls.predict(X_te)
    acc = np.mean(preds == y_te)
    logger.info(f"[GPU-Pool] {method} done -> {ds} Fold={fold} Acc={acc:.4f}")
    return (ds, method, fold, acc)

# ----------------------------
# Main logic
# ----------------------------
if __name__ == '__main__':
    # initialize results file
    with open('ragresults.txt', 'w') as f:
        f.write('Dataset,Method,MeanAccuracy\n')

    logger.info(f"Running in MODE='{MODE}', SAMPLE_FRAC={SAMPLE_FRAC}")
    # datasets = ['wine']
    datasets = [

        # "student-portuguese_hotencoded_50PassFail",   
        "student-math_hotencoded_50PassFail",  
        # "adult_hotencoded",
        "Telco-Customer-Churn_hotencoded_missingto0",
        # "diabetes_012_health_indicators_BRFSS2015",  
        "diabetes_binary_5050split_health_indicators_BRFSS2015", 
        "diabetes_binary_health_indicators_BRFSS2015",
        # "pendigits",
        # "satellite",
        # "segment",
        # "sign",
        # "sick_hotencoded",
        # "letter-recog",
        # "localization_hotencoded",
        "census-income_hotencoded"
        # "covtype",
        # "magic",

        # 'bank-full_hotencoded', 
        # 'Credit_Card_Applications', 
        # 'wine', 
        # 'zoo'

        # student-portuguese_hotencoded_50PassFail

        #Datasets for test. 
# adult_hotencoded
# diabetes_012_health_indicators_BRFSS2015
# pendigits
# satellite
# segment
# sign
# sick_hotencoded
# letter-recog
# localization_hotencoded
# covtype
# magic
# bank-full_hotencoded
# Credit_Card_Applications
# wine
# zoo
    ]
    n_splits = 2
    methods = ['KNN']
    k_neighbors = 2

    gpu_count = detect_gpu_count() or 1
    logger.info(f"Detected {gpu_count} GPU(s)")
    start = time.time()

    for ds in datasets:
        csvp = f'Datasets/{ds}/{ds}.csv'
        mp = f'Datasets/{ds}/{ds}_metadata.json'
        if not os.path.exists(csvp) or not os.path.exists(mp):
            logger.warning(f"Skipping {ds}: missing files")
            continue

        df = pd.read_csv(csvp, header=None)
        if SAMPLE_FRAC < 1.0:
            n = int(len(df) * SAMPLE_FRAC)
            df = df.head(n)

        meta = json.load(open(mp))
        tgt = meta['target_column_index']
        X, y = df.drop(columns=[tgt]), df.iloc[:, tgt]

        splits = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42).split(X, y))

        tasks = []
        for fold_idx, (tr, te) in enumerate(splits, start=1):
            X_tr, X_te = X.iloc[tr], X.iloc[te]
            y_tr, y_te = y.iloc[tr], y.iloc[te]
            for method in methods:
                tasks.append((ds, fold_idx, method, X_tr, y_tr, X_te, y_te))

        if MODE == 'single':
            for ds, fold, method, X_tr, y_tr, X_te, y_te in tasks:
                # only KNN will run here
                acc = np.mean(knn_predict_skl(X_tr, y_tr, X_te, k_neighbors) == y_te)
                with open('ragresults.txt', 'a') as f:
                    f.write(f"{ds},{method},{acc:.4f}\n")
        else:
            # only KNN tasks exist, so only CPU pooling is used
            cpu_tasks = tasks
            with ProcessPoolExecutor(max_workers=os.cpu_count()) as cpu_exec:
                for ds, fold, _, X_tr, y_tr, X_te, y_te in cpu_tasks:
                    cpu_exec.submit(run_knn_task, ds, fold, X_tr, y_tr, X_te, y_te, k_neighbors)

    total = time.time() - start
    logger.info(f"All done in {total:.2f}s")
