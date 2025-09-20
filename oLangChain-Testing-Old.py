import pysqlite3
import sys

# Monkey-patch SQLite to use pysqlite3 as sqlite3
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import pandas as pd
import numpy as np
import re
import json
import time
import logging
from sklearn.model_selection import StratifiedKFold
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama.llms import OllamaLLM
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

import math
from collections import Counter

import shutil

# For RAG:
import chromadb
import ollama
from chromadb.config import Settings
from langchain_core.documents import Document
import os
import shutil

# ----------------------------
# Logging Configuration
# ----------------------------

class JsonFormatter(logging.Formatter):
    """
    A custom logging formatter that outputs log records in JSON format.

    This formatter attempts to parse the log message as JSON, and if successful,
    places the parsed JSON object under the "message" key in the formatted output.
    If parsing fails, it treats the log message as a simple string and wraps it 
    in JSON.
    """
    def format(self, record):
        try:
            log_record = {
                "timestamp": self.formatTime(record, self.datefmt),
                "level": record.levelname,
                "message": json.loads(record.getMessage())
            }
            return json.dumps(log_record)
        except json.JSONDecodeError:
            return json.dumps({
                "timestamp": self.formatTime(record, self.datefmt),
                "level": record.levelname,
                "message": record.getMessage()
            })

def setup_logging(dataset_name):
    """
    Sets up two loggers (general and detailed) with dynamic file names based on the dataset name.
    
    The general logger writes logs to a file `<dataset_name>.log` in plain text,
    while the detailed logger writes logs to a file `<dataset_name>.jsonl` in JSON format.
    
    Parameters:
        dataset_name (str): The name of the dataset for which logs are being created.
        
    Returns:
        tuple: A tuple containing the general logger and the detailed logger. 
               The first element is the general logger (text logs),
               and the second element is the detailed logger (JSON logs).
    """
    # General logger
    general_logger = logging.getLogger(f'general_logger_{dataset_name}')
    general_logger.setLevel(logging.INFO)
    general_handler = logging.FileHandler(f'Logging/{dataset_name}.log')
    general_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    general_handler.setFormatter(general_formatter)
    general_logger.addHandler(general_handler)

    # Detailed logger
    detailed_logger = logging.getLogger(f'detailed_logger_{dataset_name}')
    detailed_logger.setLevel(logging.INFO)
    detailed_handler = logging.FileHandler(f'Logging/{dataset_name}.jsonl')
    detailed_formatter = JsonFormatter()
    detailed_handler.setFormatter(detailed_formatter)
    detailed_logger.addHandler(detailed_handler)

    # Return loggers for use
    return general_logger, detailed_logger

def log_general(message):
    """
    Logs a general message to both the console (via print) and the global general_logger.
    
    Parameters:
        message (str): The message to be logged in plain text.
    """
    print(message)
    try:
        general_logger.info(message)
    except NameError:
        logging.getLogger("default").info(message)


def log_detailed(message_dict):
    """
    Logs a detailed message to the global detailed_logger in JSON format.
    
    Parameters:
        message_dict (dict): A dictionary containing the message data to be logged as JSON.
    """
    json_message = json.dumps(message_dict)
    detailed_logger.info(json_message)

# ----------------------------
# Data Loading Function
# ----------------------------

def load_data(csv_file, json_file):
    """""
    Load dataset features and target labels from CSV and JSON metadata.

    Args:
        csv_file (str): Path to the CSV data file.
        json_file (str): Path to the JSON metadata file.

    Returns:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target labels.
        target_col_idx (int): Index of the target column.
    """
    df = pd.read_csv(csv_file, header=None)
    with open(json_file, "r") as f:
        metadata = json.load(f)

    target_col_idx = metadata["target_column_index"]
    log_general(f"Loaded data from {csv_file} and {json_file}. Target column index: {target_col_idx}")

    # Should we drop rows or fill with median????
    # df.dropna(subset=[target_col_idx], inplace=True)

    y = df.iloc[:, target_col_idx]
    X = df.drop(columns=[target_col_idx])

    return X, y, target_col_idx

# ----------------------------
# Pydantic Model for Classification
# ----------------------------

class ClassificationWithReasoning(BaseModel):
    classification: int = Field(..., description="The classification as an integer")
    reasoning: str = Field(..., description="The reasoning steps taken to arrive at the classification")

# ----------------------------
# LangChain Llama Classifier
# ----------------------------

class LangChainOllamaClassifier:
    """
    A simple classifier using a LangChain-based Llama model.
    
    This class implements fit, predict, score, and cross_validate
    methods that mimic scikit-learn's estimator interface.
    """

    def __init__(self, model="llama3.1:70b", max_training_examples=10):
        """
        Initializes the classifier with a specific model and optional limit on training examples.
        
        Parameters:
            model (str): The name or path of the Llama model to be used.
            max_training_examples (int): Maximum number of training examples to include in prompts.
        """
        self.model = model
        self.max_training_examples = max_training_examples
        self.X_train_ = None
        self.y_train_ = None
        self.current_fold = None

#         self.template = """
# You are a classification model. Below are some training examples in a Q&A format where Q is the input features and A is the correct classification.

# {few_shot_examples}

# Now, classify the following new case:

# Q: {test_case}

# Return your answer **exclusively** as a single JSON object with no additional text. The JSON must include exactly two keys: "classification" (an integer) and "reasoning" (a string). Do not include markdown, code blocks, or any extra text. 

# Example output:
# {{"classification": 1, "reasoning": "The features indicate a strong similarity to class 1 based on [reason]."}}

# Only output the JSON object.
# """.strip()

        self.template = """
You are a classification model. Below are some training examples in a Q&A format where Q is the input features and A is the correct classification.

{few_shot_examples}

Now, classify the following new case:

Q: {test_case}

Please provide your final answer **only** in JSON format, including both the classification and a brief reasoning. **Do not include any markdown, code blocks, or additional text.** The response should look exactly like this:

{{ "classification": <integer classification>, "reasoning": "<your reasoning here>" }}
""".strip()

        self.parser = PydanticOutputParser(pydantic_object=ClassificationWithReasoning)
        self.prompt = ChatPromptTemplate.from_template(self.template)
        self.llm = OllamaLLM(model=self.model, temperature=0)
        # self.llm = OllamaLLM(model=self.model)
        self.chain = self.prompt | self.llm

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the model by storing training data. 
        
        Parameters:
            X (pd.DataFrame): Training features.
            y (pd.Series): Training labels.
        
        Returns:
            self: The fitted classifier.
        """
        self.X_train_ = X.reset_index(drop=True)
        self.y_train_ = y.reset_index(drop=True)
        log_general(f"Fitted model {self.model} with {len(X)} training examples.")
        return self

    def extract_json(self, response: str) -> str:
        """
        Extracts valid JSON from a response string if wrapped in backticks, otherwise returns the original.
        
        Parameters:
            response (str): The response string from the model.
        
        Returns:
            str: Extracted JSON or the raw response if no JSON pattern is found.
        """
        pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1)
        else:
            return response.strip()

    def predict(self, X: pd.DataFrame, y_true: pd.Series, max_retries=3) -> np.ndarray:
        """
        Uses the trained model to predict labels for each sample in X.
        
        Parameters:
            X (pd.DataFrame): Test features.
            y_true (pd.Series): Ground truth labels for accuracy logging.
            max_retries (int): Maximum number of times to retry parsing a response.
        
        Returns:
            np.ndarray: Array of predicted labels. A value of -1 indicates a parsing failure.
        """
        if self.current_fold is None:
            log_general("Warning: current_fold is not set. Predictions may not be associated with any fold.")

        train_sample = pd.concat([self.X_train_, self.y_train_], axis=1)

        # Sample from the entire training set (existing approach)
        if len(train_sample) > self.max_training_examples:
            train_sample = train_sample.sample(n=self.max_training_examples, random_state=42)

        few_shot_examples_lines = []
        for _, row in train_sample.iterrows():
            features = row.iloc[:-1].tolist()
            label = row.iloc[-1]
            q_line = f"Q: {', '.join(map(str, features))}"
            a_line = f"A: {label}"
            few_shot_examples_lines.append(q_line)
            few_shot_examples_lines.append(a_line)

        few_shot_examples = "\n".join(few_shot_examples_lines)
        predictions = []

        for i in range(len(X)):
            row = X.iloc[i]
            true_label = y_true.iloc[i]
            test_case_str = ", ".join(map(str, row.tolist()))

            prompt_str = self.prompt.format(
                few_shot_examples=few_shot_examples, 
                test_case=test_case_str
            )

            log_detailed({
                "event": "prompt",
                "fold": self.current_fold,
                "test_case": i + 1,
                "prompt": prompt_str,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })

            response = self.chain.invoke({"few_shot_examples": few_shot_examples, "test_case": test_case_str})
            
            log_detailed({
                "event": "response",
                "fold": self.current_fold,
                "test_case": i + 1,
                "response": response,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })

            retries = 0
            success = False
            while retries < max_retries and not success:
                try:
                    clean_response = self.extract_json(response)
                    parsed_output = self.parser.parse(clean_response)
                    prediction = parsed_output.classification
                    correct = bool(prediction == true_label)
                    predictions.append(prediction)

                    log_detailed({
                        "event": "prediction_accuracy",
                        "fold": self.current_fold,
                        "test_case": i + 1,
                        "correct": correct,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                    success = True
                except Exception as e:
                    retries += 1
                    log_general(f"Attempt {retries}: Error parsing response for fold {self.current_fold}, test case {i + 1}: {e}")
                    if retries == max_retries:
                        log_general(f"Max retries reached for fold {self.current_fold}, test case {i + 1}. Marking prediction as -1.")
                        predictions.append(-1)
                    else:
                        log_general(f"Retrying fold {self.current_fold}, test case {i + 1}...")

        return np.array(predictions)

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Computes accuracy by comparing predictions to true labels, ignoring failed predictions (-1).
        
        Parameters:
            X (pd.DataFrame): Test features.
            y (pd.Series): True labels for scoring.
        
        Returns:
            float: The accuracy of the classifier on the given test data.
        """
        predictions = self.predict(X, y)
        valid_indices = predictions != -1
        if np.sum(valid_indices) == 0:
            log_general("No valid predictions to calculate accuracy.")
            return 0.0
        accuracy = np.mean(predictions[valid_indices] == y[valid_indices])
        return accuracy

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, n_splits=5):
        """
        Performs stratified k-fold cross-validation on the dataset.
        
        Parameters:
            X (pd.DataFrame): Input features.
            y (pd.Series): Labels corresponding to the features.
            n_splits (int): Number of folds for cross-validation.
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_accuracies = []
        start_time = time.time()

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            self.current_fold = fold_idx + 1
            log_general(f"=== Starting Fold {self.current_fold}/{n_splits} ===")
            fold_start_time = time.time()

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            self.fit(X_train, y_train)
            accuracy = self.score(X_test, y_test)
            fold_accuracies.append(accuracy)

            fold_time = time.time() - fold_start_time
            log_general(f"=== Completed Fold {self.current_fold}/{n_splits} ===")
            log_general(f"Fold {self.current_fold} Accuracy: {accuracy:.4f}")
            log_general(f"Fold {self.current_fold} Duration: {fold_time:.2f} seconds\n")

            self.current_fold = None

        total_time = time.time() - start_time
        mean_accuracy = np.mean(fold_accuracies)
        log_general(f"=== Cross-Validation Complete ===")
        log_general(f"Mean Accuracy: {mean_accuracy:.4f}")
        log_general(f"Total Duration: {total_time:.2f} seconds\n")

# ----------------------------
# Extended Class Using N Examples per Class prompt
# ----------------------------

class LangChainOllamaClassifierByClass(LangChainOllamaClassifier):
    """
    Extends LangChainLlamaClassifier to create few-shot examples based on classes 
    rather than a fixed total sample size. Each class gets a specified number 
    of examples for prompt construction.
    """
    def __init__(self, model="llama3.1:70b", examples_per_class=2):
        """
        Initializes the classifier, specifying how many examples to use per class.
        
        Parameters:
            model (str): The name or path of the Llama model to be used.
            examples_per_class (int): Number of training examples to include 
                                      in the prompt for each class.
        """
        # Parent constructor with a large max_training_examples (not used directly here).
        super().__init__(model=model, max_training_examples=999999)
        self.examples_per_class = examples_per_class

        # Overwrite the parent's template to show examples grouped by class.
#         self.template = """
# You are a classification model. Below are example data points grouped by class. Each data point lists its features (comma-separated). Then, you will be given a new data point to classify.

# Training examples by class:

# {few_shot_examples}

# Now, classify the following new data point (features comma-separated):
# {test_case}

# Return ONLY a JSON object with exactly two keys:
# - "classification": an integer representing the predicted class.
# - "reasoning": a brief explanation for your decision.

# Do NOT include any extra text, markdown, or formatting. The JSON must follow this exact format:

# {{"classification": <integer>, "reasoning": "<your reasoning here>"}}
# """.strip()

        self.template = """
You are a classification model. Below are example data points grouped by class. Each data point lists its features (comma separated). Then, you will be given a new data point to classify.

Training examples by class:

{few_shot_examples}

Now, classify the following new data point (features comma separated):
{test_case}

Please provide your final answer **only** in JSON format, including both the classification and a brief reasoning. **Do not include any markdown, code blocks, or additional text.** The response should look exactly like this:

{{ "classification": <integer classification>, "reasoning": "<your reasoning here>" }}
""".strip()

        # Reinitialize parser, prompt, llm, and chain to reflect the new template.
        self.parser = PydanticOutputParser(pydantic_object=ClassificationWithReasoning)
        self.prompt = ChatPromptTemplate.from_template(self.template)
        self.llm = OllamaLLM(model=self.model, temperature=0)
        # self.llm = OllamaLLM(model=self.model)
        self.chain = self.prompt | self.llm

    def _build_few_shot_examples_by_class(self) -> str:
        """
        Builds a string of few-shot examples for each class, using 
        up to `examples_per_class` training samples per class.
        
        Returns:
            str: The text to be inserted into the prompt for few-shot learning.
        """
        train_df = pd.concat([self.X_train_, self.y_train_], axis=1)
        class_labels = train_df.iloc[:, -1].unique()  # unique classes

        lines = []
        for class_label in class_labels:
            df_class = train_df[train_df.iloc[:, -1] == class_label]
            df_class_sampled = df_class.sample(
                n=min(self.examples_per_class, len(df_class)), 
                random_state=42
            )

            lines.append(f"Class {class_label} examples:")
            for _, row in df_class_sampled.iterrows():
                features = row.iloc[:-1].tolist()
                features_str = ", ".join(map(str, features))
                lines.append(f"({features_str})")

            lines.append("")

        return "\n".join(lines).strip()

    def predict(self, X: pd.DataFrame, y_true: pd.Series, max_retries=3) -> np.ndarray:
        """
        Predicts class labels for the given DataFrame using examples grouped by class.
        
        Parameters:
            X (pd.DataFrame): Test features.
            y_true (pd.Series): Ground truth labels for logging accuracy.
            max_retries (int): Maximum times to retry parsing the model's response.
        
        Returns:
            np.ndarray: Predicted labels for each row in X (-1 for parsing failures).
        """
        if self.current_fold is None:
            log_general("Warning: current_fold is not set. Predictions may not be associated with any fold.")

        # Build few-shot prompt from examples grouped by class
        few_shot_examples = self._build_few_shot_examples_by_class()
        predictions = []

        for i in range(len(X)):
            row = X.iloc[i]
            true_label = y_true.iloc[i]
            test_case_str = ", ".join(map(str, row.tolist()))

            prompt_str = self.prompt.format(
                few_shot_examples=few_shot_examples,
                test_case=test_case_str
            )

            log_detailed({
                "event": "prompt",
                "fold": self.current_fold,
                "test_case": i + 1,
                "prompt": prompt_str,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })

            response = self.chain.invoke({
                "few_shot_examples": few_shot_examples, 
                "test_case": test_case_str
            })

            log_detailed({
                "event": "response",
                "fold": self.current_fold,
                "test_case": i + 1,
                "response": response,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })

            retries = 0
            success = False
            while retries < max_retries and not success:
                try:
                    clean_response = self.extract_json(response)
                    parsed_output = self.parser.parse(clean_response)
                    prediction = parsed_output.classification
                    correct = bool(prediction == true_label)
                    predictions.append(prediction)

                    log_detailed({
                        "event": "prediction_accuracy",
                        "fold": self.current_fold,
                        "test_case": i + 1,
                        "correct": correct,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                    success = True
                except Exception as e:
                    retries += 1
                    log_general(f"Attempt {retries}: Error parsing response for fold {self.current_fold}, test case {i + 1}: {e}")
                    if retries == max_retries:
                        log_general(f"Max retries reached for fold {self.current_fold}, test case {i + 1}. Marking prediction as -1.")
                        predictions.append(-1)
                    else:
                        log_general(f"Retrying fold {self.current_fold}, test case {i + 1}...")

        return np.array(predictions)
        
# --------------------------------------------------
# Chroma Vector Store Manager
# --------------------------------------------------

class ChromaVectorStoreManager:
    """
    A reusable class for creating and managing a Chroma-based vector store
    that stores embedded training data rows for a given dataset.

    This class relies on Ollama for generating embeddings, then stores
    them in a local Chroma database (SQLite) in a specified persistent directory.
    """

    def __init__(self, persist_path):
        """
        Initializes the manager with fixed settings.

        Args:
            persist_path (str): Path to the folder where the Chroma DB will be stored.
        """
        # The collection name is fixed as "VecDB"
        self.dataset_name = "VecDB"
        self.persist_path = persist_path
        self.model_name = "all-minilm"

        # Create a persistent Chroma client pointed at our unique folder
        self.client = chromadb.PersistentClient(path=self.persist_path)

    def build_vectorstore(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Converts each training row (X, y) into text, obtains an embedding via Ollama,
        and stores the results in a Chroma collection named 'VecDB'.

        If a collection named 'VecDB' already exists, it is deleted first.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
        """
        # Check for existing collections
        existing_collections = self.client.list_collections()
        if self.dataset_name in existing_collections:
            # If the 'VecDB' collection exists, delete it before recreating
            self.client.delete_collection(name=self.dataset_name)
            print(f"Deleted existing collection '{self.dataset_name}'.")

        # Create a new collection named 'VecDB'
        collection = self.client.create_collection(name=self.dataset_name)
        print(f"Created collection '{self.dataset_name}'.")

        # For each row of X_train, generate an embedding and store
        for i in range(len(X_train)):
            # Convert the row (features) and label into a textual representation
            row_values = X_train.iloc[i].tolist()
            row_str = ", ".join(map(str, row_values))
            label_str = str(y_train.iloc[i])
            content = f"Features: {row_str}. Label: {label_str}"

            # Generate embedding via Ollama
            response = ollama.embeddings(model=self.model_name, prompt=content)
            embedding = response["embedding"]

            # Add this document + embedding to the Chroma collection
            collection.add(
                documents=[content],
                embeddings=[embedding],
                ids=[str(i)]
            )

        print(f"Successfully loaded and embedded {len(X_train)} rows "
              f"into Chroma collection '{self.dataset_name}'.")

    def query(self, query_text: str, n_results: int = 3):
        """
        Performs a similarity search in the Chroma collection using the given text
        and returns up to n_results documents.

        Args:
            query_text (str): The text prompt used for the similarity query.
            n_results (int): Number of results to retrieve.

        Returns:
            str: A single formatted string containing the retrieved documents.
        """
        # Retrieve the existing collection
        collection = self.client.get_collection(name=self.dataset_name)

        # Perform a similarity query
        response = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=["documents"]
        )

        # response["documents"] is a list of lists of strings; we flatten it
        all_documents = [doc for sublist in response["documents"] for doc in sublist]
        # Join all documents with a line break
        formatted_response = " ,\n ".join(all_documents)
        return formatted_response

class LangChainOllamaRAGClassifier:
    """
    A classifier that uses a Llama model combined with a retrieval-augmented generation (RAG) approach.
    
    It retrieves the most relevant training examples from a vectorstore
    and uses them as context for classification.
    """
    def __init__(
        self,
        model="llama3.1",
        k_neighbors=3,
        persist_directory="chroma_db"
    ):
        """
        Initialize the classifier and prepare the vectorstore.

        Args:
            model (str): Name of the Llama model to use (e.g., "llama3.1" or "llama3.1:70b").
            k_neighbors (int): How many training examples to retrieve at inference time.
            persist_directory (str): Where to store the Chroma database.
        """
        self.model = model
        self.k_neighbors = k_neighbors
        self.persist_directory = persist_directory

        shutil.rmtree(self.persist_directory, ignore_errors=True)
        os.makedirs(self.persist_directory, exist_ok=True)
        log_detailed(f"Cleared and prepared Chroma database directory for dataset {datasetname}.")

        # Initialize ChromaVectorStoreManager
        self.vec_store_manager = ChromaVectorStoreManager(self.persist_directory)

        # Prompt template for the LLM
#         self.template = """
# You are a classification model.

# Follow these instructions EXACTLY:
# 1) Return ONLY a JSON object with exactly two keys:
#    - "classification": an integer that is your predicted class.
#    - "reasoning": a brief explanation (string) for your decision.
# 2) Do NOT include any additional text, markdown, or code formatting.
# 3) The JSON must start with '{{' and end with '}}' without any extra content.

# Here are some relevant training examples:

# {retrieved_examples}

# Now, classify the following new data point:

# {test_case}

# The JSON must match this exact format:

# {{"classification": <integer>, "reasoning": "<your reasoning here>"}}
# """.strip()

        self.template = """
You are a classification model.

You MUST follow these rules:
1) Return the final answer ONLY in JSON format.
2) Do NOT include any additional text before or after the JSON.
3) Do NOT include markdown formatting or code fences.

Here are some relevant training examples:

{retrieved_examples}

Now classify the following new data point:

You must Return ONLY valid JSON in the format:
{{ "classification": <integer>, "reasoning": "<some text>" }}
""".strip()

        # Setup LLM and chain
        self.parser = PydanticOutputParser(pydantic_object=ClassificationWithReasoning)
        self.prompt = ChatPromptTemplate.from_template(self.template)
        self.llm = OllamaLLM(model=self.model, temperature=0)
        # self.llm = OllamaLLM(model=self.model)
        self.chain = self.prompt | self.llm

        # Placeholder for current fold during cross-validation
        self.current_fold = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit step: build the vectorstore from training data for retrieval.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Training labels.

        Returns:
            self: For method chaining.
        """
        self.vec_store_manager.build_vectorstore(X, y)
        log_general(f"Built vectorstore with {len(X)} training examples.")
        return self

    def extract_json(self, response: str) -> str:
        """
        Attempt to extract a JSON object from the response text.

        Args:
            response (str): Raw response text from the LLM.

        Returns:
            str: Extracted JSON string or the original response.
        """
        pattern = r"\{.*?\}"  # Naive pattern to extract the first {...}
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(0)
        else:
            return response.strip()

    def predict(self, X_test: pd.DataFrame, y_true: pd.Series, max_retries=3) -> np.ndarray:
        """        
        Predict classifications for a test dataset.

        Args:
            X_test (pd.DataFrame): Test features.
            y_true (pd.Series, optional): True labels for logging accuracy.
            max_retries (int): Max attempts to parse a valid JSON response.

        Returns:
            np.ndarray: Predicted class labels.
        """
        if self.current_fold is None:
            log_general("Warning: current_fold is not set. Predictions may not be associated with any fold.")

        predictions = []

        # Loop over each row in X_test
        for i in range(len(X_test)):
            row = X_test.iloc[i]
            test_case_str = "Features: " + ", ".join(map(str, row.tolist()))
            true_label = y_true.iloc[i] if y_true is not None else None

            # 1. Retrieve top-k similar documents from Chroma
            if not self.vec_store_manager.client:
                raise ValueError("Vectorstore is not initialized. Call fit() first.")

            retrieved_docs = self.vec_store_manager.query(
                query_text=test_case_str,
                n_results=self.k_neighbors
            )

            # Turn the retrieved docs string into a multi-line string
            # so it can fit into the prompt easily
            retrieved_examples_str = "\n".join(retrieved_docs.split(" ,\n "))

            # 2. Format the entire prompt
            prompt_str = self.prompt.format(
                retrieved_examples=retrieved_examples_str,
                test_case=test_case_str
            )

            # 3. Log the prompt details
            log_detailed({
                "event": "prompt",
                "fold": self.current_fold,
                "test_case": i + 1,
                "prompt": prompt_str,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })

            # 4. Invoke the LLM chain
            response = self.chain.invoke({
                "retrieved_examples": retrieved_examples_str,
                "test_case": test_case_str
            })

            # 5. Log the raw response
            log_detailed({
                "event": "response",
                "fold": self.current_fold,
                "test_case": i + 1,
                "response": response,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })

            # 6. Try to parse the JSON response up to max_retries times
            retries = 0
            success = False
            prediction = -1  # default if we fail
            while retries < max_retries and not success:
                try:
                    # Attempt to parse as JSON -> Pydantic model
                    parsed_output = self.parser.parse(response)
                    prediction = parsed_output.classification
                    success = True
                except Exception as e:
                    retries += 1
                    log_general(f"Attempt {retries}: Error parsing response for fold {self.current_fold}, test case {i+1}: {e}")

                    if retries == max_retries:
                        log_general(f"Max retries reached for fold {self.current_fold}, test case {i+1}. Marking prediction as -1.")
                        prediction = -1
                    else:
                        log_general(f"Retrying fold {self.current_fold}, test case {i+1}...")

            # 7. Record the prediction and (optionally) log whether it matched true_label
            predictions.append(prediction)
            if true_label is not None and prediction != -1:
                correct = bool(prediction == true_label)
                log_detailed({
                    "event": "prediction_accuracy",
                    "fold": self.current_fold,
                    "test_case": i + 1,
                    "correct": correct,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })

        return np.array(predictions, dtype=int)

    def score(self, X_test: pd.DataFrame, y_test: pd.Series) -> float:
        """
        Calculate accuracy of predictions.

        Args:
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): True labels for the test set.

        Returns:
            float: Accuracy score.
        """
        preds = self.predict(X_test, y_test)
        valid = preds != -1
        if valid.sum() == 0:
            log_general("No valid predictions to calculate accuracy.")
            return 0.0
        return np.mean(preds[valid] == y_test[valid])

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, n_splits=5):
        """
        Perform cross-validation with stratified folds.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target labels.
            n_splits (int): Number of cross-validation folds.

        Returns:
            float: Average accuracy across folds.
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        fold_accuracies = []
        start_time = time.time()

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            self.current_fold = fold_idx + 1
            log_general(f"=== Starting Fold {self.current_fold}/{n_splits} ===")
            fold_start_time = time.time()
            log_general(f"building vectorstore and predicting...")

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Build vectorstore on train
            self.fit(X_train, y_train)

            # Score on test
            acc = self.score(X_test, y_test)
            fold_accuracies.append(acc)

            fold_time = time.time() - fold_start_time
            log_general(f"=== Completed Fold {self.current_fold}/{n_splits} ===")
            log_general(f"Fold {self.current_fold} Accuracy: {acc:.4f}")
            log_general(f"Fold {self.current_fold} Duration: {fold_time:.2f} seconds\n")

            self.current_fold = None

        total_time = time.time() - start_time
        mean_accuracy = np.mean(fold_accuracies)
        log_general(f"=== Cross-Validation Complete ===")
        log_general(f"Mean Accuracy: {mean_accuracy:.4f}")
        log_general(f"Total Duration: {total_time:.2f} seconds\n")

        return mean_accuracy

class LangChainCoTClassifier(LangChainOllamaClassifier):
    """
    A specialized classifier that handles chain-of-thought (CoT) text by 
    extracting everything before the final JSON block and placing it 
    inside the 'reasoning' field of the JSON.
    
    Inherits LangChainLlamaClassifier for everything else (fit, predict, etc.)
    but overrides only 'extract_json'.
    """
    def extract_json(self, response: str) -> str:
        """
        1. Extract chain-of-thought from <think>...</think> if present.
        2. Remove that <think> block from 'response'.
        3. Look for final JSON block. If none found:
             - Try to parse classification from 'Answer: \\boxed{...}' or fallback patterns.
             - Construct a JSON object from that classification + chain_of_thought.
        4. Return final JSON string (or raw response on complete failure).
        """

        # -----------------------------
        # 1) Grab chain-of-thought in <think>...</think>
        # -----------------------------
        think_pattern = r"<think>(.*?)</think>"
        think_match = re.search(think_pattern, response, flags=re.DOTALL)

        chain_of_thought = ""
        if think_match:
            chain_of_thought = think_match.group(1).strip()

        # Remove the <think> block from text so it doesn't interfere with other parsing
        response_no_think = re.sub(think_pattern, "", response, flags=re.DOTALL).strip()
        log_general(f"DEBUG response_no_think:{repr(response_no_think)}")
        # -----------------------------
        # 2) Check if there's a valid JSON block
        #    (maybe the model sometimes DOES produce JSON).
        # -----------------------------
        block_pattern = r"\{[^{}]*\}"
        matches = re.findall(block_pattern, response_no_think)

        if matches:
            # If we found at least one top-level curly brace block,
            # try to parse the last one as valid JSON
            last_block = matches[-1]
            try:
                data = json.loads(last_block)

                # Merge chain-of-thought if needed
                if "reasoning" not in data:
                    data["reasoning"] = chain_of_thought

                return json.dumps(data, ensure_ascii=False)
            except json.JSONDecodeError:
                pass  # Continue to fallback

        # -----------------------------
        # 3) No valid JSON found or JSON parse failed
        #    => fallback: parse classification from pattern
        # -----------------------------

        # Example pattern:  "Answer: \boxed{1}"
        # or maybe "Answer: 1", "classification = 0", etc.
        # Adjust your pattern if needed.
        classification_pattern = r"Answer.*?\\boxed\{(\d+)\}"
        match_answer = re.search(classification_pattern, response_no_think, flags=re.DOTALL)
        classification = None

        if match_answer:
            classification = match_answer.group(1)  # e.g. "1"
        else:
            # Optionally, a second fallback if the box isn't there
            # e.g. "Answer: 1"
            fallback_answer_pattern = r"Answer:\s*(\d+)"
            match_answer2 = re.search(fallback_answer_pattern, response_no_think)
            if match_answer2:
                classification = match_answer2.group(1)

        if classification:
            # Construct JSON string from classification + chain_of_thought
            data = {
                "classification": int(classification),
                "reasoning": chain_of_thought
            }
            return json.dumps(data, ensure_ascii=False)

        # -----------------------------
        # 4) If everything fails, just return the raw response
        #    -> your existing retry logic can handle or skip
        # -----------------------------
        return response.strip()
         
class LangChainCoTClassifierByClass(LangChainOllamaClassifierByClass):
    """
    A specialized classifier that handles chain-of-thought (CoT) text by 
    extracting everything before the final JSON block and placing it 
    inside the 'reasoning' field of the JSON.
    
    Inherits LangChainLlamaClassifier for everything else (fit, predict, etc.)
    but overrides only 'extract_json'.
    """
    def extract_json(self, response: str) -> str:
        """
        1. Extract chain-of-thought from <think>...</think> if present.
        2. Remove that <think> block from 'response'.
        3. Look for final JSON block. If none found:
             - Try to parse classification from 'Answer: \\boxed{...}' or fallback patterns.
             - Construct a JSON object from that classification + chain_of_thought.
        4. Return final JSON string (or raw response on complete failure).
        """

        # -----------------------------
        # 1) Grab chain-of-thought in <think>...</think>
        # -----------------------------
        think_pattern = r"<think>(.*?)</think>"
        think_match = re.search(think_pattern, response, flags=re.DOTALL)

        chain_of_thought = ""
        if think_match:
            chain_of_thought = think_match.group(1).strip()

        # Remove the <think> block from text so it doesn't interfere with other parsing
        response_no_think = re.sub(think_pattern, "", response, flags=re.DOTALL).strip()
        log_general(f"DEBUG response_no_think:{repr(response_no_think)}")
        # -----------------------------
        # 2) Check if there's a valid JSON block
        #    (maybe the model sometimes DOES produce JSON).
        # -----------------------------
        block_pattern = r"\{[^{}]*\}"
        matches = re.findall(block_pattern, response_no_think)

        if matches:
            # If we found at least one top-level curly brace block,
            # try to parse the last one as valid JSON
            last_block = matches[-1]
            try:
                data = json.loads(last_block)

                # Merge chain-of-thought if needed
                if "reasoning" not in data:
                    data["reasoning"] = chain_of_thought

                return json.dumps(data, ensure_ascii=False)
            except json.JSONDecodeError:
                pass  # Continue to fallback

        # -----------------------------
        # 3) No valid JSON found or JSON parse failed
        #    => fallback: parse classification from pattern
        # -----------------------------

        # Example pattern:  "Answer: \boxed{1}"
        # or maybe "Answer: 1", "classification = 0", etc.
        # Adjust your pattern if needed.
        classification_pattern = r"Answer.*?\\boxed\{(\d+)\}"
        match_answer = re.search(classification_pattern, response_no_think, flags=re.DOTALL)
        classification = None

        if match_answer:
            classification = match_answer.group(1)  # e.g. "1"
        else:
            # Optionally, a second fallback if the box isn't there
            # e.g. "Answer: 1"
            fallback_answer_pattern = r"Answer:\s*(\d+)"
            match_answer2 = re.search(fallback_answer_pattern, response_no_think)
            if match_answer2:
                classification = match_answer2.group(1)

        if classification:
            # Construct JSON string from classification + chain_of_thought
            data = {
                "classification": int(classification),
                "reasoning": chain_of_thought
            }
            return json.dumps(data, ensure_ascii=False)

        # -----------------------------
        # 4) If everything fails, just return the raw response
        #    -> your existing retry logic can handle or skip
        # -----------------------------
        return response.strip()
# ----------------------------
# Main Execution
# ----------------------------

if __name__ == "__main__":
    
    # Start timing the total execution
    total_start_time = time.time()

    # List of datasets to test
    datasets = [        
        # "bank-full_hotencoded",    
        # "Credit_Card_Applications",
        # "wine",        
        # "zoo",
        "student-portuguese_hotencoded_50PassFail",   
        "student-math_hotencoded_50PassFail",  
        "adult_hotencoded",
        "Telco-Customer-Churn_hotencoded_missingto0",
        # "diabetes_012_health_indicators_BRFSS2015",  
        # "diabetes_binary_5050split_health_indicators_BRFSS2015", 
        # "diabetes_binary_health_indicators_BRFSS2015",
        # "pendigits",
        # "satellite",
        # "segment",
        # "sign",
        # "sick_hotencoded",
        # "letter-recog",
        # "localization_hotencoded",
        # "census-income_hotencoded",
        # "covtype",
        # "magic",
        ]

    # Define the number of splits
    n_splits = 3
    few_shot_examples = 10
    class_examples = 5
    MAX_DATA_SIZE = 10000

    
    # Here we calculate a recommended MIN_PER_CLASS
    # Explanation:
    #   - We need at least `class_examples` examples per class in each training fold.
    #   - Each training fold is ~ (n_splits-1)/n_splits of the data.
    #   - Multiply by (n_splits / (n_splits - 1)) to invert that fraction.
    #   - Include a "buffer_factor" so random splits don’t leave us short.
    buffer_factor = 2.0  # Tune
    fold_factor = n_splits / (n_splits - 1)  # e.g. for 10 folds, 10/9 ≈ 1.11
    MIN_PER_CLASS = math.ceil(class_examples * fold_factor * buffer_factor)


    # Initialize overall results
    overall_results = {}

    for datasetname in datasets:
        try:
            dataset_start_time = time.time()  # Start timing for this dataset

            # Set up logging for the specific dataset
            general_logger, detailed_logger = setup_logging("Log_Testing_withMax500_and_10fold_" + datasetname)
            log_general(f"\n=== Testing Dataset: {datasetname} ===")
            
            # File paths for the dataset
            data_folder = f"Datasets/{datasetname}"
            csv_file = f"{data_folder}/{datasetname}.csv"
            json_file = f"{data_folder}/{datasetname}_metadata.json"
            
            # Check that the required dataset files exist
            if not os.path.exists(csv_file):
                log_general(f"[ERROR] CSV file not found for dataset {datasetname} at {csv_file}. Skipping.")
                continue
            if not os.path.exists(json_file):
                log_general(f"[ERROR] JSON metadata file not found for dataset {datasetname} at {json_file}. Skipping.")
                continue
            
            # Load data
            X, y, target_col_idx = load_data(csv_file, json_file)
            
            # (The rest of your dataset processing code goes here, including downsampling, classifier evaluation, etc.)
            
            dataset_end_time = time.time()  # End timing for this dataset
            dataset_duration = dataset_end_time - dataset_start_time
            log_general(f"=== Results for {datasetname} ===")
            # (Output per-dataset results, etc.)
            log_general(f"Total Time for {datasetname}: {dataset_duration:.2f} seconds")
            log_general(f"=== End of Cross-Validation for {datasetname} ===\n")
            
        except Exception as e:
            log_general(f"[ERROR] Skipping dataset {datasetname} due to error: {e}")
            continue


        # Downsample if dataset too large
        # from sklearn.model_selection import train_test_split

        # if len(X) > MAX_DATA_SIZE:
        #     current_size = MAX_DATA_SIZE

        #     while True:
        #         # Attempt a stratified sample of size `current_size`
        #         X_sub, _, y_sub, _ = train_test_split(
        #             X,
        #             y,
        #             train_size=current_size,
        #             stratify=y,
        #             random_state=42
        #         )
        #         # Check class counts
        #         counts = Counter(y_sub)
        #         if all(count >= MIN_PER_CLASS for count in counts.values()):
        #             # We have enough examples per class; accept this subset
        #             X, y = X_sub, y_sub
        #             log_general(
        #                 f"Downsampled {datasetname} to {len(X)} rows. "
        #                 f"Class distribution: {dict(counts)}"
        #             )
        #             break
        #         else:
        #             current_size += 25  # increase in small steps
        #             if current_size >= len(X):
        #                 # If we exceed total data size, just use the full dataset
        #                 log_general(
        #                     f"Could not ensure {MIN_PER_CLASS} min per class with "
        #                     f"downsampling; using full dataset of {len(X)} rows."
        #                 )
        #                 break


        classifiers = [
            #             {
            #     "name": "LLaMA-8b-FewShot",
            #     "clf": LangChainOllamaClassifier(
            #         model="llama3.1", 
            #         max_training_examples=few_shot_examples
            #     ),
            # }
            {
                "name": "LLaMA-8b-RAG",
                "clf": LangChainOllamaRAGClassifier(
                    model="llama3.1",  # 8b version
                    k_neighbors=few_shot_examples,  
                    persist_directory=f"/home/lenora/LLM-Tabular-benchmarking-code/databases/{datasetname}"
                ),
            },
            # {
            #     "name": "LLaMA-8b-NSamples",
            #     "clf": LangChainOllamaClassifierByClass(
            #         model="llama3.1", 
            #         examples_per_class=class_examples
            #     ),
            # },
            # {
            #     "name": "LLaMA-70b-FewShot",
            #     "clf": LangChainOllamaClassifier(
            #         model="llama3.1:70b", 
            #         max_training_examples=few_shot_examples
            #     ),
            # },
            # {
            #     "name": "LLaMA-70b-NSamples",
            #     "clf": LangChainOllamaClassifierByClass(
            #         model="llama3.1:70b", 
            #         examples_per_class=class_examples
            #     ),
            # },
            {
                "name": "LLaMA-70b-RAG",
                "clf": LangChainOllamaRAGClassifier(
                    model="llama3.1:70b",  # 70b version
                    k_neighbors=few_shot_examples,  
                    persist_directory=f"/home/lenora/LLM-Tabular-benchmarking-code/databases/{datasetname}_70B"
                ),
            },
            # {
            #     "name": "LogisticRegression",
            #     "clf": LogisticRegression(C=1.0, max_iter=10000),
            # },
            #{
            #    "name": "Deepseek-r1-FewShot",
            #    "clf": LangChainCoTClassifier(
            #        model="Deepseek-r1", 
            #        max_training_examples=few_shot_examples
            #    ),
            #},
            #{
            #    "name": "Deepseek-r1-NSamples",
            #    "clf": LangChainCoTClassifierByClass(
            #        model="Deepseek-r1", 
            #        examples_per_class=class_examples
            #    ),
            #},
        ]

        # Cross-validation setup
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_accuracies = {clf_info["name"]: [] for clf_info in classifiers}

        log_general(f"=== Starting Cross-Validation for {datasetname} ===")

        # Perform cross-validation
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            log_general(f"=== Starting Fold {fold}/{n_splits} ===")
            fold_start_time = time.time()

            # Split the data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Evaluate each classifier
            for clf_info in classifiers:
                name = clf_info["name"]
                clf = clf_info["clf"]

                # Set current fold for logging or debugging inside the classifier
                clf.current_fold = fold

                log_general(f"--- Training {name} ---")

                # Fit the model (both LLaMA-based and LR need .fit)
                clf.fit(X_train, y_train)

                # Generate predictions and compute accuracy
                if name == "LogisticRegression":
                    # For scikit-learn:
                    predictions = clf.predict(X_test)
                    accuracy = np.mean(predictions == y_test)
                else:
                    # For LLaMA-based classifiers:
                    accuracy = clf.score(X_test, y_test)

                fold_accuracies[name].append(accuracy)

                # Logging
                log_general(f"{name} Fold {fold} Accuracy: {accuracy:.4f}")
                log_detailed({
                    "event": "cross_validation_accuracy",
                    "classifier": name,
                    "fold": fold,
                    "accuracy": accuracy,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })

            # Calculate fold duration
            fold_time = time.time() - fold_start_time
            log_general(f"=== Completed Fold {fold}/{n_splits} ===")
            log_general(f"Fold {fold} Duration: {fold_time:.2f} seconds\n")

        # Compute average accuracies
        mean_accuracies = {
            name: np.mean(acc_list) 
            for name, acc_list in fold_accuracies.items()
        }

        # Store results for this dataset
        overall_results[datasetname] = mean_accuracies

        dataset_end_time = time.time()  # End timing for this dataset
        dataset_duration = dataset_end_time - dataset_start_time

        # Log results for this dataset
        log_general(f"=== Results for {datasetname} ===")
        for name, mean_acc in mean_accuracies.items():
            log_general(f"{name} Mean Accuracy: {mean_acc:.4f}")
        log_general(f"Total Time for {datasetname}: {dataset_duration:.2f} seconds")
        log_general(f"=== End of Cross-Validation for {datasetname} ===\n")

    general_logger, detailed_logger = setup_logging("Overal_Results_Max500_10Fold")

    # Final overall results
    log_general("\n=== Overall Results ===")
    for dataset, results in overall_results.items():
        log_general(f"Dataset: {dataset}")
        for classifier, accuracy in results.items():
            log_general(f"  {classifier}: {accuracy:.4f}")

    # Log total time for all datasets
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    log_general(f"=== Total Time for All Datasets: {total_duration:.2f} seconds ===")

    log_general("=== End of Testing ===")
