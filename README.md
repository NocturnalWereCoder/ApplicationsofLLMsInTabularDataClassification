# LLaMA-Based Tabular Data Classification with Logging & Benchmarking

This repository contains a Python script for benchmarking classification performance using both LLaMA-based large language models (LLMs) and traditional methods (e.g., Logistic Regression). The code includes:

- Multiple custom classifier classes:
  - **Few-shot** Ollama classifier
  - **Retriever-Augmented Generation (RAG)** Ollama classifier
  - **Class-based few-shot** Ollama classifier
  - **Logistic Regression** (baseline)
- A logging framework that outputs both text logs and JSON logs.
- Stratified k-fold cross-validation, with optional dataset downsampling for large datasets.
- Support for Pydantic-based output parsing from Ollama responses.

## Table of Contents
1. [Features](#features)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Project Structure](#project-structure)
5. [Usage](#usage)
6. [Logging Output](#logging-output)
7. [Adding New Datasets](#adding-new-datasets)
8. [Extending Classifiers](#extending-classifiers)
9. [License](#license)

## Features

1. **Multiple Classification Approaches**  
   - **Ollama Few-Shot**: Uses in-context learning with a limited number of training examples in the prompt.  
   - **Ollama with RAG**: Retrieves the most relevant training examples from a vector store (ChromaDB) and uses them as context for classification.  
   - **Class-Based Few-Shot**: Ensures a fixed number of examples *per class* in the prompt.  
   - **Chain-of-Thought (CoT) Extraction**: Optionally captures the model’s reasoning process.  
   - **Logistic Regression**: Traditional baseline for comparison.

2. **Stratified K-Fold Cross-Validation**  
   Automatically divides datasets into stratified folds to ensure balanced class distributions.

3. **Downsampling for Large Datasets**  
   If a dataset exceeds a set threshold (e.g., 500 rows), the code attempts to downsample while maintaining representative class distributions.

4. **Extensive Logging**  
   - **Text-based general logs**  
   - **JSON-based detailed logs** for storing prompt, response, and accuracy details.

5. **Embeddings & Vector Store**  
   For RAG, it uses **ChromaDB** as a persistent client, storing embeddings generated via `ollama.embeddings`.

## Requirements

- **Python 3.8+** (recommended; tested on Python 3.9+)
- [pysqlite3](https://pysqlite3.readthedocs.io/) (monkey-patched as `sqlite3`)
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [scikit-learn](https://scikit-learn.org/)
- [pydantic](https://docs.pydantic.dev/)
- [langchain-core](https://pypi.org/project/langchain-core/) (or your internal variant)
- [chromadb](https://docs.trychroma.com/) for RAG
- [ollama](https://ollama.ai/) Python bindings for generating embeddings and calling LLaMA-based models (among others)

**Optional**:
- A GPU Without a GPU you will need lots of RAM and it will take significantly longer to run benchmarks.

## Installation

1. **Clone** the repository:
```bash
git clone https://github.com/Mondotrasho/LLM-Tabular-benchmarking-code.git
cd LLM-Tabular-benchmarking-code
```

2. **Create and activate a virtual environment** (recommended):
```bash
python3 -m venv venv
source venv/bin/activate     # On Linux/Mac
# or
venv\Scripts\activate        # On Windows
```

3. **Install dependencies**:
```bash
pip install pysqlite3 pandas numpy scikit-learn pydantic langchain-core chromadb ollama
```

4. **Set up your environment** to point to the correct Ollama or other model paths if needed. For instance, if you use `ollama`, ensure the `ollama` CLI is installed and your model files are in the right location.

- **Install the Ollama CLI**  
   - Installers for **macOS**, **Windows**, and **Linux** are available at:
     ```bash
     https://ollama.com/
     ```
   Follow the instructions provided there to install Ollama on your platform of choice.

- **Pull the Model Files**  
   - Once the Ollama CLI is installed, pull your desired model(s) locally, for example:
     ```bash
     ollama pull llama3.1:8b
     ollama pull llama3.1:70b
     ```

## Project Structure

.  
├── Datasets/  
│   ├── dataset_name/  
│   │   ├── dataset_name.csv  
│   │   └── dataset_name_metadata.json  
│   └── ...  
├── Logging/  
│   └── ... (log files go here) ...  
├── PreprocessingScripts/  
│   └── ... (Some preprocessing scripts are included) ...  
├── LangChain-Testing.py (the script with all classes and the main execution)  
├── requirements.txt  
├── README.md  
└── ...

- **`Datasets/`**: Each dataset has its own folder, containing a CSV file and corresponding metadata JSON file.
- **`Logging/`**: Contains log files generated for each dataset. There will be `.log` files and `.jsonl` files.
- **`main.py`**: The main script that loads data, configures classifiers, runs cross-validation, and writes logs.

## Usage

1. **Place your datasets** inside `Datasets/`. For each dataset:
   - `dataset.csv`: The actual data, one row per sample, columns are features plus the target column.
   - `dataset_metadata.json`: Must contain at least a `target_column_index` field that indicates which column is the label.

2. **Edit the list of datasets** in `main.py` under:
```bash
datasets = [
    "donation_missingto0",
    "waveform",
    "WearableComputing_hotencoded"
]
```
Add or remove dataset names from this list.

3. **Adjust parameters** like `MAX_DATA_SIZE`, `n_splits`, `few_shot_examples`, and so on in `main.py` as needed.

4. **Run the script**:
```bash
python LangChain-Testing.py
```
- The script will run cross-validation on each dataset, training and evaluating each specified classifier in `classifiers`.
- Logs will be saved in the `Logging/` directory.

5. **Check your results**:
   - The console will show progress and final summary.
   - **Text logs**: `<prefix>.log` in the `Logging/` folder contain a readable summary.
   - **JSON logs**: `<prefix>.jsonl` in the same folder contain detailed prompts, responses, and accuracy checks.

## Logging Output

- **General Logger** (`.log` file):  
  Contains high-level status updates, training messages, fold results, etc.

- **Detailed Logger** (`.jsonl` file):  
  Each line is a JSON object with events such as:
```bash
{
  "timestamp": "2023-01-01 12:00:00",
  "level": "INFO",
  "message": {
    "event": "prompt",
    "fold": 1,
    "test_case": 1,
    "prompt": "<the full prompt sent to LLaMA>",
    "timestamp": "2023-01-01 12:00:00"
  }
}
```
This is invaluable for debugging model responses and analyzing performance in detail.

## Adding New Datasets

1. Create a folder under `Datasets/` with the dataset name (e.g. `iris`).
2. Copy your CSV file into `Datasets/iris/iris.csv`.
3. Create a `Datasets/iris/iris_metadata.json` file with the following structure:
```bash
{
    "target_column_index": 4
}
```
(Adjust the index to match your target column in the CSV.)
4. Add `"iris"` to the `datasets` list in `main.py`.


## Changing the Ollama Model

By default, this script might use "llama3.1" or "llama3.1:70b" in Ollama. However, you can easily switch to any other model (e.g., "deepseek-r1:671b") by changing the `model` parameter in the classifier definitions. For example:

```
{
    "name": "My-Custom-Classifier",
    "clf": LangChainOllamaClassifier(
        model="deepseek-r1:671b",
        max_training_examples=few_shot_examples
    )
}
```

Make sure you’ve already pulled the model via the Ollama CLI (for instance, ```bash ollama pull deepseek-r1:671b```) so that it’s available locally. The same approach applies to all other LLaMA-based classifiers in this repository.



## Extending Classifiers

- The code defines multiple classes (e.g. `LangChainOllamaClassifier`, `LangChainOllamaRAGClassifier`, etc.).
- You can create your own custom logic by inheriting from these classes or by writing new classes with a `.fit()`, `.predict()`, and `.score()` method.
- For more advanced logging or prompt engineering, overwrite relevant methods or add new logic for prompt construction and JSON extraction.

## License

[MIT License](LICENSE) – You are free to use, modify, and distribute this code as long as attribution is provided.

---

**Happy benchmarking!** If you have questions or feature requests, feel free to open an Issue or submit a Pull Request.
