# config.py

# ----------------------------
# USER CONFIGURATION
# ----------------------------
MODE = 'single'
SAMPLE_FRAC = 1.0

# Ollama LLM model to use: e.g. 'llama3.1:8b' or 'llama3.1:70b'
LLM_MODEL = 'llama3.1:70b'

# ----------------------------
# DATASETS & CROSS-VAL SETTINGS
# ----------------------------
DATASETS = [
    # "adult_hotencoded",
    # "bank-full_hotencoded",
    "covtype",
    # "Credit_Card_Applications",
    # "letter-recog",
     "localization_hotencoded",
    # "magic",
    # "pendigits",
    # "satellite",
    # "segment",
    # "sick_hotencoded",
    # "sign",
    # "student-portuguese_hotencoded_50PassFail",
     "waveform",
     "WearableComputing_hotencoded"
    # "wine"
]


N_SPLITS    = 5
METHODS = ['MetaSelectorLLM']  

# METHODS = ['MetaSelectorLLM','RAG', 'Hybrid_model',]  
# # or run alongside your others

# METHODS = ['quantile_binned_fewshot_tabular_llm_classifier']
# METHODS     = [ 'XGB', 'KNN', 'CatBoost','RandomForest']
K_NEIGHBORS = 3
# K_NEIGHBORS = 200

RANDOM_K = 1

