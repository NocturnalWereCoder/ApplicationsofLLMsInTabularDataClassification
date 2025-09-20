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
METHODS = ['MetaSelectorLLM','MetaSelectorLLM']  

# METHODS = ['MetaSelectorLLM','RAG', 'Hybrid_model',]  
# # or run alongside your others

# METHODS = ['quantile_binned_fewshot_tabular_llm_classifier']
# METHODS     = [ 'XGB', 'KNN', 'CatBoost','RandomForest']
K_NEIGHBORS = 3
# K_NEIGHBORS = 200

RANDOM_K = 1


#Can LLMs act as meta-learners that guide model selection, preprocessing, or feature construction for tabular classification?

#Next steps run meta on large datasets (if enough time with higher fold variance and also run the ML models on the same.) (potentially also do mid sized afterwards if enough time)

#Run highest preffered classifier from the LLM and use a tuner of some sort to identify the best tuning settings for highest preffered method by the LLM. Get the LLM to return Hyperparameters. 

#Final run before First half results complete. 