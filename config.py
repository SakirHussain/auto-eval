"""
Centralized configuration for the Auto-Evaluation project.
"""

# File Paths
DATA_PATH = "procot_eval_with_ideal.json"
MODEL_PATH = "softener_gbr.joblib"
IDEAL_ANSWERS_PATH = "qnia.json"
STUDENT_ANSWERS_PATH = "student_answers.json"
PDF_PATH = "os.pdf"
CORPORA_DIR = "corpora"

# Model Names
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "gemma3:latest"
OLLAMA_EMBEDDING_MODEL = "bge-m3"
OLLAMA_LLM_MODEL = "gemma3:latest"
ZERO_SHOT_CLASSIFIER_MODEL = "facebook/bart-large-mnli"

# Thresholds and Parameters
CLUSTERING_THRESHOLD = 0.80
SIMILARITY_THRESHOLD = 0.8
FILTERING_WINDOW_SIZE = 3
FILTERING_TOLERANCE = 0.034
ITERATIVE_REFINEMENT_TURNS = 3
