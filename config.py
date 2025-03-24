import os

# Model and Dataset Config
MODEL_NAME = "distilbert-base-uncased"
DATASET_NAME = "sst2"

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "distilbert-sst2")

# Training Config
BATCH_SIZE = 32 
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5

# Pruning Config
PRUNE_PERCENTAGE = 0.3  # Prune 40% of the least relevant layers

