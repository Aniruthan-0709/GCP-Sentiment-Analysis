import mlflow
import logging
import os

# ========== Set up Logging ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("experiment_tracking.log"),
        logging.StreamHandler()
    ]
)

logging.info("Starting Experiment Tracking with MLflow...")

# ========== Fix: Use absolute path to avoid PermissionError ==========
MLFLOW_DIR = os.path.abspath("ML_OPS_Sentiment_Analyser/mlruns")
os.makedirs(MLFLOW_DIR, exist_ok=True)

# Set tracking URI to the absolute directory
mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")
mlflow.set_experiment("Sentiment_Model_Experiments")

# ========== Start the MLflow run ==========
with mlflow.start_run():
    # Log hyperparameters
    mlflow.log_param("alpha", 1.0)
    mlflow.log_param("max_features", 5000)

    # Log example metric
    mlflow.log_metric("accuracy", 0.85)

    # Log model version if available
    version_file = os.path.join("ML_OPS_Sentiment_Analyser", "models", "model_version.txt")
    if os.path.exists(version_file):
        with open(version_file, "r") as f:
            model_version = f.read().strip()
        mlflow.log_param("model_version", model_version)

    logging.info("Experiment tracked with MLflow.")
