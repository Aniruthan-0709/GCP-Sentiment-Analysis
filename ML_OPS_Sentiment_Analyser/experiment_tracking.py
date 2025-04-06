import mlflow
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("experiment_tracking.log"),
        logging.StreamHandler()
    ]
)

logging.info("Starting Experiment Tracking with MLflow...")

# ✅ Make sure MLflow tracking directory exists (local path inside repo)
MLFLOW_DIR = "ML_OPS_Sentiment_Analyser/mlruns"
os.makedirs(MLFLOW_DIR, exist_ok=True)

# ✅ Set tracking URI to a relative directory (not root `/mlruns`)
mlflow.set_tracking_uri(f"file://./{MLFLOW_DIR}")
mlflow.set_experiment("Sentiment_Model_Experiments")

with mlflow.start_run():
    # Example hyperparameters — replace with actual ones if needed
    mlflow.log_param("alpha", 1.0)
    mlflow.log_param("max_features", 5000)

    # Example metric — replace with actual metric from your pipeline
    mlflow.log_metric("accuracy", 0.85)

    # Log model version if version file exists
    version_file = os.path.join("models", "model_version.txt")
    if os.path.exists(version_file):
        with open(version_file, "r") as f:
            model_version = f.read().strip()
        mlflow.log_param("model_version", model_version)

    logging.info("Experiment tracked with MLflow.")
