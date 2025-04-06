import mlflow
import logging
import os

# ==================== Logging ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("experiment_tracking.log"),
        logging.StreamHandler()
    ]
)

logging.info("Starting Experiment Tracking with MLflow...")

# ==================== Local MLflow Tracking ====================
# Create local tracking folder (inside repo workspace)
MLFLOW_TRACKING_DIR = os.path.abspath("ML_OPS_Sentiment_Analyser/mlruns")
os.makedirs(MLFLOW_TRACKING_DIR, exist_ok=True)

mlflow.set_tracking_uri(f"file://{MLFLOW_TRACKING_DIR}")

# ==================== Remote Artifact Storage ====================
# GCS path to store artifacts
GCS_ARTIFACT_PATH = "gs://mlops-dataset123/mlruns/artifacts"

# Set experiment and artifact location
mlflow.set_experiment(
    experiment_name="Sentiment_Model_Experiments",
    artifact_location=GCS_ARTIFACT_PATH
)

# ==================== Start Run ====================
with mlflow.start_run():
    # Log example hyperparameters and metrics
    mlflow.log_param("alpha", 1.0)
    mlflow.log_param("max_features", 5000)
    mlflow.log_metric("accuracy", 0.85)

    # Log model version from version file if it exists
    version_file = os.path.join("ML_OPS_Sentiment_Analyser", "models", "model_version.txt")
    if os.path.exists(version_file):
        with open(version_file, "r") as f:
            model_version = f.read().strip()
        mlflow.log_param("model_version", model_version)

    logging.info("MLflow experiment logged successfully.")
