import mlflow
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("experiment_tracking.log"), logging.StreamHandler()]
)
logging.info("Starting Experiment Tracking with MLflow...")

# Use file-based tracking URI for CI environments
mlflow.set_tracking_uri("file:///mlruns")
mlflow.set_experiment("Sentiment_Model_Experiments")

with mlflow.start_run():
    mlflow.log_param("alpha", 1.0)
    mlflow.log_param("max_features", 5000)
    mlflow.log_metric("accuracy", 0.85)

    version_file = os.path.join("models", "model_version.txt")
    if os.path.exists(version_file):
        with open(version_file, "r") as f:
            model_version = f.read().strip()
        mlflow.log_param("model_version", model_version)

    logging.info("Experiment tracked with MLflow.")
