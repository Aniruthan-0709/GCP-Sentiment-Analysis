import mlflow
import logging
import os
from mlflow.tracking import MlflowClient

# ========== Logging ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("experiment_tracking.log"),
        logging.StreamHandler()
    ]
)

logging.info("Starting Experiment Tracking with MLflow...")

# ========== Local tracking location ==========
TRACKING_URI = os.path.abspath("ML_OPS_Sentiment_Analyser/mlruns")
ARTIFACT_URI = "gs://mlops-dataset123/mlruns/artifacts"
EXPERIMENT_NAME = "Sentiment_Model_Experiments"

os.makedirs(TRACKING_URI, exist_ok=True)
mlflow.set_tracking_uri(f"file://{TRACKING_URI}")

# ========== Create experiment with custom GCS artifact location ==========
client = MlflowClient(tracking_uri=f"file://{TRACKING_URI}")

# Check if experiment exists
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    experiment_id = client.create_experiment(
        name=EXPERIMENT_NAME,
        artifact_location=ARTIFACT_URI
    )
    logging.info(f"Created new experiment: {EXPERIMENT_NAME}")
else:
    experiment_id = experiment.experiment_id
    logging.info(f"Using existing experiment: {EXPERIMENT_NAME}")

mlflow.set_experiment(EXPERIMENT_NAME)

# ========== Start tracking run ==========
with mlflow.start_run():
    mlflow.log_param("alpha", 1.0)
    mlflow.log_param("max_features", 5000)
    mlflow.log_metric("accuracy", 0.85)

    version_file = os.path.join("ML_OPS_Sentiment_Analyser", "models", "model_version.txt")
    if os.path.exists(version_file):
        with open(version_file, "r") as f:
            model_version = f.read().strip()
        mlflow.log_param("model_version", model_version)

    logging.info("MLflow experiment run completed and pushed to GCP artifacts.")
