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

# ========== Configs ==========
TRACKING_URI = os.path.abspath("ML_OPS_Sentiment_Analyser/mlruns")
ARTIFACT_URI = "gs://mlops_dataset123/mlruns/artifacts"
EXPERIMENT_NAME = "Sentiment_Model_Experiments"

# ========== Set Tracking URI ==========
os.makedirs(TRACKING_URI, exist_ok=True)
mlflow.set_tracking_uri(f"file://{TRACKING_URI}")

# ========== Create Experiment if Needed ==========
client = MlflowClient(tracking_uri=f"file://{TRACKING_URI}")
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

# ========== Run Tracking ==========
with mlflow.start_run():
    # Log params and metrics
    mlflow.log_param("alpha", 1.0)
    mlflow.log_param("max_features", 5000)
    mlflow.log_metric("accuracy", 0.85)

    # Log model version if available
    version_file = os.path.join("ML_OPS_Sentiment_Analyser", "models", "model_version.txt")
    if os.path.exists(version_file):
        with open(version_file, "r") as f:
            model_version = f.read().strip()
        mlflow.log_param("model_version", model_version)
        mlflow.log_artifact(version_file)  # ✅ Log the model version file

    # Log SHAP plot if exists
    shap_plot_path = os.path.join("ML_OPS_Sentiment_Analyser", "artifacts", "shap_summary.png")
    if os.path.exists(shap_plot_path):
        mlflow.log_artifact(shap_plot_path)
        logging.info("Logged SHAP summary plot to MLflow.")

    # Create and log a dummy file to verify artifact push
    dummy_artifact = "artifact_test.txt"
    with open(dummy_artifact, "w") as f:
        f.write("This is a test artifact to ensure GCS upload works.")

    mlflow.log_artifact(dummy_artifact)
    logging.info("Logged dummy artifact to trigger GCS upload.")

    logging.info("MLflow experiment run completed successfully.")
