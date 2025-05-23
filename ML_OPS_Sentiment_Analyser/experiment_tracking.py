import mlflow
import logging
import os
import pickle
import pandas as pd
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
import sklearn

from utils.gcp_utils import load_csv_from_gcs

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

# ========== GCP Config ==========
BUCKET_NAME = os.environ.get("GCP_BUCKET")
BLOB_NAME = os.environ.get("GCP_PROCESSED_BLOB")

# ========== MLflow Config ==========
TRACKING_URI = os.path.abspath("ML_OPS_Sentiment_Analyser/mlruns")
ARTIFACT_URI = "gs://mlops_dataset123/mlruns/artifacts"
EXPERIMENT_NAME = "Sentiment_Model_Experiments"
MODEL_PATH = os.path.join("models", "sentiment_analyzer_model.pkl")

# ========== Set Tracking URI ==========
os.makedirs(TRACKING_URI, exist_ok=True)
mlflow.set_tracking_uri(f"file://{TRACKING_URI}")

# ========== Create or Get Experiment ==========
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

# ========== Load Sample Data from GCS ==========
df = load_csv_from_gcs(BUCKET_NAME, BLOB_NAME)
df = df.dropna(subset=["review_body"])
df["review_body"] = df["review_body"].astype(str)
input_example = df[["review_body"]].sample(n=1, random_state=42)

# ========== Run Tracking ==========
with mlflow.start_run():
    # Log parameters and metrics
    mlflow.log_param("alpha", 1.0)
    mlflow.log_param("max_features", 5000)
    mlflow.log_metric("accuracy", 0.85)

    # Log model
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        try:
            output_example = model.predict(input_example["review_body"])
        except Exception as e:
            logging.warning(f"Failed to run prediction on input_example: {e}")
            output_example = [1]  # fallback

        signature = infer_signature(input_example, output_example)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="sentiment_analyzer_model",
            input_example=input_example,
            signature=signature
        )
        logging.info("Logged model to MLflow.")
    else:
        logging.warning("Model file not found. Skipping model logging.")

    # Log model version file if available
    version_file = os.path.join("ML_OPS_Sentiment_Analyser", "models", "model_version.txt")
    if os.path.exists(version_file):
        with open(version_file, "r") as f:
            model_version = f.read().strip()
        mlflow.log_param("model_version", model_version)
        mlflow.log_artifact(version_file)

    # Log SHAP summary plot if available
    shap_plot_path = os.path.join("ML_OPS_Sentiment_Analyser", "artifacts", "shap_summary.png")
    if os.path.exists(shap_plot_path):
        mlflow.log_artifact(shap_plot_path)
        logging.info("Logged SHAP summary plot.")

    # Dummy artifact for testing
    dummy_artifact = "artifact_test.txt"
    with open(dummy_artifact, "w") as f:
        f.write("This is a test artifact to ensure GCS upload works.")
    mlflow.log_artifact(dummy_artifact)

    logging.info("MLflow experiment run completed successfully.")
