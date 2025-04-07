import os
import logging
import shutil
import re
from google.cloud import storage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("model_versioning.log"), logging.StreamHandler()]
)

logging.info("Starting Model Versioning...")

# ====================== GCP CONFIG ======================
BUCKET_NAME = "mlops_dataset123"
GCS_MODEL_FOLDER = "models/"
LOCAL_MODEL_PATH = "models/sentiment_analyzer_model.pkl"

# ====================== GCS Setup ======================
client = storage.Client()
bucket = client.bucket(BUCKET_NAME)

# List existing model versions in GCS
blobs = list(bucket.list_blobs(prefix=GCS_MODEL_FOLDER))

# Extract version numbers from filenames
version_pattern = re.compile(r"sentiment_analyzer_model_v(\d+)\.pkl")
existing_versions = []

for blob in blobs:
    match = version_pattern.search(blob.name)
    if match:
        existing_versions.append(int(match.group(1)))

# Determine next version
current_version = max(existing_versions) if existing_versions else 0
new_version = current_version + 1

# Create new versioned filename
versioned_filename = f"sentiment_analyzer_model_v{new_version}.pkl"
versioned_local_path = os.path.join("ML_OPS_Sentiment_Analyser", "models", versioned_filename)

# Copy the base model to the versioned file
shutil.copy(LOCAL_MODEL_PATH, versioned_local_path)
logging.info(f"Created versioned model file: {versioned_filename}")

# Upload to GCS
blob = bucket.blob(os.path.join(GCS_MODEL_FOLDER, versioned_filename))
blob.upload_from_filename(versioned_local_path)
logging.info(f"Uploaded {versioned_filename} to GCS: gs://{BUCKET_NAME}/{GCS_MODEL_FOLDER}{versioned_filename}")

# Update version tracker (optional)
version_file = os.path.join("ML_OPS_Sentiment_Analyser", "models", "model_version.txt")
with open(version_file, "w") as f:
    f.write(str(new_version))

logging.info(f"Updated version file with v{new_version}")
