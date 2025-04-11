from google.cloud import storage
import pickle
import os
import tempfile
import re

def load_latest_model(bucket_name="mlops_dataset123", model_prefix="models/"):
    try:
        # Init GCP Storage Client
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        # List all model blobs in the prefix
        blobs = list(bucket.list_blobs(prefix=model_prefix))

        # Filter only .pkl files and sort by updated timestamp
        model_blobs = [blob for blob in blobs if blob.name.endswith(".pkl")]
        if not model_blobs:
            raise FileNotFoundError("No model files found in GCS bucket")

        # Sort by latest update time
        latest_blob = sorted(model_blobs, key=lambda x: x.updated, reverse=True)[0]

        # Download to temp file
        temp_dir = tempfile.gettempdir()
        local_path = os.path.join(temp_dir, "model.pkl")
        latest_blob.download_to_filename(local_path)

        # Load model
        with open(local_path, "rb") as f:
            model = pickle.load(f)

        # Extract version from file name (e.g. model_v3.pkl → 3)
        version_match = re.search(r"_v(\d+)", latest_blob.name)
        version = version_match.group(1) if version_match else "Unknown"

        print(f"✅ Loaded model: {latest_blob.name} (Version: {version})")
        return model, version

    except Exception as e:
        print(f"🚨 Failed to load model from GCP bucket: {e}")
        raise
