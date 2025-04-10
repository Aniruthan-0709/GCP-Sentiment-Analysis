import os
from google.cloud import storage

# ===== Update file name to match new pipeline file =====
BUCKET_NAME = "mlops_dataset123"
LOCAL_MODEL_PATH = "models/sentiment_analyzer_model.pkl"
GCS_MODEL_PATH = "models/sentiment_analyzer_model.pkl"

def upload_to_gcp(bucket_name, source_file_name, destination_blob_name):
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        raise EnvironmentError("GOOGLE_APPLICATION_CREDENTIALS is not set.")
    if not os.path.exists(source_file_name):
        raise FileNotFoundError(f"{source_file_name} not found.")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"Uploaded {source_file_name} to gs://{bucket_name}/{destination_blob_name}")

#upload_to_gcp(BUCKET_NAME, LOCAL_MODEL_PATH, GCS_MODEL_PATH)
