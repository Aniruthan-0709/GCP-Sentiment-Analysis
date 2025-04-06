import os
from google.cloud import storage

# ======================= GCP CONFIG =======================
BUCKET_NAME = os.environ.get("GCP_BUCKET")          # From GitHub secret
FILE_NAME = os.environ.get("GCP_PROCESSED_BLOB")  
LOCAL_DIR = "Data/"
LOCAL_FILE_NAME = "Data.csv"
LOCAL_FILE_PATH = os.path.join(LOCAL_DIR, LOCAL_FILE_NAME)

# Ensure the Data folder exists
os.makedirs(LOCAL_DIR, exist_ok=True)

# ======================= DOWNLOAD DATA FROM GCP =======================
def download_from_gcp(bucket_name, source_blob_name, destination_file_name):
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        raise EnvironmentError("GOOGLE_APPLICATION_CREDENTIALS is not set.")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} from GCP bucket {bucket_name} to {destination_file_name}")

download_from_gcp(BUCKET_NAME, FILE_NAME, LOCAL_FILE_PATH)
