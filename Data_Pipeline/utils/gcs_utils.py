import os
from google.cloud import storage

def download_from_gcp(bucket_name, source_blob_name, destination_file_name):
    """Downloads a file from GCP Cloud Storage to the local filesystem."""
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        raise EnvironmentError("GOOGLE_APPLICATION_CREDENTIALS is not set.")

    os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(f"✅ Downloaded {source_blob_name} from GCP bucket {bucket_name} to {destination_file_name}")
