import os
from google.cloud import storage
from io import BytesIO


def download_from_gcp(bucket_name, source_blob_name, destination_file_name):
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        raise EnvironmentError("GOOGLE_APPLICATION_CREDENTIALS not set")

    os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"✅ Downloaded {source_blob_name} to {destination_file_name}")

def upload_to_gcp(bucket_name, local_file_path, destination_blob_name):
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        raise EnvironmentError("GOOGLE_APPLICATION_CREDENTIALS not set")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_file_path)
    print(f"✅ Uploaded {local_file_path} to gs://{bucket_name}/{destination_blob_name}")

def read_csv_from_gcs(bucket_name, blob_name):
    from google.cloud import storage
    import pandas as pd

    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        raise EnvironmentError("GOOGLE_APPLICATION_CREDENTIALS not set")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    content = blob.download_as_bytes()
    return pd.read_csv(BytesIO(content))
