# utils/gcp_utils.py
import os
import pandas as pd
from io import BytesIO
from google.cloud import storage

def load_csv_from_gcs(bucket_name: str, blob_name: str) -> pd.DataFrame:
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        raise EnvironmentError("GOOGLE_APPLICATION_CREDENTIALS not set.")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    content = blob.download_as_bytes()
    return pd.read_csv(BytesIO(content))
