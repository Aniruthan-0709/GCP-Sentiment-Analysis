import os
import logging
import pandas as pd
from utils.gcs_utils import upload_to_gcp

# Paths
BASE_DIR = os.path.dirname(__file__)
LOG_DIR = os.path.join(BASE_DIR, "logs")
PARQUET_PATH = os.path.join(BASE_DIR, "data/processed/reviews.parquet")
CSV_PATH = os.path.join(BASE_DIR, "data/processed/reviews.csv")

# GCP environment variables
gcs_bucket = os.getenv("GCP_BUCKET")
destination_blob = os.getenv("GCP_PROCESSED_BLOB")  # e.g., processed/reviews.csv

# Setup logging
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "mlops_upload_pipeline.log"), encoding="utf-8"),
        logging.StreamHandler()
    ]
)

def run_upload():
    try:
        logging.info(f"🔍 Checking for processed file at: {PARQUET_PATH}")
        if not os.path.exists(PARQUET_PATH):
            nearby_files = os.listdir(os.path.dirname(PARQUET_PATH))
            logging.error(f"❌ Processed file not found.\nContents of directory:\n{nearby_files}")
            raise FileNotFoundError(f"Processed file not found: {PARQUET_PATH}")

        # Convert to CSV
        logging.info(f"🧪 Converting {PARQUET_PATH} to CSV...")
        df = pd.read_parquet(PARQUET_PATH)
        df.to_csv(CSV_PATH, index=False)
        logging.info(f"✅ CSV saved at: {CSV_PATH}")

        # Upload CSV to GCP
        logging.info(f"☁️ Uploading {CSV_PATH} to gs://{gcs_bucket}/{destination_blob}")
        upload_to_gcp(gcs_bucket, CSV_PATH, destination_blob)
        logging.info("✅ Upload complete.")

    except Exception as e:
        logging.error(f"❌ Upload failed: {e}")

if __name__ == "__main__":
    run_upload()
