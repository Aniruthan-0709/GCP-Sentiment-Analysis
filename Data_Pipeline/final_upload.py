import os
import logging
from utils.gcs_utils import upload_to_gcp

BASE_DIR = os.path.dirname(__file__)
LOG_DIR = os.path.join(BASE_DIR, "logs")
PROCESSED_PATH = os.path.join(BASE_DIR, "data/processed/reviews.parquet")

gcs_bucket = os.getenv("GCP_BUCKET")
destination_blob = os.getenv("GCP_PROCESSED_BLOB")

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
        if not os.path.exists(PROCESSED_PATH):
            raise FileNotFoundError(f"Processed file not found: {PROCESSED_PATH}")

        logging.info(f"☁️ Uploading {PROCESSED_PATH} to {gcs_bucket}/{destination_blob}")
        upload_to_gcp(gcs_bucket, PROCESSED_PATH, destination_blob)
        logging.info("✅ Upload complete.")
    except Exception as e:
        logging.error(f"❌ Upload failed: {e}")

if __name__ == "__main__":
    run_upload()
