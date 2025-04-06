import os
import logging
from utils.gcs_utils import upload_to_gcp

# Paths
BASE_DIR = os.path.dirname(__file__)
LOG_DIR = os.path.join(BASE_DIR, "logs")
PROCESSED_PATH = os.path.join(BASE_DIR, "data/processed/reviews.parquet")

# GCP environment variables
gcs_bucket = os.getenv("GCP_BUCKET")
destination_blob = os.getenv("GCP_PROCESSED_BLOB")

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
        logging.info(f"🔍 Checking for processed file at: {PROCESSED_PATH}")
        if not os.path.exists(PROCESSED_PATH):
            # Helpful debug: show what files exist in the directory
            nearby_files = os.listdir(os.path.dirname(PROCESSED_PATH))
            logging.error(f"❌ Processed file not found.\nContents of directory:\n{nearby_files}")
            raise FileNotFoundError(f"Processed file not found: {PROCESSED_PATH}")

        logging.info(f"☁️ Uploading {PROCESSED_PATH} to gs://{gcs_bucket}/{destination_blob}")
        upload_to_gcp(gcs_bucket, PROCESSED_PATH, destination_blob)
        logging.info("✅ Upload complete.")

    except Exception as e:
        logging.error(f"❌ Upload failed: {e}")

if __name__ == "__main__":
    run_upload()
