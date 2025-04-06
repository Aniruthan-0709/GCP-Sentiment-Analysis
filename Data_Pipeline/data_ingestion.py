import os
import logging
from utils.gcs_utils import download_from_gcp

# Setup logging
BASE_DIR = os.path.dirname(__file__)
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "mlops_ingestion_pipeline.log"), encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# Environment variables (set via GitHub Actions)
BUCKET_NAME = os.getenv("GCP_BUCKET")
SOURCE_BLOB = os.getenv("SOURCE_BLOB")  # e.g. "raw/sample_reviews.csv"
LOCAL_FILE_PATH = os.path.join(BASE_DIR, "data/raw/reviews.csv")

def run_ingestion():
    try:
        logging.info(f"⬇️ Downloading from GCP: {BUCKET_NAME}/{SOURCE_BLOB}")
        download_from_gcp(BUCKET_NAME, SOURCE_BLOB, LOCAL_FILE_PATH)
        logging.info(f"✅ File downloaded to {LOCAL_FILE_PATH}")
        return LOCAL_FILE_PATH
    except Exception as e:
        logging.error(f"❌ Failed to download data: {e}")
        return None

if __name__ == "__main__":
    run_ingestion()
