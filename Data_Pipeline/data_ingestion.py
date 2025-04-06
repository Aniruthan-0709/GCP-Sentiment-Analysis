import os
import logging
import hashlib
from datetime import datetime
from utils.gcs_utils import download_from_gcp

BASE_DIR = os.path.dirname(__file__)
LOG_DIR = os.path.join(BASE_DIR, "logs")
DVC_DIR = os.path.join(BASE_DIR, "dvc")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DVC_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "mlops_ingestion_pipeline.log"), encoding="utf-8"),
        logging.StreamHandler()
    ]
)

BUCKET_NAME = os.getenv("GCP_BUCKET")
SOURCE_BLOB = os.getenv("SOURCE_BLOB")
LOCAL_FILE_PATH = os.path.join(BASE_DIR, "data/raw/reviews.csv")

def save_dvc_hash(file_path, dvc_dir=DVC_DIR):
    """Generate and save a SHA256 hash of the input file."""
    hash_obj = hashlib.sha256()

    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hash_obj.update(chunk)

    hash_hex = hash_obj.hexdigest()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(dvc_dir, f"hash_raw_reviews_{timestamp}.txt")

    with open(output_file, "w") as out:
        out.write(f"SHA256: {hash_hex}\n")
        out.write(f"File: {file_path}\n")
        out.write(f"Generated: {timestamp}\n")

    logging.info(f"📦 DVC-style hash written to {output_file}")

def run_ingestion():
    try:
        logging.info(f"⬇️ Downloading from GCP: {BUCKET_NAME}/{SOURCE_BLOB}")
        download_from_gcp(BUCKET_NAME, SOURCE_BLOB, LOCAL_FILE_PATH)
        logging.info(f"✅ File downloaded to {LOCAL_FILE_PATH}")

        # 🧬 Save DVC-style hash
        save_dvc_hash(LOCAL_FILE_PATH)

        return LOCAL_FILE_PATH
    except Exception as e:
        logging.error(f"❌ Failed to download data: {e}")
        return None

if __name__ == "__main__":
    run_ingestion()
