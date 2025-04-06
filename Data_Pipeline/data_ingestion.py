import os
import gcsfs
import pandas as pd
import logging
from tqdm import tqdm

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

# Environment variables set in GitHub Actions
BUCKET_NAME = os.getenv("GCP_BUCKET")
FILE_PATH = os.getenv("SOURCE_BLOB")
LOCAL_SAVE_PATH = os.path.join(BASE_DIR, "data/raw/reviews.csv")

def download_data(chunk_size=8192):
    try:
        if os.path.exists(LOCAL_SAVE_PATH):
            logging.info(f"✅ File already exists: {LOCAL_SAVE_PATH}")
            return LOCAL_SAVE_PATH

        logging.info("🔹 Connecting to GCS...")
        fs = gcsfs.GCSFileSystem()  # Uses GOOGLE_APPLICATION_CREDENTIALS from env

        logging.info(f"🔹 Downloading {FILE_PATH} from {BUCKET_NAME}...")
        file_size = fs.size(FILE_PATH)

        os.makedirs(os.path.dirname(LOCAL_SAVE_PATH), exist_ok=True)
        with fs.open(FILE_PATH, 'rb') as remote_file:
            with open(LOCAL_SAVE_PATH, 'wb') as local_file:
                with tqdm(total=file_size, unit='B', unit_scale=True, desc=FILE_PATH) as pbar:
                    while True:
                        chunk = remote_file.read(chunk_size)
                        if not chunk:
                            break
                        local_file.write(chunk)
                        pbar.update(len(chunk))

        logging.info(f"✅ Dataset downloaded successfully to: {LOCAL_SAVE_PATH}")
        return LOCAL_SAVE_PATH

    except Exception as e:
        logging.error(f"❌ Error during dataset download: {e}")
        return None

if __name__ == "__main__":
    download_data()
