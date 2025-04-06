import os
import gcsfs
import pandas as pd
import logging
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables (for local testing or GitHub secrets)
load_dotenv()

# Setup logging directory
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

# Config from environment
BUCKET_NAME = os.getenv("GCP_BUCKET")
FILE_PATH = os.getenv("SOURCE_BLOB")
SERVICE_ACCOUNT_KEY = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "config/key.json")
LOCAL_SAVE_PATH = os.path.join(BASE_DIR, "data/raw/reviews.csv")

# Set GCP credentials path for gcsfs
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_KEY

def download_data(chunk_size=8192):
    """Downloads dataset from GCP bucket and saves it locally."""
    try:
        if os.path.exists(LOCAL_SAVE_PATH):
            logging.info(f"✅ File already exists locally: {LOCAL_SAVE_PATH}")
            return LOCAL_SAVE_PATH

        logging.info("🔹 Connecting to GCS...")
        fs = gcsfs.GCSFileSystem(token=SERVICE_ACCOUNT_KEY)

        file_size = fs.size(FILE_PATH)
        os.makedirs(os.path.dirname(LOCAL_SAVE_PATH), exist_ok=True)

        logging.info(f"🔹 Downloading {FILE_PATH} from {BUCKET_NAME}...")
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
