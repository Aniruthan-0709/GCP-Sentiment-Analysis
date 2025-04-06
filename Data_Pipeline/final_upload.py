import os
import logging
import gcsfs
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Logging setup
BASE_DIR = os.path.dirname(__file__)
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "final_upload.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Configuration
PARQUET_FILE = os.path.join(BASE_DIR, "data/processed/reviews.parquet")
CSV_FILE = PARQUET_FILE.replace(".parquet", ".csv")

GCP_BUCKET = os.getenv("GCP_BUCKET")
DESTINATION_PATH = os.getenv("GCP_PROCESSED_BLOB", "processed_data/reviews.csv")  # Ensure it's .csv
GCP_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "config/key.json")

# Set credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_CREDENTIALS

def upload_csv_to_gcs():
    try:
        if not os.path.exists(PARQUET_FILE):
            logging.error(f"❌ Processed file not found: {PARQUET_FILE}")
            return

        # Convert to CSV
        df = pd.read_parquet(PARQUET_FILE)
        df.to_csv(CSV_FILE, index=False)
        logging.info(f"📄 Converted parquet to CSV: {CSV_FILE}")

        # Upload to GCS
        fs = gcsfs.GCSFileSystem(token=GCP_CREDENTIALS)
        with open(CSV_FILE, "rb") as f:
            with fs.open(f"{GCP_BUCKET}/{DESTINATION_PATH}", "wb") as gcs_file:
                gcs_file.write(f.read())

        logging.info(f"✅ Uploaded CSV file to: {GCP_BUCKET}/{DESTINATION_PATH}")
    except Exception as e:
        logging.error(f"❌ Failed to upload CSV file: {e}")

if __name__ == "__main__":
    upload_csv_to_gcs()
