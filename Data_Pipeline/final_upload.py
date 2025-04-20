import os
import logging
import pandas as pd
from io import StringIO
from utils.gcs_utils import load_csv_from_gcs, upload_to_gcp

# ========== Environment ==========
gcs_bucket = os.getenv("GCP_BUCKET")
gcs_blob = os.getenv("GCP_PROCESSED_BLOB")  # e.g., "processed/reviews.csv"

# ========== Logging ==========
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
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
        logging.info(f"📥 Reading processed CSV from gs://{gcs_bucket}/{gcs_blob}")
        df = load_csv_from_gcs(gcs_bucket, gcs_blob)

        logging.info(f"✅ Loaded data: {len(df)} rows, {df.shape[1]} columns")

        # (Optional) Transform df here
        # df = df[df["star_rating"].notnull()]

        # Convert to CSV in memory
        buffer = StringIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)

        logging.info(f"☁️ Uploading updated CSV to gs://{gcs_bucket}/{gcs_blob}")
        upload_to_gcp(gcs_bucket, buffer.getvalue(), gcs_blob, from_memory=True)

        logging.info("✅ Upload complete.")

    except Exception as e:
        logging.error(f"❌ Upload failed: {e}")

if __name__ == "__main__":
    run_upload()
