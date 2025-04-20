import os
import logging
import pandas as pd
from utils.gcp_utils import load_csv_from_gcs, upload_to_gcs
from io import BytesIO

# ========== Environment Variables ==========
gcs_bucket = os.getenv("GCP_BUCKET")
gcs_blob = os.getenv("GCP_PROCESSED_BLOB")  # e.g., processed/reviews.csv

# ========== Logging Setup ==========
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

# ========== Upload Logic ==========
def run_upload():
    try:
        logging.info(f"📥 Reading processed CSV from: gs://{gcs_bucket}/{gcs_blob}")
        df = load_csv_from_gcs(gcs_bucket, gcs_blob)

        logging.info(f"✅ Data loaded. Rows: {len(df)}, Columns: {df.shape[1]}")

        # Perform optional transformation (identity in this case)
        # e.g., df = df[df["star_rating"] > 1]

        # Save to in-memory CSV
        buffer = BytesIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)

        logging.info(f"☁️ Uploading updated CSV back to: gs://{gcs_bucket}/{gcs_blob}")
        upload_to_gcs(gcs_bucket, buffer, gcs_blob, is_fileobj=True)
        logging.info("✅ Upload complete.")

    except Exception as e:
        logging.error(f"❌ Upload failed: {e}")

if __name__ == "__main__":
    run_upload()
