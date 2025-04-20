import tensorflow_data_validation as tfdv
import pandas as pd
import os
import logging
import sys
from utils.gcs_utils import read_csv_from_gcs, upload_to_gcp
from io import StringIO

# Setup directories
BASE_DIR = os.path.dirname(__file__)
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Setup logging
LOG_FILE = os.path.join(LOG_DIR, "mlops_anomalies_pipeline.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

# GCS env
GCP_BUCKET = os.environ["GCP_BUCKET"]
GCP_PROCESSED_BLOB = os.environ["GCP_PROCESSED_BLOB"]
SCHEMA_PATH = os.path.join(BASE_DIR, "validation/schema.pbtxt")
NEW_STATS_PATH = os.path.join(BASE_DIR, "validation/new_stats.tfrecord")

def detect_anomalies(schema_path=SCHEMA_PATH):
    """Detects data anomalies using TFDV and updates processed GCS blob if needed."""
    try:
        if not os.path.exists(schema_path):
            logging.error(f"❌ Schema file not found: {schema_path}")
            return {"error": "Schema file not found"}

        logging.info("🔹 Loading schema...")
        schema = tfdv.load_schema_text(schema_path)

        logging.info("🔹 Reading processed CSV from GCS...")
        df = read_csv_from_gcs(GCP_BUCKET, GCP_PROCESSED_BLOB)

        logging.info("🔹 Generating statistics from processed data...")
        new_stats = tfdv.generate_statistics_from_dataframe(df)

        os.makedirs(os.path.dirname(NEW_STATS_PATH), exist_ok=True)
        tfdv.write_stats_text(new_stats, NEW_STATS_PATH)

        logging.info("🔹 Running anomaly detection against schema...")
        anomalies = tfdv.validate_statistics(new_stats, schema)

        anomalies_dict = {}
        updated = False

        if anomalies.anomaly_info:
            logging.warning("⚠️ Anomalies detected:")
            for feature, anomaly in anomalies.anomaly_info.items():
                description = anomaly.description
                anomalies_dict[feature] = description
                logging.warning(f"⚠️ {feature}: {description}")

                # Example fix: if rating column has out-of-range values
                if "values outside the range" in description.lower() and feature == "star_rating":
                    logging.info("🔧 Fixing out-of-range star_rating values")
                    df["star_rating"] = df["star_rating"].clip(1, 5)
                    updated = True

        else:
            logging.info("✅ No anomalies found.")

        # Upload updated data if modified
        if updated:
            logging.info("☁️ Uploading updated DataFrame back to GCS...")
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            upload_to_gcp(GCP_BUCKET, csv_buffer.getvalue().encode("utf-8"), GCP_PROCESSED_BLOB, from_memory=True)
            logging.info(f"✅ Updated dataset uploaded to: gs://{GCP_BUCKET}/{GCP_PROCESSED_BLOB}")

        return anomalies_dict

    except Exception as e:
        logging.error(f"❌ Exception during anomaly detection: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    detect_anomalies()
