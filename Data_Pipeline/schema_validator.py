import tensorflow_data_validation as tfdv
import tensorflow as tf
import pandas as pd
import os
import logging
import sys
from utils.gcs_utils import read_parquet_from_gcs  # NEW import
from tensorflow_data_validation.utils import schema_util

# Directory setup
BASE_DIR = os.path.dirname(__file__)
LOG_DIR = os.path.join(BASE_DIR, "logs")
VALIDATION_DIR = os.path.join(BASE_DIR, "validation")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(VALIDATION_DIR, exist_ok=True)

# Logging setup
LOG_FILE = os.path.join(LOG_DIR, "mlops_schema_pipeline.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

# GCS ENV
BUCKET_NAME = os.environ.get("GCP_BUCKET")
PROCESSED_BLOB = os.environ.get("GCP_PROCESSED_PARQUET")  # e.g., processed/reviews.parquet

# Local schema/stat paths
SCHEMA_PATH = os.path.join(VALIDATION_DIR, "schema.pbtxt")
REFERENCE_STATS_PATH = os.path.join(VALIDATION_DIR, "reference_stats.tfrecord")

def save_statistics_as_tfrecord(stats, path):
    """Save statistics to TFRecord format."""
    with tf.io.TFRecordWriter(path) as writer:
        writer.write(stats.SerializeToString())

def validate_schema(schema_path=SCHEMA_PATH, stats_path=REFERENCE_STATS_PATH):
    """Validates schema using TFDV and saves schema/stats as needed."""
    try:
        logging.info("🔹 Reading processed dataset directly from GCS...")
        df = read_parquet_from_gcs(BUCKET_NAME, PROCESSED_BLOB)

        logging.info("🔹 Generating statistics from dataset...")
        stats = tfdv.generate_statistics_from_dataframe(df)

        if os.path.exists(schema_path):
            logging.info("🔍 Schema exists. Validating against it...")
            schema = tfdv.load_schema_text(schema_path)
            anomalies = tfdv.validate_statistics(stats, schema)

            if anomalies.anomaly_info:
                logging.warning("⚠️ Schema anomalies found.")
                for feature, detail in anomalies.anomaly_info.items():
                    logging.warning(f" - {feature}: {detail.description}")

                for feature, detail in anomalies.anomaly_info.items():
                    if "column is completely missing" in detail.description.lower():
                        raise ValueError(f"🚨 Critical schema change! Column '{feature}' is missing.")
            else:
                logging.info("✅ No schema anomalies found.")
        else:
            logging.info("📥 No existing schema. Inferring and saving new schema...")
            schema = tfdv.infer_schema(stats)
            tfdv.write_schema_text(schema, schema_path)
            logging.info(f"✅ New schema saved to: {schema_path}")

            save_statistics_as_tfrecord(stats, stats_path)
            logging.info(f"✅ Reference stats saved to: {stats_path}")

        return schema_path, stats_path

    except Exception as e:
        logging.error(f"❌ Error during schema validation: {e}")
        return None, None

if __name__ == "__main__":
    validate_schema()
