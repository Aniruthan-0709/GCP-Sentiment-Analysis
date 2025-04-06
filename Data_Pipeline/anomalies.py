import tensorflow_data_validation as tfdv
import pandas as pd
import os
import logging
import sys

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

# Paths
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data/processed/reviews.parquet")
SCHEMA_PATH = os.path.join(BASE_DIR, "validation/schema.pbtxt")
NEW_STATS_PATH = os.path.join(BASE_DIR, "validation/new_stats.tfrecord")

def detect_anomalies(input_path=PROCESSED_DATA_PATH, schema_path=SCHEMA_PATH):
    """Detects data anomalies using TFDV and returns a dictionary of anomalies."""
    try:
        if not os.path.exists(schema_path):
            logging.error(f"❌ Schema file not found: {schema_path}")
            return {"error": "Schema file not found"}

        if not os.path.exists(input_path):
            logging.error(f"❌ Processed dataset not found: {input_path}")
            return {"error": "Processed dataset not found"}

        logging.info("🔹 Loading schema...")
        schema = tfdv.load_schema_text(schema_path)

        logging.info("🔹 Reading processed data...")
        df = pd.read_parquet(input_path)

        logging.info("🔹 Generating statistics from processed data...")
        new_stats = tfdv.generate_statistics_from_dataframe(df)

        os.makedirs(os.path.dirname(NEW_STATS_PATH), exist_ok=True)
        tfdv.write_stats_text(new_stats, NEW_STATS_PATH)

        logging.info("🔹 Running anomaly detection against schema...")
        anomalies = tfdv.validate_statistics(new_stats, schema)

        anomalies_dict = {}
        if anomalies.anomaly_info:
            logging.warning("⚠️ Anomalies detected:")
            for feature, anomaly in anomalies.anomaly_info.items():
                description = anomaly.description
                anomalies_dict[feature] = description
                logging.warning(f"⚠️ {feature}: {description}")
        else:
            logging.info("✅ No anomalies found.")

        return anomalies_dict

    except Exception as e:
        logging.error(f"❌ Exception during anomaly detection: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    detect_anomalies()
