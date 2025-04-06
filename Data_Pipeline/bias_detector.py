import tensorflow_data_validation as tfdv
import tensorflow as tf
import pandas as pd
import os
import logging
import sys
from tensorflow_metadata.proto.v0 import statistics_pb2

# Setup directories
BASE_DIR = os.path.dirname(__file__)
LOG_DIR = os.path.join(BASE_DIR, "logs")
VALIDATION_DIR = os.path.join(BASE_DIR, "validation")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(VALIDATION_DIR, exist_ok=True)

# Logging setup
LOG_FILE = os.path.join(LOG_DIR, "mlops_bias_pipeline.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

# File paths
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data/processed/reviews.parquet")
SCHEMA_PATH = os.path.join(VALIDATION_DIR, "schema.pbtxt")
REFERENCE_STATS_PATH = os.path.join(VALIDATION_DIR, "reference_stats.tfrecord")
NEW_STATS_PATH = os.path.join(VALIDATION_DIR, "new_stats.tfrecord")
BIAS_REPORT_PATH = os.path.join(VALIDATION_DIR, "bias_report.txt")

def load_statistics_from_tfrecord(path):
    """Load statistics from TFRecord file."""
    dataset = tf.data.TFRecordDataset([path])
    for record in dataset:
        stats = statistics_pb2.DatasetFeatureStatisticsList()
        stats.ParseFromString(record.numpy())
        return stats
    return None

def save_statistics_as_tfrecord(stats, path):
    """Save statistics as TFRecord."""
    with tf.io.TFRecordWriter(path) as writer:
        writer.write(stats.SerializeToString())

def detect_bias(input_path=PROCESSED_DATA_PATH, schema_path=SCHEMA_PATH, reference_stats_path=REFERENCE_STATS_PATH):
    """Detects dataset bias by comparing current and reference statistics."""
    try:
        if not os.path.exists(schema_path):
            logging.error(f"❌ Schema not found at: {schema_path}")
            return None

        if not os.path.exists(input_path):
            logging.error(f"❌ Processed dataset not found at: {input_path}")
            return None

        logging.info("🔹 Loading processed dataset...")
        df = pd.read_parquet(input_path)

        logging.info("🔹 Generating statistics for current dataset...")
        new_stats = tfdv.generate_statistics_from_dataframe(df)

        # Save current stats
        save_statistics_as_tfrecord(new_stats, NEW_STATS_PATH)

        drift_results = []

        if os.path.exists(reference_stats_path):
            logging.info("🔄 Reference statistics found. Comparing for drift...")
            reference_stats = load_statistics_from_tfrecord(reference_stats_path)

            for feature in new_stats.datasets[0].features:
                feature_name = feature.path.step[0]
                ref_feature = next((f for f in reference_stats.datasets[0].features if f.path.step[0] == feature_name), None)

                if ref_feature and feature.HasField("num_stats") and ref_feature.HasField("num_stats"):
                    old_mean = ref_feature.num_stats.mean
                    new_mean = feature.num_stats.mean
                    percent_drift = ((new_mean - old_mean) / old_mean) * 100 if old_mean != 0 else 0

                    drift_results.append(f"{feature_name}: Old Mean = {old_mean:.3f}, New Mean = {new_mean:.3f}, Drift = {percent_drift:.2f}%")

            # Save to text file
            with open(BIAS_REPORT_PATH, "w", encoding="utf-8") as f:
                f.write("📊 Bias Detection Report\n")
                f.write("="*40 + "\n")
                for entry in drift_results:
                    f.write(entry + "\n")

            logging.info(f"📁 Drift report saved to: {BIAS_REPORT_PATH}")
        else:
            logging.warning("⚠️ No reference statistics found. Saving current stats as baseline.")
            save_statistics_as_tfrecord(new_stats, reference_stats_path)

        logging.info("✅ Bias detection complete.")
        return BIAS_REPORT_PATH

    except Exception as e:
        logging.error(f"❌ Error in bias detection: {e}")
        return None

if __name__ == "__main__":
    detect_bias()
