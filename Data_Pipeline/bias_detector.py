import tensorflow_data_validation as tfdv
import tensorflow as tf
import pandas as pd
import os
import logging
import sys
from io import StringIO
from tensorflow_metadata.proto.v0 import statistics_pb2
from utils.gcs_utils import read_csv_from_gcs, upload_to_gcp

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

# GCS env
GCP_BUCKET = os.environ["GCP_BUCKET"]
GCP_PROCESSED_BLOB = os.environ["GCP_PROCESSED_BLOB"]

# Local paths
SCHEMA_PATH = os.path.join(VALIDATION_DIR, "schema.pbtxt")
REFERENCE_STATS_PATH = os.path.join(VALIDATION_DIR, "reference_stats.tfrecord")
NEW_STATS_PATH = os.path.join(VALIDATION_DIR, "new_stats.tfrecord")
BIAS_REPORT_PATH = os.path.join(VALIDATION_DIR, "bias_report.txt")

def load_statistics_from_tfrecord(path):
    dataset = tf.data.TFRecordDataset([path])
    for record in dataset:
        stats = statistics_pb2.DatasetFeatureStatisticsList()
        stats.ParseFromString(record.numpy())
        return stats
    return None

def save_statistics_as_tfrecord(stats, path):
    with tf.io.TFRecordWriter(path) as writer:
        writer.write(stats.SerializeToString())

def detect_bias(schema_path=SCHEMA_PATH, reference_stats_path=REFERENCE_STATS_PATH):
    try:
        if not os.path.exists(schema_path):
            logging.error(f"❌ Schema not found at: {schema_path}")
            return None

        logging.info("🔹 Reading processed CSV from GCS...")
        df = read_csv_from_gcs(GCP_BUCKET, GCP_PROCESSED_BLOB)

        logging.info("🔹 Generating current statistics...")
        new_stats = tfdv.generate_statistics_from_dataframe(df)
        save_statistics_as_tfrecord(new_stats, NEW_STATS_PATH)

        drift_results = []
        updated = False

        if os.path.exists(reference_stats_path):
            logging.info("🔍 Comparing with reference statistics...")
            reference_stats = load_statistics_from_tfrecord(reference_stats_path)

            for feature in new_stats.datasets[0].features:
                feature_name = feature.path.step[0]
                ref_feature = next((f for f in reference_stats.datasets[0].features if f.path.step[0] == feature_name), None)

                if ref_feature and feature.HasField("num_stats") and ref_feature.HasField("num_stats"):
                    old_mean = ref_feature.num_stats.mean
                    new_mean = feature.num_stats.mean
                    percent_drift = ((new_mean - old_mean) / old_mean) * 100 if old_mean != 0 else 0

                    drift_results.append(f"{feature_name}: Old Mean = {old_mean:.3f}, New Mean = {new_mean:.3f}, Drift = {percent_drift:.2f}%")

                    if abs(percent_drift) > 50 and feature_name == "star_rating":
                        logging.warning(f"⚠️ High drift in {feature_name}. Capping values to 1–5.")
                        df[feature_name] = df[feature_name].clip(1, 5)
                        updated = True

            # Save report
            with open(BIAS_REPORT_PATH, "w", encoding="utf-8") as f:
                f.write("📊 Bias Detection Report\n")
                f.write("=" * 40 + "\n")
                for entry in drift_results:
                    f.write(entry + "\n")

            logging.info(f"📁 Drift report saved at: {BIAS_REPORT_PATH}")
        else:
            logging.warning("⚠️ No reference stats found. Saving current stats as baseline.")
            save_statistics_as_tfrecord(new_stats, reference_stats_path)

        if updated:
            logging.info("☁️ Uploading corrected CSV to GCS...")
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            upload_to_gcp(GCP_BUCKET, csv_buffer.getvalue().encode("utf-8"), GCP_PROCESSED_BLOB, from_memory=True)
            logging.info(f"✅ Corrected file uploaded to gs://{GCP_BUCKET}/{GCP_PROCESSED_BLOB}")
        else:
            logging.info("✅ No bias corrections needed. No file upload performed.")

        logging.info("✅ Bias detection complete.")
        return BIAS_REPORT_PATH

    except Exception as e:
        logging.error(f"❌ Error in bias detection: {e}")
        return None

if __name__ == "__main__":
    detect_bias()
