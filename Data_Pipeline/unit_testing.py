import os
import pytest
import pandas as pd
from utils.gcs_utils import read_csv_from_gcs

# GCS env variables
GCP_BUCKET = os.environ["GCP_BUCKET"]
GCP_PROCESSED_BLOB = os.environ["GCP_PROCESSED_BLOB"]

# ==== Load Data Once for All Tests ====

@pytest.fixture(scope="session")
def processed_df():
    return read_csv_from_gcs(GCP_BUCKET, GCP_PROCESSED_BLOB)

# ==== Data Quality Checks ====

def test_sentiment_column_exists(processed_df):
    assert "review_sentiment" in processed_df.columns, "Missing 'review_sentiment' column in processed data"

def test_star_rating_within_bounds(processed_df):
    assert processed_df["star_rating"].between(1, 5).all(), "Some star_rating values are outside 1–5"

def test_review_body_not_empty(processed_df):
    assert processed_df["review_body"].str.strip().ne("").all(), "Empty review_body found in processed data"

def test_product_category_encoded_present(processed_df):
    assert "product_category_encoded" in processed_df.columns, "Missing encoded product_category column"

def test_sentiment_class_distribution(processed_df):
    classes = set(processed_df["review_sentiment"].unique())
    expected_classes = {"negative", "neutral", "positive"}
    assert expected_classes.issubset(classes), f"Expected sentiment classes missing. Found: {classes}"

# ==== Schema and Logs Validation (LOCAL FILE CHECKS ONLY) ====

def test_schema_file_exists():
    assert os.path.exists("Data_Pipeline/validation/schema.pbtxt"), "Schema file not generated"

def test_bias_report_exists():
    assert os.path.exists("Data_Pipeline/validation/bias_report.txt"), "Bias report not found"

def test_logs_generated():
    log_files = [
        "mlops_ingestion_pipeline.log",
        "mlops_preprocessing_pipeline.log",
        "mlops_schema_pipeline.log",
        "mlops_anomalies_pipeline.log",
        "mlops_bias_pipeline.log",
        "mlops_upload_pipeline.log"
    ]
    for log in log_files:
        path = os.path.join("Data_Pipeline/logs", log)
        assert os.path.exists(path), f"Expected log not found: {log}"
