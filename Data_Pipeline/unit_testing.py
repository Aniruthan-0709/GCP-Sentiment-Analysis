import os
import pytest
import pandas as pd

# ==== Basic File Tests ====

def test_raw_data_file_exists():
    assert os.path.exists("Data_Pipeline/data/raw/reviews.csv"), "Raw data file (reviews.csv) not found."

def test_processed_parquet_exists():
    assert os.path.exists("Data_Pipeline/data/processed/reviews.parquet"), "Processed .parquet file not found."

def test_processed_csv_exists():
    assert os.path.exists("Data_Pipeline/data/processed/reviews.csv"), "Processed .csv file not found."


# ==== Data Quality Checks ====

@pytest.fixture
def processed_df():
    return pd.read_csv("Data_Pipeline/data/processed/reviews.csv")

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


# ==== Schema and Validation ====

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
