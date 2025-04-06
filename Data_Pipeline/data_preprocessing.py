import pandas as pd
import os
import logging
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler

# Setup directory paths
BASE_DIR = os.path.dirname(__file__)
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "mlops_preprocessing_pipeline.log"), encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# File paths
RAW_DATA_PATH = os.path.join(BASE_DIR, "data/raw/reviews.csv")
PROCESSED_PARQUET_PATH = os.path.join(BASE_DIR, "data/processed/reviews.parquet")
PROCESSED_CSV_PATH = PROCESSED_PARQUET_PATH.replace(".parquet", ".csv")

def preprocess_data(input_path=RAW_DATA_PATH, parquet_path=PROCESSED_PARQUET_PATH, csv_path=PROCESSED_CSV_PATH):
    """Cleans, encodes, balances, and adds sentiment labels to the dataset."""
    try:
        logging.info("🔹 Loading raw data...")
        df = pd.read_csv(input_path)

        logging.info("🔹 Dropping duplicates & handling missing values...")
        df = df.drop_duplicates()
        df = df.dropna(subset=["star_rating", "review_body", "product_category"])

        if df["star_rating"].min() < 1 or df["star_rating"].max() > 5:
            logging.warning(f"⚠️ Star ratings out of range. Scaling to 1-5...")
            scaler = MinMaxScaler(feature_range=(1, 5))
            df["star_rating"] = scaler.fit_transform(df[["star_rating"]]).round().astype(int)

        logging.info("🔹 Cleaning review text...")
        df["review_body"] = df["review_body"].str.lower().str.replace(r"[^\w\s]", "", regex=True).str.strip()

        logging.info("🔹 Encoding product category...")
        df["product_category_encoded"] = df["product_category"].astype("category").cat.codes

        logging.info("🔹 Adding sentiment labels...")
        df["review_sentiment"] = df["star_rating"].apply(
            lambda x: "negative" if x in [1, 2]
            else "neutral" if x == 3
            else "positive"
        )

        # Check sentiment distribution
        sentiment_counts = Counter(df["review_sentiment"])
        logging.info(f"🔹 Initial Sentiment Distribution: {sentiment_counts}")

        if len(sentiment_counts) > 2 and max(sentiment_counts.values()) / min(sentiment_counts.values()) > 1.5:
            logging.info("🔹 Applying SMOTE for class balancing...")
            smote = SMOTE(sampling_strategy="auto", random_state=42)
            X = df.drop(columns=["review_sentiment", "review_body", "product_category"])  # exclude text/categorical
            y = df["review_sentiment"]

            X_resampled, y_resampled = smote.fit_resample(X.select_dtypes(include=["number"]), y)

            df_resampled = pd.DataFrame(X_resampled, columns=X.select_dtypes(include=["number"]).columns)
            df_resampled["review_sentiment"] = y_resampled

            logging.info("🔹 Merging back text columns for output...")
            df_text = df[["review_body", "product_category"]].iloc[:len(df_resampled)].reset_index(drop=True)
            df = pd.concat([df_resampled, df_text], axis=1)

        # Save files
        os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
        df.to_parquet(parquet_path, index=False)
        df.to_csv(csv_path, index=False)

        logging.info(f"✅ Preprocessing complete. Saved to:\n  - {parquet_path}\n  - {csv_path}")
        logging.info(f"🔹 Balanced Sentiment Distribution: {Counter(df['review_sentiment'])}")

        return parquet_path, csv_path

    except Exception as e:
        logging.error(f"❌ Error during preprocessing: {e}")
        return None, None

if __name__ == "__main__":
    preprocess_data()
