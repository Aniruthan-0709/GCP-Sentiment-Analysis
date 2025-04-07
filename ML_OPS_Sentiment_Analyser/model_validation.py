import pandas as pd
import logging
import os
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# ========== Logging Setup ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("model_validation.log"),
        logging.StreamHandler()
    ]
)
logging.info("Starting Model Validation...")

# ========== Load Pickled Pipeline ==========
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "sentiment_analyzer_model.pkl")

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError("Pipeline model file not found.")

with open(MODEL_FILE, "rb") as f:
    model_pipeline = pickle.load(f)

# ========== Load and Prepare Data ==========
DATA_PATH = os.path.join("Data", "Data.csv")
df = pd.read_csv(DATA_PATH)

def map_sentiment(rating):
    if rating <= 2:
        return "Negative"
    elif rating == 3:
        return "Neutral"
    else:
        return "Positive"

df["label"] = df["star_rating"].apply(map_sentiment)
df = df.dropna(subset=["review_body"])
df["review_body"] = df["review_body"].astype(str)

label_mapping = {"Negative": 0, "Neutral": 1, "Positive": 2}
df["label"] = df["label"].map(label_mapping)

# ========== Balance the Dataset ==========
positive = df[df["label"] == 2]
negative = df[df["label"] == 0]
neutral = df[df["label"] == 1]
negative_upsampled = negative.sample(len(positive), replace=True, random_state=42)
neutral_upsampled = neutral.sample(len(positive), replace=True, random_state=42)
balanced_df = pd.concat([positive, negative_upsampled, neutral_upsampled])
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# ========== Split Test Set ==========
_, X_test, _, y_test = train_test_split(
    balanced_df["review_body"], balanced_df["label"], test_size=0.2, random_state=42
)

# ========== Predict and Evaluate ==========
y_pred = model_pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

logging.info(f"Validation Accuracy: {accuracy:.4f}")
logging.info(f"Confusion Matrix:\n{cm}")
