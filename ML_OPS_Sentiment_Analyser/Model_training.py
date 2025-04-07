import pandas as pd
import logging
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

# ========== Logging ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("naive_bayes_training.log"), logging.StreamHandler()]
)
logging.info("Starting Sentiment Analysis Training with Naïve Bayes...")

# ========== Config ==========
DATA_PATH = "Data/Data.csv"
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "sentiment_analyzer_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# ========== Load Data ==========
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

# ========== Balance the Data ==========
logging.info("Balancing dataset...")
positive = df[df["label"] == 2]
negative = df[df["label"] == 0]
neutral = df[df["label"] == 1]
negative_upsampled = negative.sample(len(positive), replace=True, random_state=42)
neutral_upsampled = neutral.sample(len(positive), replace=True, random_state=42)
balanced_df = pd.concat([positive, negative_upsampled, neutral_upsampled])
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# ========== Train/Test Split ==========
X_train, X_test, y_train, y_test = train_test_split(
    balanced_df["review_body"], balanced_df["label"], test_size=0.2, random_state=42
)

# ========== Pipeline ==========
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
    ("nb", MultinomialNB())
])

logging.info("Training pipeline...")
pipeline.fit(X_train, y_train)

# ========== Evaluation ==========
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
logging.info(f"Validation Accuracy: {accuracy:.4f}")

# ========== Save Pipeline using Pickle ==========
with open(MODEL_FILE, "wb") as f:
    pickle.dump(pipeline, f)
logging.info(f"Pipeline model saved at {MODEL_FILE}")
