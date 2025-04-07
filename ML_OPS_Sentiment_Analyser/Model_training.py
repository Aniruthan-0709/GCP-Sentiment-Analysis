import pandas as pd
import logging
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# ========== Logging ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("naive_bayes_training.log"), logging.StreamHandler()]
)
logging.info("Starting Sentiment Analysis Training with Naïve Bayes...")

# ========== Custom Wrapper Class ==========
class SentimentModel:
    def __init__(self, model, vectorizer, label_map_rev):
        self.model = model
        self.vectorizer = vectorizer
        self.label_map_rev = label_map_rev

    def predict(self, texts):
        tfidf = self.vectorizer.transform(texts)
        preds = self.model.predict(tfidf)
        return [self.label_map_rev[p] for p in preds]

# ========== Config ==========
DATA_PATH = "Data/Data.csv"
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "naive_bayes_sentiment.pkl")

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
label_map_rev = {v: k for k, v in label_mapping.items()}
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

# ========== Vectorizer & Model ==========
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = MultinomialNB()
logging.info("Training model...")
model.fit(X_train_tfidf, y_train)

# ========== Evaluation ==========
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
logging.info(f"Validation Accuracy: {accuracy:.4f}")

# ========== Save the Wrapper ==========
serving_model = SentimentModel(model, vectorizer, label_map_rev)
joblib.dump(serving_model, MODEL_FILE)
logging.info(f"Wrapped model saved at {MODEL_FILE}")
