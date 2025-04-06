import pandas as pd
import logging
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV, train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("hyperparameter_tuning.log"), logging.StreamHandler()]
)
logging.info("Starting Hyperparameter Tuning...")

# Load dataset
data_path = os.path.join("Data", "Data.csv")
df = pd.read_csv(data_path)

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

# Balance the dataset
positive = df[df["label"] == 2]
negative = df[df["label"] == 0]
neutral = df[df["label"] == 1]
negative_upsampled = negative.sample(len(positive), replace=True, random_state=42)
neutral_upsampled = neutral.sample(len(positive), replace=True, random_state=42)
balanced_df = pd.concat([positive, negative_upsampled, neutral_upsampled]).sample(frac=1, random_state=42)

X_train, _, y_train, _ = train_test_split(
    balanced_df["review_body"], balanced_df["label"], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)

param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0]}
model = MultinomialNB()

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_tfidf, y_train)

logging.info(f"Best Parameters: {grid_search.best_params_}")
logging.info(f"Best CV Accuracy: {grid_search.best_score_:.4f}")

os.makedirs("artifacts", exist_ok=True)
with open(os.path.join("artifacts", "best_params.txt"), "w") as f:
    f.write(str(grid_search.best_params_))
