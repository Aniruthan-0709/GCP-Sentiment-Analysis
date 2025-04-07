import pandas as pd
import logging
import os
import pickle
import shap
import matplotlib.pyplot as plt

# ========== Logging ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("sensitivity_analysis.log"),
        logging.StreamHandler()
    ]
)
logging.info("Starting Sensitivity Analysis using SHAP...")

# ========== Load Pickled Pipeline ==========
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "sentiment_analyzer_model.pkl")

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError("Pipeline model file not found.")

with open(MODEL_FILE, "rb") as f:
    model_pipeline = pickle.load(f)

# Extract vectorizer and model from pipeline
vectorizer = model_pipeline.named_steps["tfidf"]
model = model_pipeline.named_steps["nb"]

# ========== Load and Prepare Data ==========
df = pd.read_csv("Data/Data.csv")

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

# ========== Sample Data ==========
sample_df = df.sample(n=5, random_state=42)
texts = sample_df["review_body"].tolist()
X_sample = vectorizer.transform(texts)

# ========== SHAP Explanation ==========
shap.initjs()

# KernelExplainer for scikit-learn Naive Bayes
explainer = shap.Explainer(model.predict_proba, X_sample.toarray())
shap_values = explainer(X_sample.toarray())

# ========== Save SHAP Summary Plot ==========
plt.figure()
shap.summary_plot(
    shap_values,
    X_sample.toarray(),
    feature_names=vectorizer.get_feature_names_out(),
    show=False
)

os.makedirs("artifacts", exist_ok=True)
plot_path = os.path.join("artifacts", "shap_summary.png")
plt.savefig(plot_path, bbox_inches="tight")
plt.close()

logging.info(f"SHAP summary plot saved at {plot_path}")
