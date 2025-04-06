import os
import logging
import shutil

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("model_versioning.log"), logging.StreamHandler()]
)
logging.info("Starting Model Versioning...")

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_FILE = os.path.join(MODEL_DIR, "naive_bayes_sentiment.pkl")
version_file = os.path.join(MODEL_DIR, "model_version.txt")

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"{MODEL_FILE} not found. Cannot version.")

if os.path.exists(version_file):
    with open(version_file, "r") as f:
        version = int(f.read().strip())
else:
    version = 0

new_version = version + 1
new_model_file = os.path.join(MODEL_DIR, f"naive_bayes_sentiment_v{new_version}.pkl")
shutil.copy(MODEL_FILE, new_model_file)

with open(version_file, "w") as f:
    f.write(str(new_version))

logging.info(f"Model version updated to v{new_version}. New model file: {new_model_file}")
