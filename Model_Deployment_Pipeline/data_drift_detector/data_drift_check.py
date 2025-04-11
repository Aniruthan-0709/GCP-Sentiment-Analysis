from google.cloud import storage, bigquery
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import chi2_contingency
import tempfile
import os
import numpy as np
import requests  # <-- missing import

# Get key path (for local testing only — not needed in GKE if using Workload Identity or mounted secret)
KEY_FILE = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

# Only validate locally
if KEY_FILE and not os.path.exists(KEY_FILE):
    raise ValueError(f"GCP key file not found at: {KEY_FILE}")

def load_reference_data(bucket_name="mlops_dataset123", blob_path="data/raw/Sampled_Chunk.csv"):
    client = storage.Client() if not KEY_FILE else storage.Client.from_service_account_json(KEY_FILE)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    
    _, temp_path = tempfile.mkstemp()
    blob.download_to_filename(temp_path)
    
    df = pd.read_csv(temp_path)
    df["review_texts"] = df["review_headline"].fillna("") + " " + df["review_body"].fillna("")
    return df[["review_texts"]]

def load_current_data():
    client = bigquery.Client() if not KEY_FILE else bigquery.Client.from_service_account_json(KEY_FILE)
    query = "SELECT review AS review_texts FROM `mlops-project-test-448822.sentiment_data.user_logs`"
    return client.query(query).to_dataframe()

def detect_data_drift(reference_df, current_df, output_path="drift_report.html"):
    vectorizer = CountVectorizer()
    ref_counts = vectorizer.fit_transform(reference_df["review_texts"])
    current_counts = vectorizer.transform(current_df["review_texts"])
    
    ref_word_counts = np.array(ref_counts.sum(axis=0)).flatten()
    current_word_counts = np.array(current_counts.sum(axis=0)).flatten()
    
    total_ref_counts = ref_word_counts.sum()
    total_current_counts = current_word_counts.sum()
    
    drifted_features = 0
    significant_features = []
    alpha = 0.001

    for i, word in enumerate(vectorizer.get_feature_names_out()):
        count_ref = ref_word_counts[i]
        count_curr = current_word_counts[i]
        if count_ref + count_curr < 5:
            continue

        table = [
            [count_ref, total_ref_counts - count_ref],
            [count_curr, total_current_counts - count_curr]
        ]
        chi2, p, _, _ = chi2_contingency(table, correction=False)
        if p < alpha:
            drifted_features += 1
            significant_features.append(word)
    
    total_features = len(vectorizer.get_feature_names_out())
    drift_detected = (drifted_features / total_features) > alpha

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("<html><head><title>Data Drift Report</title></head><body>")
        f.write("<h1>Data Drift Report</h1>")
        f.write(f"<p><strong>Drift Detected:</strong> {drift_detected}</p>")
        f.write(f"<p><strong>Drifted Features:</strong> {drifted_features} / {total_features}</p>")
        if significant_features:
            f.write("<ul>")
            for word in significant_features[:20]:
                f.write(f"<li>{word}</li>")
            f.write("</ul>")
        f.write("</body></html>")
    
    return drift_detected, drifted_features

def upload_report_to_gcs(local_path, bucket_name="mlops_dataset123", gcs_path="drift_report/drift_report.html"):
    client = storage.Client() if not KEY_FILE else storage.Client.from_service_account_json(KEY_FILE)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    print(f"✅ Uploaded drift report to gs://{bucket_name}/{gcs_path}")

def trigger_github_action():
    """Triggers GitHub Actions if drift is detected."""
    token = os.environ.get("GITHUB_PAT")
    if not token:
        print("⚠️ GitHub token not provided.")
        return

    repo = "Mani31899/MLOPS-Sentiment-Analyzer-Project"
    workflow_file = "data_pipeline.yaml"

    url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow_file}/dispatches"

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    response = requests.post(url, headers=headers, json={"ref": "main"})

    if response.status_code == 204:
        print("✅ GitHub Actions workflow triggered successfully.")
    else:
        print(f"❌ Failed to trigger GitHub Actions: {response.status_code}, {response.text}")


if __name__ == "__main__":
    reference_df = load_reference_data()
    current_df = load_current_data()
    
    report_path = "drift_report.html"
    drift_detected, drifted_features = detect_data_drift(reference_df, current_df, report_path)

    print(f"🚨 Drift Detected: {drift_detected} | Features Drifted: {drifted_features}")
    upload_report_to_gcs(report_path)

    if drift_detected:
        trigger_github_action()
