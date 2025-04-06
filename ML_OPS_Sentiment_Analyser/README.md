# Amazon Customer Sentiment Analysis

## IE 7374: Model Pipeline
**Group Number: 7**  

### Team Members:  
- Aniruthan Swaminathan Arulmurugan  
- Manikandan Mohan  
- Manivannan Senthilkumar  
- Janani Karthikeyan  
- Kiran Tamilselvan  
- Aravind Anbazhagan  

### GitHub Repo Link  
[Sentiment Analysis Tool for Product Reviews](https://github.com/Aniruthan-0709/Sentiment-Analysis-Tool-for-Product-Reviews)

---

## 📌 Synopsis  
This documentation outlines the **end-to-end model pipeline** for sentiment analysis of Amazon customer reviews, **automated through GitHub Actions**. The pipeline includes:  
- **Data retrieval from Google Cloud Storage (GCS)**  
- **Naïve Bayes model training for sentiment classification**  
- **MLflow for model tracking and version control**  
- **Selection of the best-performing model based on accuracy, precision, recall, and F1-score**  
- **Automated retraining triggers** to maintain model accuracy  
- **Deployment on Google Cloud Platform (GCP)** for scalable inference  

---

## 📊 Data Pipeline Integration  
The **model development pipeline** is automatically triggered after the data pipeline completes execution.  

### **Steps**  
1. **Data Retrieval:** The pipeline fetches the latest dataset from **GCP storage**.  
2. **Version Tracking:** Each dataset version is logged for reproducibility.  

---

## 🛠 Model Development & Training  
Once the data is retrieved, the following steps occur:  

1. **Hyperparameter Tuning:** Optimized using `GridSearchCV`.  
2. **MLflow Logging:** Tracks hyperparameters, performance metrics, and model versions.  
3. **Automated Retraining:** If performance metrics fall below a threshold (e.g., `F1-score < 0.7`), the model retrains up to three times.  

---

## 🏷 Model Registry & Versioning  
We use **MLflow’s Model Registry** for managing model versions:  

- **Version Creation:** Each training run produces a new model version.  
- **Staging Versions:** Promising models are marked as `"Staging"` for further evaluation.  
- **Production Versions:** The best model is promoted to `"Production"`.  
- **Archived Versions:** Older models are archived for historical tracking.  

---

## 📈 Model Evaluation & Selection  
1. **Performance Metrics:** Models are evaluated based on accuracy, precision, recall, and F1-score.  
2. **Bias Detection:** Data slicing techniques ensure fairness across different subgroups.  
3. **Sensitivity Analysis:** Measures the impact of feature variations on model performance.  

---

## 🏗 Model Pipeline DAG  
The pipeline follows a **Directed Acyclic Graph (DAG)** structure:  

### **1️⃣ Trigger Workflow**  
Initiated by **GitHub Actions** when new data is available.  

#### **Workflow: Train, Track with MLflow, and Upload Sentiment Model**  
**Triggers:**  
- On push to `"main"` branch  
- Every **Monday at midnight UTC**  

#### **Steps:**  
1. **Checkout Code**  
2. **Set Up Python (3.9) & Install Dependencies**  
3. **Authenticate with Google Cloud**  
4. **Start MLflow Tracking Server**  
5. **Train Model & Log with MLflow**  
6. **Upload Model to Google Cloud**  
7. **Save Model & MLflow Logs as Artifacts**  

---

## 📦 Data Processing  
The pipeline reads raw data from GCS and preprocesses it.  

```python
BUCKET_NAME = "mlops_dataset123"  # Replace with your GCS bucket name  
FILE_NAME = "data/raw/Sampled_Chunk.csv"  # Replace with your file name in GCS  
LOCAL_DIR = "Data/"  # Local folder to save the file  
LOCAL_FILE_NAME = "Data.csv"

def download_from_gcp(bucket_name, source_blob_name, destination_file_name):
    # Function to download data from GCP
