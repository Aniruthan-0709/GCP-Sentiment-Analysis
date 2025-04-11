# 🚀 MLOps Model Deployment GKE: Sentiment Analysis + Data Drift Detection

This repository contains two Dockerfiles to deploy a FastAPI-based sentiment analysis service and a data drift detection job on Google Kubernetes Engine (GKE).

---

## 📦 Contents

- `Dockerfile.deploy` → Sentiment analyzer app with UI
- `Dockerfile.drift` → Data drift detection and GitHub trigger job

---

## 🧰 Prerequisites

Make sure you have the following:

- ✅ A Google Cloud project with billing enabled
- ✅ [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) installed
- ✅ Docker installed and authenticated with GCP
- ✅ Kubernetes CLI (`kubectl`) installed
- ✅ A service account JSON key with the following roles:
  - Storage Admin
  - BigQuery Data Viewer
  - Kubernetes Engine Admin
  - Artifact Registry Writer

---

## ⚙️ Step-by-Step Setup

### 🔑 1. Authenticate with Google Cloud

```bash
gcloud auth login
gcloud auth configure-docker
gcloud config set project <YOUR_PROJECT_ID>
gcloud config set compute/zone us-east1-b
```


### 2. Create a GKE Cluster

```bash
gcloud container clusters create sentiment-cluster --num-nodes=1 --zone=us-east1-b
gcloud container clusters get-credentials sentiment-cluster
```

### 3. Push Docker Images

#### Build and push sentiment analyzer

```bash
docker tag sentiment-analyzer-app gcr.io/<YOUR_PROJECT_ID>/sentiment-analyzer-app
docker push gcr.io/<YOUR_PROJECT_ID>/sentiment-analyzer-app
```

#### Build and push data drift detector

```bash
cd data_drift_detector
docker tag data-drift-detector gcr.io/<YOUR_PROJECT_ID>/data-drift-detector
docker push gcr.io/<YOUR_PROJECT_ID>/data-drift-detector
```

### 🔐 4. Create Required Secrets in GKE

```bash
kubectl create secret generic gcp-key-secret --from-file=gcp_key.json=<path-to-your-service-account-key.json>

kubectl create secret generic github-pat --from-literal=token=<YOUR_GITHUB_PAT>
```

### 📦 5. Deploy to GKE

#### Deploy the sentiment analyzer (web app)

```bash
kubectl apply -f k8s.yaml
```

#### 🌐 6. Access the Web App
Get the External Web address

```bash
kubectl get service sentiment-service
```

### 📊 7. Monitor Data Drift

```bash
kubectl get pods
kubectl logs <data-drift-pod-name>

```

