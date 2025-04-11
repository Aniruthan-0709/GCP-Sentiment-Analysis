
# 🚀 MLOps on GKE using Prebuilt Docker Images

This repo shows how to deploy a sentiment analysis web app and a data drift detection job on Google Kubernetes Engine (GKE), using **prebuilt Docker images** stored in Google Container Registry (GCR).

---

## 🧰 Prerequisites

- GCP project with billing enabled
- GKE cluster created (e.g., `sentiment-cluster`)
- Required APIs enabled:
  - Kubernetes Engine API
  - Container Registry API

---

## 🔧 Deployment Steps

### 1. Authenticate with GCP & Configure

```bash
gcloud auth login
gcloud config set project <your-project-id>
gcloud config set compute/zone us-east1-b
gcloud container clusters get-credentials sentiment-cluster
