#Authenticate Google Cloud Account and set Project
gcloud auth login
gcloud config set project <YOUR_BUCKET_ID>

#Create a GKE cluster and authenticate it
gcloud container clusters create sentiment-cluster --num-nodes=1 --zone=us-east1-b
gcloud container clusters get-credentials testsentiment-cluster --zone=us-east1-b

#Create a Big Query Table to store the user logs"
bq query --use_legacy_sql=false '
CREATE OR REPLACE TABLE `<YOUR_BUCKET_ID>.sentiment_data.user_logs` (
  review STRING,
  sentiment STRING,
  timestamp TIMESTAMP
)'

#Authenticate your Github Repo with GCP Cloud Build
gcloud alpha builds connections create github --repository-owner="YOUR_GITHUB_USERNAME" --repository-name="YOUR_REPO_NAME" --connection-name="github-connection"

#Create a Cloud Build Trigger that triggers the GKE cluster 
gcloud beta builds triggers create github --name="deploymodelgke" --region=us-east1 \
  --repo-name="YOUR_REPO_NAME" \
  --repo-owner="YOUR_GITHUB_USERNAME" \
  --branch-pattern="^main$" \
  --build-config="cloudbuild.yaml" \
  --include-logs-with-status \
  --substitutions="_DEPLOY_ENV=prod"


#Create a Pub/Sub topic to trigger Cloud Build if there is Model Push in GCS
gsutil notification create -t model-updates-topic -f json -e OBJECT_FINALIZE -p models/ gs://<YOUR_BUCKET_ID>


