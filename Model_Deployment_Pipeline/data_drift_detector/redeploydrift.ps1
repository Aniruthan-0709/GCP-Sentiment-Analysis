Write-Host "Building Docker image for data drift..."
docker build -t data-drift-detector .

Write-Host "Create a Github Secret key for Triggering"
#kubectl create secret generic github-pat --from-literal=token=YOUR_GITHUB_PAT_HERE
#kubectl create secret generic gmail-secret --from-literal=EMAIL_APP_PASSWORD='your_app_password'

Write-Host "Tagging image..."
docker tag data-drift-detector gcr.io/mlops-project-test-448822/data-drift-detector

Write-Host "Pushing image to Google Container Registry (GCR)..."
docker push gcr.io/mlops-project-test-448822/data-drift-detector

Write-Host "Applying Kubernetes Job to run drift detection..."
kubectl apply -f data-drift-job.yaml

Write-Host "Job submitted. Check logs using Pod Name using:"
Write-Host "kubectl get pods  # find your pod name"
Write-Host "kubectl logs <pod-name>  # replace <pod-name> with the actual pod"
