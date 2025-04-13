Write-Host "Building Docker image..."
docker build -t sentiment-analyzer-app .

Write-Host "Tagging image..."
docker tag sentiment-analyzer-app gcr.io/mlops-project-test-448822/sentiment-analyzer-app

Write-Host "Pushing to GCR..."
docker push gcr.io/mlops-project-test-448822/sentiment-analyzer-app

Write-Host "Restarting Kubernetes deployment..."
kubectl rollout restart deployment sentiment-analyzer


Write-Host "Google Cloud Secret Key"
kubectl create secret generic gcp-key-secret --from-file=gcp_key.json=./.secrets/gcp_key.json

Write-Host "Re-applying Kubernetes YAML (if updated)..."
kubectl apply -f k8s.yaml

Write-Host "Fetching External IP of LoadBalancer..."
kubectl get service sentiment-service
