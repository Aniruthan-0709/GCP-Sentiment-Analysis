Write-Host "Building Docker image for data drift..."
docker build -t data-drift-detector .

Write-Host "Create a Github Secret key for Triggering"
# Load Github and GMail secrets from secrets/secrets.env
$secrets = Get-Content "./.secrets/secrets.env" | ForEach-Object {
    if ($_ -match "=") {
        $parts = $_ -split '=', 2
        $key = $parts[0].Trim()
        $value = $parts[1].Trim('"').Trim()
        @{ Key = $key; Value = $value }
    }
}

# Convert to hashtable
$secretMap = @{}
foreach ($pair in $secrets) {
    $secretMap[$pair.Key] = $pair.Value
}

Write-Host "Tagging image..."
docker tag data-drift-detector gcr.io/mlops-project-test-448822/data-drift-detector

Write-Host "Pushing image to Google Container Registry (GCR)..."
docker push gcr.io/mlops-project-test-448822/data-drift-detector

# Create GCP Key secret
Write-Host "Creating GCP Key secret"
kubectl create secret generic gcp-key-secret --from-file=gcp_key.json=./.secrets/gcp_key.json --dry-run=client -o yaml | kubectl apply -f -

# Create GitHub PAT secret
Write-Host "Creating GitHub Secret..."
kubectl create secret generic github-pat `--from-literal=token=$($secretMap["GITHUB_PAT"]) `--dry-run=client -o yaml | kubectl apply -f -

# Create Gmail secret
Write-Host "Creating Gmail Secret..."
kubectl create secret generic gmail-secret `--from-literal=EMAIL_APP_PASSWORD=$($secretMap["EMAIL_APP_PASSWORD"]) `--dry-run=client -o yaml | kubectl apply -f -

Write-Host "Applying Kubernetes Job to run drift detection..."
kubectl apply -f data-drift-job.yaml

Write-Host "Job submitted. Check logs using Pod Name using:"
Write-Host "kubectl get pods  # find your pod name"
Write-Host "kubectl logs <pod-name>  # replace <pod-name> with the actual pod"
