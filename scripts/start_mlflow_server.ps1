# Start a local MLflow tracking server with Model Registry support.
# UI: http://127.0.0.1:5000

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

New-Item -ItemType Directory -Force -Path "mlflow_db" | Out-Null
New-Item -ItemType Directory -Force -Path "mlflow_artifacts" | Out-Null

Write-Host "Starting MLflow server at http://127.0.0.1:5000 ..."
Write-Host "Backend: sqlite:///mlflow_db/mlflow.db"
Write-Host "Artifacts: ./mlflow_artifacts"
Write-Host "Press Ctrl+C to stop."

mlflow server `
  --backend-store-uri "sqlite:///mlflow_db/mlflow.db" `
  --default-artifact-root "./mlflow_artifacts" `
  --host "127.0.0.1" `
  --port 5000
