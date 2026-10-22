# Windows PowerShell version of demo_k3d.sh. Run from anywhere: powershell -File scripts\demo_k3d.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
k3d cluster create carbon-demo --agents 1 --wait
docker build -t carbon-demo-app:latest demo_app
k3d image import carbon-demo-app:latest -c carbon-demo
kubectl apply -f k8s/demo-app.yaml
kubectl rollout status deploy/demo-app --timeout=120s
Write-Host @"

Cluster ready. Open 4 terminals (venv active in the Python ones):
  T1: kubectl port-forward svc/demo-app 8080:80
  T2: python scripts/loadgen.py --rps 15 --duration 1200
  T3: carbon-autoscaler run --interval 15 --cooldown 45 --dry-run   (drop --dry-run for real scaling)
  T4: kubectl get deploy demo-app -w
"@
