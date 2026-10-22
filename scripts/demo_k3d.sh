#!/usr/bin/env bash
# One-shot setup of the live demo: k3d cluster + demo app. Then follow the printed next steps.
set -euo pipefail
cd "$(dirname "$0")/.."
k3d cluster create carbon-demo --agents 1 --wait || true
docker build -t carbon-demo-app:latest demo_app
k3d image import carbon-demo-app:latest -c carbon-demo
kubectl apply -f k8s/demo-app.yaml
kubectl rollout status deploy/demo-app --timeout=120s
cat <<MSG

Cluster ready. Open 3 more terminals (venv active, 'pip install -e .' done):
  T1: kubectl port-forward svc/demo-app 8080:80
  T2: python scripts/loadgen.py --url http://localhost:8080/work --rps 15 --duration 600
  T3: carbon-autoscaler run --interval 15 --cooldown 45
  T4: watch kubectl get deploy demo-app
MSG
