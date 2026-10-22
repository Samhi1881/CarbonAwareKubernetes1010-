#!/usr/bin/env bash
# Usage: scripts/push_to_github.sh git@github.com:<user>/carbon-aware-k8s-autoscaler.git
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d .git ] || git init -b main
git add -A
git commit -m "Carbon-aware Kubernetes autoscaler: forecasting + CVXPY optimizer + safety layer" || true
git remote remove origin 2>/dev/null || true
git remote add origin "${1:?repo url required}"
git push -u origin main
