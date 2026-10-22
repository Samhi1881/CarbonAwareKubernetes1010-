# Demo guide

## Provenance (say this up front)
This repository is a clean reference implementation written *from* the paper. It is not the code that produced the
paper's reported measurements, and its simulator numbers differ from the paper's tables (synthetic data, different
latency model). The headline result - ~50 % less carbon than over-provisioning with the SLO held - is reproduced.

## 7-minute script
1. **Idea (30 s)** - HPA/VPA/CA ignore carbon; grid intensity swings 2-4x/day. Forecast -> optimise -> safety-check -> scale.
2. `carbon-autoscaler forecast-eval` - Table 1 equivalent; ARIMA is weak on hourly data, ensemble is mostly seasonal.
3. `carbon-autoscaler optimize-once --replicas 4 --p95 160 --cpu 0.28 --carbon 400` - one solve, 4 -> 2.
4. `carbon-autoscaler simulate --plot` - three scenarios; open `results/simulation.png`.
5. `carbon-autoscaler simulate --load spike --initial-replicas 2 --outdir results_spike --plot` - emergency override + cooldown.
6. Live k3d (optional) - see below. `head -2 results/live_decisions.jsonl` shows the audit trail.

## Live k3d
macOS/Linux/WSL: `bash scripts/demo_k3d.sh`   Windows PowerShell: `powershell -File scripts\demo_k3d.ps1`

    kubectl port-forward svc/demo-app 8080:80                       # T1 (keep running)
    python scripts/loadgen.py --rps 15 --duration 1200               # T2
    carbon-autoscaler run --interval 15 --cooldown 45 --dry-run      # T3 (drop --dry-run to actually scale)
    kubectl get deploy demo-app -w                                   # T4

Raise `--rps` to ~40 mid-run to see scale-up (max +3 per step); stop the load to see scale-down.
Cleanup: `k3d cluster delete carbon-demo`

## Troubleshooting
- `HOLD - all probes failed`: port-forward (T1) died; restart it. The controller deliberately takes no action meanwhile.
- `ImagePullBackOff`: rerun `k3d image import carbon-demo-app:latest -c carbon-demo`.
- CPU stuck at 50 %: metrics-server not ready; check `kubectl top pods` after ~1 min.
- Install errors for cvxpy/ecos: use Python 3.10-3.12.

## Likely questions
- *Carbon-aware equals baseline?* The saving is from right-sizing vs over-provisioning; temporal carbon-awareness acts only at the margin via the carbon term. Say so yourself.
- *Why convex?* Global optimum in milliseconds; latency ~ 1/r^2 is convex.
- *Deviations from the paper?* See README (integer relaxation, base-latency floor, SLO penalty scale, Algorithm 1 line 22).
- *Weaknesses?* Synthetic carbon data, single node, simple latency model, probe-based latency.
