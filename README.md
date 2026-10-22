# Carbon-Aware Kubernetes Autoscaling (Multi-Objective Optimization)

> Reference implementation written from the paper; it is not the code behind the paper's reported measurements (see DEMO.md).

Implementation of *"Carbon-Aware Kubernetes Autoscaling Using Multi-Objective Optimization for Sustainable Cloud Computing"* (Bhavishya, Samhitha, Beena — FaceIT 2026).

```
Grid carbon + history ─► Hybrid forecaster (WMA + ARIMA + Seasonal)
                              ─► CVXPY optimizer (carbon + cost + SLO, 17 checks)
                                    ─► Controller + safety layer ─► Kubernetes scale API
```

| Module | Paper section |
|---|---|
| `forecasting.py` | 3.1 – ensemble, 66-config weight grid search, Table 1 |
| `optimizer.py`, `validator.py` | 3.2 – Eq. 4-7, 17 constraints |
| `safety.py`, `controller.py` | 3.3 – Algorithm 1 (rate limit, oscillation, emergency, ±3 clamp, cooldown) |
| `simulator.py` | 4.3 / 4.4 – three-scenario benchmark (Tables 2-4) |
| `backends.py` | live k3d / Kubernetes mode |

## Setup
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q
```

## Demo A — no Kubernetes needed (2 minutes)
```bash
carbon-autoscaler forecast-eval                                  # Table 1: WMA / ARIMA / Seasonal / Hybrid MAPE
carbon-autoscaler optimize-once --replicas 4 --p95 160 --cpu 0.28 --carbon 400   # one optimizer solve
carbon-autoscaler simulate --plot                                # Tables 3 & 4: baseline vs over-prov vs carbon-aware
carbon-autoscaler simulate --load spike --initial-replicas 2 --outdir results_spike --plot   # safety layer under a 3x burst
cat results/decisions.jsonl | head -3                            # audit log of every decision
```
Outputs land in `results/` (`results.json`, `decisions.jsonl`, `simulation.png`).

See [DEMO.md](DEMO.md) for the interview script and troubleshooting. Windows: `powershell -File scripts\demo_k3d.ps1`.

## Demo B — live on k3d (needs Docker, k3d, kubectl)
```bash
bash scripts/demo_k3d.sh                       # creates cluster, builds + imports demo app, deploys 4 replicas
kubectl port-forward svc/demo-app 8080:80      # terminal 1
python scripts/loadgen.py --rps 15 --duration 600          # terminal 2
carbon-autoscaler run --interval 15 --cooldown 45          # terminal 3  (add --dry-run to only log)
watch kubectl get deploy demo-app              # terminal 4
```
Raise load (`--rps 40`) mid-run to watch it scale up; stop the load to watch it scale back down.
Cleanup: `k3d cluster delete carbon-demo`.

Real grid data: `export ELECTRICITYMAPS_TOKEN=...` then `carbon-autoscaler run --carbon-source electricitymaps --zone IN-SO`.

## Push to GitHub
```bash
git init -b main && git add -A && git commit -m "Initial commit"
git remote add origin git@github.com:<you>/carbon-aware-k8s-autoscaler.git
git push -u origin main
# or: bash scripts/push_to_github.sh git@github.com:<you>/carbon-aware-k8s-autoscaler.git
```

## Deviations from the paper (please read before citing numbers)
1. **Integer program → convex relaxation + rounding.** ECOS has no integer support; with one variable in [1,10] we round to the best feasible floor/ceil.
2. **Latency model has a base-latency floor** (`L_base + (L-L_base)·(r_cur/r_new)²`); the paper scales the whole latency, which blocks sensible scale-downs.
3. **`slo_penalty_scale=200`** on f_SLO; with the paper's weights alone, cost savings outweigh the SLO penalty and P95 would sit at the +20 % tolerance edge.
4. **Seasonal phase** is expressed in hours (default 5 h → trough at 13:00); the paper's `φ=-π/2` is added to an hour value, which is dimensionally inconsistent.
5. **Algorithm 1 line 22** prints `if r* = S.r`; implemented as `!=`. Emergency override bypasses cooldown/rate limits.
6. **ARIMA is skipped when its ensemble weight is 0** (paper's weights = 0.3/0.0/0.7), saving ~80 ms per refresh.
7. Latency is measured by HTTP probes in live mode (not Prometheus); CPU comes from metrics-server.
8. The simulator (not the paper's k3d numbers) produces the benchmark tables — they follow the same structure but will not match digit-for-digit. Forecast data is synthetic; the paper itself lists real ISO data as future work.
9. The "17 constraints" are 17 named checks (`validator.py`); several (budgets) are deliberately non-binding by default.
