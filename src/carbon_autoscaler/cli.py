import argparse
import json
import os
import time

import numpy as np

from .carbon import ElectricityMapsSource, SyntheticCarbonSource
from .config import Config
from .forecasting import evaluate_forecasters
from .models import State
from .optimizer import optimize


def cmd_forecast(a):
    cfg = Config()
    src = SyntheticCarbonSource(seed=a.seed)
    series = src.history(24 * (a.days + 5), a.days * 24)
    res = evaluate_forecasters(cfg, series)
    print(f"Grid-searched {res['n_configs']} weight configs on the train split -> best (WMA, ARIMA, Seasonal) = {res['best_weights']}")
    print(f"\n{'Method':<16}{'MAE':>8}{'RMSE':>8}{'MAPE %':>9}")
    for k, m in res["metrics"].items():
        print(f"{k:<16}{m['mae']:>8.1f}{m['rmse']:>8.1f}{m['mape']:>9.2f}")


def cmd_optimize(a):
    cfg = Config(l_threshold_ms=a.slo_ms)
    cf = np.full(cfg.horizon_steps, a.carbon)
    r = optimize(cfg, State(a.replicas, a.p95, a.cpu, a.carbon), cf)
    print(json.dumps(r.__dict__, indent=2))


def cmd_simulate(a):
    from .simulator import plot, print_table, run_all
    cfg = Config(l_threshold_ms=a.slo_ms)
    res = run_all(cfg, outdir=a.outdir, cycles=a.cycles, start_hour=a.start_hour, load=a.load,
                  rps=a.rps, seed=a.seed, initial_replicas=a.initial_replicas)
    print_table(res, cfg)
    print(f"\nSaved: {a.outdir}/results.json, {a.outdir}/decisions.jsonl")
    if a.plot:
        print("Plot :", plot(res, a.outdir, cfg))


def cmd_run(a):
    from .backends import KubernetesBackend
    from .controller import Controller
    cfg = Config(l_threshold_ms=a.slo_ms, control_interval_s=a.interval, cooldown_s=a.cooldown,
                 rate_limit_window_s=a.cooldown * 2, l_base_ms=a.base_ms, r_max=a.max_replicas,
                 horizon_steps=max(1, int(3600 / a.interval)))
    carbon = ElectricityMapsSource(a.zone) if a.carbon_source == "electricitymaps" else SyntheticCarbonSource()
    be = KubernetesBackend(a.deployment, a.namespace, a.probe_url, a.probe_n, a.cpu_request_m, a.selector)
    os.makedirs(os.path.dirname(a.log) or ".", exist_ok=True)
    ctl = Controller(cfg, be, carbon, log_path=a.log, dry_run=a.dry_run)
    print(f"Carbon-aware autoscaler for {a.namespace}/{a.deployment}  interval={a.interval}s  dry_run={a.dry_run}")
    try:
        while True:
            d = ctl.step(time.time())
            if d["reason"] == "state_unavailable":
                print(f"[{time.strftime('%H:%M:%S')}] HOLD - {d['error']}", flush=True)
                time.sleep(a.interval)
                continue
            print(f"[{time.strftime('%H:%M:%S')}] r={d['replicas']} p95={d['p95_ms']}ms cpu={d['cpu']*100:.0f}% "
                  f"C={d['carbon_now']} -> r*={d['r_star']} target={d['target']} [{d['reason']}] "
                  f"solve={d['solve_ms']}ms viol={len(d['violations'])}", flush=True)
            time.sleep(a.interval)
    except KeyboardInterrupt:
        ctl.close()


def main():
    p = argparse.ArgumentParser(prog="carbon-autoscaler", description="Carbon-aware Kubernetes autoscaler")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("forecast-eval", help="Compare WMA / ARIMA / Seasonal / Hybrid (paper Table 1)")
    f.add_argument("--days", type=int, default=40); f.add_argument("--seed", type=int, default=1)
    f.set_defaults(fn=cmd_forecast)

    o = sub.add_parser("optimize-once", help="Solve a single optimisation and print the result")
    o.add_argument("--replicas", type=int, default=4); o.add_argument("--p95", type=float, default=160)
    o.add_argument("--cpu", type=float, default=0.28); o.add_argument("--carbon", type=float, default=400)
    o.add_argument("--slo-ms", type=float, default=400)
    o.set_defaults(fn=cmd_optimize)

    s = sub.add_parser("simulate", help="3-scenario benchmark (paper Table 3/4) on a simulated cluster")
    s.add_argument("--cycles", type=int, default=156); s.add_argument("--rps", type=float, default=50)
    s.add_argument("--load", choices=["constant", "spike", "diurnal"], default="constant")
    s.add_argument("--start-hour", type=int, default=8); s.add_argument("--seed", type=int, default=1)
    s.add_argument("--initial-replicas", type=int, default=4); s.add_argument("--slo-ms", type=float, default=400)
    s.add_argument("--outdir", default="results"); s.add_argument("--plot", action="store_true")
    s.set_defaults(fn=cmd_simulate)

    r = sub.add_parser("run", help="Run the controller against a real Kubernetes Deployment")
    r.add_argument("--deployment", default="demo-app"); r.add_argument("--namespace", default="default")
    r.add_argument("--selector", default="app=demo-app", help="pod label selector for metrics")
    r.add_argument("--probe-url", default="http://localhost:8080/work", help="URL probed to measure P95")
    r.add_argument("--probe-n", type=int, default=20); r.add_argument("--cpu-request-m", type=float, default=200)
    r.add_argument("--interval", type=float, default=15, help="control interval (s); paper uses 60")
    r.add_argument("--cooldown", type=float, default=45, help="cooldown (s); paper uses 300")
    r.add_argument("--slo-ms", type=float, default=400); r.add_argument("--base-ms", type=float, default=30)
    r.add_argument("--max-replicas", type=int, default=10)
    r.add_argument("--carbon-source", choices=["synthetic", "electricitymaps"], default="synthetic")
    r.add_argument("--zone", default="IN-SO"); r.add_argument("--log", default="results/live_decisions.jsonl")
    r.add_argument("--dry-run", action="store_true")
    r.set_defaults(fn=cmd_run)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
