"""Deterministic cluster simulator so the whole pipeline can be demoed without Kubernetes."""
import json
import math
import os

import numpy as np

from .carbon import SyntheticCarbonSource
from .config import Config
from .controller import Controller
from .models import State


class SimBackend:
    """Replicas serve `mu` req/s each. p95 = base + a*rho^2 (rho = load / capacity)."""

    def __init__(self, cfg, carbon, replicas, load_fn, mu=45.0, base_ms=100.0, a_ms=650.0, seed=0):
        self.cfg, self.carbon, self.r = cfg, carbon, replicas
        self.load_fn, self.mu, self.base, self.a = load_fn, mu, base_ms, a_ms
        self.rng = np.random.default_rng(seed)
        self.t_start = None
        # accounting
        self.carbon_g = self.energy_kwh = 0.0
        self.requests = self.ok_requests = 0.0
        self.samples, self.weights = [], []
        self.rep_series, self.cpu_series, self.carbon_series, self.p95_series = [], [], [], []

    def _rho(self, now):
        return self.load_fn(now - self.t_start) / (self.r * self.mu)

    def _p95(self, rho):
        return min(self.base + self.a * rho ** 2, 5000.0)

    def get_state(self, now):
        if self.t_start is None:
            self.t_start = now
        rho = self._rho(now) * (1 + self.rng.normal(0, 0.02))
        return State(self.r, self._p95(max(rho, 0)), min(max(rho, 0), 1.0), self.carbon.at(now))

    def scale(self, n):
        self.r = int(n)
        return True

    def advance(self, now, dt):
        """Account one control interval with the *current* replica count."""
        rho = self._rho(now)
        p95 = self._p95(rho)
        reqs = self.load_fn(now - self.t_start) * dt
        succ = 1.0 if rho <= 1.0 else 1.0 / rho
        scale = max(p95 - self.base, 1e-6) / math.log(20)           # exponential tail with this p95
        s = self.base + self.rng.exponential(scale, 400)
        self.samples.append(s)
        self.weights.append(np.full(400, reqs / 400))
        c = self.carbon.at(now)
        e = self.r * self.cfg.p_pod_kw * dt / 3600
        self.energy_kwh += e
        self.carbon_g += e * c
        self.requests += reqs
        self.ok_requests += reqs * succ
        self.rep_series.append(self.r); self.cpu_series.append(min(rho, 1.0))
        self.carbon_series.append(c); self.p95_series.append(p95)

    def percentile(self, q):
        x, w = np.concatenate(self.samples), np.concatenate(self.weights)
        o = np.argsort(x)
        cw = np.cumsum(w[o]) / w.sum()
        return float(x[o][np.searchsorted(cw, q)])


def make_load(kind, rps):
    if kind == "constant":
        return lambda t: rps
    if kind == "spike":      # 3x burst between 40% and 55% of a 2.6h run
        return lambda t: rps * (3.0 if 3744 <= t < 5148 else 1.0)
    if kind == "diurnal":
        return lambda t: rps * (1 + 0.5 * math.sin(2 * math.pi * t / 7200))
    raise ValueError(kind)


def run_scenario(name, cfg, mode, cycles=156, start_hour=8, load="constant", rps=50.0,
                 replicas=2, seed=1, outdir=None):
    carbon = SyntheticCarbonSource(seed=seed)
    t0 = (24 * 20 + start_hour) * 3600.0       # 20 days of history precede the run
    be = SimBackend(cfg, carbon, replicas, make_load(load, rps), seed=seed)
    ctl = None
    if mode == "carbon-aware":
        log = os.path.join(outdir, "decisions.jsonl") if outdir else None
        if log and os.path.exists(log):
            os.remove(log)
        ctl = Controller(cfg, be, carbon, log_path=log)
    dt = cfg.control_interval_s
    for i in range(cycles):
        now = t0 + i * dt
        be.get_state(now)
        if ctl:
            ctl.step(now)
        be.advance(now, dt)
    if ctl:
        ctl.close()
    p50, p95 = be.percentile(0.5), be.percentile(0.95)
    out = {
        "name": name, "total_carbon_g": be.carbon_g, "cost_usd": be.energy_kwh * cfg.price_per_kwh,
        "avg_power_kw": be.energy_kwh / (cycles * dt / 3600), "requests": be.requests,
        "success_rate": 100 * be.ok_requests / be.requests, "p50_ms": p50, "p95_ms": p95,
        "slo_pass": p95 <= cfg.l_threshold_ms,
        "slo_headroom_pct": 100 * (cfg.l_threshold_ms - p95) / cfg.l_threshold_ms,
        "avg_replicas": float(np.mean(be.rep_series)), "avg_cpu_pct": 100 * float(np.mean(be.cpu_series)),
        "_series": {"replicas": be.rep_series, "carbon": be.carbon_series, "p95": be.p95_series},
    }
    if ctl:
        d = ctl.decisions
        solve = np.array([x["solve_ms"] for x in d])
        out["controller"] = {
            "cycles": len(d), "scaling_actions": sum(1 for x in d if x["reason"] in ("scaled", "emergency_override")),
            "held": {k: sum(1 for x in d if x["reason"] == k) for k in ("rate_limit", "oscillation_detected", "cooldown")},
            "emergency_overrides": sum(1 for x in d if x["reason"] == "emergency_override"),
            "optimal": sum(1 for x in d if x["status"] == "OPTIMAL"), "fallbacks": sum(1 for x in d if x["fallback"]),
            "constraint_checks": 17 * len(d), "constraint_violations": sum(len(x["violations"]) for x in d),
            "solve_ms": {"mean": float(solve.mean()), "median": float(np.median(solve)), "std": float(solve.std()),
                         "p95": float(np.percentile(solve, 95)), "max": float(solve.max())},
            "loop_ms_mean": float(np.mean([x["loop_ms"] for x in d])),
        }
    return out


def run_all(cfg=None, outdir=None, **kw):
    cfg = cfg or Config()
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    init = kw.pop("initial_replicas", 4)
    base_r = kw.pop("baseline_replicas", 2)
    over_r = kw.pop("overprov_replicas", 4)
    res = [run_scenario("Baseline (static)", cfg, "static", replicas=base_r, **kw),
           run_scenario("Over-provisioned (static)", cfg, "static", replicas=over_r, **kw),
           run_scenario("Carbon-aware (dynamic)", cfg, "carbon-aware", replicas=init, outdir=outdir, **kw)]
    over = res[1]["total_carbon_g"]
    for r in res:
        r["carbon_reduction_vs_overprov_pct"] = 100 * (over - r["total_carbon_g"]) / over
    if outdir:
        with open(os.path.join(outdir, "results.json"), "w") as f:
            json.dump([{k: v for k, v in r.items() if k != "_series"} for r in res], f, indent=2)
    return res


def print_table(res, cfg):
    rows = [("Total carbon (gCO2)", "total_carbon_g", "{:.2f}"), ("Reduction vs over-prov (%)", "carbon_reduction_vs_overprov_pct", "{:.1f}"),
            ("Operational cost ($)", "cost_usd", "{:.5f}"), ("Avg power (kW)", "avg_power_kw", "{:.3f}"),
            ("Total requests", "requests", "{:,.0f}"), ("Success rate (%)", "success_rate", "{:.2f}"),
            ("P50 latency (ms)", "p50_ms", "{:.1f}"), ("P95 latency (ms)", "p95_ms", "{:.1f}"),
            ("SLO headroom (%)", "slo_headroom_pct", "{:.1f}"), ("Avg replicas", "avg_replicas", "{:.2f}"),
            ("CPU utilisation (%)", "avg_cpu_pct", "{:.1f}")]
    w = 28
    print(f"\n{'Metric':<{w}}" + "".join(f"{r['name']:>28}" for r in res))
    print("-" * (w + 28 * len(res)))
    for label, key, fmt in rows:
        print(f"{label:<{w}}" + "".join(f"{fmt.format(r[key]):>28}" for r in res))
    print(f"{'SLO (P95 <= %d ms)' % cfg.l_threshold_ms:<{w}}" + "".join(f"{'PASS' if r['slo_pass'] else 'FAIL':>28}" for r in res))
    c = res[-1].get("controller")
    if c:
        s = c["solve_ms"]
        print(f"\nController: {c['cycles']} cycles, {c['scaling_actions']} scaling actions, held={c['held']}, "
              f"emergency={c['emergency_overrides']}")
        print(f"Optimizer : {c['optimal']}/{c['cycles']} OPTIMAL, fallbacks={c['fallbacks']}, "
              f"solve mean={s['mean']:.1f}ms median={s['median']:.1f}ms p95={s['p95']:.1f}ms max={s['max']:.1f}ms")
        print(f"Constraints: {c['constraint_violations']} violations / {c['constraint_checks']} checks")


def plot(res, outdir, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ca = res[-1]["_series"]
    t = np.arange(len(ca["replicas"])) * cfg.control_interval_s / 60
    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    ax[0].plot(t, ca["carbon"], color="tab:green"); ax[0].set_ylabel("gCO2/kWh"); ax[0].set_title("Grid carbon intensity")
    for r in res:
        ax[1].step(t, r["_series"]["replicas"], where="post", label=r["name"])
    ax[1].set_ylabel("replicas"); ax[1].legend(fontsize=8); ax[1].set_title("Replica count")
    for r in res:
        ax[2].plot(t, r["_series"]["p95"], label=r["name"])
    ax[2].axhline(cfg.l_threshold_ms, color="r", ls="--", label="SLO"); ax[2].set_ylabel("P95 (ms)")
    ax[2].set_xlabel("minutes"); ax[2].legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(outdir, "simulation.png")
    fig.savefig(p, dpi=130)
    return p
