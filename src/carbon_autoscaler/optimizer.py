"""Multi-objective convex optimiser (paper Sect. 3.2, Eq. 4-7) using CVXPY.

The paper formulates a mixed-integer program. ECOS has no integer support in current
CVXPY, so we solve the convex relaxation and round to the best *feasible* neighbouring
integer (floor / ceil) - exact for a 1-D problem with r in [1, 10].
"""
import math
import time
from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np

from .models import State, predict_latency
from .validator import validate


@dataclass
class OptResult:
    replicas: int
    relaxed: float
    status: str
    solve_ms: float
    objective: float
    fallback: bool = False
    solver: str = ""


def objective_value(cfg, st: State, cf, r) -> float:
    """Numpy evaluation of Eq. 4 (normalised terms) for an integer candidate."""
    lat = predict_latency(cfg, st, r)
    f_carbon = r * float(np.mean(cf)) / (cfg.r_max * cfg.c_ref)
    f_cost = r / cfg.r_max
    f_slo = (max(0.0, lat - cfg.l_threshold_ms) / cfg.l_threshold_ms) ** 2
    return cfg.alpha * f_carbon + cfg.beta * f_cost + cfg.gamma * cfg.slo_penalty_scale * f_slo


def _fallback(cfg, st: State, status, t0, why="") -> OptResult:
    """Conservative default: hold, or scale up if the SLO is currently violated."""
    r = min(st.r + cfg.delta_r_max, cfg.r_max) if st.p95_ms > cfg.l_threshold_ms else st.r
    return OptResult(int(r), float(r), status + why, (time.perf_counter() - t0) * 1e3,
                     float("nan"), fallback=True)


def optimize(cfg, st: State, cf) -> OptResult:
    t0 = time.perf_counter()
    cf = np.asarray(cf, float)
    checks = validate(cfg, st, st.r, cf)
    if not (checks["forecast_valid"] and checks["state_valid"]):
        return _fallback(cfg, st, "invalid_input", t0)

    P, dt_h, L = cfg.p_pod_kw, cfg.control_interval_s / 3600, cfg.l_threshold_ms
    base = min(cfg.l_base_ms, st.p95_ms)
    k = (st.p95_ms - base) * st.r ** 2

    r = cp.Variable()
    lat = base + k * cp.power(r, -2)                       # convex in r (Eq. 7)
    f_carbon = r * float(np.mean(cf)) / (cfg.r_max * cfg.c_ref)   # = sum_t r*P*C(t)*dt, normalised
    f_cost = r / cfg.r_max                                  # = sum_t r*P*dt*price, normalised
    f_slo = cp.square(cp.pos(lat - L) / L)                  # Eq. 5, convex
    obj = cfg.alpha * f_carbon + cfg.beta * f_cost + cfg.gamma * cfg.slo_penalty_scale * f_slo

    cons = [
        r >= cfg.r_min, r <= cfg.r_max,
        r * P * dt_h <= cfg.e_budget_kwh,
        r * P * cf[0] <= cfg.c_budget_g_per_h,
        r * P * cfg.price_per_kwh <= cfg.cost_budget_per_h,
        r * P <= cfg.p_max_kw,
        lat <= L * (1 + cfg.tol),
        st.cpu_util * st.r <= cfg.rho_max * r,
        r - st.r <= cfg.delta_r_max, st.r - r <= cfg.delta_r_max,
        r >= math.ceil(st.r / 2),
        r * P * dt_h * float(cf.sum()) <= cfg.horizon_carbon_budget_g,
    ]
    if st.p95_ms > L:
        cons.append(r >= st.r)
    if st.p95_ms > cfg.emergency_multiplier * L:
        cons.append(r >= min(st.r + 1, cfg.r_max))

    prob = cp.Problem(cp.Minimize(obj), cons)
    used = ""
    for name in cfg.solver_order:
        if name not in cp.installed_solvers():
            continue
        try:
            prob.solve(solver=name)
            used = name
            if prob.status in ("optimal", "optimal_inaccurate"):
                break
        except cp.error.SolverError:
            continue
    if prob.status not in ("optimal", "optimal_inaccurate") or r.value is None:
        return _fallback(cfg, st, prob.status or "no_solver", t0)

    rv = float(r.value)
    cands = {int(min(max(math.floor(rv + 1e-6), cfg.r_min), cfg.r_max)),
             int(min(max(math.ceil(rv - 1e-6), cfg.r_min), cfg.r_max))}
    feasible = [c for c in cands if all(validate(cfg, st, c, cf).values())]
    if not feasible:
        return _fallback(cfg, st, "rounding_infeasible", t0)
    best = min(feasible, key=lambda c: objective_value(cfg, st, cf, c))
    return OptResult(best, rv, "OPTIMAL", (time.perf_counter() - t0) * 1e3,
                     objective_value(cfg, st, cf, best), False, used)
