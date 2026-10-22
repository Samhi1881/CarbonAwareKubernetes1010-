"""The 17 constraint checks evaluated for every control decision."""
import math

import numpy as np

from .models import State, predict_latency

CONSTRAINT_NAMES = [
    "replica_min", "replica_max", "integrality", "energy_budget", "carbon_budget",
    "cost_budget", "power_cap", "slo_latency", "utilisation_cap", "step_up",
    "step_down", "no_downscale_when_violating", "emergency_scale_up", "forecast_valid",
    "state_valid", "disruption_budget", "horizon_carbon_budget",
]
assert len(CONSTRAINT_NAMES) == 17


def validate(cfg, st: State, r, cf) -> dict:
    cf = np.asarray(cf, float)
    P, dt_h = cfg.p_pod_kw, cfg.control_interval_s / 3600
    L_cur = st.p95_ms
    ok_state = all(map(math.isfinite, [L_cur, st.cpu_util, st.carbon])) and L_cur >= 0 \
        and 0 <= st.cpu_util <= 1.5 and st.r >= 0
    ok_fc = bool(np.all(np.isfinite(cf)) and np.all(cf >= 0) and np.all(cf <= 1500) and len(cf) > 0)
    # the model-based checks are only meaningful on valid input
    lat = predict_latency(cfg, st, max(r, 1e-9)) if ok_state else float("inf")
    c0 = cf[0] if ok_fc else float("inf")
    eps = 1e-9
    res = {
        "replica_min": r >= cfg.r_min,
        "replica_max": r <= cfg.r_max,
        "integrality": float(r).is_integer(),
        "energy_budget": r * P * dt_h <= cfg.e_budget_kwh + eps,
        "carbon_budget": r * P * c0 <= cfg.c_budget_g_per_h + eps,
        "cost_budget": r * P * cfg.price_per_kwh <= cfg.cost_budget_per_h + eps,
        "power_cap": r * P <= cfg.p_max_kw + eps,
        "slo_latency": lat <= cfg.l_threshold_ms * (1 + cfg.tol) + eps,
        "utilisation_cap": (st.cpu_util * st.r <= cfg.rho_max * r + eps) if ok_state else False,
        "step_up": r - st.r <= cfg.delta_r_max,
        "step_down": st.r - r <= cfg.delta_r_max,
        "no_downscale_when_violating": not (L_cur > cfg.l_threshold_ms) or r >= st.r,
        "emergency_scale_up": not (L_cur > cfg.emergency_multiplier * cfg.l_threshold_ms)
                              or r > st.r or r >= cfg.r_max,
        "forecast_valid": ok_fc,
        "state_valid": ok_state,
        "disruption_budget": r >= math.ceil(st.r / 2),
        "horizon_carbon_budget": (r * P * dt_h * float(cf.sum()) <= cfg.horizon_carbon_budget_g + eps) if ok_fc else False,
    }
    return {k: bool(res[k]) for k in CONSTRAINT_NAMES}
