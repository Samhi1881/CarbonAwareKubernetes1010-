"""Central configuration. Defaults follow the paper (Bhavishya et al., FaceIT 2026)."""
from dataclasses import dataclass


@dataclass
class Config:
    # --- objective weights (Eq. 4) ---
    alpha: float = 0.4   # carbon
    beta: float = 0.3    # cost
    gamma: float = 0.3   # SLO violation
    slo_penalty_scale: float = 200.0  # DEVIATION: scales f_SLO so a small latency excess is never "worth" saving a pod
    # --- power / price (Eq. 5) ---
    p_pod_kw: float = 0.05
    price_per_kwh: float = 0.12
    c_ref: float = 500.0             # gCO2/kWh used to normalise the carbon term
    # --- replica bounds / scaling (Eq. 6) ---
    r_min: int = 1
    r_max: int = 10
    delta_r_max: int = 3
    # --- SLO ---
    l_threshold_ms: float = 400.0
    tol: float = 0.2
    l_base_ms: float = 100.0         # latency floor of the queueing model (service time)
    rho_max: float = 0.8             # utilisation cap (convexity region of Eq. 7)
    # --- control loop ---
    control_interval_s: float = 60.0
    horizon_steps: int = 60          # optimisation horizon (control steps)
    # --- budgets (non-binding by default) ---
    e_budget_kwh: float = 10 * 0.05 * (60 / 3600)   # per control interval
    c_budget_g_per_h: float = 10 * 0.05 * 800
    cost_budget_per_h: float = 10 * 0.05 * 0.12
    p_max_kw: float = 0.5
    horizon_carbon_budget_g: float = 10 * 0.05 * (60 / 3600) * 60 * 800
    # --- safety layer ---
    rate_limit_actions: int = 3
    rate_limit_window_s: float = 600.0
    oscillation_changes: int = 3
    oscillation_history: int = 6
    emergency_multiplier: float = 2.0
    cooldown_s: float = 300.0
    # --- forecasting ---
    ensemble_weights: tuple = (0.3, 0.0, 0.7)   # (WMA, ARIMA, Seasonal)
    wma_lambda: float = 0.1
    wma_days: int = 7
    seasonal_mu: float = 350.0
    seasonal_amp: float = 150.0
    seasonal_phase_hours: float = 5.0           # puts the minimum at 13:00 (solar trough)
    solver_order: tuple = ("ECOS", "CLARABEL", "SCS")
