from dataclasses import dataclass


@dataclass
class State:
    """S_t = <r_t, L_t, U_t, C_t> (paper Sect. 3)."""
    r: int
    p95_ms: float
    cpu_util: float   # 0..1
    carbon: float     # gCO2/kWh


def predict_latency(cfg, st: State, r: float) -> float:
    """Queueing-inspired latency model (paper Eq. 7).

    L_pred = L_base + (L_cur - L_base) * (rho_new / rho_cur)^2, with rho ~ 1/r,
    i.e. (r_cur / r_new)^2. The paper scales the whole latency; keeping a base
    latency floor makes the model less pessimistic when scaling down.
    """
    base = min(cfg.l_base_ms, st.p95_ms)
    return base + (st.p95_ms - base) * (st.r / r) ** 2
