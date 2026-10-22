import numpy as np
import pytest

from carbon_autoscaler.carbon import SyntheticCarbonSource
from carbon_autoscaler.config import Config
from carbon_autoscaler.forecasting import HybridForecaster, mape, weight_grid
from carbon_autoscaler.models import State
from carbon_autoscaler.optimizer import optimize
from carbon_autoscaler.safety import SafetyLayer
from carbon_autoscaler.simulator import run_all
from carbon_autoscaler.validator import CONSTRAINT_NAMES, validate

cfg = Config()
CF = np.full(60, 400.0)


def test_weight_grid_has_66_configs():
    g = weight_grid(0.1)
    assert len(g) == 66 and all(abs(sum(w) - 1) < 1e-9 for w in g)


def test_forecast_shape_and_accuracy():
    src = SyntheticCarbonSource(seed=3)
    hist = src.history(24 * 20, 24 * 14)
    f = HybridForecaster(cfg).forecast(hist, 0, 24)
    assert f.shape == (24,)
    assert mape(src.history(24 * 21, 24)[:24], f) < 15


def test_constraint_count_is_17():
    assert len(CONSTRAINT_NAMES) == 17
    assert len(validate(cfg, State(2, 300, 0.5, 400), 2, CF)) == 17


def test_scale_down_when_overprovisioned():
    r = optimize(cfg, State(4, 150, 0.28, 400), CF)
    assert r.status == "OPTIMAL" and r.replicas < 4


def test_scale_up_when_slo_violated():
    r = optimize(cfg, State(2, 900, 0.95, 400), CF)
    assert r.replicas > 2


def test_step_limit_respected():
    r = optimize(cfg, State(2, 3000, 1.0, 400), CF)
    assert r.replicas - 2 <= cfg.delta_r_max


def test_invalid_input_falls_back_safely():
    r = optimize(cfg, State(3, float("nan"), 0.5, 400), CF)
    assert r.fallback and r.replicas == 3


def test_safety_rate_limit_and_oscillation():
    s = SafetyLayer(cfg)
    for i, (a, b) in enumerate([(2, 3), (3, 2), (2, 3)]):
        s.record(100 * i, a, b)
    assert s.rate_limited(250) and not s.rate_limited(5000)
    s.record(300, 3, 2)
    assert s.oscillating()
    assert s.in_cooldown(400) and not s.in_cooldown(1000)
    assert s.clamp(10, 2) == 5 and s.clamp(1, 6) == 3


def test_simulation_end_to_end():
    res = run_all(cfg, cycles=60)
    base, over, ca = res
    assert ca["slo_pass"] and ca["success_rate"] > 99.9
    assert ca["total_carbon_g"] < over["total_carbon_g"] * 0.75
    assert ca["controller"]["constraint_violations"] == 0
    assert ca["controller"]["fallbacks"] == 0


def test_controller_holds_when_state_unavailable():
    from carbon_autoscaler.controller import Controller

    class Broken:
        scaled = False
        def get_state(self, now):
            raise RuntimeError("all probes failed")
        def scale(self, n):
            self.scaled = True
            return True

    b = Broken()
    d = Controller(cfg, b, SyntheticCarbonSource()).step(1e6)
    assert d["reason"] == "state_unavailable" and not b.scaled
