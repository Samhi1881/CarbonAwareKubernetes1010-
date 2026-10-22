"""Autonomous controller with integrated safety (paper Algorithm 1).

Deviations from the printed pseudo-code:
  * line 22 prints `if r* = S.r` - obviously meant `!=`; we scale only when r* differs.
  * emergency override bypasses rate-limit / oscillation / cooldown (otherwise it could
    never fire while those guards are active).
"""
import json
import time

import numpy as np

from .forecasting import HybridForecaster
from .optimizer import optimize
from .safety import SafetyLayer
from .validator import validate


class Controller:
    def __init__(self, cfg, backend, carbon, log_path=None, dry_run=False):
        self.cfg, self.backend, self.carbon = cfg, backend, carbon
        self.fc, self.safety = HybridForecaster(cfg), SafetyLayer(cfg)
        self.log_path, self.dry_run = log_path, dry_run
        self._fc_cache = (None, None)
        self.decisions = []
        self._fh = open(log_path, "a") if log_path else None

    def close(self):
        if self._fh:
            self._fh.close()

    def _hourly_forecast(self, now):
        abs_hour = int(now // 3600)
        if self._fc_cache[0] != abs_hour:
            hist = self.carbon.history(abs_hour, 24 * 14)
            f = self.fc.forecast(hist, abs_hour % 24, 24)
            self._fc_cache = (abs_hour, f)
        return abs_hour, self._fc_cache[1]

    def _horizon(self, now):
        abs_hour, hourly = self._hourly_forecast(now)
        steps = np.arange(self.cfg.horizon_steps)
        t = now + steps * self.cfg.control_interval_s
        idx = np.clip((t // 3600).astype(int) - abs_hour, 0, 23)
        return hourly[idx]

    def step(self, now) -> dict:
        t0 = time.perf_counter()
        cfg, sf = self.cfg, self.safety
        try:
            st = self.backend.get_state(now)
        except Exception as e:                       # e.g. all probes failed: hold, never act blindly
            rec = {"t": now, "reason": "state_unavailable", "error": str(e), "violations": []}
            self.decisions.append(rec)
            if self._fh:
                self._fh.write(json.dumps(rec) + "\n")
                self._fh.flush()
            return rec
        if not st.carbon:
            st.carbon = float(self.carbon.at(now))   # live backends don't know the grid intensity
        tf = time.perf_counter()
        cf = self._horizon(now)
        forecast_ms = (time.perf_counter() - tf) * 1e3
        opt = optimize(cfg, st, cf)

        reason, bypass = "", False
        if sf.emergency(st) and st.r < cfg.r_max:
            target, reason, bypass = min(st.r + cfg.delta_r_max, cfg.r_max), "emergency_override", True
        else:
            target = sf.clamp(opt.replicas, st.r)
        checks = validate(cfg, st, target, cf)

        applied = st.r
        if target != st.r:
            if bypass:
                pass
            elif sf.rate_limited(now):
                reason = "rate_limit"
            elif sf.oscillating():
                reason = "oscillation_detected"
            elif sf.in_cooldown(now):
                reason = "cooldown"
            if not reason or bypass:
                ok = True if self.dry_run else self.backend.scale(target)
                if ok:
                    sf.record(now, st.r, target)
                    applied = target
                    reason = reason or "scaled"
                else:
                    reason = "scale_failed"
        else:
            reason = "no_change"

        rec = {
            "t": now, "replicas": st.r, "p95_ms": round(st.p95_ms, 2), "cpu": round(st.cpu_util, 3),
            "carbon_now": round(st.carbon, 1), "carbon_fc_mean": round(float(cf.mean()), 1),
            "r_star": opt.replicas, "r_relaxed": round(opt.relaxed, 3), "target": target,
            "applied": applied, "reason": reason, "status": opt.status, "fallback": opt.fallback,
            "solve_ms": round(opt.solve_ms, 2), "forecast_ms": round(forecast_ms, 2),
            "loop_ms": round((time.perf_counter() - t0) * 1e3, 2),
            "violations": [k for k, v in checks.items() if not v],
        }
        self.decisions.append(rec)
        if self._fh:
            self._fh.write(json.dumps(rec) + "\n")
            self._fh.flush()
        return rec
