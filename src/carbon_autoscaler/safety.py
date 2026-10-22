"""Four safety mechanisms (+ cooldown) from paper Sect. 3.3 / Algorithm 1."""
from collections import deque


class SafetyLayer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.actions = deque()          # (time, r_from, r_to)
        self.last_action_t = float("-inf")

    def rate_limited(self, now) -> bool:
        w = self.cfg.rate_limit_window_s
        return sum(1 for t, *_ in self.actions if t >= now - w) >= self.cfg.rate_limit_actions

    def oscillating(self) -> bool:
        deltas = [b - a for _, a, b in list(self.actions)[-self.cfg.oscillation_history:]]
        signs = [1 if d > 0 else -1 for d in deltas if d != 0]
        changes = sum(1 for x, y in zip(signs, signs[1:]) if x != y)
        return changes >= self.cfg.oscillation_changes

    def emergency(self, st) -> bool:
        return st.p95_ms > self.cfg.emergency_multiplier * self.cfg.l_threshold_ms

    def in_cooldown(self, now) -> bool:
        return now - self.last_action_t < self.cfg.cooldown_s

    def clamp(self, r_target, r_now) -> int:
        d = self.cfg.delta_r_max
        return int(min(max(r_target, r_now - d), r_now + d))

    def record(self, now, r_from, r_to):
        self.actions.append((now, r_from, r_to))
        self.last_action_t = now
        while len(self.actions) > 50:
            self.actions.popleft()
