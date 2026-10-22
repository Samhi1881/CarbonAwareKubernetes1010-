"""Carbon-intensity sources: synthetic (default), CSV, ElectricityMaps."""
import math
import os

import numpy as np


class SyntheticCarbonSource:
    """Deterministic diurnal grid profile (200-500 gCO2/kWh) with AR(1) noise.

    value(h) is the intensity of absolute hour h (so history is reproducible).
    """

    def __init__(self, mu=350.0, amp=150.0, phase_hours=5.0, noise=12.0, seed=1, n_hours=24 * 400):
        self.mu, self.amp, self.phase = mu, amp, phase_hours
        rng = np.random.default_rng(seed)
        eps = rng.normal(0, noise, n_hours)
        ar = np.zeros(n_hours)
        for i in range(1, n_hours):
            ar[i] = 0.7 * ar[i - 1] + eps[i]
        day_scale = rng.uniform(0.92, 1.08, n_hours // 24 + 1)
        h = np.arange(n_hours)
        base = mu + amp * day_scale[h // 24] * np.sin(2 * math.pi * (h % 24 + phase_hours) / 24)
        self._v = np.clip(base + ar, 50, 900)

    def value(self, abs_hour: int) -> float:
        return float(self._v[int(abs_hour) % len(self._v)])

    def at(self, t_s: float) -> float:
        return self.value(int(t_s // 3600))

    def history(self, end_abs_hour: int, n: int = 24 * 14) -> np.ndarray:
        """Hourly values for hours [end-n, end)."""
        idx = np.arange(end_abs_hour - n, end_abs_hour)
        return self._v[idx % len(self._v)]


class CsvCarbonSource:
    """CSV with hourly gCO2/kWh in the last column (>= 8 days)."""

    def __init__(self, path):
        data = np.genfromtxt(path, delimiter=",", usecols=-1, invalid_raise=False)
        self._v = data[~np.isnan(data)]
        if len(self._v) < 24 * 8:
            raise ValueError("need >= 8 days of hourly data")

    def value(self, abs_hour):
        return float(self._v[int(abs_hour) % len(self._v)])

    def at(self, t_s):
        return self.value(t_s // 3600)

    def history(self, end_abs_hour, n=24 * 14):
        return self._v[np.arange(end_abs_hour - n, end_abs_hour) % len(self._v)]


class ElectricityMapsSource:
    """Live intensity + history from ElectricityMaps (needs ELECTRICITYMAPS_TOKEN).

    Not exercised by the test-suite (needs network + API key).
    """

    BASE = "https://api.electricitymap.org/v3/carbon-intensity"

    def __init__(self, zone="IN-SO", token=None):
        import requests
        self.requests, self.zone = requests, zone
        self.token = token or os.environ.get("ELECTRICITYMAPS_TOKEN", "")

    def _get(self, path):
        r = self.requests.get(f"{self.BASE}/{path}", params={"zone": self.zone},
                              headers={"auth-token": self.token}, timeout=10)
        r.raise_for_status()
        return r.json()

    def at(self, t_s):
        return float(self._get("latest")["carbonIntensity"])

    def history(self, end_abs_hour, n=24 * 14):
        h = [d["carbonIntensity"] for d in self._get("history")["history"]]
        h = (h * (n // max(len(h), 1) + 1))[-n:]
        return np.asarray(h, dtype=float)
