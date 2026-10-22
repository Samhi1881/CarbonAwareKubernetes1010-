"""Hybrid forecasting engine: WMA + ARIMA(1,1,1) + sinusoidal seasonal (paper Sect. 3.1).

Convention: `history` is a flat hourly array whose LAST element is the hour just before
the first forecast hour. forecast(...) returns `steps` (<=24) hourly values.
"""
import itertools
import math
import warnings

import numpy as np


def mape(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.mean(np.abs((y - yhat) / y)) * 100)


class HybridForecaster:
    def __init__(self, cfg):
        self.cfg = cfg

    # ---- components --------------------------------------------------------------
    def wma(self, history, steps=24):
        """Exponentially-decaying weighted mean of the same hour over the past N days."""
        h, n = np.asarray(history, float), len(history)
        out = np.empty(steps)
        for k in range(1, steps + 1):
            vals, ws = [], []
            for j in range(1, self.cfg.wma_days + 1):
                idx = n + k - 1 - 24 * j
                if 0 <= idx < n:
                    vals.append(h[idx])
                    ws.append(math.exp(-self.cfg.wma_lambda * (j - 1)))
            out[k - 1] = np.average(vals, weights=ws) if vals else h[-1]
        return out

    def arima(self, history, steps=24):
        """ARIMA(1,1,1) on the last 7 days; falls back to persistence on failure."""
        h = np.asarray(history, float)[-24 * 7:]
        try:
            from statsmodels.tsa.arima.model import ARIMA
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = ARIMA(h, order=(1, 1, 1)).fit()
                return np.asarray(fit.forecast(steps), float)
        except Exception:
            return np.full(steps, h[-1])

    def seasonal(self, start_hour_of_day, steps=24):
        c = self.cfg
        hrs = (start_hour_of_day + np.arange(steps)) % 24
        return c.seasonal_mu + c.seasonal_amp * np.sin(2 * np.pi * (hrs + c.seasonal_phase_hours) / 24)

    # ---- ensemble ----------------------------------------------------------------
    def components(self, history, start_hour_of_day, steps=24, need_arima=True):
        return {
            "wma": self.wma(history, steps),
            "arima": self.arima(history, steps) if need_arima else np.zeros(steps),
            "seasonal": self.seasonal(start_hour_of_day, steps),
        }

    def forecast(self, history, start_hour_of_day, steps=24, weights=None):
        w = weights or self.cfg.ensemble_weights
        comp = self.components(history, start_hour_of_day, steps, need_arima=w[1] > 0)
        return np.clip(w[0] * comp["wma"] + w[1] * comp["arima"] + w[2] * comp["seasonal"], 0, None)


def weight_grid(step=0.1):
    """All (wma, arima, seasonal) weights summing to 1 -> 66 configs at step 0.1 (as in the paper)."""
    n = round(1 / step)
    return [(i * step, j * step, (n - i - j) * step)
            for i, j in itertools.product(range(n + 1), repeat=2) if i + j <= n]


def evaluate_forecasters(cfg, series, train_frac=0.8):
    """Rolling day-ahead evaluation. Weights are grid-searched on the *train* part only,
    metrics are reported on the held-out last 20 % (days)."""
    fc = HybridForecaster(cfg)
    n_days = len(series) // 24
    split = int(n_days * train_frac)
    warm = cfg.wma_days + 1

    def per_day(day):
        hist = series[: day * 24]
        comp = fc.components(hist, 0, 24, need_arima=True)
        return comp, series[day * 24:(day + 1) * 24]

    train = [per_day(d) for d in range(warm, split)]
    test = [per_day(d) for d in range(split, n_days)]

    def score(days, w):
        errs = [mape(y, w[0] * c["wma"] + w[1] * c["arima"] + w[2] * c["seasonal"]) for c, y in days]
        return float(np.mean(errs))

    grid = weight_grid(0.1)
    best = min(grid, key=lambda w: score(train, w))
    rows = {}
    for name, w in [("WMA", (1, 0, 0)), ("ARIMA(1,1,1)", (0, 1, 0)), ("Seasonal", (0, 0, 1)), ("Hybrid Ens.", best)]:
        preds = np.concatenate([w[0] * c["wma"] + w[1] * c["arima"] + w[2] * c["seasonal"] for c, _ in test])
        ys = np.concatenate([y for _, y in test])
        rows[name] = {"mae": float(np.mean(np.abs(ys - preds))),
                      "rmse": float(np.sqrt(np.mean((ys - preds) ** 2))),
                      "mape": mape(ys, preds)}
    return {"best_weights": tuple(round(x, 2) for x in best), "n_configs": len(grid), "metrics": rows}
