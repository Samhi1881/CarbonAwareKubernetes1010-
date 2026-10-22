#!/usr/bin/env python3
"""Open-loop-ish load generator: python scripts/loadgen.py --url http://localhost:8080/work --rps 20 --duration 300"""
import argparse
import threading
import time

import requests


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8080/work")
    p.add_argument("--rps", type=float, default=20)
    p.add_argument("--duration", type=float, default=300)
    p.add_argument("--workers", type=int, default=64)
    a = p.parse_args()
    stop, stats, lock = time.time() + a.duration, {"ok": 0, "err": 0}, threading.Lock()
    per_worker = a.rps / a.workers

    def worker():
        s = requests.Session()
        nxt = time.time()
        while time.time() < stop:
            try:
                s.get(a.url, timeout=10).raise_for_status()
                k = "ok"
            except Exception:
                k = "err"
            with lock:
                stats[k] += 1
            nxt += 1 / per_worker
            time.sleep(max(0, nxt - time.time()))

    ts = [threading.Thread(target=worker, daemon=True) for _ in range(a.workers)]
    [t.start() for t in ts]
    while time.time() < stop:
        time.sleep(10)
        print(f"sent ok={stats['ok']} err={stats['err']}", flush=True)


if __name__ == "__main__":
    main()
