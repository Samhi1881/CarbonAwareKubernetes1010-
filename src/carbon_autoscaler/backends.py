"""Live Kubernetes backend (works with k3d / kind / any cluster)."""
import time

import numpy as np
import requests


def _cpu_to_milli(s: str) -> float:
    if s.endswith("n"):
        return float(s[:-1]) / 1e6
    if s.endswith("u"):
        return float(s[:-1]) / 1e3
    if s.endswith("m"):
        return float(s[:-1])
    return float(s) * 1000


class KubernetesBackend:
    def __init__(self, deployment, namespace, probe_url, probe_n=20, cpu_request_m=200, selector="app=demo-app"):
        from kubernetes import client, config
        try:
            config.load_kube_config()
        except Exception:
            config.load_incluster_config()
        self.apps, self.core, self.custom = client.AppsV1Api(), client.CoreV1Api(), client.CustomObjectsApi()
        self.dep, self.ns, self.url, self.n = deployment, namespace, probe_url, probe_n
        self.cpu_req, self.selector = cpu_request_m, selector

    def _replicas(self) -> int:
        return int(self.apps.read_namespaced_deployment(self.dep, self.ns).spec.replicas or 0)

    def _p95(self) -> float:
        lat, failed = [], 0
        for _ in range(self.n):
            t = time.perf_counter()
            try:
                requests.get(self.url, timeout=5).raise_for_status()
                lat.append((time.perf_counter() - t) * 1e3)
            except Exception:
                failed += 1
                lat.append(5000.0)          # a failed probe counts as a very slow request
        if failed == self.n:                # nothing answered: measurement is unusable, don't act on it
            raise RuntimeError(f"all {self.n} probes to {self.url} failed - is 'kubectl port-forward' running?")
        return float(np.percentile(lat, 95))

    def _cpu(self, replicas) -> float:
        try:
            pods = self.custom.list_namespaced_custom_object("metrics.k8s.io", "v1beta1", self.ns, "pods",
                                                             label_selector=self.selector)["items"]
            used = sum(_cpu_to_milli(c["usage"]["cpu"]) for p in pods for c in p["containers"])
            return min(used / max(replicas * self.cpu_req, 1), 1.5)
        except Exception:
            return 0.5     # metrics-server not ready yet: neutral default

    def get_state(self, now):
        from .models import State
        r = self._replicas()
        return State(r, self._p95(), self._cpu(r), 0.0)

    def scale(self, n) -> bool:
        try:
            self.apps.patch_namespaced_deployment_scale(self.dep, self.ns, {"spec": {"replicas": int(n)}})
            return True
        except Exception as e:
            print("scale failed:", e)
            return False
