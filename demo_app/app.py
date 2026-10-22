"""Tiny CPU-bound HTTP service used as the autoscaling target (GET /work, GET /healthz)."""
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

WORK_MS = float(os.environ.get("WORK_MS", "25"))


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/work"):
            end = time.perf_counter() + WORK_MS / 1000
            while time.perf_counter() < end:      # burn CPU so latency rises with load
                sum(i * i for i in range(200))
        body = b"ok\n"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


ThreadingHTTPServer(("0.0.0.0", 8000), H).serve_forever()
