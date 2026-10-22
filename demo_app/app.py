import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

WORK_MS = float(os.environ.get("WORK_MS", "10"))


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/work"):
            end = time.thread_time() + WORK_MS / 1000   # real CPU time, so latency rises with load
            while time.thread_time() < end:
                pass
        body = b"ok\n"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


ThreadingHTTPServer.request_queue_size = 128
ThreadingHTTPServer(("0.0.0.0", 8000), H).serve_forever()
