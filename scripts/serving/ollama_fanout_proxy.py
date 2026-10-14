"""Round-robin reverse proxy over several single-slot Ollama instances.

Ollama 0.33.2 pins every qwen3-family model to one llama-server slot
("model architecture does not currently support parallel requests"), so a
single daemon serialises requests no matter what OLLAMA_NUM_PARALLEL says.
Running N independent daemons and fanning out across them restores
concurrency without overriding the per-architecture restriction.

The Ollama HTTP API is stateless per request, so round-robin is safe.
"""

from __future__ import annotations

import itertools
import os
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BACKENDS = [b.strip() for b in os.environ.get("FANOUT_BACKENDS", "").split(",") if b.strip()]
LISTEN_PORT = int(os.environ.get("FANOUT_PORT", "11500"))
TIMEOUT = float(os.environ.get("FANOUT_TIMEOUT", "1800"))

if not BACKENDS:
    sys.exit("FANOUT_BACKENDS is required, e.g. http://127.0.0.1:11435,http://127.0.0.1:11436")

_cycle = itertools.cycle(BACKENDS)
_lock = threading.Lock()

# Hop-by-hop headers must not be forwarded.
_DROP = {"connection", "keep-alive", "transfer-encoding", "upgrade", "host", "content-length"}


def next_backend() -> str:
    with _lock:
        return next(_cycle)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # keep the proxy quiet
        pass

    def _forward(self, method: str) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        headers = {k: v for k, v in self.headers.items() if k.lower() not in _DROP}
        backend = next_backend()
        request = urllib.request.Request(
            backend + self.path, data=body, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                payload = response.read()
                self.send_response(response.status)
                for key, value in response.headers.items():
                    if key.lower() not in _DROP:
                        self.send_header(key, value)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
        except urllib.error.HTTPError as error:
            payload = error.read()
            self.send_response(error.code)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except Exception as error:  # backend down or timed out
            payload = f"proxy error via {backend}: {error}".encode()
            self.send_response(502)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    def do_GET(self):
        self._forward("GET")

    def do_POST(self):
        self._forward("POST")

    def do_HEAD(self):
        self._forward("HEAD")

    def do_DELETE(self):
        self._forward("DELETE")


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", LISTEN_PORT), Handler)
    server.daemon_threads = True
    print(f"fanout proxy on 127.0.0.1:{LISTEN_PORT} -> {', '.join(BACKENDS)}", flush=True)
    server.serve_forever()
