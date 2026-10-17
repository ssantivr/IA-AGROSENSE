"""A provider's API played by a local HTTP server: it records each request and answers what it is told to."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeProvider:
    def __init__(self):
        self.requests: list[dict] = []
        # Each answer is (status, body as dict, seconds to wait first); the last one repeats.
        self.answers: list[tuple[int, dict, float]] = [(200, {}, 0)]
        provider = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                content_type = self.headers.get("Content-Type", "")
                provider.requests.append({
                    "path": self.path,
                    "headers": {key.lower(): value for key, value in self.headers.items()},
                    "json": json.loads(raw) if content_type.startswith("application/json") else None,
                    "raw": raw,
                })
                status, body, delay = provider.answers[0] if len(provider.answers) == 1 else provider.answers.pop(0)
                if delay:
                    time.sleep(delay)
                payload = json.dumps(body).encode("utf-8")
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except OSError:
                    pass  # the client gave up waiting

            def log_message(self, format, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def answer(self, *answers):
        """Answers in this order: each is ``(status, body)`` or ``(status, body, delay_seconds)``."""
        self.answers = [answer if len(answer) == 3 else (*answer, 0) for answer in answers]

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def openai_reply(text: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}


def anthropic_reply(text: str, stop_reason: str = "end_turn") -> dict:
    return {"content": [{"type": "thinking", "thinking": "..."}, {"type": "text", "text": text}],
            "stop_reason": stop_reason}


def gemini_reply(text: str) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}
