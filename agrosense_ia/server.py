"""Runs the service with uvicorn: ``python -m agrosense_ia`` in production and ``create_server`` in tests."""

import logging
import os
import socket
import threading

import uvicorn

from . import providers
from .app import MAX_BODY_BYTES, create_app  # noqa: F401  (MAX_BODY_BYTES is part of what tests import)
from .config import Settings, from_environment

log = logging.getLogger("agrosense_ia")


class ServiceServer:
    """The service on a socket of its own, with the start and stop calls of the standard library's servers."""

    def __init__(self, app, host: str, port: int):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((host, port))
        # Listening before the server starts, so a client that connects right away waits instead of failing.
        self.socket.listen(128)
        self.server_address = self.socket.getsockname()
        self.server = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="off",
                                                    server_header=False))
        self._stopped = threading.Event()

    def serve_forever(self) -> None:
        try:
            self.server.run(sockets=[self.socket])
        finally:
            self._stopped.set()

    def shutdown(self) -> None:
        self.server.should_exit = True
        self._stopped.wait(10)

    def server_close(self) -> None:
        self.socket.close()


def create_server(host: str = "127.0.0.1", port: int = 0, token: str = "", settings: Settings | None = None,
                  model: providers.ChatModel | None = None) -> ServiceServer:
    """A server ready to ``serve_forever``; port 0 lets the system pick a free one (used by the tests).

    With a ``token``, predictions are only given to callers that send it as ``Authorization: Bearer``.
    """
    settings = settings or Settings(service_token=token.strip())
    return ServiceServer(create_app(settings, model), host, port)


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper(),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # httpx logs every provider URL it calls; keys travel in headers, but the providers' paths are noise.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings = from_environment()
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    model = providers.chosen(settings)
    log.info("AgroSense AI service listening on %s:%s; predictions %s; assistant %s, answered by %s", host, port,
             "need a service credential" if settings.guarded else "open to any caller",
             "needs X-Internal-Key" if settings.guarded else "closed until INTERNAL_API_KEY is set",
             model.name if model else "no provider (it answers 503)")
    # The access log has method, path and status only: no path of this service carries a secret.
    uvicorn.run(create_app(settings, model), host=host, port=port, log_level="info", server_header=False,
                proxy_headers=False)
