import logging
import os
import socket
import threading

import uvicorn

from . import ai_factory, providers
from .app import MAX_BODY_BYTES, create_app  # noqa: F401
from .config import Settings, from_environment

log = logging.getLogger("agrosense_ia")


class ServiceServer:
    def __init__(self, app, host: str, port: int):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((host, port))
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
    settings = settings or Settings(service_token=token.strip())
    return ServiceServer(create_app(settings, model), host, port)


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper(),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings = from_environment()
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    model = ai_factory.chosen(settings)
    log.info("AgroSense AI service listening on %s:%s; predictions %s; assistant %s, answered by %s", host, port,
             "need a service credential" if settings.guarded else "open to any caller",
             "needs X-Internal-Key" if settings.guarded else "closed until INTERNAL_API_KEY is set",
             model.name if model else "no provider (it answers 503)")
    uvicorn.run(create_app(settings, model), host=host, port=port, log_level="info", server_header=False,
                proxy_headers=False)
