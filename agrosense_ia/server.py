"""HTTP front of the irrigation model, on the standard library alone.

    GET  /                     -> what this service is
    GET  /health               -> {"status": "UP"}
    POST /predict/irrigation   -> the prediction (see README.md)

With AI_SERVICE_TOKEN set, predictions are only given to callers that send it as a bearer token.

Every error answers JSON with a ``message`` written for the reader, like the AgroSense backend does.
"""

import hmac
import json
import logging
import math
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .model import DEFAULT_HUMIDITY_RANGE, DEFAULT_PH_RANGE, predict_irrigation

log = logging.getLogger("agrosense_ia")

PREDICT_PATH = "/predict/irrigation"
HEALTH_PATH = "/health"
ROOT_PATH = "/"
# Paths that only answer GET (and HEAD, which monitors use).
READ_ONLY_PATHS = (ROOT_PATH, HEALTH_PATH)
# A prediction request is a handful of numbers; anything larger is not one.
MAX_BODY_BYTES = 16 * 1024
# A client that stops sending is dropped instead of holding a thread.
SOCKET_TIMEOUT_S = 10


class BadRequest(Exception):
    """The request cannot be answered; the message says what to fix."""

    def __init__(self, message: str, status: HTTPStatus = HTTPStatus.BAD_REQUEST):
        super().__init__(message)
        self.status = status


def _number(body: dict, name: str, low: float, high: float, required: bool = True) -> float | None:
    value = body.get(name)
    if value is None:
        if required:
            raise BadRequest(f"Falta el campo '{name}'.")
        return None
    # bool is a kind of int in Python; true and false are not measurements.
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise BadRequest(f"El campo '{name}' debe ser un número.")
    if not low <= value <= high:
        raise BadRequest(f"El campo '{name}' debe estar entre {low:g} y {high:g}.")
    return float(value)


def _range(body: dict, low_name: str, high_name: str, low: float, high: float,
           default: tuple[float, float]) -> tuple[float, float]:
    """An optional pair of limits: both or neither, with the minimum below the maximum."""
    minimum = _number(body, low_name, low, high, required=False)
    maximum = _number(body, high_name, low, high, required=False)
    if minimum is None and maximum is None:
        return default
    if minimum is None or maximum is None:
        raise BadRequest(f"Los campos '{low_name}' y '{high_name}' deben enviarse juntos.")
    if minimum >= maximum:
        raise BadRequest(f"El campo '{low_name}' debe ser menor que '{high_name}'.")
    return minimum, maximum


def answer_prediction(body: object) -> dict:
    """Validates a decoded request body and returns the body of the answer."""
    if not isinstance(body, dict):
        raise BadRequest("El cuerpo debe ser un objeto JSON.")
    crop_id = body.get("id_crop")
    if crop_id is not None and (isinstance(crop_id, bool) or not isinstance(crop_id, int) or crop_id <= 0):
        raise BadRequest("El campo 'id_crop' debe ser un entero positivo.")
    prediction = predict_irrigation(
        soil_moisture=_number(body, "soil_moisture", 0, 100),
        air_temperature=_number(body, "air_temperature", -50, 80, required=False),
        ph=_number(body, "ph", 0, 14, required=False),
        humidity_range=_range(body, "humidity_min", "humidity_max", 0, 100, DEFAULT_HUMIDITY_RANGE),
        ph_range=_range(body, "ph_min", "ph_max", 0, 14, DEFAULT_PH_RANGE),
    )
    return {
        "id_crop": crop_id,
        "irrigate": prediction.irrigate,
        "duration_minutes": prediction.duration_minutes,
        "recommendation": prediction.recommendation,
        "confidence": prediction.confidence,
        "model_used": prediction.model_used,
        "factors": prediction.factors,
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "AgroSenseIA"
    sys_version = ""
    protocol_version = "HTTP/1.1"
    timeout = SOCKET_TIMEOUT_S

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == HEALTH_PATH:
            self._send(HTTPStatus.OK, {"status": "UP"})
        elif path == ROOT_PATH:
            self._send(HTTPStatus.OK, {"service": "AgroSense IA", "status": "UP", "health": HEALTH_PATH,
                                       "predict": f"POST {PREDICT_PATH}"})
        elif path == PREDICT_PATH:
            self._send(HTTPStatus.METHOD_NOT_ALLOWED, {"message": "Usa POST en esta ruta."}, allow="POST")
        else:
            self._send(HTTPStatus.NOT_FOUND, {"message": "No se encontró el recurso solicitado."})

    def do_HEAD(self) -> None:
        # The same answer as GET without its body, which is all an uptime monitor wants.
        self.head_only = True
        try:
            self.do_GET()
        finally:
            # One handler serves every request of a kept-alive connection.
            self.head_only = False

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in READ_ONLY_PATHS:
            self._send(HTTPStatus.METHOD_NOT_ALLOWED, {"message": "Usa GET en esta ruta."}, allow="GET, HEAD")
            return
        if path != PREDICT_PATH:
            self._send(HTTPStatus.NOT_FOUND, {"message": "No se encontró el recurso solicitado."})
            return
        if not self._authorized():
            # The body is left unread, so the connection cannot be used again.
            self.close_connection = True
            self._send(HTTPStatus.UNAUTHORIZED, {"message": "Falta el token del servicio o no es válido."},
                       authenticate="Bearer")
            return
        try:
            self._send(HTTPStatus.OK, answer_prediction(self._read_json()))
        except BadRequest as error:
            self._send(error.status, {"message": str(error)})
        except Exception:
            # The details stay in the log; the caller only learns that it was not its fault.
            log.exception("Unexpected error answering a prediction")
            self._send(HTTPStatus.INTERNAL_SERVER_ERROR,
                       {"message": "Ocurrió un error inesperado. Inténtalo de nuevo."})

    def _authorized(self) -> bool:
        """Whether the caller may ask for predictions: always when no token is configured."""
        expected = self.server.token
        if not expected:
            return True
        scheme, _, sent = self.headers.get("Authorization", "").partition(" ")
        # compare_digest takes the same time whatever the guess, so the token cannot be found bit by bit.
        return scheme.lower() == "bearer" and hmac.compare_digest(sent.strip().encode("utf-8"),
                                                                  expected.encode("utf-8"))

    def _read_json(self) -> object:
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise BadRequest("Envía el cuerpo como application/json.", HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            raise BadRequest("Falta la cabecera Content-Length.", HTTPStatus.LENGTH_REQUIRED) from None
        if length < 0:
            raise BadRequest("La cabecera Content-Length no es válida.")
        if length > MAX_BODY_BYTES:
            # The body is left unread, so the connection cannot be used again.
            self.close_connection = True
            raise BadRequest("La solicitud es demasiado grande.", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"), parse_constant=_reject_constant)
        except (ValueError, UnicodeDecodeError):
            raise BadRequest("El cuerpo no es JSON válido.") from None

    def _send(self, status: HTTPStatus, body: dict, allow: str | None = None,
              authenticate: str | None = None) -> None:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if allow:
            self.send_header("Allow", allow)
        if authenticate:
            self.send_header("WWW-Authenticate", authenticate)
        self.end_headers()
        if not getattr(self, "head_only", False):
            self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        # Method, path and status only: readings and recommendations are not written to the log.
        log.info("%s %s", self.address_string(), format % args)


def _reject_constant(name: str) -> float:
    """JSON has no NaN or Infinity; Python's parser would accept them."""
    raise ValueError(name)


def create_server(host: str = "127.0.0.1", port: int = 0, token: str = "") -> ThreadingHTTPServer:
    """A server ready to ``serve_forever``; port 0 lets the system pick a free one (used by the tests).

    With a ``token``, predictions are only given to callers that send it as ``Authorization: Bearer``.
    """
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    server.token = token.strip()
    return server


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper(),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    server = create_server(host, port, os.environ.get("AI_SERVICE_TOKEN", ""))
    log.info("AgroSense IA listening on %s:%s (%s)", host, port,
             "predictions need the service token" if server.token else "no service token: open to any caller")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
