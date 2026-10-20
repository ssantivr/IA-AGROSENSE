import hmac
import logging

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from . import providers
from .assistant import Assistant, load_prompt
from .config import Settings
from .validation import BadRequest, answer_analysis, answer_prediction, chat_request, decode_json, farm_context

log = logging.getLogger("agrosense_ia")

PREDICT_PATH = "/predict/irrigation"
ANALYZE_PATH = "/analizar"
HEALTH_PATH = "/health"
CHAT_PATH = "/chat"
VOICE_PATH = "/voice"
MAX_BODY_BYTES = 16 * 1024
MAX_CHAT_BYTES = 128 * 1024
MAX_AUDIO_BYTES = 3 * 1024 * 1024
MAX_MESSAGE_LENGTH = 1000

AUDIO_TYPES = {"audio/webm", "audio/ogg", "audio/wav", "audio/x-wav", "audio/mp4", "audio/mpeg", "audio/mp3",
               "audio/aac", "audio/flac"}

SOURCE_MODEL = "model"


class Json(JSONResponse):
    media_type = "application/json; charset=utf-8"


def _message(status: int, message: str, **headers: str) -> Json:
    return Json({"message": message}, status_code=status, headers=headers or None)


def _same(sent: str, expected: str) -> bool:
    return bool(expected) and hmac.compare_digest(sent.strip().encode("utf-8"), expected.encode("utf-8"))


def create_app(settings: Settings, model: providers.ChatModel | None = None, prompt: str | None = None) -> FastAPI:
    assistant = Assistant(model if model is not None else providers.chosen(settings),
                          prompt if prompt is not None else load_prompt())
    app = FastAPI(title="AgroSense IA", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.assistant = assistant

    def authorized(request: Request) -> bool:
        if _same(request.headers.get("X-Internal-Key", ""), settings.internal_key):
            return True
        scheme, _, sent = request.headers.get("Authorization", "").partition(" ")
        return scheme.lower() == "bearer" and _same(sent, settings.service_token)

    async def read_json(request: Request, limit: int) -> object:
        content_type = request.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise BadRequest("Envía el cuerpo como application/json.", 415)
        try:
            length = int(request.headers.get("Content-Length", ""))
        except ValueError:
            raise BadRequest("Falta la cabecera Content-Length.", 411) from None
        if length < 0:
            raise BadRequest("La cabecera Content-Length no es válida.")
        if length > limit:
            raise BadRequest("La solicitud es demasiado grande.", 413)
        return decode_json(await request.body())

    @app.middleware("http")
    async def no_store(request: Request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, error: HTTPException):
        if error.status_code == 404:
            return _message(404, "No se encontró el recurso solicitado.")
        if error.status_code == 405:
            allowed = ", ".join(sorted((error.headers or {}).get("Allow", "").replace(" ", "").split(",")))
            return _message(405, f"Usa {'GET' if 'GET' in allowed else allowed} en esta ruta.", Allow=allowed)
        return _message(error.status_code, str(error.detail))

    @app.exception_handler(BadRequest)
    async def bad_request(request: Request, error: BadRequest):
        headers = {"Connection": "close"} if error.status == 413 else {}
        return _message(error.status, str(error), **headers)

    @app.exception_handler(providers.AssistantError)
    async def assistant_error(request: Request, error: providers.AssistantError):
        return _message(error.status, str(error))

    @app.exception_handler(Exception)
    async def unexpected(request: Request, error: Exception):
        log.error("Unexpected error answering %s %s", request.method, request.url.path, exc_info=error)
        return _message(500, "Ocurrió un error inesperado. Inténtalo de nuevo.")

    @app.api_route("/", methods=["GET", "HEAD"])
    async def root():
        return Json({"service": "AgroSense IA", "status": "UP", "health": HEALTH_PATH,
                     "predict": f"POST {PREDICT_PATH}", "analyze": f"POST {ANALYZE_PATH}", "chat": f"POST {CHAT_PATH}", "voice": f"POST {VOICE_PATH}"})

    @app.api_route(HEALTH_PATH, methods=["GET", "HEAD"])
    async def health():
        return Json({"status": "UP"})

    def unauthorized() -> Json:
        return _message(401, "Falta el token del servicio o no es válido.", **{"WWW-Authenticate": "Bearer",
                                                                                "Connection": "close"})

    @app.post(PREDICT_PATH)
    async def predict(request: Request):
        if settings.guarded and not authorized(request):
            return unauthorized()
        return Json(answer_prediction(await read_json(request, MAX_BODY_BYTES)))

    @app.post(ANALYZE_PATH)
    async def analyze(request: Request):
        if settings.guarded and not authorized(request):
            return unauthorized()
        return Json(answer_analysis(await read_json(request, MAX_BODY_BYTES)))

    def refused() -> Json:
        return _message(401, "Falta la llave interna del servicio o no es válida.", Connection="close")

    @app.post(CHAT_PATH)
    async def chat(request: Request):
        if not authorized(request):
            return refused()
        question, history, farm = chat_request(await read_json(request, MAX_CHAT_BYTES))
        reply = await run_in_threadpool(assistant.chat, question, history, farm)
        return Json({"reply": reply, "source": SOURCE_MODEL})

    @app.post(VOICE_PATH)
    async def voice(request: Request):
        if not authorized(request):
            return refused()
        if int(request.headers.get("Content-Length", "0") or 0) > MAX_AUDIO_BYTES + MAX_CHAT_BYTES:
            raise BadRequest("El audio es demasiado largo. Graba un mensaje más corto.", 413)
        try:
            form = await request.form(max_files=1, max_fields=4)
        except Exception:
            raise BadRequest("Envía la pregunta como formulario multipart.") from None
        farm = farm_context(form.get("farm_context"))
        transcript = form.get("transcript")
        spoken = transcript.strip() if isinstance(transcript, str) else ""
        if spoken:
            if len(spoken) > MAX_MESSAGE_LENGTH:
                raise BadRequest("El mensaje de voz es demasiado largo.")
            reply = await run_in_threadpool(assistant.chat, spoken, [], farm)
            return Json({"reply": reply, "transcript": spoken, "source": SOURCE_MODEL})
        audio = form.get("audio")
        if audio is None or isinstance(audio, str):
            raise BadRequest("No se recibió ningún audio.")
        data = await audio.read()
        if not data:
            raise BadRequest("No se recibió ningún audio.")
        if len(data) > MAX_AUDIO_BYTES:
            raise BadRequest("El audio es demasiado largo. Graba un mensaje más corto.", 413)
        mime_type = (audio.content_type or "").split(";", 1)[0].strip().lower()
        if mime_type not in AUDIO_TYPES:
            raise BadRequest("Ese formato de audio no es compatible.", 415)
        reply, heard = await run_in_threadpool(assistant.voice, data, mime_type, farm)
        return Json({"reply": reply, "transcript": heard, "source": SOURCE_MODEL})

    return app
