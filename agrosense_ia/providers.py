"""The language models the assistant can use, each reached over its own HTTP API.

Every failure becomes an ``AssistantError`` with a status and a message the end user can read, in the same
terms the AgroSense backend uses: 429 out of allowance, 502 broken, 504 too slow, 422 refused. A provider's
own error text may quote the request or the key, so only its status code is logged.
"""

import base64
import json
import logging
from dataclasses import dataclass

import httpx

log = logging.getLogger("agrosense_ia.providers")

USER = "user"
ASSISTANT = "assistant"

# The version of the Anthropic API the request and answer shapes below follow.
ANTHROPIC_VERSION = "2023-06-01"

# The file name tells a transcription API how to read the recording; the type alone is not enough for it.
EXTENSIONS = {"audio/webm": "webm", "audio/ogg": "ogg", "audio/wav": "wav", "audio/x-wav": "wav",
              "audio/mp4": "mp4", "audio/mpeg": "mp3", "audio/mp3": "mp3", "audio/flac": "flac", "audio/aac": "aac"}

REFUSED = "No puedo responder a esa consulta. Prueba a formularla de otra manera."
EMPTY = "El asistente no generó una respuesta. Inténtalo de nuevo."
CANNOT_LISTEN = "Ahora no puedo escuchar mensajes de voz. Escribe tu pregunta y te respondo."


class AssistantError(Exception):
    """The question could not be answered; ``status`` says why and the message is for the end user."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class Said:
    """One side's part of a conversation; ``role`` is ``user`` or ``assistant``."""
    role: str
    text: str


def _send(call, model: str) -> dict:
    """Runs one HTTP call and turns each way it can fail into an ``AssistantError``."""
    try:
        response = call()
    except httpx.TimeoutException:
        log.warning("The AI provider did not answer in time (model %s)", model)
        raise AssistantError(504, "El asistente tardó demasiado en responder. Inténtalo de nuevo.") from None
    except httpx.HTTPError as error:
        # Connection failures; the key travels in a header and is not part of the error.
        log.warning("Could not reach the AI provider: %s", type(error).__name__)
        raise AssistantError(504, "El asistente tardó demasiado en responder. Inténtalo de nuevo.") from None
    if response.status_code == 429:
        log.warning("The AI provider answered 429 to model %s", model)
        raise AssistantError(429, "El asistente está recibiendo muchas consultas. Inténtalo de nuevo en un minuto.")
    if response.status_code >= 300:
        log.warning("The AI provider answered %s to model %s", response.status_code, model)
        raise AssistantError(502, "El asistente no está disponible en este momento. Inténtalo más tarde.")
    try:
        answer = response.json()
    except ValueError:
        answer = None
    if not isinstance(answer, dict):
        raise AssistantError(502, "El asistente no devolvió ninguna respuesta. Inténtalo de nuevo.")
    return answer


class ChatModel:
    """A model that answers written questions; ``listens`` says whether it can also transcribe recordings."""

    name = "model"

    def chat(self, instructions: str, conversation: list[Said]) -> str:
        raise NotImplementedError

    def listens(self) -> bool:
        return False

    def transcribe(self, audio: bytes, mime_type: str) -> str:
        raise AssistantError(503, CANNOT_LISTEN)


class OpenAiCompatible(ChatModel):
    """Any provider that speaks the API OpenAI defined (Groq, OpenRouter, Mistral, OpenAI...)."""

    name = "openai-compatible"

    def __init__(self, api_key: str, base_url: str, models: tuple[str, ...], transcription_model: str,
                 timeout_s: float, max_output_tokens: int):
        if not models:
            raise ValueError("AI_CHAT_MODEL must name at least one model")
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.models = models
        self.transcription_model = transcription_model
        self.timeout_s = timeout_s
        self.max_output_tokens = max_output_tokens

    def listens(self) -> bool:
        return bool(self.transcription_model)

    def chat(self, instructions: str, conversation: list[Said]) -> str:
        messages = [{"role": "system", "content": instructions}]
        messages += [{"role": turn.role, "content": turn.text} for turn in conversation]
        last_failure = None
        for model in self.models:
            try:
                return self._complete(model, messages)
            except AssistantError as failure:
                # Out of allowance or broken for this model: the next one may still answer. A refusal is
                # about the question and a provider that does not respond will not for the next model either.
                if failure.status not in (429, 502):
                    raise
                last_failure = failure
        raise last_failure

    def _complete(self, model: str, messages: list[dict]) -> str:
        body = {"model": model, "messages": messages, "temperature": 0.4, "max_tokens": self.max_output_tokens}
        answer = _send(lambda: httpx.post(f"{self.base_url}/chat/completions", json=body, headers=self.headers,
                                          timeout=self.timeout_s), model)
        choices = answer.get("choices") or [{}]
        choice = choices[0] if isinstance(choices[0], dict) else {}
        reply = ((choice.get("message") or {}).get("content") or "").strip()
        if reply:
            return reply
        if choice.get("finish_reason") == "content_filter":
            raise AssistantError(422, REFUSED)
        raise AssistantError(502, EMPTY)

    def transcribe(self, audio: bytes, mime_type: str) -> str:
        if not self.listens():
            return super().transcribe(audio, mime_type)
        files = {"file": (f"voice.{EXTENSIONS.get(mime_type, 'webm')}", audio, mime_type)}
        data = {"model": self.transcription_model, "language": "es", "response_format": "json"}
        answer = _send(lambda: httpx.post(f"{self.base_url}/audio/transcriptions", files=files, data=data,
                                          headers=self.headers, timeout=self.timeout_s), self.transcription_model)
        heard = (answer.get("text") or "").strip()
        if not heard:
            raise AssistantError(422, "No se entendió el mensaje de voz. Inténtalo de nuevo o escribe tu pregunta.")
        return heard


class Anthropic(ChatModel):
    """Claude, through the Messages API. It does not transcribe recordings."""

    name = "anthropic"

    def __init__(self, api_key: str, base_url: str, model: str, effort: str, timeout_s: float,
                 max_output_tokens: int):
        if not model:
            raise ValueError("ANTHROPIC_MODEL must name a model")
        self.url = f"{base_url.rstrip('/')}/v1/messages"
        # In headers, never in the address, so the key does not reach a log of requested URLs.
        self.headers = {"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION}
        self.model = model
        self.effort = effort
        self.timeout_s = timeout_s
        self.max_output_tokens = max_output_tokens

    def chat(self, instructions: str, conversation: list[Said]) -> str:
        body = {"model": self.model, "max_tokens": self.max_output_tokens, "system": instructions,
                "messages": [{"role": turn.role, "content": turn.text} for turn in conversation]}
        # Current models refuse sampling settings such as temperature; how much they think is the lever.
        if self.effort:
            body["output_config"] = {"effort": self.effort}
        answer = _send(lambda: httpx.post(self.url, json=body, headers=self.headers, timeout=self.timeout_s),
                       self.model)
        if answer.get("stop_reason") == "refusal":
            raise AssistantError(422, REFUSED)
        # Only the text blocks are the answer; thinking blocks, when the model sends them, are not shown.
        reply = "".join(block.get("text", "") for block in answer.get("content") or []
                        if isinstance(block, dict) and block.get("type") == "text").strip()
        if not reply:
            raise AssistantError(502, EMPTY)
        return reply


class Gemini(ChatModel):
    """Google Gemini. It also listens: a recording is transcribed and answered in the same call."""

    name = "gemini"

    VOICE_INSTRUCTION = "Transcribe el audio en español y responde a lo que pregunta."

    def __init__(self, api_key: str, base_url: str, model: str, timeout_s: float, max_output_tokens: int):
        self.url = f"{base_url.rstrip('/')}/v1beta/models/{model}:generateContent"
        self.headers = {"x-goog-api-key": api_key}
        self.model = model
        self.timeout_s = timeout_s
        self.max_output_tokens = max_output_tokens

    def listens(self) -> bool:
        return True

    def chat(self, instructions: str, conversation: list[Said]) -> str:
        # Gemini calls the assistant's side "model".
        contents = [{"role": "user" if turn.role == USER else "model", "parts": [{"text": turn.text}]}
                    for turn in conversation]
        return self._text(self._generate(instructions, contents, voice=False))

    def listen_and_answer(self, instructions: str, audio: bytes, mime_type: str) -> tuple[str, str | None]:
        """The answer to a recording and what was understood from it, in a single call."""
        audio_part = {"inline_data": {"mime_type": mime_type, "data": base64.b64encode(audio).decode("ascii")}}
        contents = [{"role": "user", "parts": [audio_part, {"text": self.VOICE_INSTRUCTION}]}]
        text = self._text(self._generate(instructions, contents, voice=True))
        try:
            parsed = json.loads(text)
            reply = str(parsed.get("reply", "")).strip()
            if reply:
                heard = str(parsed.get("transcript", "")).strip()
                return reply, heard or None
        except (ValueError, AttributeError):
            log.warning("Gemini did not return the requested JSON for a voice message")
        # The model ignored the requested shape: its text is still an answer worth showing.
        return text, None

    def _generate(self, instructions: str, contents: list[dict], voice: bool) -> dict:
        config = {"temperature": 0.4, "maxOutputTokens": self.max_output_tokens}
        if voice:
            config["responseMimeType"] = "application/json"
            config["responseSchema"] = {"type": "OBJECT",
                                        "properties": {"transcript": {"type": "STRING"}, "reply": {"type": "STRING"}},
                                        "required": ["transcript", "reply"]}
        body = {"system_instruction": {"parts": [{"text": instructions}]}, "contents": contents,
                "generationConfig": config}
        return _send(lambda: httpx.post(self.url, json=body, headers=self.headers, timeout=self.timeout_s),
                     self.model)

    @staticmethod
    def _text(answer: dict) -> str:
        candidates = answer.get("candidates") or [{}]
        parts = ((candidates[0] or {}).get("content") or {}).get("parts") or []
        reply = "".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip()
        if reply:
            return reply
        if (answer.get("promptFeedback") or {}).get("blockReason"):
            raise AssistantError(422, REFUSED)
        raise AssistantError(502, EMPTY)


def chosen(settings) -> ChatModel | None:
    """The provider that answers, by priority: OpenAI-compatible, Claude, Gemini; ``None`` without any key."""
    timeout_s = settings.timeout_ms / 1000
    if settings.chat_api_key:
        return OpenAiCompatible(settings.chat_api_key, settings.chat_base_url, settings.chat_models,
                                settings.transcription_model, timeout_s, settings.max_output_tokens)
    if settings.anthropic_api_key:
        return Anthropic(settings.anthropic_api_key, settings.anthropic_base_url, settings.anthropic_model,
                         settings.anthropic_effort, timeout_s, settings.max_output_tokens)
    if settings.gemini_api_key:
        return Gemini(settings.gemini_api_key, settings.gemini_base_url, settings.gemini_model, timeout_s,
                      settings.max_output_tokens)
    return None
