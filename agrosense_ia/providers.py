import base64
import json
import logging
from dataclasses import dataclass

import httpx

log = logging.getLogger("agrosense_ia.providers")

USER = "user"
ASSISTANT = "assistant"

ANTHROPIC_VERSION = "2023-06-01"

EXTENSIONS = {"audio/webm": "webm", "audio/ogg": "ogg", "audio/wav": "wav", "audio/x-wav": "wav",
              "audio/mp4": "mp4", "audio/mpeg": "mp3", "audio/mp3": "mp3", "audio/flac": "flac", "audio/aac": "aac"}

REFUSED = "No puedo responder a esa consulta. Prueba a formularla de otra manera."
EMPTY = "El asistente no generó una respuesta. Inténtalo de nuevo."
CANNOT_LISTEN = "Ahora no puedo escuchar mensajes de voz. Escribe tu pregunta y te respondo."


class AssistantError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class Said:
    role: str
    text: str


def _send(call, model: str) -> dict:
    try:
        response = call()
    except httpx.TimeoutException:
        log.warning("The AI provider did not answer in time (model %s)", model)
        raise AssistantError(504, "El asistente tardó demasiado en responder. Inténtalo de nuevo.") from None
    except httpx.HTTPError as error:
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


@dataclass(frozen=True)
class Heard:
    """What a recording said. A provider that answers in the same call also brings the reply."""
    transcript: str | None
    reply: str | None = None


class Transcriber:
    def hear(self, instructions: str, audio: bytes, mime_type: str) -> Heard:
        raise NotImplementedError


class ChatModel:
    name = "model"
    transcriber: Transcriber | None = None

    def chat(self, instructions: str, conversation: list[Said]) -> str:
        raise NotImplementedError

    def listens(self) -> bool:
        return self.transcriber is not None


class OpenAiCompatAdapter(ChatModel):
    name = "openai-compatible"

    def __init__(self, api_key: str, base_url: str, models: tuple[str, ...], timeout_s: float,
                 max_output_tokens: int):
        if not models:
            raise ValueError("AI_CHAT_MODEL must name at least one model")
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.models = models
        self.timeout_s = timeout_s
        self.max_output_tokens = max_output_tokens

    def chat(self, instructions: str, conversation: list[Said]) -> str:
        messages = [{"role": "system", "content": instructions}]
        messages += [{"role": turn.role, "content": turn.text} for turn in conversation]
        last_failure = None
        for model in self.models:
            try:
                return self._complete(model, messages)
            except AssistantError as failure:
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


class OpenAiCompatTranscriber(Transcriber):
    def __init__(self, api_key: str, base_url: str, model: str, timeout_s: float):
        self.url = f"{base_url.rstrip('/')}/audio/transcriptions"
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.model = model
        self.timeout_s = timeout_s

    def hear(self, instructions: str, audio: bytes, mime_type: str) -> Heard:
        files = {"file": (f"voice.{EXTENSIONS.get(mime_type, 'webm')}", audio, mime_type)}
        data = {"model": self.model, "language": "es", "response_format": "json"}
        answer = _send(lambda: httpx.post(self.url, files=files, data=data, headers=self.headers,
                                          timeout=self.timeout_s), self.model)
        heard = (answer.get("text") or "").strip()
        if not heard:
            raise AssistantError(422, "No se entendió el mensaje de voz. Inténtalo de nuevo o escribe tu pregunta.")
        return Heard(heard)


class AnthropicAdapter(ChatModel):
    name = "anthropic"

    def __init__(self, api_key: str, base_url: str, model: str, effort: str, timeout_s: float,
                 max_output_tokens: int):
        if not model:
            raise ValueError("ANTHROPIC_MODEL must name a model")
        self.url = f"{base_url.rstrip('/')}/v1/messages"
        self.headers = {"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION}
        self.model = model
        self.effort = effort
        self.timeout_s = timeout_s
        self.max_output_tokens = max_output_tokens

    def chat(self, instructions: str, conversation: list[Said]) -> str:
        body = {"model": self.model, "max_tokens": self.max_output_tokens, "system": instructions,
                "messages": [{"role": turn.role, "content": turn.text} for turn in conversation]}
        if self.effort:
            body["output_config"] = {"effort": self.effort}
        answer = _send(lambda: httpx.post(self.url, json=body, headers=self.headers, timeout=self.timeout_s),
                       self.model)
        if answer.get("stop_reason") == "refusal":
            raise AssistantError(422, REFUSED)
        reply = "".join(block.get("text", "") for block in answer.get("content") or []
                        if isinstance(block, dict) and block.get("type") == "text").strip()
        if not reply:
            raise AssistantError(502, EMPTY)
        return reply


class GeminiAdapter(ChatModel):
    name = "gemini"

    VOICE_INSTRUCTION = "Transcribe el audio en español y responde a lo que pregunta."

    def __init__(self, api_key: str, base_url: str, model: str, timeout_s: float, max_output_tokens: int):
        self.url = f"{base_url.rstrip('/')}/v1beta/models/{model}:generateContent"
        self.headers = {"x-goog-api-key": api_key}
        self.model = model
        self.timeout_s = timeout_s
        self.max_output_tokens = max_output_tokens

    def chat(self, instructions: str, conversation: list[Said]) -> str:
        contents = [{"role": "user" if turn.role == USER else "model", "parts": [{"text": turn.text}]}
                    for turn in conversation]
        return self._text(self._generate(instructions, contents, voice=False))

    def listen_and_answer(self, instructions: str, audio: bytes, mime_type: str) -> tuple[str, str | None]:
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


class GeminiTranscriber(Transcriber):
    """Gemini hears the recording and answers it in one call, with the farm's instructions."""

    def __init__(self, chat: GeminiAdapter):
        self.chat = chat

    def hear(self, instructions: str, audio: bytes, mime_type: str) -> Heard:
        reply, heard = self.chat.listen_and_answer(instructions, audio, mime_type)
        return Heard(heard, reply)
