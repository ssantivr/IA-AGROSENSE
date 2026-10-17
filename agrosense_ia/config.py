import os
from dataclasses import dataclass

DEFAULT_CHAT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_CHAT_MODEL = "qwen/qwen3.8-27b"
DEFAULT_TRANSCRIPTION_MODEL = "whisper-large-v3-turbo"
DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-5-5"
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com"
DEFAULT_GEMINI_MODEL = "gemini-flash-latest"


@dataclass(frozen=True)
class Settings:
    internal_key: str = ""
    service_token: str = ""
    chat_api_key: str = ""
    chat_base_url: str = DEFAULT_CHAT_BASE_URL
    chat_models: tuple[str, ...] = (DEFAULT_CHAT_MODEL,)
    transcription_model: str = DEFAULT_TRANSCRIPTION_MODEL
    anthropic_api_key: str = ""
    anthropic_base_url: str = DEFAULT_ANTHROPIC_BASE_URL
    anthropic_model: str = DEFAULT_ANTHROPIC_MODEL
    anthropic_effort: str = "low"
    gemini_api_key: str = ""
    gemini_base_url: str = DEFAULT_GEMINI_BASE_URL
    gemini_model: str = DEFAULT_GEMINI_MODEL
    timeout_ms: int = 20000
    max_output_tokens: int = 2048

    @property
    def guarded(self) -> bool:
        return bool(self.internal_key or self.service_token)


def _models(value: str) -> tuple[str, ...]:
    return tuple(name.strip() for name in value.split(",") if name.strip())


def _int(environ: dict, name: str, default: int) -> int:
    value = environ.get(name, "").strip()
    return int(value) if value else default


def from_environment(environ: dict | None = None) -> Settings:
    env = os.environ if environ is None else environ

    def text(name: str, default: str = "") -> str:
        value = env.get(name, "").strip()
        return value or default

    return Settings(
        internal_key=text("INTERNAL_API_KEY"),
        service_token=text("AI_SERVICE_TOKEN"),
        chat_api_key=text("AI_CHAT_API_KEY"),
        chat_base_url=text("AI_CHAT_BASE_URL", DEFAULT_CHAT_BASE_URL),
        chat_models=_models(text("AI_CHAT_MODEL", DEFAULT_CHAT_MODEL)) or (DEFAULT_CHAT_MODEL,),
        transcription_model=env.get("AI_TRANSCRIPTION_MODEL", DEFAULT_TRANSCRIPTION_MODEL).strip(),
        anthropic_api_key=text("ANTHROPIC_API_KEY"),
        anthropic_base_url=text("ANTHROPIC_BASE_URL", DEFAULT_ANTHROPIC_BASE_URL),
        anthropic_model=text("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL),
        anthropic_effort=env.get("ANTHROPIC_EFFORT", "low").strip(),
        gemini_api_key=text("GEMINI_API_KEY"),
        gemini_base_url=text("GEMINI_BASE_URL", DEFAULT_GEMINI_BASE_URL),
        gemini_model=text("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        timeout_ms=_int(env, "AI_TIMEOUT_MS", 20000),
        max_output_tokens=_int(env, "AI_MAX_OUTPUT_TOKENS", 2048),
    )
