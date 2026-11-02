import logging
from pathlib import Path

from .providers import ASSISTANT, CANNOT_LISTEN, EMPTY, USER, AssistantError, ChatModel, Said

log = logging.getLogger("agrosense_ia.assistant")

PROMPT_FILE = Path(__file__).resolve().parent.parent / "prompts" / "system-es.txt"

MAX_HISTORY_SENT = 8

NO_FARM = "No hay datos de la finca disponibles en este momento."
NOT_CONFIGURED = "El asistente de IA no está configurado todavía."


def _answer_of(call, *arguments):
    """The adapters read what a provider should send; what it sends instead is a failed answer, not a crash."""
    try:
        return call(*arguments)
    except (KeyError, IndexError, TypeError, AttributeError) as odd:
        log.warning("The AI provider answered something unexpected: %s", type(odd).__name__)
        raise AssistantError(502, EMPTY) from None


def load_prompt(path: Path = PROMPT_FILE) -> str:
    return path.read_text(encoding="utf-8").strip()


def latest(history: list[Said]) -> list[Said]:
    return history[-MAX_HISTORY_SENT:]


def conversation(history: list[Said], question: str) -> list[Said]:
    roles: list[str] = []
    texts: list[str] = []
    for turn in [*latest(history), Said(USER, question)]:
        role = ASSISTANT if turn.role == ASSISTANT else USER
        if not roles and role == ASSISTANT:
            continue
        if roles and roles[-1] == role:
            texts[-1] += "\n" + turn.text
            continue
        roles.append(role)
        texts.append(turn.text)
    return [Said(role, text) for role, text in zip(roles, texts)]


class Assistant:
    def __init__(self, model: ChatModel | None, prompt: str):
        self.model = model
        self.prompt = prompt

    @property
    def configured(self) -> bool:
        return self.model is not None

    def instructions(self, farm_context: str | None) -> str:
        farm = (farm_context or "").strip() or NO_FARM
        return f"{self.prompt}\n\n{farm}"

    def chat(self, question: str, history: list[Said], farm_context: str | None) -> str:
        if self.model is None:
            raise AssistantError(503, NOT_CONFIGURED)
        return _answer_of(self.model.chat, self.instructions(farm_context), conversation(history, question.strip()))

    def voice(self, audio: bytes, mime_type: str, farm_context: str | None) -> tuple[str, str | None]:
        if self.model is None:
            raise AssistantError(503, NOT_CONFIGURED)
        if not self.model.listens():
            raise AssistantError(503, CANNOT_LISTEN)
        heard = _answer_of(self.model.transcriber.hear, self.instructions(farm_context), audio, mime_type)
        if heard.reply is not None:
            return heard.reply, heard.transcript
        said = heard.transcript[:1000]
        return self.chat(said, [], farm_context), said
