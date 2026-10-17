"""The conversational assistant: the system prompt, the farm the backend describes and the conversation.

The service never reads the database. Everything it knows about the user's farm arrives from the backend in
``farm_context``, already limited to that user's own data.
"""

from pathlib import Path

from .providers import ASSISTANT, CANNOT_LISTEN, USER, AssistantError, ChatModel, Gemini, Said

PROMPT_FILE = Path(__file__).resolve().parent.parent / "prompts" / "system-es.txt"

# Earlier turns sent to the model with a question; older ones cost tokens and rarely change the answer.
MAX_HISTORY_SENT = 8

NO_FARM = "No hay datos de la finca disponibles en este momento."
NOT_CONFIGURED = "El asistente de IA no está configurado todavía."


def load_prompt(path: Path = PROMPT_FILE) -> str:
    """Kept as a text file so it can be reworded without touching the code; a missing one stops the start."""
    return path.read_text(encoding="utf-8").strip()


def latest(history: list[Said]) -> list[Said]:
    return history[-MAX_HISTORY_SENT:]


def conversation(history: list[Said], question: str) -> list[Said]:
    """The turns in the shape models accept: the user speaks first and the two sides alternate.

    A conversation kept by a browser does not always look like that (a question left without an answer, or
    older turns dropped from the start), so answers with nothing before them are left out and turns in a row
    from the same side become one.
    """
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
        return self.model.chat(self.instructions(farm_context), conversation(history, question.strip()))

    def voice(self, audio: bytes, mime_type: str, farm_context: str | None) -> tuple[str, str | None]:
        """The answer to a recording and what was understood from it."""
        if self.model is None:
            raise AssistantError(503, NOT_CONFIGURED)
        if not self.model.listens():
            raise AssistantError(503, CANNOT_LISTEN)
        if isinstance(self.model, Gemini):
            return self.model.listen_and_answer(self.instructions(farm_context), audio, mime_type)
        # Listening and answering in two steps: once the recording is text it is a written question.
        heard = self.model.transcribe(audio, mime_type)[:1000]
        return self.chat(heard, [], farm_context), heard
