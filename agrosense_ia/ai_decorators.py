"""Decorators of the chat model: each adds one concern and keeps the ChatModel interface, so the assistant does not
know how many it is talking through."""
import hashlib
import logging
import threading
import time
from collections import OrderedDict, deque

from .providers import AssistantError, ChatModel, Said

log = logging.getLogger("agrosense_ia.decorators")

RATE_WINDOW_S = 60.0
MAX_CACHED_REPLIES = 128

TOO_MANY = "El asistente está recibiendo muchas consultas. Inténtalo de nuevo en un minuto."


class ChatModelDecorator(ChatModel):
    def __init__(self, inner: ChatModel):
        self.inner = inner

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def transcriber(self):
        return self.inner.transcriber

    def chat(self, instructions: str, conversation: list[Said]) -> str:
        return self.inner.chat(instructions, conversation)


class LoggingChatModel(ChatModelDecorator):
    """Records who answered and how long it took. Never the text: it holds the farm's data."""

    def __init__(self, inner: ChatModel, clock=time.monotonic):
        super().__init__(inner)
        self.clock = clock

    def chat(self, instructions: str, conversation: list[Said]) -> str:
        started = self.clock()
        try:
            reply = self.inner.chat(instructions, conversation)
        except AssistantError as failure:
            log.warning("%s failed with %s after %.0f ms", self.name, failure.status,
                        (self.clock() - started) * 1000)
            raise
        log.info("%s answered in %.0f ms", self.name, (self.clock() - started) * 1000)
        return reply


class RateLimitChatModel(ChatModelDecorator):
    """A limit for the whole service, which protects the provider's quota; the backend has its own per user."""

    def __init__(self, inner: ChatModel, requests_per_minute: int, clock=time.monotonic):
        super().__init__(inner)
        self.limit = requests_per_minute
        self.clock = clock
        self.recent: deque[float] = deque()
        self.lock = threading.Lock()

    def chat(self, instructions: str, conversation: list[Said]) -> str:
        with self.lock:
            now = self.clock()
            while self.recent and now - self.recent[0] >= RATE_WINDOW_S:
                self.recent.popleft()
            if len(self.recent) >= self.limit:
                raise AssistantError(429, TOO_MANY)
            self.recent.append(now)
        return self.inner.chat(instructions, conversation)


class CacheChatModel(ChatModelDecorator):
    """The key covers the instructions, which carry the farm's data, so an answer never reaches another farm."""

    def __init__(self, inner: ChatModel, ttl_s: float, clock=time.monotonic):
        super().__init__(inner)
        self.ttl_s = ttl_s
        self.clock = clock
        self.replies: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self.lock = threading.Lock()

    @staticmethod
    def _key(instructions: str, conversation: list[Said]) -> str:
        digest = hashlib.sha256(instructions.encode("utf-8"))
        for turn in conversation:
            digest.update(b"\0" + turn.role.encode("utf-8") + b"\0" + turn.text.encode("utf-8"))
        return digest.hexdigest()

    def chat(self, instructions: str, conversation: list[Said]) -> str:
        key = self._key(instructions, conversation)
        with self.lock:
            cached = self.replies.get(key)
            if cached is not None and self.clock() < cached[0]:
                return cached[1]
            self.replies.pop(key, None)
        reply = self.inner.chat(instructions, conversation)
        with self.lock:
            self.replies[key] = (self.clock() + self.ttl_s, reply)
            while len(self.replies) > MAX_CACHED_REPLIES:
                self.replies.popitem(last=False)
        return reply
