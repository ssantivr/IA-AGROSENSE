"""Abstract Factory: each provider family makes a chat model and the transcriber that goes with it, so the rest of
the service never mixes, say, a Gemini chat with an OpenAI-compatible transcription."""
from abc import ABC, abstractmethod

from .ai_decorators import CacheChatModel, LoggingChatModel, RateLimitChatModel
from .providers import AnthropicAdapter, ChatModel, GeminiAdapter, GeminiTranscriber, OpenAiCompatAdapter, \
    OpenAiCompatTranscriber, Transcriber


class AiProviderFactory(ABC):
    def __init__(self, settings):
        self.settings = settings
        self.timeout_s = settings.timeout_ms / 1000

    @abstractmethod
    def create_chat(self) -> ChatModel:
        ...

    @abstractmethod
    def create_transcription(self, chat: ChatModel) -> Transcriber | None:
        ...

    def create(self) -> ChatModel:
        chat = self.create_chat()
        chat.transcriber = self.create_transcription(chat)
        return chat


class OpenAiCompatProviderFactory(AiProviderFactory):
    def create_chat(self) -> ChatModel:
        s = self.settings
        return OpenAiCompatAdapter(s.chat_api_key, s.chat_base_url, s.chat_models, self.timeout_s, s.max_output_tokens)

    def create_transcription(self, chat: ChatModel) -> Transcriber | None:
        s = self.settings
        if not s.transcription_model:
            return None
        return OpenAiCompatTranscriber(s.chat_api_key, s.chat_base_url, s.transcription_model, self.timeout_s)


class AnthropicProviderFactory(AiProviderFactory):
    def create_chat(self) -> ChatModel:
        s = self.settings
        return AnthropicAdapter(s.anthropic_api_key, s.anthropic_base_url, s.anthropic_model, s.anthropic_effort,
                                self.timeout_s, s.max_output_tokens)

    def create_transcription(self, chat: ChatModel) -> Transcriber | None:
        return None


class GeminiProviderFactory(AiProviderFactory):
    def create_chat(self) -> ChatModel:
        s = self.settings
        return GeminiAdapter(s.gemini_api_key, s.gemini_base_url, s.gemini_model, self.timeout_s, s.max_output_tokens)

    def create_transcription(self, chat: ChatModel) -> Transcriber | None:
        return GeminiTranscriber(chat)


def factory_for(settings) -> AiProviderFactory | None:
    if settings.chat_api_key:
        return OpenAiCompatProviderFactory(settings)
    if settings.anthropic_api_key:
        return AnthropicProviderFactory(settings)
    if settings.gemini_api_key:
        return GeminiProviderFactory(settings)
    return None


def decorate(model: ChatModel, settings) -> ChatModel:
    """A cached answer spends no quota, so the cache goes outside the limit; the log sees everything."""
    if settings.provider_requests_per_minute > 0:
        model = RateLimitChatModel(model, settings.provider_requests_per_minute)
    if settings.reply_cache_seconds > 0:
        model = CacheChatModel(model, settings.reply_cache_seconds)
    return LoggingChatModel(model)


def chosen(settings) -> ChatModel | None:
    factory = factory_for(settings)
    return None if factory is None else decorate(factory.create(), settings)
