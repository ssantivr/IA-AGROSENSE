"""Design patterns of the AI service. Plain unittest and no FastAPI, so they run with `python -m unittest
tests.test_patterns` on machines that cannot load pydantic's native module."""
import logging
import unittest

from agrosense_ia.ai_decorators import CacheChatModel, LoggingChatModel, RateLimitChatModel
from agrosense_ia.ai_factory import AnthropicProviderFactory, GeminiProviderFactory, OpenAiCompatProviderFactory, \
    chosen, decorate, factory_for
from agrosense_ia.assistant import Assistant
from agrosense_ia.config import Settings, from_environment
from agrosense_ia.providers import USER, AssistantError, ChatModel, GeminiAdapter, GeminiTranscriber, Heard, \
    OpenAiCompatAdapter, OpenAiCompatTranscriber, Said, Transcriber

from .fake_provider import FakeProvider, anthropic_reply, gemini_reply, openai_reply

QUESTION = [Said(USER, "¿Riego hoy?")]


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class Scripted(ChatModel):
    name = "scripted"

    def __init__(self, failure: AssistantError | None = None):
        self.calls = 0
        self.failure = failure

    def chat(self, instructions, conversation):
        self.calls += 1
        if self.failure:
            raise self.failure
        return f"respuesta {self.calls}"


class AbstractFactoryTests(unittest.TestCase):
    def test_each_family_makes_its_own_chat_and_transcription(self):
        openai = OpenAiCompatProviderFactory(Settings(chat_api_key="k")).create()
        self.assertIsInstance(openai, OpenAiCompatAdapter)
        self.assertIsInstance(openai.transcriber, OpenAiCompatTranscriber)

        gemini = GeminiProviderFactory(Settings(gemini_api_key="k")).create()
        self.assertIsInstance(gemini, GeminiAdapter)
        self.assertIsInstance(gemini.transcriber, GeminiTranscriber)

    def test_a_family_may_not_listen(self):
        self.assertFalse(AnthropicProviderFactory(Settings(anthropic_api_key="k")).create().listens())
        without_model = Settings(chat_api_key="k", transcription_model="")
        self.assertFalse(OpenAiCompatProviderFactory(without_model).create().listens())

    def test_the_family_follows_the_priority_of_the_keys(self):
        every = {"chat_api_key": "a", "anthropic_api_key": "b", "gemini_api_key": "c"}
        self.assertIsInstance(factory_for(Settings(**every)), OpenAiCompatProviderFactory)
        del every["chat_api_key"]
        self.assertIsInstance(factory_for(Settings(**every)), AnthropicProviderFactory)
        del every["anthropic_api_key"]
        self.assertIsInstance(factory_for(Settings(**every)), GeminiProviderFactory)
        self.assertIsNone(factory_for(Settings()))
        self.assertIsNone(chosen(Settings()))


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.provider = FakeProvider()
        self.addCleanup(self.provider.close)

    def assistant(self, **settings) -> Assistant:
        return Assistant(chosen(Settings(**settings)), "Instrucciones")

    def test_the_same_call_reaches_each_vendor_in_its_own_format(self):
        self.provider.answer((200, openai_reply("hola")))
        reply = self.assistant(chat_api_key="k", chat_base_url=self.provider.url).chat("¿Riego?", [], "Finca")
        self.assertEqual(reply, "hola")
        sent = self.provider.requests[-1]
        self.assertEqual(sent["path"], "/chat/completions")
        self.assertEqual(sent["json"]["messages"][0], {"role": "system", "content": "Instrucciones\n\nFinca"})

        self.provider.answer((200, gemini_reply("hola")))
        reply = self.assistant(gemini_api_key="k", gemini_base_url=self.provider.url, gemini_model="g").chat(
            "¿Riego?", [], "Finca")
        self.assertEqual(reply, "hola")
        sent = self.provider.requests[-1]
        self.assertEqual(sent["path"], "/v1beta/models/g:generateContent")
        self.assertEqual(sent["json"]["contents"], [{"role": "user", "parts": [{"text": "¿Riego?"}]}])

        self.provider.answer((200, anthropic_reply("hola")))
        reply = self.assistant(anthropic_api_key="k", anthropic_base_url=self.provider.url).chat("¿Riego?", [], None)
        self.assertEqual(reply, "hola")
        self.assertEqual(self.provider.requests[-1]["path"], "/v1/messages")

    def test_openai_voice_takes_two_calls_and_gemini_one_even_through_the_decorators(self):
        self.provider.answer((200, {"text": "¿Riego hoy?"}), (200, openai_reply("Sí")))
        answer = self.assistant(chat_api_key="k", chat_base_url=self.provider.url).voice(b"audio", "audio/webm", "F")
        self.assertEqual(answer, ("Sí", "¿Riego hoy?"))
        self.assertEqual([r["path"] for r in self.provider.requests],
                         ["/audio/transcriptions", "/chat/completions"])

        self.provider.requests.clear()
        self.provider.answer((200, gemini_reply('{"transcript": "¿Riego?", "reply": "No"}')))
        answer = self.assistant(gemini_api_key="k", gemini_base_url=self.provider.url, gemini_model="g").voice(
            b"audio", "audio/webm", "F")
        self.assertEqual(answer, ("No", "¿Riego?"))
        self.assertEqual(len(self.provider.requests), 1)

    def test_a_provider_that_cannot_listen_answers_503(self):
        assistant = self.assistant(anthropic_api_key="k", anthropic_base_url=self.provider.url)
        with self.assertRaises(AssistantError) as caught:
            assistant.voice(b"audio", "audio/webm", "F")
        self.assertEqual(caught.exception.status, 503)


class FixedTranscriber(Transcriber):
    def hear(self, instructions, audio, mime_type):
        return Heard("texto", "respuesta")


class DecoratorTests(unittest.TestCase):
    def test_a_decorator_keeps_the_interface_of_what_it_wraps(self):
        inner = Scripted()
        inner.transcriber = FixedTranscriber()
        model = decorate(inner, Settings(provider_requests_per_minute=5, reply_cache_seconds=5))
        self.assertIsInstance(model, LoggingChatModel)
        self.assertEqual(model.name, "scripted")
        self.assertTrue(model.listens())

    def test_the_decorators_are_only_there_when_configured(self):
        inner = Scripted()
        self.assertIs(decorate(inner, Settings()).inner, inner)

    def test_the_environment_turns_on_limit_and_cache_with_sane_defaults(self):
        settings = from_environment({})
        self.assertEqual((settings.provider_requests_per_minute, settings.reply_cache_seconds), (60, 60))
        settings = from_environment({"AI_PROVIDER_REQUESTS_PER_MINUTE": "0", "AI_REPLY_CACHE_SECONDS": "0"})
        self.assertEqual((settings.provider_requests_per_minute, settings.reply_cache_seconds), (0, 0))

    def test_the_rate_limit_refuses_with_429_and_recovers_after_the_window(self):
        clock = Clock()
        inner = Scripted()
        limited = RateLimitChatModel(inner, 2, clock)
        limited.chat("i", QUESTION)
        limited.chat("i", QUESTION)
        with self.assertRaises(AssistantError) as caught:
            limited.chat("i", QUESTION)
        self.assertEqual(caught.exception.status, 429)
        self.assertEqual(inner.calls, 2)

        clock.now += 60
        limited.chat("i", QUESTION)
        self.assertEqual(inner.calls, 3)

    def test_the_cache_repeats_an_answer_until_it_expires(self):
        clock = Clock()
        inner = Scripted()
        cached = CacheChatModel(inner, 30, clock)
        self.assertEqual(cached.chat("i", QUESTION), "respuesta 1")
        self.assertEqual(cached.chat("i", QUESTION), "respuesta 1")
        self.assertEqual(inner.calls, 1)

        clock.now += 30
        self.assertEqual(cached.chat("i", QUESTION), "respuesta 2")

    def test_the_cache_never_shares_an_answer_between_farms_or_keeps_a_failure(self):
        inner = Scripted()
        cached = CacheChatModel(inner, 30, Clock())
        cached.chat("Finca A", QUESTION)
        cached.chat("Finca B", QUESTION)
        self.assertEqual(inner.calls, 2)

        failing = Scripted(AssistantError(502, "caído"))
        cached = CacheChatModel(failing, 30, Clock())
        for _ in range(2):
            with self.assertRaises(AssistantError):
                cached.chat("i", QUESTION)
        self.assertEqual(failing.calls, 2)

    def test_a_cached_answer_spends_no_quota(self):
        inner = Scripted()
        model = decorate(inner, Settings(provider_requests_per_minute=1, reply_cache_seconds=30))
        model.chat("i", QUESTION)
        self.assertEqual(model.chat("i", QUESTION), "respuesta 1")

    def test_the_log_shows_failures_and_never_the_text(self):
        with self.assertLogs("agrosense_ia.decorators", logging.INFO) as logs:
            LoggingChatModel(Scripted()).chat("Datos privados de la finca", QUESTION)
            with self.assertRaises(AssistantError):
                LoggingChatModel(Scripted(AssistantError(504, "tarde"))).chat("i", QUESTION)
        text = "\n".join(logs.output)
        self.assertIn("scripted answered", text)
        self.assertIn("failed with 504", text)
        self.assertNotIn("privados", text)
        self.assertNotIn("Riego", text)


if __name__ == "__main__":
    unittest.main()
