"""The assistant's routes, with every provider played by a local HTTP server."""

import logging
import socket

import pytest
from fastapi.testclient import TestClient

from agrosense_ia.app import MAX_AUDIO_BYTES, create_app
from agrosense_ia.assistant import MAX_HISTORY_SENT, conversation
from agrosense_ia.config import Settings, from_environment
from agrosense_ia.providers import ASSISTANT, USER, Said

from .fake_provider import FakeProvider, anthropic_reply, gemini_reply, openai_reply

KEY = "llave-interna-de-prueba"
TOKEN = "token-anterior-de-prueba"
PROVIDER_KEY = "llave-del-proveedor"
FARM = "Fecha y hora actual: 2026-10-16 08:00.\nDatos de la finca del usuario: 1 cultivos, 2 sensores, 0 alertas sin atender."
PROMPT = "Eres el asistente de AgroSense."


@pytest.fixture
def provider():
    fake = FakeProvider()
    yield fake
    fake.close()


def client_for(**settings) -> TestClient:
    values = {"internal_key": KEY, "service_token": TOKEN, "timeout_ms": 2000}
    values.update(settings)
    return TestClient(create_app(Settings(**values), prompt=PROMPT))


def ask(client, body=None, headers=None, path="/chat"):
    sent = {"X-Internal-Key": KEY}
    sent.update(headers or {})
    return client.post(path, json=body if body is not None else {"message": "¿Necesito regar hoy?",
                                                                    "farm_context": FARM}, headers=sent)


def closed_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


# --- Who may call -----------------------------------------------------------------------------------------

def test_the_assistant_answers_only_callers_with_the_internal_key(provider):
    provider.answer((200, openai_reply("Sí, riega.")))
    client = client_for(chat_api_key=PROVIDER_KEY, chat_base_url=provider.url)

    for headers in ({"X-Internal-Key": ""}, {"X-Internal-Key": "otra"}, {"X-Internal-Key": KEY + "x"},
                    {"X-Internal-Key": "", "Authorization": f"Bearer {KEY}"}):
        response = ask(client, headers=headers)
        assert response.status_code == 401
        assert KEY not in response.text
    assert provider.requests == []
    assert ask(client).status_code == 200


def test_the_bearer_token_the_backend_sent_before_is_still_accepted(provider):
    provider.answer((200, openai_reply("Sí.")))
    client = client_for(chat_api_key=PROVIDER_KEY, chat_base_url=provider.url)

    response = ask(client, headers={"X-Internal-Key": "", "Authorization": f"Bearer {TOKEN}"})

    assert response.status_code == 200


def test_without_a_configured_key_the_assistant_answers_nobody():
    client = TestClient(create_app(Settings(chat_api_key=PROVIDER_KEY), prompt=PROMPT))

    assert ask(client, headers={"X-Internal-Key": ""}).status_code == 401
    assert client.get("/health").json() == {"status": "UP"}
    assert client.post("/predict/irrigation", json={"soil_moisture": 30}).status_code == 200


def test_predictions_also_accept_the_internal_key():
    client = client_for()

    assert client.post("/predict/irrigation", json={"soil_moisture": 30}).status_code == 401
    assert client.post("/predict/irrigation", json={"soil_moisture": 30},
                       headers={"X-Internal-Key": KEY}).status_code == 200


def test_secrets_are_never_logged(provider, caplog):
    provider.answer((500, {"error": {"message": f"bad key {PROVIDER_KEY}"}}))
    client = client_for(chat_api_key=PROVIDER_KEY, chat_base_url=provider.url)

    with caplog.at_level(logging.DEBUG):
        ask(client)
        ask(client, headers={"X-Internal-Key": "adivinada"})

    for secret in (KEY, TOKEN, PROVIDER_KEY, "adivinada"):
        assert secret not in caplog.text


# --- OpenAI-compatible providers -------------------------------------------------------------------------

def test_an_openai_compatible_provider_gets_the_prompt_the_farm_and_the_question(provider):
    provider.answer((200, openai_reply("Sí, el café está en 31 %.")))
    client = client_for(chat_api_key=PROVIDER_KEY, chat_base_url=provider.url + "/", chat_models=("m1",))

    response = ask(client)

    assert response.status_code == 200
    assert response.json() == {"reply": "Sí, el café está en 31 %.", "source": "model"}
    sent = provider.requests[0]
    assert sent["path"] == "/chat/completions"
    assert sent["headers"]["authorization"] == f"Bearer {PROVIDER_KEY}"
    body = sent["json"]
    assert body["model"] == "m1"
    assert body["max_tokens"] == 2048
    assert body["messages"][0] == {"role": "system", "content": f"{PROMPT}\n\n{FARM}"}
    assert body["messages"][-1] == {"role": "user", "content": "¿Necesito regar hoy?"}


def test_only_the_last_eight_turns_are_sent(provider):
    provider.answer((200, openai_reply("Bien.")))
    client = client_for(chat_api_key=PROVIDER_KEY, chat_base_url=provider.url)
    history = [{"role": USER if index % 2 == 0 else ASSISTANT, "text": f"turno {index}"} for index in range(12)]

    ask(client, {"message": "¿Y ahora?", "history": history})

    messages = provider.requests[0]["json"]["messages"]
    assert [message["content"] for message in messages[1:-1]] == [f"turno {index}" for index in range(4, 12)]
    assert "No hay datos de la finca" in messages[0]["content"]


def test_the_next_model_answers_when_one_is_out_of_allowance(provider):
    provider.answer((429, {}), (200, openai_reply("Respuesta del segundo.")))
    client = client_for(chat_api_key=PROVIDER_KEY, chat_base_url=provider.url, chat_models=("primero", "segundo"))

    response = ask(client)

    assert response.json()["reply"] == "Respuesta del segundo."
    assert [request["json"]["model"] for request in provider.requests] == ["primero", "segundo"]


def test_every_model_out_of_allowance_is_a_429(provider):
    provider.answer((429, {}))
    client = client_for(chat_api_key=PROVIDER_KEY, chat_base_url=provider.url, chat_models=("a", "b"))

    response = ask(client)

    assert response.status_code == 429
    assert "un minuto" in response.json()["message"]


def test_a_broken_provider_is_a_502_and_one_that_cannot_be_reached_a_504(provider):
    provider.answer((500, {}))
    assert ask(client_for(chat_api_key=PROVIDER_KEY, chat_base_url=provider.url)).status_code == 502

    down = client_for(chat_api_key=PROVIDER_KEY, chat_base_url=f"http://127.0.0.1:{closed_port()}")
    assert ask(down).status_code == 504


def test_a_provider_that_takes_too_long_is_a_504(provider):
    provider.answer((200, openai_reply("tarde"), 1.5))
    client = client_for(chat_api_key=PROVIDER_KEY, chat_base_url=provider.url, timeout_ms=300)

    response = ask(client)

    assert response.status_code == 504
    assert "tardó" in response.json()["message"]


def test_a_filtered_answer_is_a_refusal(provider):
    provider.answer((200, {"choices": [{"message": {"content": ""}, "finish_reason": "content_filter"}]}))

    response = ask(client_for(chat_api_key=PROVIDER_KEY, chat_base_url=provider.url))

    assert response.status_code == 422


# --- Claude ------------------------------------------------------------------------------------------------

def test_claude_gets_its_headers_and_the_instructions_as_system(provider):
    provider.answer((200, anthropic_reply("Riega 20 minutos.")))
    client = client_for(anthropic_api_key=PROVIDER_KEY, anthropic_base_url=provider.url, anthropic_model="claude-x")

    response = ask(client)

    assert response.json() == {"reply": "Riega 20 minutos.", "source": "model"}
    sent = provider.requests[0]
    assert sent["path"] == "/v1/messages"
    assert sent["headers"]["x-api-key"] == PROVIDER_KEY
    assert sent["headers"]["anthropic-version"] == "2023-06-01"
    body = sent["json"]
    assert body["model"] == "claude-x"
    assert body["system"] == f"{PROMPT}\n\n{FARM}"
    assert body["messages"] == [{"role": "user", "content": "¿Necesito regar hoy?"}]
    assert body["max_tokens"] == 2048
    assert body["output_config"] == {"effort": "low"}
    assert "temperature" not in body


def test_claude_s_refusal_and_allowance(provider):
    client = client_for(anthropic_api_key=PROVIDER_KEY, anthropic_base_url=provider.url)
    provider.answer((200, anthropic_reply("", stop_reason="refusal")))
    assert ask(client).status_code == 422
    provider.answer((429, {}))
    assert ask(client).status_code == 429


# --- Gemini ------------------------------------------------------------------------------------------------

def test_gemini_gets_its_key_in_a_header_and_the_model_in_the_path(provider):
    provider.answer((200, gemini_reply("Todo bien.")))
    client = client_for(gemini_api_key=PROVIDER_KEY, gemini_base_url=provider.url, gemini_model="gemini-x")

    response = ask(client)

    assert response.json()["reply"] == "Todo bien."
    sent = provider.requests[0]
    assert sent["path"] == "/v1beta/models/gemini-x:generateContent"
    assert sent["headers"]["x-goog-api-key"] == PROVIDER_KEY
    assert PROVIDER_KEY not in sent["path"]
    assert sent["json"]["system_instruction"]["parts"][0]["text"] == f"{PROMPT}\n\n{FARM}"


# --- Which provider answers --------------------------------------------------------------------------------

def test_providers_are_chosen_by_priority():
    every = {"chat_api_key": "a", "anthropic_api_key": "b", "gemini_api_key": "c"}
    assert create_app(Settings(**every), prompt=PROMPT).state.assistant.model.name == "openai-compatible"
    del every["chat_api_key"]
    assert create_app(Settings(**every), prompt=PROMPT).state.assistant.model.name == "anthropic"
    del every["anthropic_api_key"]
    assert create_app(Settings(**every), prompt=PROMPT).state.assistant.model.name == "gemini"


def test_only_the_first_provider_with_a_key_is_called(provider):
    other = FakeProvider()
    try:
        provider.answer((200, openai_reply("de OpenAI")))
        other.answer((200, anthropic_reply("de Claude")))
        client = client_for(chat_api_key=PROVIDER_KEY, chat_base_url=provider.url,
                            anthropic_api_key=PROVIDER_KEY, anthropic_base_url=other.url)

        assert ask(client).json()["reply"] == "de OpenAI"
        assert other.requests == []
    finally:
        other.close()


def test_without_any_provider_key_the_answer_is_a_503_the_backend_can_act_on():
    response = ask(client_for())

    assert response.status_code == 503
    assert "no está configurado" in response.json()["message"]


def test_settings_come_from_the_environment():
    settings = from_environment({"INTERNAL_API_KEY": " k ", "AI_CHAT_MODEL": "uno, dos,,", "AI_TIMEOUT_MS": "5000",
                                 "AI_MAX_OUTPUT_TOKENS": "512", "AI_TRANSCRIPTION_MODEL": ""})

    assert settings.internal_key == "k"
    assert settings.chat_models == ("uno", "dos")
    assert settings.timeout_ms == 5000
    assert settings.max_output_tokens == 512
    assert settings.transcription_model == ""
    assert settings.anthropic_model == "claude-haiku-5-5"
    assert from_environment({}).timeout_ms == 20000


# --- What is asked -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("body, expected", [
    ({"message": "  "}, "Escribe tu pregunta"),
    ({"message": "x" * 1001}, "1000"),
    ({"message": "hola", "history": [{"role": "user", "text": "a"}] * 13}, "demasiado larga"),
    ({"message": "hola", "history": [{"role": "system", "text": "a"}]}, "autor"),
    ({"message": "hola", "farm_context": 3}, "farm_context"),
])
def test_bad_questions_are_refused_with_a_message(body, expected):
    response = ask(client_for(chat_api_key=PROVIDER_KEY), body)

    assert response.status_code == 400
    assert expected in response.json()["message"]


def test_the_conversation_starts_with_the_user_and_alternates():
    history = [Said(ASSISTANT, "hola"), Said(USER, "a"), Said(USER, "b"), Said(ASSISTANT, "c")]

    assert conversation(history, "d") == [Said(USER, "a\nb"), Said(ASSISTANT, "c"), Said(USER, "d")]
    assert MAX_HISTORY_SENT == 8


# --- Voice -------------------------------------------------------------------------------------------------

def speak(client, files=None, data=None, headers=None):
    sent = {"X-Internal-Key": KEY}
    sent.update(headers or {})
    return client.post("/voice", files=files, data=data, headers=sent)


def test_a_transcript_is_answered_like_a_written_question_and_the_audio_is_not_sent(provider):
    provider.answer((200, openai_reply("Sí.")))
    client = client_for(chat_api_key=PROVIDER_KEY, chat_base_url=provider.url)

    response = speak(client, files={"audio": ("v.webm", b"ruido", "audio/webm")},
                     data={"transcript": " ¿Riego? ", "farm_context": FARM})

    assert response.json() == {"reply": "Sí.", "transcript": "¿Riego?", "source": "model"}
    assert [request["path"] for request in provider.requests] == ["/chat/completions"]


def test_a_recording_is_transcribed_then_answered(provider):
    provider.answer((200, {"text": "¿Cómo está el pH?"}), (200, openai_reply("El pH es 6,4.")))
    client = client_for(chat_api_key=PROVIDER_KEY, chat_base_url=provider.url, transcription_model="whisper-x")

    response = speak(client, files={"audio": ("v.webm", b"\x1aE\xdf\xa3audio", "audio/webm;codecs=opus")},
                     data={"farm_context": FARM})

    assert response.status_code == 200
    assert response.json() == {"reply": "El pH es 6,4.", "transcript": "¿Cómo está el pH?", "source": "model"}
    transcription = provider.requests[0]
    assert transcription["path"] == "/audio/transcriptions"
    assert transcription["headers"]["content-type"].startswith("multipart/form-data")
    assert b"whisper-x" in transcription["raw"] and b'filename="voice.webm"' in transcription["raw"]
    assert provider.requests[1]["json"]["messages"][-1]["content"] == "¿Cómo está el pH?"


def test_gemini_listens_and_answers_in_one_call(provider):
    provider.answer((200, gemini_reply('{"transcript": "¿Riego hoy?", "reply": "No hace falta."}')))
    client = client_for(gemini_api_key=PROVIDER_KEY, gemini_base_url=provider.url)

    response = speak(client, files={"audio": ("v.ogg", b"OggS", "audio/ogg")})

    assert response.json() == {"reply": "No hace falta.", "transcript": "¿Riego hoy?", "source": "model"}
    parts = provider.requests[0]["json"]["contents"][0]["parts"]
    assert parts[0]["inline_data"]["mime_type"] == "audio/ogg"


def test_claude_cannot_listen_so_a_recording_is_a_503(provider):
    client = client_for(anthropic_api_key=PROVIDER_KEY, anthropic_base_url=provider.url)

    response = speak(client, files={"audio": ("v.webm", b"audio", "audio/webm")})

    assert response.status_code == 503
    assert "Escribe tu pregunta" in response.json()["message"]
    assert provider.requests == []


def test_recordings_that_cannot_be_used_are_refused():
    client = client_for(chat_api_key=PROVIDER_KEY)

    assert speak(client, data={"farm_context": FARM}).status_code == 400
    assert speak(client, files={"audio": ("v.txt", b"hola", "text/plain")}).status_code == 415
    big = speak(client, files={"audio": ("v.webm", b"0" * (MAX_AUDIO_BYTES + 1), "audio/webm")})
    assert big.status_code == 413
    assert speak(client, files={"audio": ("v.webm", b"a", "audio/webm")},
                 headers={"X-Internal-Key": "no"}).status_code == 401
