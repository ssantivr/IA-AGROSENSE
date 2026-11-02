import json
import math

from .model import DEFAULT_HUMIDITY_RANGE, DEFAULT_PH_RANGE, predict_irrigation
from .providers import ASSISTANT, USER, Said

MAX_MESSAGE_LENGTH = 1000
MAX_HISTORY_TURNS = 12
MAX_TURN_LENGTH = 4000
MAX_FARM_CONTEXT_LENGTH = 8000
MAX_HISTORY_POINTS = 48

# The measurements of a reading as the devices name them in "datos".
READING_MEASUREMENTS = ("temperatura", "humedad_aire", "humedad_suelo")
# The factors of the irrigation rules that need someone to act; a short preventive watering is not one.
ALERT_FACTORS = ("humedad_baja", "humedad_alta", "riesgo_de_helada", "ph_acido", "ph_alcalino")


class BadRequest(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def decode_json(raw: bytes) -> object:
    try:
        return json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
    except (ValueError, UnicodeDecodeError, RecursionError):
        raise BadRequest("El cuerpo no es JSON válido.") from None


def _reject_constant(name: str) -> float:
    raise ValueError(name)


def _number(body: dict, name: str, low: float, high: float, required: bool = True) -> float | None:
    value = body.get(name)
    if value is None:
        if required:
            raise BadRequest(f"Falta el campo '{name}'.")
        return None
    # Only floats: math.isfinite() cannot take an integer too big for a float, and JSON has those.
    if isinstance(value, bool) or not isinstance(value, (int, float)) or \
            (isinstance(value, float) and not math.isfinite(value)):
        raise BadRequest(f"El campo '{name}' debe ser un número.")
    if not low <= value <= high:
        raise BadRequest(f"El campo '{name}' debe estar entre {low:g} y {high:g}.")
    return float(value)


def _range(body: dict, low_name: str, high_name: str, low: float, high: float,
           default: tuple[float, float]) -> tuple[float, float]:
    minimum = _number(body, low_name, low, high, required=False)
    maximum = _number(body, high_name, low, high, required=False)
    if minimum is None and maximum is None:
        return default
    if minimum is None or maximum is None:
        raise BadRequest(f"Los campos '{low_name}' y '{high_name}' deben enviarse juntos.")
    if minimum >= maximum:
        raise BadRequest(f"El campo '{low_name}' debe ser menor que '{high_name}'.")
    return minimum, maximum


def _moisture_history(body: dict) -> list[tuple[float, float]] | None:
    history = body.get("moisture_history")
    if history is None:
        return None
    if not isinstance(history, list) or len(history) > MAX_HISTORY_POINTS:
        raise BadRequest(f"El campo 'moisture_history' debe ser una lista de hasta {MAX_HISTORY_POINTS} puntos.")
    points = []
    for point in history:
        if not isinstance(point, dict):
            raise BadRequest("Cada punto de 'moisture_history' debe tener 'hours_ago' y 'value'.")
        points.append((_number(point, "hours_ago", 0, 72), _number(point, "value", 0, 100)))
    return points


def answer_prediction(body: object) -> dict:
    if not isinstance(body, dict):
        raise BadRequest("El cuerpo debe ser un objeto JSON.")
    crop_id = body.get("id_crop")
    if crop_id is not None and (isinstance(crop_id, bool) or not isinstance(crop_id, int) or crop_id <= 0):
        raise BadRequest("El campo 'id_crop' debe ser un entero positivo.")
    prediction = predict_irrigation(
        soil_moisture=_number(body, "soil_moisture", 0, 100),
        air_temperature=_number(body, "air_temperature", -50, 80, required=False),
        ph=_number(body, "ph", 0, 14, required=False),
        humidity_range=_range(body, "humidity_min", "humidity_max", 0, 100, DEFAULT_HUMIDITY_RANGE),
        ph_range=_range(body, "ph_min", "ph_max", 0, 14, DEFAULT_PH_RANGE),
        moisture_history=_moisture_history(body),
        rain_probability=_number(body, "rain_probability", 0, 100, required=False),
        rain_mm=_number(body, "rain_mm", 0, 1000, required=False),
        moisture_gain_per_minute=_number(body, "moisture_gain_per_minute", 0, 100, required=False),
    )
    return {
        "id_crop": crop_id,
        "irrigate": prediction.irrigate,
        "duration_minutes": prediction.duration_minutes,
        "recommendation": prediction.recommendation,
        "confidence": prediction.confidence,
        "model_used": prediction.model_used,
        "factors": prediction.factors,
    }


def answer_analysis(body: object) -> dict:
    """The irrigation rules applied to one reading in the devices' form, {"datos": {"humedad_suelo", ...}}."""
    if not isinstance(body, dict):
        raise BadRequest("El cuerpo debe ser un objeto JSON.")
    datos = body.get("datos")
    if not isinstance(datos, dict):
        raise BadRequest("El campo 'datos' debe ser un objeto JSON con las medidas.")
    if any(name not in READING_MEASUREMENTS for name in datos):
        raise BadRequest(f"Medida desconocida en 'datos'. Usa: {', '.join(READING_MEASUREMENTS)}.")
    # Checked like the others, though no rule depends on the air's humidity yet.
    _number(datos, "humedad_aire", 0, 100, required=False)
    prediction = predict_irrigation(
        soil_moisture=_number(datos, "humedad_suelo", 0, 100),
        air_temperature=_number(datos, "temperatura", -50, 80, required=False),
        humidity_range=_range(body, "humidity_min", "humidity_max", 0, 100, DEFAULT_HUMIDITY_RANGE),
    )
    return {"alerta": any(factor in ALERT_FACTORS for factor in prediction.factors),
            "recomendacion": prediction.recommendation}


def farm_context(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise BadRequest("El campo 'farm_context' debe ser texto.")
    if len(value) > MAX_FARM_CONTEXT_LENGTH:
        raise BadRequest("El campo 'farm_context' es demasiado largo.")
    return value


def chat_request(body: object) -> tuple[str, list[Said], str | None]:
    if not isinstance(body, dict):
        raise BadRequest("El cuerpo debe ser un objeto JSON.")
    message = body.get("message")
    if not isinstance(message, str) or not message.strip():
        raise BadRequest("Escribe tu pregunta.")
    if len(message) > MAX_MESSAGE_LENGTH:
        raise BadRequest(f"La pregunta no puede superar los {MAX_MESSAGE_LENGTH} caracteres.")
    history = body.get("history") or []
    if not isinstance(history, list):
        raise BadRequest("El campo 'history' debe ser una lista.")
    if len(history) > MAX_HISTORY_TURNS:
        raise BadRequest("La conversación enviada es demasiado larga.")
    turns = []
    for turn in history:
        if not isinstance(turn, dict) or turn.get("role") not in (USER, ASSISTANT):
            raise BadRequest("El autor de un mensaje no es válido.")
        text = turn.get("text")
        if not isinstance(text, str) or not text.strip():
            raise BadRequest("Los mensajes de la conversación no pueden estar vacíos.")
        if len(text) > MAX_TURN_LENGTH:
            raise BadRequest("Un mensaje de la conversación es demasiado largo.")
        turns.append(Said(turn["role"], text))
    return message.strip(), turns, farm_context(body.get("farm_context"))
