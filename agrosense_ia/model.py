from dataclasses import dataclass, field

MODEL_NAME = "agrosense-riego-reglas-1.0"

DEFAULT_HUMIDITY_RANGE = (40.0, 80.0)
DEFAULT_PH_RANGE = (5.5, 7.0)

HOT_C = 30.0
COLD_C = 12.0
FROST_RISK_C = 2.0

MIN_MINUTES = 10
MAX_MINUTES = 90
MINUTES_PER_POINT = 1.5
PREVENTIVE_MINUTES = 10

CLEAR_MARGIN_POINTS = 15.0
MIN_CONFIDENCE = 0.55
MAX_CONFIDENCE = 0.95

# What the moisture of the last hours may add. These are rules with fixed thresholds, not a trained model: each one
# leaves its name in `factors`, and a person can follow why a recommendation came out as it did.
MIN_TREND_POINTS = 3
MIN_TREND_HOURS = 1.0
FALLING_POINTS_PER_HOUR = -1.0
RISING_POINTS_PER_HOUR = 1.0
HOURS_TO_ANTICIPATE = 6.0
RAIN_LIKELY_PERCENT = 70.0
RAIN_ENOUGH_MM = 5.0
RAIN_CAN_WAIT_POINTS = 10.0
STUCK_POINTS = 6
JUMP_POINTS = 25.0


@dataclass(frozen=True)
class Prediction:
    irrigate: bool
    duration_minutes: int
    recommendation: str
    confidence: float
    factors: list[str] = field(default_factory=list)
    model_used: str = MODEL_NAME


def moisture_trend(history: list[tuple[float, float]]) -> float | None:
    """Points per hour, by least squares over (hours_ago, value); None when the history is too short to say."""
    if len(history) < MIN_TREND_POINTS:
        return None
    times = [-hours_ago for hours_ago, _ in history]
    if max(times) - min(times) < MIN_TREND_HOURS:
        return None
    mean_time = sum(times) / len(times)
    mean_value = sum(value for _, value in history) / len(history)
    spread = sum((time - mean_time) ** 2 for time in times)
    return sum((time - mean_time) * (value - mean_value) for time, (_, value) in zip(times, history)) / spread


def moisture_anomalies(history: list[tuple[float, float]]) -> list[str]:
    values = [value for _, value in sorted(history, key=lambda point: -point[0])]
    found = []
    if len(values) >= STUCK_POINTS and max(values) == min(values):
        found.append("sensor_atascado")
    if any(abs(after - before) >= JUMP_POINTS for before, after in zip(values, values[1:])):
        found.append("salto_brusco")
    return found


def predict_irrigation(soil_moisture: float, air_temperature: float | None = None, ph: float | None = None,
                       humidity_range: tuple[float, float] = DEFAULT_HUMIDITY_RANGE,
                       ph_range: tuple[float, float] = DEFAULT_PH_RANGE,
                       moisture_history: list[tuple[float, float]] | None = None,
                       rain_probability: float | None = None, rain_mm: float | None = None,
                       moisture_gain_per_minute: float | None = None) -> Prediction:
    humidity_min, humidity_max = humidity_range
    ph_min, ph_max = ph_range
    if humidity_min >= humidity_max or ph_min >= ph_max:
        raise ValueError("A range needs its minimum below its maximum")

    target = (humidity_min + humidity_max) / 2
    hot = air_temperature is not None and air_temperature >= HOT_C
    cold = air_temperature is not None and air_temperature <= COLD_C
    factors = []

    if soil_moisture < humidity_min:
        missing = target - soil_moisture
        minutes = MIN_MINUTES + missing * MINUTES_PER_POINT * (1.25 if hot else 0.85 if cold else 1.0)
        if moisture_gain_per_minute:
            minutes = missing / moisture_gain_per_minute
            factors.append("duracion_segun_riegos_anteriores")
        duration = int(min(MAX_MINUTES, max(MIN_MINUTES, round(minutes))))
        irrigate = True
        margin = humidity_min - soil_moisture
        advice = (f"Riega ahora durante unos {duration} minutos: la humedad del suelo "
                  f"({_number(soil_moisture)} %) está por debajo del mínimo del cultivo "
                  f"({_number(humidity_min)} %).")
        factors.append("humedad_baja")
    elif soil_moisture > humidity_max:
        duration = 0
        irrigate = False
        margin = soil_moisture - humidity_max
        advice = (f"No riegues: la humedad del suelo ({_number(soil_moisture)} %) supera el máximo del "
                  f"cultivo ({_number(humidity_max)} %). Revisa el drenaje para evitar encharcamiento.")
        factors.append("humedad_alta")
    elif soil_moisture < target and hot:
        duration = PREVENTIVE_MINUTES
        irrigate = True
        margin = min(target - soil_moisture, air_temperature - HOT_C)
        advice = (f"Aplica un riego corto de {duration} minutos: la humedad ({_number(soil_moisture)} %) "
                  f"está dentro del rango, pero con {_number(air_temperature)} °C el suelo se seca rápido.")
        factors.append("riego_preventivo")
    else:
        duration = 0
        irrigate = False
        margin = soil_moisture - humidity_min
        advice = (f"No hace falta regar: la humedad del suelo ({_number(soil_moisture)} %) está dentro "
                  f"del rango del cultivo ({_number(humidity_min)}–{_number(humidity_max)} %).")
        factors.append("humedad_adecuada")

    notes = []
    history = moisture_history or []
    trend = moisture_trend(history)
    if trend is not None and trend <= FALLING_POINTS_PER_HOUR:
        factors.append("tendencia_a_la_baja")
        hours_left = (soil_moisture - humidity_min) / -trend
        if not irrigate and soil_moisture <= humidity_max and hours_left <= HOURS_TO_ANTICIPATE:
            irrigate = True
            duration = PREVENTIVE_MINUTES
            advice = (f"Aplica un riego corto de {duration} minutos: la humedad ({_number(soil_moisture)} %) baja "
                      f"{_number(-trend)} puntos por hora y llegaría al mínimo en unas {_number(hours_left)} horas.")
    elif trend is not None and trend >= RISING_POINTS_PER_HOUR:
        factors.append("tendencia_al_alza")
        if irrigate:
            notes.append(f"La humedad ya sube {_number(trend)} puntos por hora: comprueba si acaba de llover o regar.")

    rain_is_likely = (rain_probability is not None and rain_probability >= RAIN_LIKELY_PERCENT) or \
                     (rain_mm is not None and rain_mm >= RAIN_ENOUGH_MM)
    if rain_is_likely:
        factors.append("lluvia_prevista")
        if irrigate and soil_moisture >= humidity_min - RAIN_CAN_WAIT_POINTS:
            irrigate = False
            duration = 0
            advice = (f"Espera antes de regar: hay lluvia prevista y la humedad del suelo "
                      f"({_number(soil_moisture)} %) puede aguantar hasta entonces.")
        elif irrigate:
            notes.append("Hay lluvia prevista, pero el suelo está demasiado seco para esperarla.")

    if hot:
        factors.append("temperatura_alta")
        if irrigate:
            notes.append("Hazlo al amanecer o al atardecer para perder menos agua por evaporación.")
    elif air_temperature is not None and air_temperature <= FROST_RISK_C:
        factors.append("riesgo_de_helada")
        notes.append(f"Con {_number(air_temperature)} °C hay riesgo de helada: evita regar de noche.")
    elif cold:
        factors.append("temperatura_baja")

    if ph is None:
        pass
    elif ph < ph_min:
        factors.append("ph_acido")
        notes.append(f"El pH ({_number(ph)}) está por debajo del rango del cultivo: valora encalar.")
    elif ph > ph_max:
        factors.append("ph_alcalino")
        notes.append(f"El pH ({_number(ph)}) está por encima del rango del cultivo: valora una enmienda "
                     "acidificante.")

    confidence = MIN_CONFIDENCE + (MAX_CONFIDENCE - MIN_CONFIDENCE) * min(1.0, margin / CLEAR_MARGIN_POINTS)
    anomalies = moisture_anomalies(history)
    if anomalies:
        factors += anomalies
        confidence = MIN_CONFIDENCE
        notes.append("Las lecturas recientes de humedad parecen anómalas: revisa el sensor antes de fiarte de esto.")
    return Prediction(irrigate=irrigate, duration_minutes=duration,
                      recommendation=" ".join([advice, *notes]), confidence=round(confidence, 4),
                      factors=factors)


def _number(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")
