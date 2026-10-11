"""The irrigation model: agronomic rules over the latest soil moisture, air temperature and pH.

It is a rule-based model, not a trained one. It needs no data set and every recommendation can be traced
to the rule that produced it. ``confidence`` says how far the reading is from the point where the
decision would change; it is not a statistical probability.
"""

from dataclasses import dataclass, field

MODEL_NAME = "agrosense-riego-reglas-1.0"

# The ranges the backend gives a crop that was registered without its own (see its Crop entity).
DEFAULT_HUMIDITY_RANGE = (40.0, 80.0)
DEFAULT_PH_RANGE = (5.5, 7.0)

# Above this air temperature water evaporates fast; below the other one the plant takes up little.
HOT_C = 30.0
COLD_C = 12.0
FROST_RISK_C = 2.0

MIN_MINUTES = 10
MAX_MINUTES = 90
# Minutes of irrigation per point of soil moisture missing to reach the middle of the crop's range.
MINUTES_PER_POINT = 1.5
PREVENTIVE_MINUTES = 10

# A reading this many points away from the limit that decides counts as a clear case.
CLEAR_MARGIN_POINTS = 15.0
MIN_CONFIDENCE = 0.55
MAX_CONFIDENCE = 0.95


@dataclass(frozen=True)
class Prediction:
    irrigate: bool
    duration_minutes: int
    recommendation: str
    confidence: float
    factors: list[str] = field(default_factory=list)
    model_used: str = MODEL_NAME


def predict_irrigation(soil_moisture: float, air_temperature: float | None = None, ph: float | None = None,
                       humidity_range: tuple[float, float] = DEFAULT_HUMIDITY_RANGE,
                       ph_range: tuple[float, float] = DEFAULT_PH_RANGE) -> Prediction:
    """Whether to irrigate now, for how long, and why.

    ``soil_moisture`` is a percentage, ``air_temperature`` is in °C. The decision rests on the soil
    moisture; a crop without a temperature or a pH reading gets the same decision without the
    adjustments those would bring, never one based on an assumed value. The ranges are the crop's own
    limits; both must have their minimum below their maximum.
    """
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
    return Prediction(irrigate=irrigate, duration_minutes=duration,
                      recommendation=" ".join([advice, *notes]), confidence=round(confidence, 4),
                      factors=factors)


def _number(value: float) -> str:
    """64.0 reads as 64 and 64.74 stays as it is."""
    return f"{value:.2f}".rstrip("0").rstrip(".")
