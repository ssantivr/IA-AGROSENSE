# IA-AGROSENSE

Servicio de predicción de riego de AgroSense. Es el servicio que el backend
([BACKEND-AGROSENSE](https://github.com/WilliamChavez-Grupo2b/BACKEND-AGROSENSE)) llama en
`AI_SERVICE_URL` cuando alguien pide una predicción para un cultivo
(`POST /api/crops/{id}/predictions`): recibe las últimas lecturas de humedad del suelo, temperatura del
aire y pH, y responde si conviene regar, cuánto tiempo y por qué.

Está escrito en Python 3.12 con la biblioteca estándar: no instala ninguna dependencia.

## Qué modelo usa

Un **modelo de reglas agronómicas** (`agrosense_ia/model.py`), no uno entrenado. No necesita datos de
entrenamiento y cada recomendación se puede explicar con la regla que la produjo:

| Situación | Decisión |
|---|---|
| Humedad por debajo del mínimo del cultivo | Regar. La duración crece con lo que falta para llegar a la mitad del rango: 10 minutos más 1,5 por punto, un 25 % más con calor (≥ 30 °C) y un 15 % menos con frío (≤ 12 °C), entre 10 y 90 minutos |
| Humedad dentro del rango, por debajo de la mitad, y calor | Riego corto preventivo de 10 minutos |
| Humedad dentro del rango | No regar |
| Humedad por encima del máximo | No regar y revisar el drenaje |

Además avisa, sin cambiar la decisión, cuando el pH sale del rango del cultivo o hay riesgo de helada
(≤ 2 °C). Si la petición no trae los rangos del cultivo usa los que el backend da por defecto: humedad
40–80 % y pH 5,5–7,0.

`confidence` (entre 0,55 y 0,95) indica lo lejos que está la lectura del límite donde cambiaría la
decisión: una lectura pegada al límite vale 0,55 y una a 15 puntos o más, 0,95. No es una probabilidad
estadística.

El asistente conversacional (texto y voz, con Google Gemini) no está aquí: vive en el backend
(`com.agrosense.backend.ai`) y en el frontend.

## Contrato

### `GET /` y `GET /health`

`GET /health` responde `200 {"status": "UP"}` y `GET /` dice qué servicio es. Las dos son públicas y
también responden a `HEAD`, que es lo que usan los monitores.

### `POST /predict/irrigation`

```json
{
  "id_crop": 7,
  "soil_moisture": 31.5,
  "air_temperature": 24,
  "ph": 6.4
}
```

| Campo | Tipo | Obligatorio | Rango |
|---|---|---|---|
| `soil_moisture` | número | sí | 0 a 100 (%) |
| `air_temperature` | número | no | −50 a 80 (°C) |
| `ph` | número | no | 0 a 14 |
| `id_crop` | entero | no | mayor que 0; se devuelve tal cual |
| `humidity_min`, `humidity_max` | número | no, pero juntos | 0 a 100, mínimo menor que máximo |
| `ph_min`, `ph_max` | número | no, pero juntos | 0 a 14, mínimo menor que máximo |

Respuesta `200`:

```json
{
  "id_crop": 7,
  "irrigate": true,
  "duration_minutes": 53,
  "recommendation": "Riega ahora durante unos 53 minutos: la humedad del suelo (31.5 %) está por debajo del mínimo del cultivo (40 %).",
  "confidence": 0.7767,
  "model_used": "agrosense-riego-reglas-1.0",
  "factors": ["humedad_baja"]
}
```

La decisión depende de la humedad del suelo, que es obligatoria. Sin temperatura o sin pH la decisión es
la misma pero sin los ajustes y avisos que aportarían: nunca se supone un valor que no se midió. Por eso
el backend no pide predicción para un cultivo que todavía no tiene lecturas de humedad del suelo.

El backend guarda `recommendation`, `confidence` y `model_used`; el resto es información adicional.
Valores de `factors`: `humedad_baja`, `humedad_alta`, `humedad_adecuada`, `riego_preventivo`,
`temperatura_alta`, `temperatura_baja`, `riesgo_de_helada`, `ph_acido`, `ph_alcalino`.

Errores, siempre como `{"message": "..."}`:

| Código | Cuándo |
|---|---|
| 400 | Falta un campo, no es un número, está fuera de rango o el cuerpo no es JSON |
| 401 | El servicio tiene `AI_SERVICE_TOKEN` y la petición no trae ese token |
| 404 | Ruta desconocida |
| 405 | Método equivocado (la cabecera `Allow` dice cuál usar) |
| 411 | Falta `Content-Length` |
| 413 | Cuerpo de más de 16 kB |
| 415 | El cuerpo no es `application/json` |

## Ejecutar

```bash
python -m agrosense_ia          # escucha en http://localhost:8000
```

| Variable | Descripción |
|---|---|
| `PORT` | Puerto (por defecto 8000) |
| `HOST` | Dirección en la que escucha (por defecto `0.0.0.0`) |
| `LOG_LEVEL` | Nivel del registro (por defecto `INFO`) |
| `AI_SERVICE_TOKEN` | Secreto que debe traer quien pida una predicción. Sin valor por defecto; sin él el servicio responde a cualquiera |

Con `AI_SERVICE_TOKEN` definido, `POST /predict/irrigation` exige la cabecera
`Authorization: Bearer <token>` y responde 401 sin ella. Úsalo siempre que el servicio quede accesible
desde internet. El token no se escribe en ningún archivo: llega por el entorno. El registro anota método,
ruta y código de cada petición, nunca las lecturas, las recomendaciones ni el token.

Con Docker:

```bash
docker build -t agrosense-ia .
docker run -p 8000:8000 agrosense-ia
```

## Conectar con el backend

En el backend, define `AI_SERVICE_URL` con la dirección de este servicio (por ejemplo
`http://localhost:8000` en local). `AI_IRRIGATION_PATH` ya vale `/predict/irrigation` por defecto. Sin
`AI_SERVICE_URL` el backend no pide predicciones.

Si el servicio tiene `AI_SERVICE_TOKEN`, define la misma variable con el mismo valor en el backend: lo
envía en cada llamada.

El backend manda las últimas lecturas del cultivo y sus propios límites de humedad y pH, de modo que la
recomendación se refiere a ese cultivo y no a uno promedio. El servicio no guarda ni lee datos: solo
calcula con los números que recibe.

El backend espera la respuesta 5 segundos (`AI_SERVICE_TIMEOUT_MS`). En un alojamiento gratuito que
duerme el servicio, la primera petición tras un rato sin uso puede tardar más: esa predicción no se
guarda y hay que pedirla de nuevo, o subir ese tiempo.

## Desplegar en Render

`render.yaml` describe el servicio: en Render, **New → Blueprint** y elige este repositorio. Render
construye el `Dockerfile`, comprueba `/health` y genera un `AI_SERVICE_TOKEN`. Después, en el servicio del
backend, define `AI_SERVICE_URL` con la dirección que Render le dé a este y `AI_SERVICE_TOKEN` con el
valor generado. Este archivo no se ha probado en Render todavía.

## Pruebas

```bash
python -m unittest
```

40 pruebas: las reglas del modelo y el servicio por HTTP real, con y sin token (validación, errores,
`HEAD` y peticiones simultáneas).

GitHub las ejecuta en cada push (`.github/workflows/tests.yml`).
