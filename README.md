# IA-AGROSENSE

Servicio de inteligencia artificial de AgroSense. Solo lo llama el backend
([BACKEND-AGROSENSE](https://github.com/WilliamChavez-Grupo2b/BACKEND-AGROSENSE)), en `AI_SERVICE_URL`, y
le devuelve la respuesta. Hace dos cosas:

- **Predicción de riego** (`POST /predict/irrigation`): recibe las últimas lecturas de humedad del suelo,
  temperatura del aire y pH de un cultivo, y responde si conviene regar, cuánto tiempo y por qué.
- **Asistente conversacional** (`POST /chat` y `POST /voice`): recibe la pregunta del usuario, la
  conversación anterior y la descripción de su finca que arma el backend, y responde con un modelo de
  lenguaje.

Está escrito en Python 3.12 con FastAPI. **No accede a la base de datos**: todo lo que sabe de una finca
llega en cada petición, ya limitado por el backend a los datos del usuario que pregunta.

## Seguridad

Todas las rutas, salvo `GET /` y `GET /health`, exigen la cabecera `X-Internal-Key` con el valor de
`INTERNAL_API_KEY`. Mientras el backend termina de pasar a esa cabecera también se acepta el token que ya
enviaba, `Authorization: Bearer <AI_SERVICE_TOKEN>`.

- El asistente (`/chat`, `/voice`) responde 401 a quien no traiga una de las dos, y también cuando el
  servicio no tiene ninguna configurada.
- La predicción conserva su comportamiento anterior: sin `INTERNAL_API_KEY` ni `AI_SERVICE_TOKEN` responde
  a cualquiera (útil en local); con una de las dos, exige la credencial.
- Los secretos se comparan en tiempo constante y nunca se escriben en el registro. El registro anota
  método, ruta y código de cada petición; ninguna ruta lleva secretos. De los proveedores solo se anota
  el código de estado, porque su texto de error puede citar la llave.

## Predicción de riego

### Qué modelo usa

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

## Asistente

### `POST /chat`

```json
{
  "message": "¿Necesito regar hoy?",
  "history": [
    {"role": "user", "text": "¿Cómo está el café?"},
    {"role": "assistant", "text": "La humedad del café está en 31 %..."}
  ],
  "farm_context": "Fecha y hora actual: 2026-10-16 08:00.\nDatos de la finca del usuario: 3 cultivos, ..."
}
```

| Campo | Obligatorio | Límite |
|---|---|---|
| `message` | sí | 1000 caracteres |
| `history` | no | 12 turnos de hasta 4000 caracteres; `role` es `user` o `assistant`. Al modelo solo van los **8 últimos** |
| `farm_context` | no | 8000 caracteres. Sin él, el modelo sabe que no hay datos de la finca |

Respuesta `200`: `{"reply": "...", "source": "model"}`.

### `POST /voice`

Formulario `multipart/form-data` con `audio` (la grabación, hasta 3 MB: WebM, Ogg, WAV, MP4, MP3, AAC o
FLAC), `transcript` (lo que el navegador ya entendió) o los dos, y `farm_context`. Con `transcript` se
responde ese texto y el audio no se envía a ningún proveedor.

Respuesta `200`: `{"reply": "...", "transcript": "lo que se entendió", "source": "model"}`.
`transcript` es `null` si el modelo no lo devolvió.

### Instrucciones del asistente

Están en `prompts/system-es.txt`, no en el código. A cada pregunta se le añade el `farm_context` que envía
el backend.

### Proveedores

Responde el primero que tenga llave, en este orden:

1. **Compatible con OpenAI** (`AI_CHAT_API_KEY`): Groq, OpenRouter, Mistral, OpenAI... `AI_CHAT_MODEL`
   admite varios modelos separados por comas: si uno se queda sin cuota (429) o falla (502), responde el
   siguiente. La voz se transcribe con `AI_TRANSCRIPTION_MODEL` y luego se responde como texto.
2. **Claude** (`ANTHROPIC_API_KEY`), por `POST https://api.anthropic.com/v1/messages` con `x-api-key` y
   `anthropic-version: 2023-06-01`. No transcribe audio: una grabación sin `transcript` recibe 503 con
   «Escribe tu pregunta».
3. **Gemini** (`GEMINI_API_KEY`), que escucha y responde la grabación en una sola llamada.

Sin ninguna llave, `/chat` y `/voice` responden **503** «El asistente de IA no está configurado todavía»,
y el backend responde entonces con sus datos (modo local, `source: "local"`).

### Errores

Siempre como `{"message": "..."}`, escrito para el usuario final:

| Código | Cuándo |
|---|---|
| 400 | Falta un campo, no es válido, está fuera de rango o el cuerpo no es JSON |
| 401 | Falta la credencial del servicio o no es válida |
| 404 / 405 | Ruta desconocida / método equivocado (la cabecera `Allow` dice cuál usar) |
| 411 / 413 / 415 | Falta `Content-Length` / cuerpo demasiado grande (16 kB en predicción, 128 kB en chat, 3 MB de audio) / tipo de contenido equivocado |
| 422 | El modelo se negó a responder esa pregunta |
| 429 | Todos los modelos configurados están sin cuota |
| 502 | El proveedor falló |
| 503 | No hay proveedor configurado, o no puede escuchar audio |
| 504 | El proveedor no respondió a tiempo (`AI_TIMEOUT_MS`) o no se pudo contactar |

## Ejecutar

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt      # en Windows: .venv\Scripts\pip
.venv/bin/python -m agrosense_ia               # escucha en http://localhost:8000
```

| Variable | Descripción |
|---|---|
| `PORT`, `HOST` | Puerto (8000) y dirección (`0.0.0.0`) en que escucha |
| `LOG_LEVEL` | Nivel del registro (`INFO`) |
| `INTERNAL_API_KEY` | Llave que el backend envía en `X-Internal-Key`. Sin valor por defecto |
| `AI_SERVICE_TOKEN` | Token anterior (`Authorization: Bearer`), aceptado durante la migración |
| `AI_CHAT_API_KEY`, `AI_CHAT_BASE_URL`, `AI_CHAT_MODEL`, `AI_TRANSCRIPTION_MODEL` | Proveedor compatible con OpenAI. Por defecto `https://api.groq.com/openai/v1`, `qwen/qwen3.8-27b` y `whisper-large-v3-turbo`. `AI_TRANSCRIPTION_MODEL` vacío desactiva la voz |
| `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`, `ANTHROPIC_EFFORT`, `ANTHROPIC_BASE_URL` | Claude. Por defecto `claude-haiku-5-5` y esfuerzo `low` |
| `GEMINI_API_KEY`, `GEMINI_MODEL`, `GEMINI_BASE_URL` | Gemini. Por defecto `gemini-flash-latest` |
| `AI_TIMEOUT_MS` | Espera máxima a un proveedor (20000) |
| `AI_MAX_OUTPUT_TOKENS` | Longitud máxima de la respuesta (2048) |

Las llaves llegan solo por el entorno; no se escriben en ningún archivo.

Con Docker:

```bash
docker build -t agrosense-ia .
docker run -p 8000:8000 -e INTERNAL_API_KEY=... agrosense-ia
```

La imagen corre con un usuario sin privilegios, lee el puerto de `PORT` y se comprueba a sí misma en
`/health`.

## Conectar con el backend

En el backend:

- `AI_SERVICE_URL`: la dirección de este servicio (en local, `http://localhost:8000`). Activa las
  predicciones de riego.
- `AI_SERVICE_KEY`: el mismo valor que `INTERNAL_API_KEY` aquí. `AI_SERVICE_TOKEN`, si se sigue usando,
  también igual en los dos.
- `AI_CHAT_VIA_SERVICE=true`: el asistente pasa por este servicio. Por defecto es `false` y el backend
  sigue respondiendo con su IA interna, para que el chat no cambie de proveedor sin decidirlo.
- `AI_SERVICE_TIMEOUT_MS`: cuánto espera el backend. En un alojamiento gratuito que duerme el servicio,
  la primera petición tras un rato sin uso puede tardar más de un minuto.

## Desplegar en Render

`render.yaml` describe el servicio: en Render, **New → Blueprint** y elige este repositorio. Render
construye el `Dockerfile`, comprueba `/health` y genera `INTERNAL_API_KEY` y `AI_SERVICE_TOKEN`. La llave
del proveedor se escribe a mano en el panel. Después, en el servicio del backend, define `AI_SERVICE_URL`,
`AI_SERVICE_KEY` y `AI_SERVICE_TOKEN` con esos valores.

El servicio de AgroSense está en `https://ia-agrosense.onrender.com` (plan gratuito: se duerme tras un
rato sin uso). Se creó con la API de Render y la misma configuración que describe `render.yaml`; el
archivo en sí, como blueprint, no se ha probado.

## Pruebas

```bash
pip install -r requirements-dev.txt
python -m pytest
```

- Las 40 de antes: las reglas del modelo y la predicción por HTTP real, con y sin token (validación,
  errores, `HEAD` y peticiones simultáneas).
- Las del asistente, con cada proveedor representado por un servidor HTTP local: cabeceras y cuerpo de
  cada uno, prioridad entre proveedores, cadena de modelos, 429, caída, tiempo agotado, negativas, voz,
  `X-Internal-Key` y que ningún secreto llegue al registro.

GitHub las ejecuta en cada push (`.github/workflows/tests.yml`).
