# RAPIRO Guardián — Documentación Técnica Completa

**TPI Intercátedra 5to año ISI — UCP 2026**

---

## Índice

1. [Arquitectura General del Sistema](#1-arquitectura-general)
2. [Red Neuronal Convolucional (CNN)](#2-red-neuronal-cnn)
3. [Pipeline de Inferencia en Tiempo Real](#3-pipeline-de-inferencia)
4. [Control Físico del Robot (Hardware)](#4-hardware)
5. [Módulo de Tutoría con IA](#5-tutoría-con-ia)
6. [Integración Cloud AWS](#6-cloud-aws)
7. [Variables y Parámetros Clave](#7-variables-y-parámetros)
8. [Flujo Completo de Datos](#8-flujo-completo)

---

## 1. Arquitectura General

El sistema tiene 4 capas bien definidas:

```
[Percepción] → [Clasificación] → [Actuación] → [Cloud]
   DroidCam       CNN TFLite      RAPIRO robot    AWS
                                  LEDs + servos   DynamoDB
                                  Voz (Polly)     S3 / IoT
```

Cada capa es independiente: la clasificación no sabe nada del robot, el cloud no sabe nada del hardware. Esto permite ejecutar el sistema en modo demo (sin robot) o sin cloud (`--no-cloud`).

---

## 2. Red Neuronal CNN

### 2.1 Arquitectura elegida: MobileNetV2

**¿Por qué MobileNetV2?**  
El sistema corre en una Raspberry Pi 2B (ARMv7, 1GB RAM). Las redes tipo ResNet50 o VGG16 requieren cientos de MB de RAM y tardan segundos en inferencia hasta en hardware moderno. MobileNetV2 fue diseñada específicamente para dispositivos móviles y embebidos:

- Usa **depthwise separable convolutions**: factoriza una convolución estándar en dos operaciones más baratas (depthwise + pointwise), reduciendo operaciones matemáticas ~8-9x
- Introduce **inverted residuals con linear bottlenecks**: bloques que expanden el canal, aplican la convolución depthwise, y comprimen de nuevo. El "invertido" es porque el residual conecta los bottlenecks (canales chicos) en lugar de los anchos
- Resultado: 3.4M parámetros vs 25M de ResNet50, pero con accuracy competitivo en tareas de clasificación visual

### 2.2 Capas de MobileNetV2

```
Input (224,224,3)
    ↓
Conv2D 32 filtros, stride 2           ← Primer capa, extrae features básicas
    ↓
Bottleneck blocks × 17                ← Núcleo de la red
  └─ Expand (t×canales) → DepthwiseConv → Project (reducir) → ReLU6
    ↓
Conv2D 1280 filtros                   ← Última convolución de features
    ↓
GlobalAveragePooling2D                ← Colapsa espacial a vector (1,1,1280)
    ↓
Dense 5 neuronas + Softmax            ← Clasificador final (5 clases)
    ↓
Output: probabilidades [p0,p1,p2,p3,p4]
```

**ReLU6**: variante de ReLU que clampea en 6 (`min(max(x,0), 6)`). Mejora cuantización INT8 porque acota los valores de activación en un rango finito.

**GlobalAveragePooling2D**: en lugar de aplanar el feature map (que perdería info espacial o sería muy grande), promedia cada canal espacialmente. Produce un vector compacto independiente del tamaño del input.

### 2.3 Transfer Learning

La red no se entrena desde cero. Se usa un modelo MobileNetV2 preentrenado en **ImageNet** (1.2M imágenes, 1000 clases) y se hace fine-tuning:

1. Se congela la base convolucional (pesos de ImageNet)
2. Se reemplaza la cabeza clasificadora (últimas capas Dense) por una nueva con 5 salidas
3. Se entrena solo la cabeza nueva con el dataset propio
4. Opcionalmente se descongelan las últimas capas de la base para un fine-tuning más fino

**¿Por qué transfer learning?**  
ImageNet ya "enseñó" a la red a detectar bordes, texturas, formas, caras. Esos features de bajo nivel son útiles para detectar si una persona sostiene un celular. Solo necesitamos enseñarle la tarea específica de distinguir 5 estados.

### 2.4 Las 5 Clases

| ID | Nombre | LED | Descripción |
|----|--------|-----|-------------|
| 0 | Estudiando | Verde | Persona presente, mirando hacia adelante |
| 1 | Usando celular | Amarillo | Persona con cabeza inclinada o celular visible |
| 2 | Puesto vacío | Rojo | Sin persona en frame |
| 3 | Confundido | (→ verde) | Gesto de duda, mapeado a "presente" |
| 4 | Aburrido | (→ verde) | Postura relajada, mapeado a "presente" |

Las clases 3 y 4 se mapean a "Estudiando" en el pipeline principal (son "persona presente pero con estado emocional"). Solo activan reacción física diferente en el módulo de tutoría.

### 2.5 Cuantización INT8

**¿Qué es cuantización?**  
El modelo original usa pesos en float32 (32 bits por valor). La cuantización convierte esos pesos a int8 (8 bits), reduciendo el tamaño del modelo ~4x y acelerando inferencia en hardware que no tiene FPU eficiente (como el ARM Cortex-A7 de la RPi 2B).

**Proceso de cuantización INT8:**
```
float32 valor ∈ [-1, 1]
    ↓ escalar con: value_int8 = round(value_float / scale) + zero_point
int8 valor ∈ [-128, 127]
```

Donde `scale` y `zero_point` son calibrados durante el proceso de exportación TFLite usando un dataset representativo.

**Cuantización en código (`classifier.py`):**
```python
# Al entrar: float32 → int8
in_det = self._input_details[0]
if in_det["dtype"] == np.int8:
    scale, zero_point = in_det["quantization"]
    if scale == 0:
        scale = 1.0 / 127.5
        zero_point = -1
    input_tensor = np.clip(
        np.round(input_tensor / scale + zero_point), -128, 127
    ).astype(np.int8)

# Al salir: int8 → float32
out_det = self._output_details[0]
if out_det["dtype"] in (np.int8, np.uint8):
    o_scale, o_zero = out_det["quantization"]
    output = (output.astype(np.float32) - o_zero) * o_scale
```

### 2.6 Formato TFLite

TFLite es el formato de inferencia de TensorFlow optimizado para edge. A diferencia del modelo Keras (.h5 o SavedModel), TFLite:
- Elimina nodos de entrenamiento (gradientes, optimizadores)
- Aplica fusión de operaciones (BatchNorm fusionado con Conv)
- Genera un flatbuffer binario compacto
- Usa un intérprete liviano (~1MB) sin necesidad de TF completo

**Backend en RPi 2B:**  
La RPi 2B no tiene soporte oficial de `tflite-runtime` en Python 3.13+. Se usa `tflite_ctypes.py` — un wrapper propio que llama directamente a `libtensorflow-lite.so` via ctypes. El clasificador prueba backends en orden:
```
tflite_runtime → ai_edge_litert → tensorflow → tflite_ctypes (fallback)
```

---

## 3. Pipeline de Inferencia

### 3.1 Captura de Video (DroidCam)

La cámara es un smartphone con DroidCam que expone un stream HTTP:
```
http://192.168.100.6:4747/video
```

OpenCV (`cv2.VideoCapture`) consume este stream. El problema es que OpenCV bufferiza frames internamente: si la inferencia tarda 5s, el frame leído sería el de 5s atrás.

**Solución — hilo drain:**
```python
# Un hilo daemon lee y descarta frames continuamente
# El frame más reciente siempre está disponible en _current_frame
def _drain(self):
    while self._running:
        ret, frame = self._cap.read()
        if ret:
            self._current_frame = frame
```

Así la inferencia siempre procesa el frame actual, no uno viejo.

### 3.2 Preprocesamiento

```python
# preprocessor.py
resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)   # OpenCV usa BGR, el modelo espera RGB
normalized = rgb.astype(np.float32) / 255.0       # [0,255] → [0,1]
tensor = np.expand_dims(normalized, axis=0)        # (224,224,3) → (1,224,224,3)
```

### 3.3 Histéresis (CONFIRM_CYCLES)

Para evitar reacciones ante detecciones ruidosas o frames atípicos, se confirma el estado N veces consecutivas antes de actuar:

```python
CONFIRM_CYCLES = 1  # actual: reacciona en el primer frame confirmado

if effective_class == pending_class_id:
    pending_count += 1
else:
    pending_class_id = effective_class
    pending_count = 1

if pending_count >= CONFIRM_CYCLES:
    rapiro.react(effective_class)
```

Con `CONFIRM_CYCLES=1` la respuesta es inmediata. Subir a 2 o 3 añade robustez pero agrega latencia de 1-2 ciclos de inferencia (~5-10s extra).

### 3.4 Umbral de Confianza

El modelo produce probabilidades para las 5 clases (suman 1.0). Solo se acepta la predicción si la clase con mayor probabilidad supera un umbral mínimo:

```python
MIN_CONFIDENCE = 0.20   # 20%
if confidence < self.min_confidence:
    return None  # predicción descartada
```

Un umbral bajo (20%) acepta casi todo. Un umbral alto (70%) descartaría muchas predicciones en condiciones de iluminación variable o ángulos no vistos durante entrenamiento.

---

## 4. Hardware

### 4.1 Raspberry Pi 2B

- **CPU**: ARM Cortex-A7 quad-core 900MHz
- **RAM**: 1GB LPDDR2
- **OS**: Raspberry Pi OS (Debian Bookworm, 64-bit)
- **Conexión**: WiFi (USB dongle) + SSH

### 4.2 RAPIRO — Comunicación Serial

El robot RAPIRO tiene un Arduino ATmega32U4 que controla 12 servos y LEDs.  
La comunicación es UART via GPIO pins:
- **Pin 14 (TX)** RPi → Arduino RX
- **Pin 15 (RX)** RPi → Arduino TX
- **Puerto**: `/dev/ttyAMA0`
- **Baud rate**: `57600` (crítico: a 9600 el Arduino ignora los comandos)

**Protocolo de comandos:**
```
#M{n}                          Movimiento preset (n=0-9)
#PR{rrr}G{ggg}B{bbb}T{ttt}    Color RGB de ojos (T en décimas de seg)
#PS{id:02d}A{ang:03d}T{ttt}   Servo individual
```

Ejemplo: `#PR200G000B000T300\r` → ojos rojos por 30 segundos.

**Por qué `\r` al final:** el firmware Arduino espera retorno de carro como terminador de comando.

### 4.3 LEDs GPIO

Los LEDs de estado son externos, conectados directo al RPi:

| Color | Pin GPIO (BCM) | Estado |
|-------|---------------|--------|
| Rojo | 17 | Ausente |
| Verde | 27 | Estudiando |
| Azul | 22 | Default/Escuchando |

**Ánodo común**: el cátodo de cada LED va al pin GPIO. `LOW` = LED encendido, `HIGH` = apagado. La librería `lgpio` maneja esto.

### 4.4 Audio Bluetooth

- **Speaker**: MM43BT (MAC: `E8:07:BF:00:63:6D`)
- **Stack**: PipeWire + bluez (Bluetooth moderno, reemplazó PulseAudio)
- **Reproducción**: `pw-play --target bluez_output.E8_07_BF_00_63_6D.1 archivo.mp3`
- **Variable necesaria**: `XDG_RUNTIME_DIR=/run/user/{uid}` para que pw-play encuentre la sesión PipeWire

---

## 5. Tutoría con IA

### 5.1 Flujo

```
Estado detectado (celular/ausente)
    ↓
¿Pasaron 90s desde última vez? (SPEAK_COOLDOWN)
    ↓ sí
Cargar material S3 (cargado al inicio en background)
    ↓
Construir prompt con contexto del tema estudiado
    ↓
Claude Haiku API → texto en español rioplatense
    ↓
¿El estado sigue siendo el mismo? (evita hablar si ya cambió)
    ↓ sí
Amazon Polly → síntesis de voz MP3 (Pedro neural, us-east-1)
    ↓
pw-play → speaker Bluetooth
```

### 5.2 Material de Estudio desde S3

Al arrancar, main.py lanza un hilo daemon que descarga el documento más reciente del S3:

```python
threading.Thread(
    target=lambda: _study_material.__setitem__(0, _load_study_material()),
    daemon=True
).start()
```

`_study_material` es una lista de 1 elemento `[""]` — patrón mutable para compartir estado entre hilos sin locks. El hilo escribe `_study_material[0] = texto`, el hilo principal lee `_study_material[0]`.

El documento se trunca a 2000 caracteres para no exceder el contexto del LLM.

### 5.3 Prompt Contextual

```python
def _build_prompt(class_id: int) -> str:
    material = _study_material[0]
    if class_id == CLASS_PHONE and material:
        return (
            f"Sos RAPIRO, un robot compañero de estudio. El estudiante está mirando el celular "
            f"en vez de estudiar. El tema que está estudiando es:\n\n{material[:800]}\n\n"
            "Decile algo corto y motivador relacionado al tema para que deje el celu y vuelva a estudiar. "
            "Máximo 2 oraciones, español rioplatense, sin markdown."
        )
```

El material de estudio se inyecta en el prompt → el LLM genera consejos específicos al tema en lugar de respuestas genéricas.

### 5.4 Amazon Polly — TTS

- **Voz**: `Pedro` — masculino latinoamericano, motor neural
- **Motor neural vs standard**: neural usa redes neuronales para síntesis más natural (prosodia variable, pausas naturales). Standard usa concatenación de fonemas grabados.
- **Región**: `us-east-1` — las voces neurales no están disponibles en `sa-east-1` (São Paulo)
- **Formato**: MP3, luego reproducido con `pw-play`

---

## 6. Cloud AWS

### 6.1 DynamoDB — Sesiones

**Tabla**: `rapiro_sessions_dev` (región: `sa-east-1`)

Cada predicción que supera el umbral genera un ítem:
```json
{
  "session_id": "uuid-v4",
  "timestamp": "2026-06-15T13:37:05Z",
  "class_id": 1,
  "label": "Usando celular",
  "confidence": 0.50,
  "latency_ms": 5351.0
}
```

**¿Por qué DynamoDB?**  
- Sin schema (cada evento puede tener campos diferentes en el futuro)
- Escala automáticamente
- Free tier cubre el volumen del proyecto
- SDK boto3 en Python es trivial

### 6.2 S3 — Documentos de Estudio

**Bucket**: `rapiro-user-documents-dev-442650748881`

El usuario sube documentos TXT/PDF desde el dashboard web. El RAPIRO los descarga al iniciar y los usa como contexto en los prompts de Claude.

Lógica de descarga del más reciente:
```python
objs = s3.list_objects_v2(Bucket=bucket).get("Contents", [])
latest = sorted(objs, key=lambda x: x["LastModified"], reverse=True)[0]
body = s3.get_object(Bucket=bucket, Key=latest["Key"])["Body"].read()
```

### 6.3 IoT Core — MQTT

El RPi publica eventos MQTT al endpoint de AWS IoT:
```
Topic: rapiro/events
Payload: {"class_id": 1, "label": "Usando celular", "confidence": 0.5}
```

Throttle: máximo 1 mensaje cada 5 segundos para no saturar el endpoint.  
AWS IoT Core actúa como broker MQTT serverless. Rules del IoT pueden rutear mensajes a Lambda, DynamoDB, SNS, etc.

### 6.4 EC2 + Dashboard

- **Instancia**: t2.micro, São Paulo (`54.94.201.85`)
- **Stack**: Docker + Flask API
- **Puerto**: 80 (HTTP)
- **Función**: interfaz web para subir documentos de estudio, ver historial de sesiones desde DynamoDB
- **Variables de entorno**: `S3_BUCKET`, `AWS_REGION` pasadas via `docker-compose.yml`

### 6.5 Amazon Polly

Servicio de TTS de AWS. Recibe texto, devuelve audio MP3 sintetizado. Se accede via boto3:
```python
resp = boto3.client("polly", region_name="us-east-1").synthesize_speech(
    Text=texto,
    VoiceId="Pedro",
    OutputFormat="mp3",
    Engine="neural"
)
audio_bytes = resp["AudioStream"].read()
```

### 6.6 Infraestructura como Código (Terraform)

Todos los recursos AWS están definidos en `terraform/`. Permite:
- Crear toda la infra con `terraform apply`
- Destruirla con `terraform destroy` (evitar costos cuando no se usa)
- Estado remoto en S3 con locking en DynamoDB (evita conflictos entre miembros del equipo)

---

## 7. Variables y Parámetros Clave

### 7.1 Parámetros del Modelo

| Variable | Valor | Efecto |
|----------|-------|--------|
| `MIN_CONFIDENCE` | 0.20 | Umbral mínimo de confianza para aceptar predicción. Más bajo = más sensible, más falsos positivos |
| `INPUT_SIZE` | (224, 224) | Tamaño del tensor de entrada. Fijo por la arquitectura MobileNetV2 |
| `NUM_CLASSES` | 5 | Estudiando, Celular, Vacío, Confundido, Aburrido |
| `num_threads` | 4 | Hilos de inferencia TFLite. RPi 2B tiene 4 cores → reduce latencia de 7.3s a ~3.5s |

### 7.2 Parámetros de Comportamiento

| Variable | Valor | Efecto |
|----------|-------|--------|
| `CONFIRM_CYCLES` | 1 | Detecciones consecutivas iguales antes de reaccionar. 1 = inmediato |
| `SPEAK_COOLDOWN` | 90s | Tiempo mínimo entre mensajes de voz para el mismo estado |
| `CAMERA_FPS` | 2 | FPS target de la cámara. El clasificador procesa más lento, el drain garantiza frame actual |
| `ABSENCE_ALERT_THRESHOLD_SEC` | 300 | 5 min de ausencia → log de alerta prolongada |
| `DO_NOT_DISTURB_DURATION_SEC` | 600 | 10 min de modo silencioso al presionar botón físico |

### 7.3 Parámetros de Hardware

| Variable | Valor | Descripción |
|----------|-------|-------------|
| `SERIAL_PORT` | `/dev/ttyAMA0` | Puerto UART GPIO del RPi (pines 14/15) |
| `SERIAL_BAUD_RATE` | `57600` | Baud rate del Arduino. A 9600 los comandos son ignorados |
| `LED_RED_PIN` | 17 | GPIO BCM pin rojo (ánodo común: LOW=ON) |
| `LED_GREEN_PIN` | 27 | GPIO BCM pin verde |
| `LED_BLUE_PIN` | 22 | GPIO BCM pin azul |
| `CAMERA_URL` | `http://192.168.100.6:4747/video` | Stream HTTP de DroidCam |

### 7.4 Parámetros Cloud

| Variable | Descripción |
|----------|-------------|
| `AWS_REGION` | `sa-east-1` — región principal (São Paulo) |
| `AWS_DYNAMODB_TABLE` | `rapiro_sessions_dev` |
| `S3_BUCKET` | `rapiro-user-documents-dev-442650748881` |
| `POLLY_VOICE` | `Pedro` — voz masculina latinoamericana neural |
| `POLLY_ENGINE` | `neural` — síntesis más natural |
| `POLLY_REGION` | auto: `us-east-1` si engine=neural (sa-east-1 no soporta neural) |
| `LLM_MODEL` | `claude-haiku-4-5` — modelo Anthropic rápido y económico |
| `MQTT_THROTTLE_SECONDS` | 5s entre mensajes IoT |

---

## 8. Flujo Completo de Datos

```
┌─────────────────────────────────────────────────────────────────┐
│                         ARRANQUE                                 │
│  1. Cargar .env → settings.py                                    │
│  2. Conectar serial /dev/ttyAMA0 @ 57600 → Arduino              │
│  3. Inicializar GPIO (pines 17,27,22)                            │
│  4. Cargar modelo TFLite INT8 con 4 threads                      │
│  5. Conectar DynamoDB + IoT Core                                 │
│  6. [Hilo daemon] Descargar documento de S3 → _study_material    │
│  7. Abrir stream DroidCam + hilo drain                           │
└─────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                    LOOP PRINCIPAL (cada ~4-5s)                   │
│                                                                  │
│  frame ← DroidCam (siempre el más reciente via drain)            │
│      ↓                                                           │
│  resize(224,224) + BGR→RGB + /255.0 + expand_dims               │
│      ↓                                                           │
│  float32→int8 (dequantize con scale/zero_point)                  │
│      ↓                                                           │
│  TFLite.invoke() → int8 output                                   │
│      ↓                                                           │
│  int8→float32 (dequantize output)                                │
│      ↓                                                           │
│  argmax(probs) → class_id, confidence                            │
│      ↓                                                           │
│  ¿confidence < MIN_CONFIDENCE? → descartar                       │
│      ↓                                                           │
│  Mapear: clase 3,4 → Estudiando | clase 1 → Celular | 2 → Vacío │
│      ↓                                                           │
│  Histéresis: pending_count >= CONFIRM_CYCLES (1)?                │
│      ↓ sí                                                        │
│  ¿Estado cambió desde last_class_id?                             │
│      ↓ sí                                                        │
│  ┌── rapiro.react(class_id) ──────────────────────────┐         │
│  │   GPIO LED → color por estado                       │         │
│  │   Serial → animación servo (sacudir/buscar/asentir) │         │
│  │   Timer(4.5s) → restaurar LED persistente           │         │
│  └─────────────────────────────────────────────────────┘         │
│      ↓                                                           │
│  ¿class_id != Estudiando Y cooldown >= 90s?                      │
│      ↓ sí                                                        │
│  [Hilo daemon] _speak_for_state(class_id):                       │
│      → Claude Haiku API (prompt + material S3)                   │
│      → ¿Estado sigue igual? sí                                   │
│      → Polly TTS → MP3                                           │
│      → pw-play → Bluetooth MM43BT                                │
│      ↓                                                           │
│  DynamoDB.put_item(session_id, timestamp, class_id, conf, lat)   │
│  IoT Core.publish(topic=rapiro/events, payload=json)             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Decisiones de Diseño Destacadas

**¿Por qué hilos daemon para TTS?**  
El habla tarda 10-15s (Claude API + Polly + reproducción). Si fuera síncrono, el robot dejaría de detectar durante ese tiempo. El hilo daemon permite que el loop principal siga clasificando mientras habla.

**¿Por qué `_current_class: list = [-1]`?**  
Python no permite que una función interna reasigne una variable del scope exterior con `=`. Una lista es mutable: el hilo TTS puede leer `_current_class[0]` para verificar si el estado sigue siendo el mismo antes de reproducir audio. Si el estudiante ya volvió a estudiar, el audio se cancela.

**¿Por qué boto3 directo y no AWS SDK alternativo?**  
boto3 es el SDK oficial de Python para AWS. Tiene soporte nativo para todos los servicios usados (Polly, DynamoDB, S3, IoT). No hay razón para agregar dependencias.

**¿Por qué no guardar el modelo en S3 y descargarlo?**  
Para el prototipo, el modelo está en el repo (3MB). En producción se podría versionar en S3 y actualizar sin re-deployar el código.
