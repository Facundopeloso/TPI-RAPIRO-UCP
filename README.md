# RAPIRO Guardián de Distracción

Sistema de monitoreo conductual y tutoría adaptativa para sesiones de estudio, construido sobre el robot **RAPIRO** y una Raspberry Pi 2B+.

**TPI Intercátedra 5to año — ISI — Universidad de la Cuenca del Plata — 2026**

| Integrante | Rol | Responsabilidades |
|---|---|---|
| Lucas Quintana | Líder Técnico / ML Engineer | Modelo CNN, pipeline de inferencia, coordinación |
| Stiven Monsalvo | Cloud & DevOps Engineer | AWS, Terraform IaC, dashboard, CloudWatch |
| Facundo Peloso | Hardware & Tutor Module | RAPIRO (LEDs/servos), módulo de tutoría LLM |

---

## Índice

1. [Para qué fue pensado](#1-para-qué-fue-pensado)
2. [Cómo funciona — visión general](#2-cómo-funciona--visión-general)
3. [Hardware real utilizado (leer antes de implementar)](#3-hardware-real-utilizado-leer-antes-de-implementar)
4. [Estado actual de las 5 clases (importante)](#4-estado-actual-de-las-5-clases-importante)
5. [Estructura del proyecto](#5-estructura-del-proyecto)
6. [La red neuronal (CNN)](#6-la-red-neuronal-cnn)
7. [Cómo se entrena el modelo](#7-cómo-se-entrena-el-modelo)
8. [Para qué usamos Google Drive](#8-para-qué-usamos-google-drive)
9. [Para qué usamos AWS](#9-para-qué-usamos-aws)
10. [El tutor con IA (Claude)](#10-el-tutor-con-ia-claude)
11. [El dashboard web](#11-el-dashboard-web)
12. [Reacciones físicas de RAPIRO](#12-reacciones-físicas-de-rapiro)
13. [Variables de entorno](#13-variables-de-entorno)
14. [Guía paso a paso — instalación y uso](#14-guía-paso-a-paso--instalación-y-uso)
15. [Tests](#15-tests)
16. [Limitaciones conocidas](#16-limitaciones-conocidas)
17. [Referencias](#17-referencias)

---

## 1. Para qué fue pensado

Cuando un estudiante se sienta a estudiar solo, no hay nadie que note si se distrajo con el celular, si se levantó y no volvió, o si lleva media hora sin entender el tema. RAPIRO Guardián cubre ese rol: una cámara observa la sesión de estudio, una red neuronal interpreta lo que está pasando, y el robot reacciona en consecuencia — con gestos físicos, con un mensaje habl ado generado por IA, y guardando un historial en la nube para que se pueda repasar después.

La idea no es vigilar para "castigar", sino actuar como un compañero de estudio que:
- Nota cuando te distrajiste y te lo recuerda con buena onda.
- Si no estás en tu lugar mucho tiempo, manda una alerta.
- Si pedís ayuda por voz ("RAPIRO ayuda"), puede explicarte un tema o tomarte un quiz usando tus propios apuntes (PDF que subís al dashboard).

## 2. Cómo funciona — visión general

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RASPBERRY PI 2B+ (512MB RAM)                     │
│                                                                     │
│  ┌──────────────┐    ┌────────────────┐    ┌──────────────────────┐ │
│  │  PERCEPCIÓN  │───▶│ CLASIFICACIÓN  │───▶│     ACTUACIÓN        │ │
│  │  camera.py   │    │  classifier.py │    │  rapiro.py           │ │
│  │  OpenCV      │    │  TFLite INT8   │    │  LEDs + 12 servos    │ │
│  │  ~2 fps      │    │  no medido aún │    │  Arduino via UART    │ │
│  └──────────────┘    └────────────────┘    └──────────────────────┘ │
│                              │                                      │
│                    ┌─────────▼──────────┐                          │
│                    │   TUTOR (Claude)    │                          │
│                    │  Claude Haiku 4.5   │                          │
│                    │  TF-IDF + tu PDF    │                          │
│                    └─────────┬──────────┘                          │
└──────────────────────────────┼──────────────────────────────────────┘
                               │ boto3 (HTTPS)
              ┌────────────────▼─────────────────────────┐
              │               AWS CLOUD                   │
              │  DynamoDB (historial) · S3 (documentos)   │
              │  EC2 + Docker (dashboard web)             │
              │  IoT Core + Lambda + SNS (alertas, opc.)  │
              └───────────────────────────────────────────┘
```

**Ciclo de un frame:** (la tabla de latencias de la sección 6 es una cifra de referencia bibliográfica para Raspberry Pi 4; en la Pi 2B+ real del proyecto, mucho más limitada, no se midió la latencia exacta — esperar valores notablemente más altos, ver sección 3)

```
Cámara (frame BGR) → resize 224×224 + normalizar → cuantizar a INT8
    → TFLite Interpreter.invoke() → clase + confianza
    → histéresis (confirmar antes de reaccionar)
    → rapiro.react(clase)            — LEDs + servos
    → si celular/ausente: Claude Haiku genera un mensaje corto → se reproduce por voz
    → se guarda el evento en DynamoDB
```

`main.py` es el punto de entrada en la Raspberry Pi con hardware real. `demo.py` es la versión sin hardware (cámara + ventana con un RAPIRO dibujado en pantalla) para mostrar el funcionamiento en cualquier PC. `python -m pytest` corre los tests unitarios sin necesitar ni cámara ni modelo real.

## 3. Hardware real utilizado (leer antes de implementar)

El proyecto se armó y se prueba sobre una **Raspberry Pi 2B+** — no una Pi 4. Esto importa para cualquiera que quiera replicar o continuar el proyecto:

| | Raspberry Pi 2B+ (la real) | Raspberry Pi 4 (la que asumen muchos tutoriales) |
|---|---|---|
| RAM | **512 MB** | 2-8 GB |
| CPU | Cortex-A7 quad-core ~900MHz | Cortex-A72 quad-core 1.5GHz |
| USB | 4x USB 2.0 | 2x USB 2.0 + 2x USB 3.0 |

**Por qué importa:**

- **512MB de RAM es muy poco.** No entra instalar y correr TensorFlow completo (~500MB sólo la librería) junto con OpenCV, boto3 y el resto del pipeline sin quedarse sin memoria o empezar a usar swap (que en una SD card es lentísimo). Por eso el clasificador (`src/classification/classifier.py`) intenta primero `tflite_runtime` (~7MB) y, si no está disponible, cae a un intérprete TFLite propio escrito en ctypes puro (`tflite_ctypes.py`) — pensado específicamente para no depender de instalar TensorFlow en un equipo con tan poca memoria.
- **La latencia de inferencia va a ser sensiblemente mayor** que los ~150-250ms que se citan en la sección 6 (esos números son de literatura sobre Raspberry Pi 4). No se hizo un benchmark formal en la 2B+ real; quien implemente esto debería medirlo en su propio equipo antes de prometer tiempos de respuesta.
- **La fuente de alimentación es de 12V** y tiene un presupuesto de corriente limitado para los puertos USB. En la práctica, **conectar más de 3 dispositivos por USB a la vez (cámara, hub, módulos extra, etc.) genera problemas de alimentación** — reinicios inesperados de la Pi, cámara que se desconecta sola, o el Arduino de los servos perdiendo el puerto serie. Si hace falta conectar más periféricos, conviene un hub USB con alimentación propia en vez de sumarlos todos a los puertos de la Pi.

Si en algún momento se migra a una Raspberry Pi 4 u otro equipo con más RAM, se puede instalar `tensorflow` completo sin problema y los tiempos de inferencia deberían acercarse a los de la tabla de la sección 6 — pero mientras el hardware sea la 2B+, hay que diseñar y probar pensando en estas limitaciones.

## 4. Estado actual de las 5 clases (importante)

La CNN está entrenada para reconocer **5 estados**: Estudiando, Usando celular, Puesto vacío, Confundido y Aburrido. Sin embargo, **hoy el pipeline en producción (`main.py` y `demo.py`) sólo actúa sobre 3 estados efectivos**:

| Clase detectada por la CNN | Estado efectivo usado por el robot |
|---|---|
| 0 — Estudiando | Estudiando (verde) |
| 1 — Usando celular | Celular (amarillo) |
| 2 — Puesto vacío | Ausente (rojo) |
| 3 — Confundido | **se agrupa como Estudiando** |
| 4 — Aburrido | **se agrupa como Estudiando** |

**Por qué:** todavía no hay suficientes fotos de "Confundido" y "Aburrido" en el dataset como para que la red las distinga de forma confiable de las otras clases. En vez de arriesgar falsos positivos en la presentación, esas dos clases quedan reconocidas por la CNN pero sin acción asociada — es un estado intencional, no un bug. El código ya tiene los métodos `react_confused()` / `react_bored()` y la lógica de re-explicación/quiz listos en `rapiro.py` y `tutor.py`; conectarlas al loop principal es trabajo futuro, una vez que el dataset de esas dos clases sea suficiente.

## 5. Estructura del proyecto

```
TPI-RAPIRO-UCP/
├── main.py                      # Orquestador principal (Raspberry Pi + hardware real)
├── demo.py                      # Pipeline completo sin hardware (PC, ventana OpenCV)
├── requirements.txt
├── .env.example
│
├── config/
│   └── settings.py              # Configuración centralizada (lee .env)
│
├── src/
│   ├── perception/
│   │   └── camera.py            # Captura de video (USB o IP/DroidCam)
│   ├── classification/
│   │   ├── preprocessor.py      # Resize 224×224, BGR→RGB, normalizar
│   │   ├── classifier.py        # Carga el .tflite y corre inferencia
│   │   └── tflite_ctypes.py     # Intérprete TFLite alternativo (sin TF/tflite-runtime)
│   ├── actuation/
│   │   ├── led_controller.py    # LEDs RGB (GPIO)
│   │   ├── servo_controller.py  # 12 servos vía Arduino/UART
│   │   └── rapiro.py            # Fachada: une LEDs + servos en "reacciones"
│   ├── tutoring/
│   │   ├── document_processor.py # Carga PDF/TXT y recupera contexto (TF-IDF)
│   │   ├── tutor.py              # Claude Haiku: responde, explica, genera/evalúa quiz
│   │   ├── voice_listener.py     # Hotword "RAPIRO ayuda" + captura de preguntas (mic)
│   │   └── polly_tts.py          # Texto a voz con Amazon Polly
│   ├── cloud/
│   │   ├── boto3_publisher.py    # Escribe eventos directo en DynamoDB (vía boto3)
│   │   └── iot_publisher.py      # Alternativa: publica por MQTT a AWS IoT Core
│   └── dashboard/
│       └── api.py                # Dashboard web (FastAPI), corre en la EC2
│
├── scripts/
│   ├── collect_dataset.py        # Captura fotos con la webcam + sync a Google Drive
│   └── train_model.py            # Entrena la CNN localmente
├── notebooks/
│   └── train_colab.ipynb         # Entrena la CNN en Google Colab (GPU gratis)
│
├── dataset/                      # Fotos de entrenamiento (no se commitean, salvo .gitkeep)
│   └── class_0..4/
├── models/
│   └── mobilenetv2_int8.tflite   # Modelo entrenado (sí se commitea, ~3.5MB)
│
├── tests/
│   └── test_classifier.py        # 6 tests unitarios (pytest + mocks)
│
├── lambda/
│   ├── handler.py                # Procesa eventos de IoT Core → DynamoDB + alerta SNS
│   └── inference_handler.py      # Inferencia de imágenes en la nube (alternativa sin RPi)
│
└── terraform/                    # Infraestructura AWS como código
    ├── main.tf
    ├── variables.tf
    └── outputs.tf
```

## 6. La red neuronal (CNN)

**Arquitectura:** MobileNetV2 (transfer learning) + cabeza de clasificación propia.

```
Input: imagen 224×224×3 (RGB, valores [0,1])
    │
    ▼ MobileNetV2 (pesos ImageNet, ~154 capas internas, 17 bloques bottleneck)
    │   salida: 7×7×1280 feature maps
    ▼ GlobalAveragePooling2D            → vector de 1280
    ▼ Dense(256, activation='relu')
    ▼ Dropout(0.4)
    ▼ Dense(5, activation='softmax')    → 5 probabilidades, suman 1.0
    │
Output: [P(estudiando), P(celular), P(ausente), P(confundido), P(aburrido)]
```

**Por qué MobileNetV2 y no algo más grande (ResNet50, VGG16):** está diseñado para correr en hardware limitado. Usa convoluciones "depthwise separable" que reducen drásticamente la cantidad de cálculos sin perder mucha precisión.

| | MobileNetV2 | ResNet50 | VGG16 |
|---|---|---|---|
| Parámetros | ~3.4M | ~25M | ~138M |
| Tamaño cuantizado (INT8) | ~3.5 MB | ~25 MB | ~135 MB |
| Latencia en Raspberry Pi 4 (referencia bibliográfica, no la Pi 2B+ real del proyecto — ver sección 3) | ~150-250 ms | >1000 ms | >3000 ms |

**Parámetros entrenables:** de los ~3.4M totales, sólo ~330K se entrenan en la fase 1 (la cabeza nueva); el resto son los pesos de ImageNet que se reutilizan.

## 7. Cómo se entrena el modelo

### Transfer learning en dos fases

- **Fase 1 (siempre):** la base de MobileNetV2 queda congelada (no se modifica). Sólo se entrena la cabeza nueva (`Dense(256)` + `Dense(5)`). Rápido, y ya da buen resultado.
- **Fase 2 (opcional, flag `--fine-tune`):** se descongelan las últimas 30 capas del base y se reentrenan con un learning rate muy bajo (1e-5) para afinar las características al dominio específico (estudiantes frente a una webcam), sin destruir lo aprendido de ImageNet.

### Dataset actual

Estado real del dataset combinado del equipo (suma de lo aportado por los 3 integrantes, sincronizado vía Drive — ver sección 8):

| Clase | Fotos actuales |
|---|---|
| 0 — Estudiando | 1005 |
| 1 — Usando celular | 1110 |
| 2 — Puesto vacío | 390 |
| 3 — Confundido | 0 (sin recolectar aún) |
| 4 — Aburrido | 0 (sin recolectar aún) |

"Puesto vacío" tiene notablemente menos fotos que las otras dos clases activas (390 contra ~1000+) — vale la pena seguir sumando fotos de esa clase para equilibrar el dataset antes del entrenamiento final.

### Augmentación de datos

Aun con un dataset de este tamaño, se generan variantes automáticas de cada foto durante el entrenamiento: rotación (±15°), desplazamiento, zoom (±15%), espejo horizontal, brillo (±30%). Esto ayuda a que el modelo generalice mejor y no memorice "esta persona con esta luz" en vez de aprender el concepto real — es más importante todavía para la clase con menos fotos ("Puesto vacío").

### Cuantización INT8

El modelo se entrena en `float32` y se convierte a `INT8` para la Raspberry Pi:

```
modelo float32 (~13MB) → calibración con 200 imágenes reales del dataset → modelo INT8 (~3.5MB)
```

Resultado: 4x menos tamaño y notablemente más rápido en CPU ARM (la cifra de "2-3x más rápido" es la que cita la literatura para Raspberry Pi 4; en la Pi 2B+ real del proyecto el beneficio relativo de cuantizar sigue aplicando, pero la velocidad absoluta es menor — ver sección 3), con una caída de precisión típicamente menor al 1%.

### Dos formas de entrenar

1. **Google Colab (recomendado):** abrir `notebooks/train_colab.ipynb`, activar GPU T4, "Ejecutar todo". Tarda 5-10 minutos y el `.tflite` queda guardado en Drive (`Mi Drive/RAPIRO_models/`). No requiere instalar nada localmente.
2. **Local (`scripts/train_model.py`):** lee imágenes de `dataset/class_0/` … `class_4/` en la carpeta del proyecto, entrena, cuantiza y guarda en `models/mobilenetv2_int8.tflite`. Mucho más lento sin GPU.

```bash
python scripts/train_model.py                       # sólo cabeza
python scripts/train_model.py --epochs 20 --fine-tune  # mejor accuracy, más lento
```

Cualquiera de las dos formas necesita las fotos **descargadas localmente primero** — entrenar lee de la carpeta `dataset/`, no de Drive directamente.

## 8. Para qué usamos Google Drive

Drive resuelve un problema de equipo: los 3 integrantes capturan fotos del dataset por separado (cada uno con su propia cámara/cara/ambiente para que el modelo generalice mejor), y hace falta juntar todas esas fotos en un solo lugar antes de entrenar.

`scripts/collect_dataset.py --upload` sube automáticamente las fotos nuevas de cada clase a una carpeta de Drive compartida (vía Google Drive API, autenticación OAuth con `credentials.json`). Cada clase tiene su propia carpeta (ver `DRIVE_FOLDER_IDS` en el script). El script evita subir duplicados comparando contra lo que ya existe en la carpeta de Drive.

Carpeta compartida del equipo: https://drive.google.com/drive/folders/1ULuPEGf_U_qH5OwrI1ps8kk1pHdq8mrP

El notebook de Colab (`train_colab.ipynb`) también usa Drive: lee el dataset combinado y, al terminar de entrenar, guarda el `.tflite` resultante en `Mi Drive/RAPIRO_models/` para que cualquiera del equipo lo pueda bajar sin pasar por git.

**Drive es sólo para compartir fotos y el modelo entrenado entre el equipo — no es parte del sistema en producción.** La Raspberry Pi nunca se conecta a Drive; usa el `.tflite` que se copia manualmente o vía `scp`.

## 9. Para qué usamos AWS

AWS le da al sistema lo que la Raspberry Pi sola no puede ofrecer: un historial accesible desde cualquier lado, alertas, y un lugar donde guardar los documentos de estudio para el tutor.

| Servicio | Para qué se usa | Costo/mes aprox. |
|---|---|---|
| **DynamoDB** | Guarda cada evento de clasificación (clase, confianza, timestamp, latencia) con TTL de 30 días | $0 (free tier) |
| **S3** | Guarda los PDF/TXT que el usuario sube desde el dashboard, para que el tutor los use como contexto | ~$0.12 |
| **EC2 + Docker** | Corre el dashboard web (FastAPI) 24/7, accesible por IP pública | ~$1.00 |
| **IoT Core + Lambda + SNS** | Camino alternativo: si la RPi publica por MQTT en vez de boto3 directo, una Lambda guarda el evento y, si la clase es "ausente", manda un email de alerta | $0 (free tier) + ~$0.08 |
| **CloudWatch** | Logs y alarmas (ej. errores de Lambda) | ~$0.50 |
| **Total estimado** | | **~$1.70/mes** |

**Dos caminos para publicar eventos (ambos existen en el código):**
- `src/cloud/boto3_publisher.py` — el que usa `main.py` hoy: escribe directo a DynamoDB con boto3, sin pasar por Lambda. Simple y es lo que está activo.
- `src/cloud/iot_publisher.py` — publica por MQTT con certificados X.509 (mTLS) a AWS IoT Core, que dispara la Lambda (`lambda/handler.py`), la cual guarda en DynamoDB y manda la alerta SNS si el puesto está vacío. Es el camino "serverless puro", pensado para cuando se quiera desacoplar la Pi de AWS directamente.

**Seguridad:** la EC2 del dashboard usa un **IAM Instance Profile** (sin claves hardcodeadas) con permisos mínimos: lectura de DynamoDB y lectura/escritura de S3 sólo en el prefijo de documentos. Todo el estado de Terraform vive en un bucket S3 con lock, así el equipo puede aplicar cambios sin pisarse.

```bash
cd terraform
terraform init
terraform plan -var-file="terraform.tfvars"
terraform apply -var-file="terraform.tfvars"
```

> `terraform.tfvars` no se commitea (tiene el email de alertas y la clave SSH pública). Cada quien crea el suyo localmente.

## 10. El tutor con IA (Claude)

El tutor usa la **API de Anthropic (Claude Haiku 4.5)**, el modelo más económico de la familia Claude — elegido a propósito para que las pruebas no salgan caras.

**Recuperación de contexto (TF-IDF):** cuando subís un PDF al dashboard, `document_processor.py` lo divide en fragmentos y, para cada pregunta, busca los 3 fragmentos más relevantes usando TF-IDF (frecuencia de término / frecuencia inversa de documento) — el mismo principio que el índice de un libro. No usamos embeddings porque para un solo documento y poco volumen, TF-IDF alcanza y no agrega una dependencia de cientos de MB.

**Modos del tutor** (`src/tutoring/tutor.py`):
- `answer()` — responde una pregunta libre con el contexto del documento.
- `explain_with_example()` — explica un tema con un ejemplo de la vida real.
- `generate_quiz()` / `evaluate_answer()` — genera preguntas multiple-choice y corrige.
- `study_session()` — combina explicación + quiz en un solo flujo.
- `audio_study_session()` / `listen_for_answer()` — versión hablada del flujo anterior (requiere micrófono).

**Texto a voz (cascada de fallback):**
1. **Amazon Polly** (voz "Conchita" en español) — mejor calidad, requiere AWS.
2. **gTTS + pygame** — gratis, requiere internet, usado en Linux/RPi cuando no hay AWS.
3. **pyttsx3** — sin internet, motor local (usado en Windows para pruebas).

**Hotword por voz ("RAPIRO ayuda"):** `voice_listener.py` mantiene el micrófono abierto en segundo plano escuchando ese hotword con `SpeechRecognition`. Hoy funciona y está probado vía `test_tutor.py --voice`, pero **todavía no está conectado a `main.py`** — en el pipeline real de la Raspberry Pi el hotword no se dispara solo; es la siguiente integración pendiente.

## 11. El dashboard web

FastAPI + Uvicorn, corre en Docker sobre la EC2.

| Método | Ruta | Qué hace |
|---|---|---|
| GET | `/` | Dashboard (HTML embebido, dos pestañas: Estadísticas y Documentos) |
| GET | `/health` | Liveness check |
| GET | `/api/stats` | Distribución porcentual de eventos por clase |
| GET | `/api/sessions` | Últimos 50 eventos |
| GET | `/api/documents` | Lista de PDFs/TXT subidos a S3 |
| POST | `/upload` | Sube un PDF o TXT (máx 10MB) a S3, para que el tutor lo use |

El JavaScript del dashboard actualiza estadísticas y eventos cada 10 segundos haciendo polling (sin WebSockets, más simple de desplegar). La pestaña "Documentos" permite arrastrar un PDF y lo sube directo a `s3://<bucket>/documents/`.

## 12. Reacciones físicas de RAPIRO

| Estado | LED | Servo |
|---|---|---|
| Estudiando | Verde sólido | Neutro |
| Usando celular | Amarillo sólido | Sacude la cabeza |
| Puesto vacío | Rojo sólido | Pose de alerta |
| Tutor pensando / generando respuesta | Azul parpadeante | Inclina cabeza |
| Tutor explicando | Azul sólido | Mira al estudiante |
| Respuesta correcta | Verde, 3 destellos | Levanta brazos |
| Score de quiz ≥ 70% | Verde, 5 destellos | Celebración completa |
| Esperando el hotword | Azul pulsante | Neutro |

El puerto serie hacia el Arduino de RAPIRO se abre **una sola vez** al iniciar (abrir/cerrar en cada comando resetea el Arduino por el pin DTR y agrega ~2 segundos de demora).

## 13. Variables de entorno

Copiar `.env.example` a `.env`. La única obligatoria para correr el demo del tutor:

```env
ANTHROPIC_API_KEY=sk-ant-api03-...     # console.anthropic.com/settings/keys
```

El resto (sólo necesarias con hardware real o integración AWS completa):

```env
# Modelo / clasificación
MODEL_PATH=models/mobilenetv2_int8.tflite
MIN_CONFIDENCE=0.35

# Hardware (sólo Raspberry Pi)
SERIAL_PORT=/dev/ttyAMA0
CAMERA_URL=http://192.168.x.x:4747/video   # DroidCam, opcional

# AWS (sólo si se usa la nube completa)
AWS_REGION=sa-east-1
AWS_IOT_ENDPOINT=xxx.iot.sa-east-1.amazonaws.com
AWS_DYNAMODB_TABLE=rapiro_sessions_dev
S3_BUCKET=rapiro-user-documents-dev-...

# Tutor / voz
LLM_MODEL=claude-haiku-4-5
TUTOR_HOTWORD=RAPIRO ayuda
SPEECH_LANGUAGE=es-AR
```

> Esta misma lista ya está en `.env.example` con valores de ejemplo — copiarlo a `.env` y completar `ANTHROPIC_API_KEY` (y `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` si se va a usar Polly o el resto de AWS).

## 14. Guía paso a paso — instalación y uso

### Requisitos
- Python 3.10+, Git, webcam (para el dataset), cuenta de Google (Drive/Colab).
- Si se va a correr sobre la Raspberry Pi 2B+ real: no conectar más de 3 dispositivos USB simultáneos (cámara, hub, etc.) — ver sección 3.

### Setup

```bash
git clone git@github.com:Facundopeloso/TPI-RAPIRO-UCP.git
cd TPI-RAPIRO-UCP

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac/RPi

pip install -r requirements.txt
```

```bash
copy .env.example .env     # Windows
# cp .env.example .env     # Linux/Mac
# completar ANTHROPIC_API_KEY en .env
```

### Verificar instalación

```bash
python -m pytest tests/ -v        # debe dar 6 passed
```

### Probar sin hardware

```bash
python demo.py --no-cam                 # clasificación simulada, 5 clases visibles
python demo.py --no-cam --class 1       # fuerza una clase específica (0-4)
python demo.py --quiz                   # tutor interactivo con quiz (requiere ANTHROPIC_API_KEY)
python demo.py --quiz --audio           # idem, con voz y micrófono
```

### Recolectar dataset

```bash
python scripts/collect_dataset.py
```
Teclas `0`-`4` guardan foto en cada clase, `s` muestra estadísticas, `x` limpia el dataset local, `q` sale. Meta mínima original: 150 fotos por clase — para "Estudiando" y "Usando celular" ya se superó por bastante (ver tabla de la sección 7); "Puesto vacío" todavía conviene seguir sumando fotos, y "Confundido"/"Aburrido" siguen sin empezar a recolectarse.

Para subir a Drive automáticamente al salir (requiere `credentials.json`, ver sección 8):
```bash
python scripts/collect_dataset.py --upload
```

### Entrenar

- Recomendado: `notebooks/train_colab.ipynb` en Google Colab con GPU.
- Local: `python scripts/train_model.py --epochs 20 --fine-tune`

### Llevar el modelo a la Raspberry Pi

```bash
scp models/mobilenetv2_int8.tflite pi@<IP_RAPIRO>:~/TPI-RAPIRO-UCP/models/
```

### Correr el sistema completo (Raspberry Pi con hardware)

```bash
python main.py --no-cloud                          # sin AWS
python main.py                                     # con AWS (DynamoDB)
```

### Desplegar la infraestructura AWS

```bash
cd terraform
terraform init
terraform apply -var-file="terraform.tfvars"
```

## 15. Tests

```bash
python -m pytest tests/ -v
python -m pytest tests/ --cov=src --cov-report=term-missing
```

6 tests unitarios sobre preprocesamiento y clasificación, usando mocks para no depender de un modelo `.tflite` real ni de cámara — corren en milisegundos y sirven como red de seguridad ante cualquier cambio.

## 16. Limitaciones conocidas

- **El hardware real (Raspberry Pi 2B+, 512MB RAM, fuente de 12V) es mucho más limitado que el que asumen la mayoría de tutoriales de RPi4** — ver sección 3. No instalar TensorFlow completo, y no conectar más de 3 dispositivos USB a la vez para evitar problemas de alimentación.
- **Clases 3 y 4 (Confundido/Aburrido) no generan reacción todavía** — ver sección 4. Es intencional, no un bug, hasta tener más fotos de esas clases.
- **El hotword por voz no está conectado a `main.py`** — funciona y está probado de forma aislada (`test_tutor.py --voice`), pero el pipeline principal de la Raspberry Pi aún no lo inicia automáticamente.
- **La reacción a celular/ausente en `main.py`/`demo.py` no tiene la cascada completa de TTS** — llama directo a `polly_tts.speak()`, que sólo tiene 2 niveles (Polly → pyttsx3), sin el paso intermedio de gTTS que sí usa `tutor.py` en los modos de quiz interactivo.

## 17. Referencias

- Howard, A. et al. (2017). *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications*. arXiv:1704.04861
- Sandler, M. et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. CVPR 2018
- Jacob, B. et al. (2018). *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*. CVPR 2018
- Salton, G. & Buckley, C. (1988). *Term-weighting approaches in automatic text retrieval*. Information Processing & Management, 24(5), 513-523
- Anthropic. *Claude API Documentation*. https://docs.anthropic.com
- TensorFlow Lite documentation: *Post-training integer quantization*
- HashiCorp. *Terraform AWS Provider Documentation*

---

*TPI Intercátedra 5to año ISI — Universidad de la Cuenca del Plata — 2026*
*Lucas Quintana · Stiven Monsalvo · Facundo Peloso*
