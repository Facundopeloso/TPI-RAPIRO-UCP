# RAPIRO Guardian de Distracción

Sistema de monitoreo conductual en tiempo real sobre el robot RAPIRO.  
**TPI Intercátedra 5to año — ISI — Universidad de la Cuenca del Plata — 2026**

---

## Equipo

| Integrante | Rol | Responsabilidades |
|---|---|---|
| Lucas Quintana | Líder Técnico / ML Engineer | Modelo CNN, pipeline de inferencia, coordinación |
| Stiven Monsalvo | Cloud & DevOps Engineer | AWS, Terraform IaC, dashboard, CloudWatch |
| Facundo Peloso | Hardware & Tutor Module | RAPIRO (LEDs/servos), módulo de tutoría LLM |

---

## Descripción

RAPIRO Guardián detecta en tiempo real si un estudiante está concentrado, usando el celular o ausente, mediante visión por computadora sobre Raspberry Pi 2B. Según el estado detectado, el robot reacciona físicamente (LEDs GPIO + 12 servos vía serial Arduino), registra eventos en AWS y genera consejos de estudio personalizados mediante Claude Haiku con texto a voz por Amazon Polly.

---

## Hardware Real

- **Raspberry Pi 2B** (ARMv7 quad-core) + microSD 32GB
- **Robot RAPIRO** con Arduino ATmega32U4 — comunicación UART `/dev/ttyAMA0` @ **57600 baud**
- **Cámara**: DroidCam (app Android) via red local — stream HTTP `http://192.168.100.6:4747/video`
- **LEDs RGB**: conectados a GPIO pines 17 (R), 27 (G), 22 (B) del RPi — ánodo común
- **Audio**: parlante Bluetooth `MM43BT` via PipeWire (`bluez_output.E8_07_BF_00_63_6D.1`)

---

## Estructura del Proyecto

```
TPI-RAPIRO-UCP/
├── src/
│   ├── perception/          # Captura de video y preprocesamiento
│   │   ├── camera.py        # Stream DroidCam con hilo drain para frame fresco
│   │   └── preprocessor.py
│   │
│   ├── classification/      # Clasificador TFLite INT8
│   │   ├── classifier.py    # Backends: tflite_runtime / ai_edge_litert / ctypes
│   │   ├── preprocessor.py
│   │   └── tflite_ctypes.py # Backend ctypes para RPi sin tflite_runtime
│   │
│   ├── actuation/           # Control físico del robot
│   │   ├── led_controller.py  # GPIO RGB via lgpio
│   │   ├── servo_controller.py # Serial UART 57600 baud → Arduino
│   │   └── rapiro.py          # Facade: reacciones por estado
│   │
│   ├── tutoring/            # Tutor inteligente
│   │   └── polly_tts.py     # Amazon Polly (voz Pedro neural, us-east-1)
│   │
│   ├── cloud/               # Integración AWS
│   │   └── boto3_publisher.py # DynamoDB + IoT Core
│   │
│   └── dashboard/           # API backend del dashboard web
│       └── api.py
│
├── models/
│   └── mobilenetv2_int8.tflite  # MobileNetV2 INT8 cuantizado
│
├── config/
│   └── settings.py
│
├── main.py                  # Punto de entrada (hardware real)
├── demo.py                  # Modo demo sin hardware (ventana OpenCV)
├── docker-compose.yml       # Dashboard en EC2
└── .env                     # Variables de entorno (NO commitear)
```

---

## Instalación (Raspberry Pi)

```bash
git clone https://github.com/Facundopeloso/TPI-RAPIRO-UCP.git
cd TPI-RAPIRO-UCP
pip install -r requirements.txt
cp .env.example .env
# Editar .env con credenciales AWS
python main.py
```

### Requisito de audio Bluetooth

```bash
# Emparejar speaker antes de iniciar
bluetoothctl connect E8:07:BF:00:63:6D
# PipeWire maneja el routing automáticamente
```

---

## Módulos

### `src/perception/` — Captura de Video
- Stream DroidCam via HTTP a 2 FPS
- Hilo daemon drain mantiene el buffer fresco (evita lag acumulado)
- Preprocesamiento: resize 224×224 px, normalización [0,1]

### `src/classification/` — Clasificador CNN
- MobileNetV2 INT8 cuantizado con TFLite
- **3 estados efectivos**: Estudiando (verde), Usando celular (amarillo), Puesto vacío (rojo)
- Clases 3 y 4 (confundido/aburrido) → mapeadas a Estudiando
- Latencia real en RPi 2B: **~3.5–5 seg** con 4 threads
- Umbral de confianza mínima: **20%**
- Histéresis: `CONFIRM_CYCLES=1` (reacción inmediata)

### `src/actuation/` — Control del Robot
- **LEDs GPIO** (pines 17/27/22): rojo/verde/amarillo por estado
- **Servos via serial** `/dev/ttyAMA0` @ **57600 baud** → Arduino ATmega32U4
- Reacciones diferenciadas:
  - **Celular** → sacude cabeza rápido (no,no,no) + brazo derecho arriba
  - **Ausente** → ambos brazos arriba + busca lento izquierda/derecha
  - **Estudiando** → asiente con la cabeza + LED verde
- `threading.Timer(4.5s)` restaura color LED persistente tras animación

### `src/tutoring/` — Tutoría con IA
- **Claude Haiku 4.5** genera consejo contextual en español rioplatense
- Contexto: material de estudio cargado desde S3 al arrancar
- **Amazon Polly** voz `Pedro` (neural, us-east-1) → audio por Bluetooth via PipeWire
- Cooldown: 90 seg por estado (no repite si el estado no cambia)
- Solo habla para estados "celular" y "ausente"

### `src/cloud/` — Integración AWS
- **DynamoDB** `rapiro_sessions_dev` (sa-east-1): evento por predicción
- **IoT Core**: publicación MQTT throttleada a 1 msg/5 seg
- **S3** `rapiro-user-documents-dev-442650748881`: documentos de estudio del usuario

### Dashboard Web
- URL: `http://54.94.201.85/` (EC2 + Docker, puerto 80)
- Sube documentos de estudio → RAPIRO los usa en el próximo consejo
- API backend en `src/dashboard/api.py`

---

## Flujo Principal

```
DroidCam → Preprocesamiento → Clasificador TFLite INT8
                                        │
              ┌─────────────────────────┼──────────────────────────┐
           Clase 0,3,4              Clase 1                    Clase 2
           (Estudiando)           (Celular)                  (Ausente)
           LED verde              LED amarillo               LED rojo
           Asiente               Sacude cabeza              Brazos arriba
                                  + brazo arriba            + busca al costado
                                  Claude Haiku              Claude Haiku
                                  → Polly Pedro             → Polly Pedro
                                  → BT speaker              → BT speaker
                                        │                        │
                              DynamoDB + IoT Core          DynamoDB + IoT Core
```

---

## Variables de Entorno (.env)

```env
# LLM
ANTHROPIC_API_KEY=sk-ant-...

# Modelo
MODEL_PATH=models/mobilenetv2_int8.tflite
MIN_CONFIDENCE=0.20

# Hardware
SERIAL_PORT=/dev/ttyAMA0
SERIAL_BAUD_RATE=57600
CAMERA_URL=http://192.168.100.6:4747/video
LED_RED_PIN=17
LED_GREEN_PIN=27
LED_BLUE_PIN=22

# AWS
AWS_REGION=sa-east-1
AWS_DYNAMODB_TABLE=rapiro_sessions_dev
S3_BUCKET=rapiro-user-documents-dev-442650748881

# Polly TTS
POLLY_VOICE=Pedro
POLLY_ENGINE=neural
# POLLY_REGION se auto-setea en us-east-1 para neural

# Comportamiento
SPEAK_COOLDOWN=90
LLM_MODEL=claude-haiku-4-5
```

---

## Modo Demo (sin hardware)

```bash
# Windows/Mac — requiere DroidCam activo
python demo.py
```

Ejecuta el pipeline completo (clasificador + DynamoDB + Claude + Polly) mostrando una ventana OpenCV que simula los ojos del RAPIRO con colores por estado. No requiere GPIO ni serial.

---

## Infraestructura AWS (sa-east-1)

| Servicio | Función |
|---|---|
| IoT Core | MQTT desde RPi |
| DynamoDB | Sesiones y eventos |
| S3 | Documentos de estudio del usuario |
| Lambda | Procesamiento de eventos |
| API Gateway | REST API dashboard |
| CloudWatch | Logs |
| EC2 (t2.micro) | Dashboard web — `54.94.201.85` |

Infraestructura como código en `terraform/`.

---

## Referencias

- Howard, A. et al. (2017). *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications*. arXiv.
- Amazon Polly — Neural TTS Documentation. AWS.
- Anthropic Claude API Documentation.
- IEEE Survey on Robotic Tutoring Systems (2022).
