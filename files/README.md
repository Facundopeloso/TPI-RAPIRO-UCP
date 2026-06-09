# 🤖 RAPIRO Guardián de Distracción

Sistema inteligente de monitoreo conductual y tutoría adaptativa basado en el robot RAPIRO.  
**TPI Intercátedra 5to año — ISI — Universidad de la Cuenca del Plata — 2026**

---

## 👥 Equipo

| Integrante | Rol | Responsabilidades |
|---|---|---|
| Lucas Quintana | Líder Técnico / ML Engineer | Modelo CNN, pipeline de inferencia, coordinación |
| Stiven Monsalvo | Cloud & DevOps Engineer | AWS, Terraform IaC, dashboard, CloudWatch |
| Facundo Peloso | Hardware & Tutor Module | RAPIRO (LEDs/servos), módulo de tutoría LLM |

---

## 📋 Descripción

RAPIRO Guardián detecta en tiempo real si un estudiante está concentrado, usando el celular o ausente, usando visión por computadora sobre Raspberry Pi 4. Según la situación, actúa físicamente (LEDs, servos), registra eventos en AWS y ofrece asistencia tutorial mediante un LLM contextualizado al material de estudio del usuario.

---

## 🗂 Estructura del Proyecto

```
rapiro_guardian/
│
├── src/
│   ├── perception/          # Captura de video y preprocesamiento
│   │   ├── __init__.py
│   │   └── camera.py
│   │
│   ├── classification/      # Modelo CNN TFLite — clasificación de estados
│   │   ├── __init__.py
│   │   ├── classifier.py
│   │   └── preprocessor.py
│   │
│   ├── actuation/           # Control físico del robot RAPIRO
│   │   ├── __init__.py
│   │   ├── led_controller.py
│   │   ├── servo_controller.py
│   │   └── rapiro.py        # Facade principal del robot
│   │
│   ├── tutoring/            # Módulo de tutoría inteligente (LLM + RAG)
│   │   ├── __init__.py
│   │   ├── document_processor.py
│   │   ├── retriever.py
│   │   └── tutor.py
│   │
│   ├── cloud/               # Integración con AWS
│   │   ├── __init__.py
│   │   ├── iot_publisher.py
│   │   └── s3_manager.py
│   │
│   └── dashboard/           # Backend API del dashboard web
│       ├── __init__.py
│       └── api.py
│
├── tests/
│   ├── test_classifier.py
│   ├── test_actuation.py
│   ├── test_tutoring.py
│   └── test_cloud.py
│
├── terraform/               # Infraestructura como Código (IaC)
│   ├── main.tf
│   ├── variables.tf
│   └── outputs.tf
│
├── scripts/
│   ├── collect_dataset.py   # Herramienta de captura de imágenes para dataset
│   └── train_model.py       # Entrenamiento del modelo CNN
│
├── config/
│   └── settings.py          # Configuración centralizada
│
├── main.py                  # Punto de entrada principal
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Requisitos

### Hardware
- Raspberry Pi 4 (4GB RAM) + microSD 32GB
- Robot RAPIRO con Arduino integrado
- Cámara USB Logitech C270 (o similar, 720p)

### Software
- Python 3.11+
- Dependencias listadas en `requirements.txt`
- Cuenta AWS con permisos en IoT Core, Lambda, S3, DynamoDB, CloudWatch

### Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/equipo-isi/rapiro-guardian.git
cd rapiro-guardian

# 2. Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno
cp .env.example .env
# Editar .env con credenciales AWS, puerto serial, etc.

# 5. Desplegar infraestructura AWS
cd terraform
terraform init
terraform apply

# 6. Ejecutar el sistema
cd ..
python main.py
```

---

## 🧠 Módulos

### `src/perception/` — Captura de Video
- Captura de fotogramas a 2 FPS con OpenCV
- Preprocesamiento: resize a 224×224 px, normalización [0,1]

### `src/classification/` — Clasificador CNN
- Modelo MobileNetV2 con transferencia de aprendizaje, compilado en TFLite INT8
- Clasifica en 3 estados: **Estudiando** (0), **Usando celular** (1), **Puesto vacío** (2)
- Accuracy: **89.3%** | Latencia de inferencia: **~157 ms**
- Umbral de confianza mínima: **70%** (evita falsos positivos)

### `src/actuation/` — Control del Robot
- LEDs RGB via GPIO (verde / amarillo intermitente / rojo / azul)
- 12 servos controlados via Arduino (pyserial)
- Modo "no molestar" activable por botón físico (10 minutos)

### `src/tutoring/` — Tutoría Inteligente
- Carga de documentos PDF/TXT, segmentación en chunks de 500 palabras
- Recuperación de contexto relevante mediante TF-IDF simplificado
- Respuestas generadas por GPT-3.5 Turbo (OpenAI) o Claude Haiku (AWS Bedrock)
- Speech-to-text (hotword: *"RAPIRO, ayuda"*) + text-to-speech (Google TTS)

### `src/cloud/` — Integración AWS
- Publicación de eventos MQTT en AWS IoT Core (throttling: 1 mensaje/5 seg)
- Almacenamiento de documentos en S3
- Eventos de sesión en DynamoDB

### `terraform/` — IaC
- Aprovisiona: IoT Core, Lambda (×2), DynamoDB, S3, API Gateway, CloudWatch
- Estado remoto en S3 con locking en DynamoDB
- `terraform destroy` para eliminar recursos cuando no se usan

---

## 🔁 Flujo Principal

```
Cámara → Preprocesamiento → Clasificador TFLite
                                    │
              ┌─────────────────────┼──────────────────────┐
           Clase 0              Clase 1                 Clase 2
         (Estudiando)        (Celular)               (Ausente)
         LED verde           LED amarillo            LED rojo
         Neutro              Mueve cabeza            Alerta sonora
         Log AWS             Alerta AWS              Alerta + SNS
```

---

## 🧪 Tests

```bash
# Ejecutar todos los tests
pytest tests/ -v

# Con cobertura
pytest tests/ --cov=src --cov-report=term-missing
```

Meta de cobertura: **≥ 60%**

---

## 🌩 Infraestructura AWS

| Servicio | Función | Costo/mes |
|---|---|---|
| IoT Core | MQTT desde Raspberry Pi | ~$0.08 |
| Lambda | Procesamiento de eventos | $0 (free tier) |
| DynamoDB | Almacenamiento de sesiones | $0 (free tier) |
| S3 | Documentos de usuario | ~$0.12 |
| CloudWatch | Logs y dashboards | ~$0.50 |
| API Gateway | REST API dashboard | $0 (free tier) |
| **TOTAL** | | **~$1.70/mes** |

---

## 🔐 Variables de Entorno

Ver `.env.example` para la lista completa. Variables críticas:

```env
AWS_IOT_ENDPOINT=xxxxx.iot.us-east-1.amazonaws.com
AWS_REGION=us-east-1
OPENAI_API_KEY=sk-...
SERIAL_PORT=/dev/ttyUSB0
CAMERA_INDEX=0
MODEL_PATH=models/mobilenetv2_int8.tflite
MIN_CONFIDENCE=0.70
```

---

## 📊 Métricas del Prototipo (Etapa 2)

| Métrica | Meta | Real |
|---|---|---|
| Accuracy del modelo | ≥ 85% | **89.3%** ✅ |
| Latencia de inferencia | ≤ 500 ms | **242 ms** ✅ |
| Disponibilidad AWS | ≥ 99% | **99.9%** ✅ |
| Respuestas LLM correctas | ≥ 80% | **86.7%** ✅ |

---

## 📄 Licencia

MIT License — libre para uso y replicación por otras instituciones educativas.

---

## 📚 Referencias

- Howard, A. et al. (2017). *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications*. arXiv.
- Redmon, J. et al. (2016). *You Only Look Once: Unified, Real-Time Object Detection*. CVPR.
- IEEE Survey on Robotic Tutoring Systems (2022).
- Capers Jones. *Applied Software Measurement* (3rd ed.).
