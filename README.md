# RAPIRO Guardian de Distraccion

Sistema inteligente de monitoreo conductual y tutoria adaptativa basado en el robot RAPIRO.
**TPI Intercatedra 5to anio - ISI - Universidad de la Cuenca del Plata - 2026**

---

## Equipo

| Integrante | Rol | Responsabilidades |
|---|---|---|
| Lucas Quintana | Lider Tecnico / ML Engineer | Modelo CNN, pipeline de inferencia, coordinacion |
| Stiven Monsalvo | Cloud & DevOps Engineer | AWS, Terraform IaC, dashboard, CloudWatch |
| Facundo Peloso | Hardware & Tutor Module | RAPIRO (LEDs/servos), modulo de tutoria LLM |

---

## Descripcion

RAPIRO Guardian detecta en tiempo real el estado conductual del estudiante usando vision por computadora sobre Raspberry Pi 4. Segun la situacion, el robot reacciona fisicamente (LEDs RGB + 12 servos), registra eventos en AWS y ofrece asistencia tutorial adaptativa mediante Claude Haiku (Anthropic) contextualizado al material de estudio.

### Que detecta

| Clase | Estado | Reaccion RAPIRO |
|---|---|---|
| 0 | Estudiando / concentrado | LED verde, servo neutro |
| 1 | Usando celular | LED amarillo, sacude cabeza |
| 2 | Puesto vacio / ausente | LED rojo, pose de alerta |
| 3 | Confundido | LED azul x2 flash, inclina cabeza — re-explica |
| 4 | Aburrido | LED amarillo, sacude cabeza — genera quiz |

---

## Guia de Ejecucion Paso a Paso

### Requisitos previos

- Python 3.10+
- Git
- Webcam (para recolectar dataset)
- Cuenta de Google (para Drive + Colab)

---

### PASO 1 — Clonar el repositorio

```bash
git clone git@github.com:Facundopeloso/TPI-RAPIRO-UCP.git
cd TPI-RAPIRO-UCP
```

---

### PASO 2 — Entorno virtual e instalacion de dependencias

```bash
# Crear entorno virtual
python -m venv venv

# Activar (Windows)
venv\Scripts\activate

# Activar (Linux / Mac / Raspberry Pi)
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

---

### PASO 3 — Configurar variables de entorno

```bash
# Windows
copy .env.example .env

# Linux / Mac
cp .env.example .env
```

Abrir `.env` y completar la unica variable obligatoria para el demo:

```env
ANTHROPIC_API_KEY=sk-ant-api03-TU_KEY_REAL_ACA
```

Obtener la key en: https://console.anthropic.com/settings/keys

> El resto de las variables (AWS, hardware) solo son necesarias en Raspberry Pi con hardware real.

---

### PASO 4 — Verificar que todo funciona

```bash
python -m pytest tests/ -v
```

Deben salir **6/6 PASSED**. Si falla alguno, revisar la instalacion de dependencias.

---

### PASO 5 — Probar el demo (sin hardware)

#### 5a. Clasificador con deteccion simulada

```bash
python demo.py --no-cam
```

Ver las 5 clases con acciones `[RAPIRO]` en consola. Detener con `Ctrl+C`.

#### 5b. Forzar una clase especifica

```bash
python demo.py --no-cam --class 0   # Estudiando
python demo.py --no-cam --class 1   # Celular
python demo.py --no-cam --class 2   # Ausente
python demo.py --no-cam --class 3   # Confundido
python demo.py --no-cam --class 4   # Aburrido
```

#### 5c. Modo tutor interactivo con quiz (requiere ANTHROPIC_API_KEY)

```bash
python demo.py --quiz
```

Flujo:
1. Presionar Enter → RAPIRO explica el tema con ejemplos reales
2. Presionar Enter → genera 3 preguntas multiple choice
3. Responder A/B/C/D a cada pregunta
4. Ver feedback personalizado y puntaje final

---

### PASO 6 — Recolectar dataset con la webcam

```bash
python scripts/collect_dataset.py
```

Controles dentro de la ventana:

| Tecla | Accion |
|---|---|
| `0` | Guardar foto — Estudiando |
| `1` | Guardar foto — Usando celular |
| `2` | Guardar foto — Puesto vacio |
| `3` | Guardar foto — Confundido (ceno fruncido, mano en barbilla) |
| `4` | Guardar foto — Aburrido (postura caida, mirada perdida) |
| `s` | Ver estadisticas del dataset actual |
| `x` | Limpiar todas las fotos locales (pide confirmacion) |
| `q` | Salir |

**Meta minima:** 150 fotos por clase = 750 total.

**Variabilidad recomendada:** distintas personas, iluminaciones, angulos, fondos.

#### Subir fotos automaticamente a Google Drive

Setup unico (una sola vez):

1. Ir a https://console.cloud.google.com
2. Crear proyecto → buscar **Google Drive API** → Habilitar
3. `Credenciales > + Crear credencial > OAuth 2.0 > App de escritorio`
4. Descargar JSON → renombrar a `credentials.json` → poner en la raiz del proyecto
5. Compartir ese `credentials.json` con los compañeros de equipo (por WhatsApp/Discord)

Uso diario:

```bash
python scripts/collect_dataset.py --upload
```

Al presionar `q`:
- Sube automaticamente solo las fotos nuevas a las carpetas de Drive
- La primera vez abre el navegador para autorizar con tu cuenta de Google
- Las siguientes veces funciona sin intervencion

Carpetas en Drive: https://drive.google.com/drive/folders/1ULuPEGf_U_qH5OwrI1ps8kk1pHdq8mrP

---

### PASO 7 — Entrenar la red neuronal en Google Colab

1. Abrir https://colab.research.google.com
2. `Archivo > Abrir > Google Drive` → seleccionar `notebooks/train_colab.ipynb`
3. `Entorno de ejecucion > Cambiar tipo de entorno > GPU T4`
4. `Entorno de ejecucion > Ejecutar todo`
5. Autorizar acceso a Drive cuando lo pida
6. Al finalizar, el modelo `.tflite` se guarda en `Mi Drive/RAPIRO_models/`

**Tiempo estimado:** 5-10 minutos con GPU T4.

**Accuracy esperada:** >85% con 150 imgs/clase, >92% con 300+ imgs/clase.

#### Alternativa: entrenar localmente (lento, sin GPU)

```bash
# Solo cabeza (rapido, menos accuracy)
python scripts/train_model.py

# Con fine-tuning (mejor accuracy, mas tiempo)
python scripts/train_model.py --epochs 20 --fine-tune

# Salida: models/mobilenetv2_int8.tflite
```

---

### PASO 8 — Copiar modelo a Raspberry Pi

Bajar `mobilenetv2_int8.tflite` de `Mi Drive/RAPIRO_models/` y copiarlo:

```bash
scp mobilenetv2_int8.tflite pi@<IP_RAPIRO>:~/TPI-RAPIRO-UCP/models/
```

---

### PASO 9 — Ejecutar sistema completo (Raspberry Pi con hardware)

```bash
# Sin AWS (recomendado para desarrollo)
python main.py --no-cloud

# Con documento para el tutor
python main.py --no-cloud --document apuntes.pdf

# Con AWS IoT
python main.py
```

Hotword para activar el tutor por voz: decir **"RAPIRO ayuda"**

---

## Tests

```bash
# Tests unitarios
python -m pytest tests/ -v

# Con cobertura de codigo
python -m pytest tests/ --cov=src --cov-report=term-missing
```

---

## Red Neuronal — Arquitectura CNN

### Tipo: MobileNetV2 con Transfer Learning

CNN basada en **MobileNetV2**, entrenada con **Transfer Learning** desde pesos pre-entrenados en ImageNet.

### Por que MobileNetV2

| Caracteristica | MobileNetV2 | ResNet50 | VGG16 |
|---|---|---|---|
| Parametros totales | ~3.4M | ~25M | ~138M |
| Tamano modelo INT8 | ~3.5 MB | ~25 MB | ~135 MB |
| Latencia en RPi 4 | ~150-250 ms | >1000 ms | >3000 ms |

### Arquitectura completa

```
Input: imagen 224x224x3
    |
Preprocessing: normalize [0,1] float32, BGR -> RGB
    |
MobileNetV2 base (155 capas, ImageNet weights, frozen en fase 1)
    |
GlobalAveragePooling2D  ->  vector 1280
    |
Dense(256, relu)
    |
Dropout(0.4)
    |
Dense(5, softmax)
    |
Output: [P(estudiando), P(celular), P(ausente), P(confundido), P(aburrido)]
```

### Transfer Learning — dos fases

**Fase 1 — Feature extraction:**
- Base MobileNetV2 congelada (pesos de ImageNet intactos)
- Solo se entrena la cabeza custom
- Learning rate: 1e-3 (Adam)

**Fase 2 — Fine-tuning (flag `--fine-tune`):**
- Se descongelan las ultimas 30 capas del base
- Learning rate: 1e-5 (preserva features aprendidos)

### Cuantizacion INT8

Para despliegue en Raspberry Pi 4:

```
Modelo float32  ->  calibracion con 200 imagenes  ->  modelo INT8
Resultado: 4x menor tamano, 2-3x mas rapido en ARM Cortex-A72
Precision: caida tipica < 1% accuracy
```

### Dataset

| Clase | Meta | Poses sugeridas |
|---|---|---|
| 0 Estudiando | 150 imgs | Leyendo, escribiendo, frente a pantalla |
| 1 Usando celular | 150 imgs | Celular visible, mirando hacia abajo |
| 2 Puesto vacio | 150 imgs | Silla vacia, fuera de cuadro |
| 3 Confundido | 150 imgs | Ceno fruncido, mano en barbilla o mejilla |
| 4 Aburrido | 150 imgs | Cabeza apoyada en mano, postura encorvada |

---

## Reacciones fisicas de RAPIRO

### Clasificador

| Clase detectada | LED | Servo |
|---|---|---|
| Estudiando | Verde solido | Neutro |
| Celular | Amarillo solido | Sacude cabeza |
| Ausente | Rojo solido | Pose de alerta |
| Confundido | Azul x2 flash | Inclina cabeza (think) |
| Aburrido | Amarillo solido | Sacude cabeza |

### Tutor LLM

| Momento | LED | Servo |
|---|---|---|
| Procesando / pensando | Azul parpadeante | Inclina cabeza |
| Explicando tema | Azul solido | Mira al estudiante |
| Haciendo pregunta | Blanco solido | Postura de escucha |
| Respuesta correcta | Verde x3 destellos | Levanta brazos (celebra) |
| Respuesta incorrecta | Amarillo solido | Inclina cabeza con empatia |
| Score >= 70% | Verde x5 destellos | Celebracion completa |
| Score 40-69% | Amarillo solido | Asiente (alienta) |
| Score < 40% | Azul solido | Mira al estudiante (apoyo) |

### Voz

| Momento | LED | Servo |
|---|---|---|
| Esperando hotword | Azul pulsante | Neutro |
| Hotword detectado | Azul brillante | Mira al estudiante |

---

## Estructura del proyecto

```
TPI-RAPIRO-UCP/
|
+-- src/
|   +-- perception/
|   |   +-- camera.py                  # Captura OpenCV
|   |
|   +-- classification/
|   |   +-- classifier.py              # StudentClassifier TFLite
|   |   +-- preprocessor.py           # Resize 224x224, normalize
|   |
|   +-- actuation/
|   |   +-- led_controller.py         # LEDs RGB via GPIO
|   |   +-- servo_controller.py       # 12 servos via serial Arduino
|   |   +-- rapiro.py                 # Facade principal del robot
|   |
|   +-- tutoring/
|   |   +-- document_processor.py     # PDF/TXT loader, TF-IDF retrieval
|   |   +-- tutor.py                  # Claude Haiku, quiz, evaluacion
|   |   +-- voice_listener.py         # Hotword + STT
|   |
|   +-- cloud/
|   |   +-- iot_publisher.py          # MQTT AWS IoT Core
|   |
|   +-- dashboard/
|       +-- api.py                    # FastAPI REST dashboard
|
+-- scripts/
|   +-- collect_dataset.py            # Captura webcam + upload a Drive
|   +-- train_model.py                # Entrena localmente
|
+-- notebooks/
|   +-- train_colab.ipynb             # Entrena en Google Colab con GPU
|
+-- tests/
|   +-- test_classifier.py            # 6 tests unitarios
|
+-- terraform/                        # IaC AWS
|   +-- main.tf
|
+-- dataset/                          # Fotos de entrenamiento (no en git)
|   +-- class_0/                      # Estudiando
|   +-- class_1/                      # Usando celular
|   +-- class_2/                      # Puesto vacio
|   +-- class_3/                      # Confundido
|   +-- class_4/                      # Aburrido
|
+-- models/                           # Modelo entrenado (no en git)
|   +-- mobilenetv2_int8.tflite
|
+-- config/
|   +-- settings.py                   # Configuracion centralizada
|
+-- main.py                           # Pipeline principal
+-- demo.py                           # Demo sin hardware
+-- requirements.txt
+-- .env.example
```

---

## Infraestructura AWS

| Servicio | Funcion | Costo/mes |
|---|---|---|
| IoT Core | MQTT desde Raspberry Pi | ~$0.08 |
| Lambda | Procesamiento de eventos | $0 (free tier) |
| DynamoDB | Almacenamiento de sesiones | $0 (free tier) |
| S3 | Documentos de usuario para tutor | ~$0.12 |
| CloudWatch | Logs y dashboards | ~$0.50 |
| **TOTAL** | | **~$1.70/mes** |

```bash
cd terraform
terraform init
terraform apply
```

---

## Variables de entorno

Ver `.env.example`. Variables criticas:

```env
ANTHROPIC_API_KEY=sk-ant-api03-...     # LLM tutor (obligatoria para demo)
MODEL_PATH=models/mobilenetv2_int8.tflite
MIN_CONFIDENCE=0.70
AWS_IOT_ENDPOINT=xxx.iot.sa-east-1.amazonaws.com  # solo Raspberry Pi
SERIAL_PORT=/dev/ttyUSB0                           # solo Raspberry Pi
```

---

## Referencias

- Howard, A. et al. (2017). *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications*. arXiv:1704.04861
- Sandler, M. et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. CVPR 2018
- Jacob, B. et al. (2018). *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*. CVPR 2018
- TensorFlow Lite documentation: Post-training integer quantization
- Anthropic Claude API documentation
