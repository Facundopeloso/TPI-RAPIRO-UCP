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

RAPIRO Guardian detecta en tiempo real si un estudiante esta concentrado, usando el celular o ausente, usando vision por computadora sobre Raspberry Pi 4. Segun la situacion, actua fisicamente (LEDs, servos), registra eventos en AWS y ofrece asistencia tutorial mediante un LLM contextualizado al material de estudio del usuario.

---

## Red Neuronal — Arquitectura CNN

### Tipo de red: MobileNetV2 (CNN con Transfer Learning)

El clasificador es una **Red Neuronal Convolucional (CNN)** basada en la arquitectura **MobileNetV2**, entrenada con la tecnica de **Transfer Learning** (aprendizaje por transferencia) desde pesos pre-entrenados en ImageNet.

### Por que MobileNetV2

MobileNetV2 fue disenada especificamente para dispositivos con recursos limitados (moviles, embebidos). Sus ventajas frente a otras arquitecturas:

| Caracteristica | MobileNetV2 | ResNet50 | VGG16 |
|---|---|---|---|
| Parametros totales | ~3.4M | ~25M | ~138M |
| Tamano modelo float32 | ~14 MB | ~98 MB | ~528 MB |
| Tamano modelo INT8 | ~3.5 MB | ~25 MB | ~135 MB |
| Latencia en RPi 4 (est.) | ~150-250 ms | >1000 ms | >3000 ms |
| Accuracy ImageNet Top-1 | 72% | 76% | 71% |

Para Raspberry Pi 4 con procesamiento a 2 FPS, MobileNetV2 es la unica opcion practica entre las CNNs preentrenadas de alta calidad.

### Componentes clave de MobileNetV2

**Depthwise Separable Convolutions:**
La operacion principal de MobileNetV2 separa la convolucion estandar en dos pasos:
1. Convolucion depthwise: aplica un filtro por cada canal de entrada (espacial)
2. Convolucion pointwise (1x1): combina los canales resultantes

Esto reduce operaciones de multiplicacion-acumulacion en ~8-9x respecto a una convolucion estandar, manteniendo capacidad representacional similar.

**Inverted Residuals con Linear Bottlenecks:**
Cada bloque MobileNetV2 expande los canales (factor 6x), aplica la convolucion depthwise, y luego contrae de vuelta. La conexion residual se aplica solo cuando entrada y salida tienen la misma forma, permitiendo gradientes fluir sin degradarse en redes profundas.

### Arquitectura completa del clasificador

```
Input: imagen BGR 224x224x3 (uint8)
    |
Preprocessing: normalize [0,1] float32, BGR->RGB
    |
MobileNetV2 base (155 capas, ImageNet weights)
    Capa 1:   Conv2D 32 filtros 3x3, stride 2  -> 112x112x32
    ...
    Bloques:  17 inverted residual blocks
    ...
    Ultima:   Conv2D 1280 filtros 1x1           ->   7x7x1280
    |
GlobalAveragePooling2D                          ->   1280
    |
Dense(256, activation='relu')                  ->   256
    |
Dropout(0.4)
    |
Dense(3, activation='softmax')                 ->   3
    |
Output: [P(estudiando), P(celular), P(ausente)]
```

**Parametros totales:** ~3.4M
**Parametros entrenables fase 1 (solo cabeza):** ~330K
**Parametros entrenables fase 2 (fine-tuning):** ~330K + ultimas 30 capas base

### Transfer Learning — dos fases

**Fase 1 — Feature extraction:**
- Base MobileNetV2 completamente congelada (pesos de ImageNet intocados)
- Solo se entrena la cabeza custom (Dense 256 + Dense 3)
- Learning rate: 1e-3 (Adam)
- El base actua como extractor de features generales (bordes, texturas, formas)

**Fase 2 — Fine-tuning (opcional, flag `--fine-tune`):**
- Se descongelan las ultimas 30 capas del base
- Learning rate reducido: 1e-5 (no destruir features aprendidas)
- Permite que el modelo adapte features de alto nivel a las imagenes del dominio especifico (estudiante en escritorio)

### Cuantizacion INT8

Para despliegue en Raspberry Pi 4 se aplica **Post-Training Integer Quantization (PTQ)**:

```
Proceso:
  1. Modelo entrenado float32
  2. Se pasan 200 imagenes representativas por el modelo
  3. Calibracion: se miden rangos min/max de activaciones por capa
  4. Mapeo: float32 -> int8  usando escala y zero_point por tensor
  5. Pesos y activaciones: todos en int8

Resultado:
  - Tamano: ~4x menor (float32 a int8)
  - Velocidad: ~2-3x mas rapido en ARM Cortex-A72 (RPi 4)
  - Precision: caida tipica < 1% accuracy vs float32
  - Formato: .tflite (TensorFlow Lite)
```

### Clases del clasificador

| ID | Nombre | Descripcion | Accion RAPIRO |
|---|---|---|---|
| 0 | Estudiando | Persona mirando libros/pantalla | LED verde, servo neutro |
| 1 | Usando celular | Persona mirando/sosteniendo celular | LED amarillo, sacude cabeza |
| 2 | Puesto vacio | Silla desocupada o persona ausente | LED rojo, pose de alerta |

### Dataset recomendado

| Clase | Cantidad minima | Descripcion |
|---|---|---|
| 0 Estudiando | 200 imagenes | Persona leyendo, escribiendo, frente a pantalla |
| 1 Usando celular | 200 imagenes | Celular visible, persona mirando hacia abajo |
| 2 Puesto vacio | 200 imagenes | Silla vacia, persona de espaldas, fuera de cuadro |

**Variabilidad recomendada:** distintas personas, iluminaciones (dia/noche/artificial), angulos de camara, fondos y ropa.

### Data Augmentation

Durante entrenamiento se aplican estas transformaciones aleatorias para aumentar la variabilidad efectiva del dataset:

```python
rotation_range    = 15      # rotacion +/- 15 grados
width_shift       = 0.1     # desplazamiento horizontal 10%
height_shift      = 0.1     # desplazamiento vertical 10%
zoom_range        = 0.15    # zoom in/out 15%
horizontal_flip   = True    # espejado horizontal
brightness_range  = [0.7, 1.3]  # variacion de brillo
shear_range       = 0.1     # deformacion de corte
```

### Metricas objetivo

| Metrica | Meta | Descripcion |
|---|---|---|
| Val Accuracy | >= 85% | Precision en conjunto de validacion |
| Latencia inferencia | <= 500 ms | Tiempo total por frame en RPi 4 |
| Tamano modelo INT8 | ~3.5 MB | Espacio en microSD |
| Confianza minima | 70% | Umbral para actuar (evita falsos positivos) |

---

## Estructura del Proyecto

```
rapiro_guardian/
|
+-- src/
|   +-- perception/          # Captura de video (OpenCV)
|   |   +-- camera.py
|   |
|   +-- classification/      # CNN TFLite — clasificacion de estados
|   |   +-- classifier.py    # StudentClassifier, ClassificationResult
|   |   +-- preprocessor.py  # resize 224x224, normalize [0,1]
|   |
|   +-- actuation/           # Control fisico del robot RAPIRO
|   |   +-- led_controller.py    # LEDs RGB via GPIO (BCM)
|   |   +-- servo_controller.py  # 12 servos via serial Arduino
|   |   +-- rapiro.py            # Facade principal
|   |
|   +-- tutoring/            # Modulo de tutoria inteligente (LLM + RAG)
|   |   +-- document_processor.py   # carga PDF/TXT, TF-IDF retrieval
|   |   +-- tutor.py                # OpenAI GPT-3.5 Turbo
|   |
|   +-- cloud/               # Integracion AWS
|       +-- iot_publisher.py     # MQTT IoT Core con throttling
|
+-- scripts/
|   +-- collect_dataset.py   # Captura imagenes para dataset (webcam)
|   +-- train_model.py       # Entrena MobileNetV2, exporta TFLite INT8
|
+-- tests/
|   +-- test_classifier.py   # 6 tests unitarios (todos passing)
|
+-- terraform/               # IaC AWS
|   +-- main.tf
|
+-- dataset/                 # Imagenes de entrenamiento (no en git)
|   +-- class_0/             # Estudiando
|   +-- class_1/             # Usando celular
|   +-- class_2/             # Puesto vacio
|
+-- models/                  # Modelo entrenado (no en git)
|   +-- mobilenetv2_int8.tflite
|
+-- config/
|   +-- settings.py          # Configuracion centralizada
|
+-- main.py                  # Pipeline principal
+-- demo.py                  # Demo sin hardware real
+-- requirements.txt
+-- .env.example
```

---

## Instalacion

```bash
# 1. Clonar
git clone git@github.com:Facundopeloso/TPI-RAPIRO-UCP.git
cd TPI-RAPIRO-UCP

# 2. Entorno virtual
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Dependencias
pip install -r requirements.txt

# 4. Variables de entorno
cp .env.example .env
# Editar .env con credenciales AWS, puerto serial, etc.
```

---

## Entrenar el modelo

```bash
# Paso 1: recolectar dataset (webcam)
python scripts/collect_dataset.py
# Teclas: [0] Estudiando  [1] Celular  [2] Ausente  [q] Salir
# Meta: 200+ imagenes por clase

# Paso 2: entrenar (solo cabeza, 15 epochs)
python scripts/train_model.py

# Paso 3: entrenar con fine-tuning (mejor accuracy)
python scripts/train_model.py --epochs 20 --fine-tune

# Salida: models/mobilenetv2_int8.tflite
```

---

## Demo (sin hardware)

```bash
# Demo con clasificador simulado
python demo.py --no-cam

# Demo con webcam real
python demo.py

# Forzar clase fija
python demo.py --no-cam --class 1   # siempre "usando celular"
```

---

## Tests

```bash
# Correr tests unitarios
python -m pytest tests/ -v

# Con cobertura
python -m pytest tests/ --cov=src --cov-report=term-missing
```

---

## Ejecutar el sistema completo

```bash
python main.py                  # Pipeline completo
python main.py --no-cloud       # Sin AWS
python main.py --demo           # Sin hardware real
```

---

## Infraestructura AWS

| Servicio | Funcion | Costo/mes |
|---|---|---|
| IoT Core | MQTT desde Raspberry Pi | ~$0.08 |
| Lambda | Procesamiento de eventos | $0 (free tier) |
| DynamoDB | Almacenamiento de sesiones | $0 (free tier) |
| S3 | Documentos de usuario | ~$0.12 |
| CloudWatch | Logs y dashboards | ~$0.50 |
| API Gateway | REST API dashboard | $0 (free tier) |
| **TOTAL** | | **~$1.70/mes** |

```bash
cd terraform
terraform init
terraform apply
```

---

## Variables de Entorno

Ver `.env.example`. Variables criticas:

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

## Flujo Principal

```
Camara -> Preprocesamiento -> Clasificador TFLite CNN
                                    |
          +-------------------------+-------------------------+
       Clase 0                  Clase 1                 Clase 2
     (Estudiando)             (Celular)               (Ausente)
     LED verde                LED amarillo             LED rojo
     Servo neutro             Mueve cabeza             Alerta sonora
     Log AWS                  Alerta AWS               Alerta + SNS
```

---

## Referencias

- Howard, A. et al. (2017). *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications*. arXiv:1704.04861
- Sandler, M. et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. CVPR 2018
- Jacob, B. et al. (2018). *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*. CVPR 2018
- IEEE Survey on Robotic Tutoring Systems (2022)
- TensorFlow Lite documentation: Post-training integer quantization
