"""
config/settings.py
Configuración centralizada del sistema RAPIRO Guardián.
Todas las constantes y parámetros configurables se definen aquí.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# HARDWARE
# ---------------------------------------------------------------------------
SERIAL_PORT = os.getenv("SERIAL_PORT", "/dev/ttyUSB0")
SERIAL_BAUD_RATE = int(os.getenv("SERIAL_BAUD_RATE", "9600"))
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
CAMERA_FPS = int(os.getenv("CAMERA_FPS", "2"))

# Pines GPIO (BCM) para LEDs RGB
LED_RED_PIN = int(os.getenv("LED_RED_PIN", "17"))
LED_GREEN_PIN = int(os.getenv("LED_GREEN_PIN", "27"))
LED_BLUE_PIN = int(os.getenv("LED_BLUE_PIN", "22"))

# Botón "no molestar"
DO_NOT_DISTURB_PIN = int(os.getenv("DO_NOT_DISTURB_PIN", "23"))
DO_NOT_DISTURB_DURATION_SEC = int(os.getenv("DO_NOT_DISTURB_DURATION_SEC", "600"))  # 10 min

# ---------------------------------------------------------------------------
# MODELO DE CLASIFICACIÓN
# ---------------------------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "models/mobilenetv2_int8.tflite")
INPUT_SIZE = (224, 224)
NUM_CLASSES = 5
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.70"))

# Clases del clasificador
CLASS_STUDYING = 0
CLASS_PHONE    = 1
CLASS_ABSENT   = 2
CLASS_CONFUSED = 3
CLASS_BORED    = 4
CLASS_LABELS = {
    CLASS_STUDYING: "Estudiando",
    CLASS_PHONE:    "Usando celular",
    CLASS_ABSENT:   "Puesto vacío",
    CLASS_CONFUSED: "Confundido",
    CLASS_BORED:    "Aburrido",
}

# ---------------------------------------------------------------------------
# CLOUD — AWS
# ---------------------------------------------------------------------------
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_IOT_ENDPOINT = os.getenv("AWS_IOT_ENDPOINT", "")
AWS_IOT_TOPIC = os.getenv("AWS_IOT_TOPIC", "rapiro/events")
AWS_IOT_CERT_PATH = os.getenv("AWS_IOT_CERT_PATH", "certs/certificate.pem.crt")
AWS_IOT_KEY_PATH = os.getenv("AWS_IOT_KEY_PATH", "certs/private.pem.key")
AWS_IOT_CA_PATH = os.getenv("AWS_IOT_CA_PATH", "certs/AmazonRootCA1.pem")
AWS_DYNAMODB_TABLE = os.getenv("AWS_DYNAMODB_TABLE", "rapiro_sessions")
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET", "rapiro-user-documents")

# Throttling MQTT: máximo 1 mensaje cada N segundos
MQTT_THROTTLE_SECONDS = int(os.getenv("MQTT_THROTTLE_SECONDS", "5"))

# Umbral de ausencia para generar alerta SNS (segundos)
ABSENCE_ALERT_THRESHOLD_SEC = int(os.getenv("ABSENCE_ALERT_THRESHOLD_SEC", "300"))  # 5 min

# ---------------------------------------------------------------------------
# TUTORÍA LLM
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-haiku-4-5")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "500"))

# Segmentación de documentos
DOCUMENT_CHUNK_SIZE_WORDS = int(os.getenv("DOCUMENT_CHUNK_SIZE_WORDS", "500"))
DOCUMENT_TOP_K_CHUNKS = int(os.getenv("DOCUMENT_TOP_K_CHUNKS", "3"))

# Hotword para activar tutoría por voz
TUTOR_HOTWORD = os.getenv("TUTOR_HOTWORD", "RAPIRO ayuda")
SPEECH_LANGUAGE = os.getenv("SPEECH_LANGUAGE", "es-AR")

# ---------------------------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------------------------
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
