"""
src/classification/classifier.py
Clasificador de estados del estudiante usando MobileNetV2 TFLite INT8.

Clases:
    0 — Estudiando / trabajando
    1 — Usando el celular
    2 — Puesto vacío

El clasificador aplica un umbral de confianza mínima (MIN_CONFIDENCE).
Si la confianza es menor al umbral, devuelve None para evitar falsos positivos.
"""

import time
import logging
import numpy as np

from config.settings import MODEL_PATH, NUM_CLASSES, MIN_CONFIDENCE, CLASS_LABELS
from src.classification.preprocessor import preprocess_frame

logger = logging.getLogger(__name__)


class ClassificationResult:
    """Resultado de una clasificación."""

    def __init__(self, class_id: int, confidence: float, latency_ms: float):
        self.class_id = class_id
        self.confidence = confidence
        self.latency_ms = latency_ms
        self.label = CLASS_LABELS.get(class_id, "Desconocido")

    def __repr__(self) -> str:
        return (
            f"ClassificationResult(class={self.class_id} '{self.label}', "
            f"conf={self.confidence:.2f}, latency={self.latency_ms:.1f}ms)"
        )


class StudentClassifier:
    """
    Clasifica el estado del estudiante en tiempo real usando TFLite.

    Carga el modelo en memoria al instanciar y reutiliza el intérprete
    para minimizar overhead entre inferencias.
    """

    def __init__(self, model_path: str = MODEL_PATH, min_confidence: float = MIN_CONFIDENCE):
        self.model_path = model_path
        self.min_confidence = min_confidence
        self._interpreter = None
        self._input_details = None
        self._output_details = None
        self._load_model()

    def _load_model(self) -> None:
        """Carga el modelo TFLite e inicializa el intérprete."""
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            # Fallback a TensorFlow completo (entorno de desarrollo)
            import tensorflow as tf
            tflite = tf.lite

        self._interpreter = tflite.Interpreter(model_path=self.model_path)
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()
        logger.info("Modelo TFLite cargado: %s", self.model_path)

    def predict(self, frame: np.ndarray) -> ClassificationResult | None:
        """
        Clasifica un fotograma.

        Args:
            frame: Imagen BGR cruda (H, W, 3).

        Returns:
            ClassificationResult si la confianza supera MIN_CONFIDENCE,
            None en caso contrario.
        """
        t_start = time.perf_counter()

        # Preprocesamiento
        input_tensor = preprocess_frame(frame)

        # Inferencia
        self._interpreter.set_tensor(self._input_details[0]["index"], input_tensor)
        self._interpreter.invoke()
        output = self._interpreter.get_tensor(self._output_details[0]["index"])

        latency_ms = (time.perf_counter() - t_start) * 1000

        # Extraer clase y confianza
        probabilities = output[0]  # shape: (NUM_CLASSES,)
        class_id = int(np.argmax(probabilities))
        confidence = float(probabilities[class_id])

        result = ClassificationResult(class_id, confidence, latency_ms)
        logger.debug("Predicción: %s", result)

        if confidence < self.min_confidence:
            logger.debug(
                "Confianza %.2f < umbral %.2f — sin acción.", confidence, self.min_confidence
            )
            return None

        return result
