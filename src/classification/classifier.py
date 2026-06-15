"""
src/classification/classifier.py
Clasificador de estados del estudiante usando MobileNetV2 TFLite INT8.

Clases:
    0 — Estudiando / trabajando
    1 — Usando el celular
    2 — Puesto vacío
"""

import time
import logging
import numpy as np

from config.settings import MODEL_PATH, MIN_CONFIDENCE, CLASS_LABELS
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
    """

    def __init__(self, model_path: str = MODEL_PATH, min_confidence: float = MIN_CONFIDENCE):
        self.model_path = model_path
        self.min_confidence = min_confidence
        self._interpreter = None
        self._input_details = None
        self._output_details = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            import tflite_runtime.interpreter as tflite
            logger.info("Backend: tflite_runtime")
        except ImportError:
            try:
                import ai_edge_litert.interpreter as tflite
                logger.info("Backend: ai_edge_litert")
            except ImportError:
                try:
                    import tensorflow as tf
                    tflite = tf.lite
                    logger.info("Backend: tensorflow")
                except ImportError:
                    from src.classification import tflite_ctypes as tflite
                    logger.info("Backend: ctypes libtensorflow-lite.so")

        self._interpreter = tflite.Interpreter(model_path=self.model_path, num_threads=4)
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()
        logger.info("Modelo TFLite cargado: %s", self.model_path)

    def predict(self, frame: np.ndarray) -> "ClassificationResult | None":
        t_start = time.perf_counter()

        input_tensor = preprocess_frame(frame)  # float32 [0,1], shape (1,H,W,3)

        # Cuantizar si el modelo espera INT8 o UINT8
        in_det = self._input_details[0]
        if in_det["dtype"] == np.int8:
            scale, zero_point = in_det["quantization"]
            if scale == 0:
                scale = 1.0 / 127.5
                zero_point = -1
            input_tensor = np.clip(
                np.round(input_tensor / scale + zero_point), -128, 127
            ).astype(np.int8)
        elif in_det["dtype"] == np.uint8:
            input_tensor = (input_tensor * 255).clip(0, 255).astype(np.uint8)

        self._interpreter.set_tensor(in_det["index"], input_tensor)
        self._interpreter.invoke()
        output = self._interpreter.get_tensor(self._output_details[0]["index"])

        latency_ms = (time.perf_counter() - t_start) * 1000

        # Descuantizar salida si es INT8
        out_det = self._output_details[0]
        if out_det["dtype"] in (np.int8, np.uint8):
            o_scale, o_zero = out_det["quantization"]
            if o_scale == 0:
                o_scale = 1.0
            output = (output.astype(np.float32) - o_zero) * o_scale

        probabilities = output[0]
        class_id = int(np.argmax(probabilities))
        confidence = float(probabilities[class_id])

        result = ClassificationResult(class_id, confidence, latency_ms)
        logger.info("Predicción: clase=%d (%s) conf=%.2f lat=%.1fms",
                    class_id, result.label, confidence, latency_ms)

        if confidence < self.min_confidence:
            logger.debug("Confianza baja (%.2f < %.2f) — ignorado.", confidence, self.min_confidence)
            return None

        return result
