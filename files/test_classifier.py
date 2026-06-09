"""
tests/test_classifier.py
Tests unitarios del módulo de clasificación.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Test: preprocessor
# ---------------------------------------------------------------------------

def test_preprocess_output_shape():
    """El preprocesador debe retornar tensor de forma (1, 224, 224, 3)."""
    from src.classification.preprocessor import preprocess_frame
    fake_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    result = preprocess_frame(fake_frame)
    assert result.shape == (1, 224, 224, 3), f"Forma inesperada: {result.shape}"


def test_preprocess_normalization():
    """Los valores deben estar en el rango [0, 1]."""
    from src.classification.preprocessor import preprocess_frame
    fake_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    result = preprocess_frame(fake_frame)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_preprocess_dtype():
    """El dtype del tensor de salida debe ser float32."""
    from src.classification.preprocessor import preprocess_frame
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = preprocess_frame(fake_frame)
    assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# Test: StudentClassifier
# ---------------------------------------------------------------------------

def _make_mock_interpreter(output_probs):
    """Crea un intérprete TFLite mockeado con probabilidades dadas."""
    interp = MagicMock()
    interp.get_input_details.return_value = [{"index": 0}]
    interp.get_output_details.return_value = [{"index": 0}]
    interp.get_tensor.return_value = np.array([output_probs], dtype=np.float32)
    return interp


@patch("src.classification.classifier.tflite_runtime", create=True)
@patch("src.classification.classifier.StudentClassifier._load_model")
def test_predict_high_confidence(mock_load, _mock_tflite):
    """Con alta confianza, debe retornar un ClassificationResult válido."""
    from src.classification.classifier import StudentClassifier, ClassificationResult
    clf = StudentClassifier.__new__(StudentClassifier)
    clf.min_confidence = 0.70
    clf._interpreter = _make_mock_interpreter([0.05, 0.90, 0.05])
    clf._input_details = [{"index": 0}]
    clf._output_details = [{"index": 0}]

    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = clf.predict(fake_frame)

    assert result is not None
    assert isinstance(result, ClassificationResult)
    assert result.class_id == 1
    assert result.confidence == pytest.approx(0.90)


@patch("src.classification.classifier.StudentClassifier._load_model")
def test_predict_low_confidence_returns_none(_mock_load):
    """Con confianza menor al umbral, debe retornar None."""
    from src.classification.classifier import StudentClassifier
    clf = StudentClassifier.__new__(StudentClassifier)
    clf.min_confidence = 0.70
    clf._interpreter = _make_mock_interpreter([0.40, 0.35, 0.25])
    clf._input_details = [{"index": 0}]
    clf._output_details = [{"index": 0}]

    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = clf.predict(fake_frame)

    assert result is None


def test_classification_result_label():
    """ClassificationResult debe asignar el label correcto."""
    from src.classification.classifier import ClassificationResult
    r0 = ClassificationResult(class_id=0, confidence=0.91, latency_ms=150)
    r1 = ClassificationResult(class_id=1, confidence=0.88, latency_ms=150)
    r2 = ClassificationResult(class_id=2, confidence=0.90, latency_ms=150)

    assert r0.label == "Estudiando"
    assert r1.label == "Usando celular"
    assert r2.label == "Puesto vacío"
