"""
src/classification/preprocessor.py
Preprocesamiento de frames para inferencia con MobileNetV2 TFLite.
"""

import cv2
import numpy as np
from config.settings import INPUT_SIZE


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Convierte un frame BGR crudo al tensor de entrada del modelo.

    Args:
        frame: Imagen BGR (H, W, 3), dtype uint8.

    Returns:
        Tensor float32 de forma (1, 224, 224, 3), valores en [0, 1].
    """
    resized = cv2.resize(frame, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)
