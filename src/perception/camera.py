"""
src/perception/camera.py
Módulo de captura de video desde la cámara USB montada en RAPIRO.
"""

import time
import logging
import cv2
import numpy as np
from config.settings import CAMERA_INDEX, CAMERA_FPS

logger = logging.getLogger(__name__)


class CameraCapture:
    """
    Gestiona la cámara USB y provee fotogramas al pipeline.

    Uso:
        with CameraCapture() as cam:
            for frame in cam.frames():
                process(frame)
    """

    def __init__(self, camera_index: int = CAMERA_INDEX, fps: int = CAMERA_FPS):
        self.camera_index = camera_index
        self.fps = fps
        self._interval = 1.0 / fps
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"No se pudo abrir la cámara en índice {self.camera_index}"
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        logger.info("Cámara abierta (índice=%d, fps_target=%d)", self.camera_index, self.fps)

    def close(self) -> None:
        if self._cap and self._cap.isOpened():
            self._cap.release()
            logger.info("Cámara liberada.")

    def read_frame(self) -> np.ndarray:
        if not self._cap or not self._cap.isOpened():
            raise RuntimeError("La cámara no está abierta.")
        ret, frame = self._cap.read()
        if not ret or frame is None:
            raise RuntimeError("Error al leer fotograma de la cámara.")
        return frame

    def frames(self):
        frame_duration = self._interval
        while True:
            t_start = time.perf_counter()
            try:
                yield self.read_frame()
            except RuntimeError as exc:
                logger.error("Error en captura: %s", exc)
                break
            elapsed = time.perf_counter() - t_start
            sleep_time = frame_duration - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()
