"""
src/perception/camera.py
Módulo de captura de video desde la cámara USB montada en RAPIRO.
"""

import os
import time
import logging
import threading
import cv2
import numpy as np
from config.settings import CAMERA_INDEX, CAMERA_FPS

logger = logging.getLogger(__name__)


class CameraCapture:
    """
    Gestiona la cámara USB/IP y provee fotogramas al pipeline.

    Para streams IP (DroidCam, IP Webcam), drena el buffer en background
    y siempre entrega el frame más reciente, evitando buffer overflow.

    Uso:
        with CameraCapture() as cam:
            for frame in cam.frames():
                process(frame)
    """

    def __init__(self, camera_index: int = CAMERA_INDEX, fps: int = CAMERA_FPS):
        url = os.getenv("CAMERA_URL", "")
        self.camera_source = url if url else camera_index
        self.fps = fps
        self._interval = 1.0 / fps
        self._cap: cv2.VideoCapture | None = None

        # Para streams IP: hilo que drena buffer continuamente
        self._is_stream = isinstance(self.camera_source, str) and self.camera_source.startswith("http")
        self._latest_frame: np.ndarray | None = None
        self._frame_lock = threading.Lock()
        self._drain_thread: threading.Thread | None = None
        self._running = False

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.camera_source)
        if not self._cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara: {self.camera_source}")
        if not self._is_stream:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        logger.info("Cámara abierta (fuente=%s, fps_target=%d)", self.camera_source, self.fps)

        if self._is_stream:
            self._running = True
            self._drain_thread = threading.Thread(target=self._drain_loop, daemon=True)
            self._drain_thread.start()

    def _drain_loop(self) -> None:
        """Drena el buffer del stream continuamente, guarda solo el frame más reciente."""
        while self._running:
            if not self._cap or not self._cap.isOpened():
                time.sleep(0.1)
                continue
            ret, frame = self._cap.read()
            if ret and frame is not None:
                with self._frame_lock:
                    self._latest_frame = frame
            else:
                time.sleep(0.05)

    def close(self) -> None:
        self._running = False
        if self._cap and self._cap.isOpened():
            self._cap.release()
            logger.info("Cámara liberada.")

    def read_frame(self) -> np.ndarray:
        if not self._cap or not self._cap.isOpened():
            raise RuntimeError("La cámara no está abierta.")
        if self._is_stream:
            # Esperar hasta 3s a que el hilo drain tenga un frame
            deadline = time.perf_counter() + 3.0
            while time.perf_counter() < deadline:
                with self._frame_lock:
                    if self._latest_frame is not None:
                        return self._latest_frame.copy()
                time.sleep(0.05)
            raise RuntimeError("Timeout esperando frame del stream.")
        ret, frame = self._cap.read()
        if not ret or frame is None:
            raise RuntimeError("Error al leer fotograma de la cámara.")
        return frame

    def frames(self):
        consecutive_errors = 0
        while True:
            t_start = time.perf_counter()
            try:
                yield self.read_frame()
                consecutive_errors = 0
            except RuntimeError as exc:
                consecutive_errors += 1
                logger.warning("Error captura #%d: %s", consecutive_errors, exc)
                if consecutive_errors > 20:
                    logger.error("Demasiados errores consecutivos, abortando.")
                    break
                time.sleep(2)
                continue
            elapsed = time.perf_counter() - t_start
            sleep_time = self._interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()
