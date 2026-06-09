"""
demo.py
Modo demo del sistema RAPIRO — sin hardware real.

Simula el pipeline completo:
  Cámara (webcam real o sintética) → Clasificador mock → Actuación en consola

Uso:
    python demo.py                   # webcam + clasificador aleatorio
    python demo.py --no-cam          # frames sintéticos (sin webcam)
    python demo.py --class 1         # fuerza clase fija (0/1/2)
    python demo.py --interval 2      # segundos entre clasificaciones
"""

import argparse
import random
import time
import sys
import os

# Fuerza UTF-8 en terminales Windows que usan cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("MODEL_PATH", "models/mobilenetv2_int8.tflite")

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# ---------------------------------------------------------------------------
# Colores ANSI para terminal
# ---------------------------------------------------------------------------
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BLUE   = "\033[94m"
RESET  = "\033[0m"
BOLD   = "\033[1m"
GRAY   = "\033[90m"

CLASS_STYLE = {
    0: (GREEN,  "[0] ESTUDIANDO    ", "LED verde    | Servos neutros"),
    1: (YELLOW, "[1] USANDO CELULAR", "LED amarillo | Mueve cabeza"),
    2: (RED,    "[2] PUESTO VACIO  ", "LED rojo     | Pose de alerta"),
}

# ---------------------------------------------------------------------------
# Clasificador mock
# ---------------------------------------------------------------------------

class MockClassifier:
    """Devuelve resultados simulados sin modelo real."""

    def __init__(self, fixed_class: int | None = None):
        self._fixed = fixed_class
        # Probabilidades base para simular variabilidad
        self._probs = [0.7, 0.15, 0.15]

    def predict(self, frame: np.ndarray):
        if self._fixed is not None:
            class_id = self._fixed
            confidence = random.uniform(0.75, 0.98)
        else:
            class_id = random.choices([0, 1, 2], weights=self._probs)[0]
            confidence = random.uniform(0.68, 0.97)

        latency_ms = random.uniform(120, 280)
        return class_id, confidence, latency_ms


# ---------------------------------------------------------------------------
# Captura de frames
# ---------------------------------------------------------------------------

def get_frame(cap) -> np.ndarray:
    """Devuelve frame de webcam o sintético."""
    if cap is not None and CV2_AVAILABLE:
        ret, frame = cap.read()
        if ret and frame is not None:
            return frame
    # Frame sintético: ruido RGB
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------

def print_header():
    print(f"\n{BOLD}{'='*58}{RESET}")
    print(f"{BOLD}   RAPIRO Guardian -- MODO DEMO{RESET}")
    print(f"{BOLD}{'='*58}{RESET}")
    print(f"{GRAY}   TPI Intercatedra 5to anio -- UCP 2026{RESET}")
    print(f"{GRAY}   Hardware simulado | AWS offline | Sin modelo TFLite{RESET}")
    print(f"{BOLD}{'='*58}{RESET}\n")


def print_detection(frame_n: int, class_id: int, confidence: float, latency_ms: float):
    color, label, action = CLASS_STYLE[class_id]
    bar_len = int(confidence * 20)
    bar = "#" * bar_len + "-" * (20 - bar_len)

    print(f"{GRAY}Frame #{frame_n:04d}{RESET}  {color}{BOLD}{label}{RESET}")
    print(f"  Confianza : {color}[{bar}]{RESET} {confidence*100:.1f}%")
    print(f"  Latencia  : {latency_ms:.1f} ms")
    print(f"  Actuación : {BOLD}{action}{RESET}")
    print()


def show_opencv_window(frame: np.ndarray, class_id: int, confidence: float):
    """Muestra frame en ventana OpenCV con overlay de clasificación."""
    if not CV2_AVAILABLE:
        return

    COLOR_MAP = {0: (0, 200, 0), 1: (0, 200, 200), 2: (0, 0, 200)}
    LABEL_MAP = {0: "Estudiando", 1: "Usando celular", 2: "Puesto vacio"}

    display = frame.copy()
    color = COLOR_MAP[class_id]
    label = f"{LABEL_MAP[class_id]}  {confidence*100:.1f}%"

    cv2.rectangle(display, (0, 0), (frame.shape[1], 60), color, -1)
    cv2.putText(display, label, (10, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.imshow("RAPIRO Demo", display)
    cv2.waitKey(1)


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_demo(fixed_class: int | None, no_cam: bool, interval: float):
    print_header()

    # Abrir webcam
    cap = None
    if not no_cam and CV2_AVAILABLE:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print(f"{GREEN}OK Webcam abierta{RESET}")
        else:
            cap = None
            print(f"{YELLOW}!! Webcam no disponible -- usando frames sinteticos{RESET}")
    elif not CV2_AVAILABLE:
        print(f"{YELLOW}!! opencv-python no instalado -- usando frames sinteticos{RESET}")
    else:
        print(f"{GRAY}  Modo --no-cam: frames sintéticos{RESET}")

    print(f"{GRAY}  Clasificador: mock (sin modelo TFLite){RESET}")
    print(f"{GRAY}  AWS IoT: offline (logs locales){RESET}")
    print(f"\n{GRAY}Presioná Ctrl+C para detener.\n{RESET}")

    clf = MockClassifier(fixed_class=fixed_class)
    frame_n = 0
    absence_streak = 0

    try:
        while True:
            frame = get_frame(cap)
            class_id, confidence, latency_ms = clf.predict(frame)

            print_detection(frame_n, class_id, confidence, latency_ms)

            if CV2_AVAILABLE and (cap is not None or not no_cam):
                show_opencv_window(frame, class_id, confidence)

            # Simular alerta de ausencia prolongada
            if class_id == 2:
                absence_streak += 1
                if absence_streak >= 3:
                    print(f"  {RED}{BOLD}!! ALERTA SNS: ausencia prolongada detectada{RESET}\n")
                    absence_streak = 0
            else:
                absence_streak = 0

            frame_n += 1
            time.sleep(interval)

    except KeyboardInterrupt:
        print(f"\n{GRAY}Demo detenido. Frames procesados: {frame_n}{RESET}\n")
    finally:
        if cap is not None:
            cap.release()
        if CV2_AVAILABLE:
            cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="RAPIRO demo sin hardware")
    parser.add_argument("--no-cam",   action="store_true", help="No usar webcam")
    parser.add_argument("--class",    dest="fixed_class", type=int, choices=[0, 1, 2],
                        default=None, help="Forzar clase fija (0=estudio 1=celular 2=ausente)")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="Segundos entre clasificaciones (default: 2)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_demo(
        fixed_class=args.fixed_class,
        no_cam=args.no_cam,
        interval=args.interval,
    )
