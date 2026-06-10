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
    python demo.py --quiz            # modo tutor interactivo (quiz + explicaciones)
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
# Modo tutor interactivo
# ---------------------------------------------------------------------------

SAMPLE_TEXT = """
La fotosintesis es el proceso mediante el cual las plantas producen su propio alimento
usando luz solar, agua y dioxido de carbono. Ocurre principalmente en las hojas, en
estructuras llamadas cloroplastos que contienen clorofila, el pigmento verde responsable
de captar la energia luminica.

El proceso se divide en dos etapas: las reacciones luminosas (que ocurren en los tilacoides
y convierten la luz en energia quimica ATP y NADPH) y el Ciclo de Calvin (que ocurre en
el estroma y usa esa energia para fijar el CO2 en azucares como la glucosa).

La ecuacion general es: 6CO2 + 6H2O + luz -> C6H12O6 + 6O2

La glucosa producida sirve como fuente de energia para la planta y como materia prima
para construir celulosa, proteinas y otros compuestos organicos. El oxigeno liberado
es el subproducto que hace posible la vida animal en la Tierra.

La eficiencia de la fotosintesis depende de factores como la intensidad luminica,
la concentracion de CO2, la temperatura y la disponibilidad de agua y nutrientes.
"""


def run_quiz_mode():
    """Modo tutor interactivo: explicacion + quiz multiple choice."""
    print(f"\n{BOLD}{'='*58}{RESET}")
    print(f"{BOLD}   RAPIRO Guardian -- MODO TUTOR INTERACTIVO{RESET}")
    print(f"{BOLD}{'='*58}{RESET}\n")

    # Configurar tutor
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    os.environ.setdefault("MODEL_PATH", "models/mobilenetv2_int8.tflite")

    from src.tutoring.document_processor import DocumentProcessor
    from src.tutoring.tutor import IntelligentTutor

    doc = DocumentProcessor()
    n_chunks = doc.load_from_text(SAMPLE_TEXT)
    print(f"{GRAY}  Documento de ejemplo cargado ({n_chunks} chunks){RESET}")

    tutor = IntelligentTutor(doc)

    if not tutor._available():
        print(f"{RED}  ANTHROPIC_API_KEY no configurada.{RESET}")
        print(f"  Crea un archivo .env con: ANTHROPIC_API_KEY=sk-ant-...")
        return

    print(f"{GREEN}  Claude Haiku conectado{RESET}")
    print(f"{GRAY}  Tema de ejemplo: Fotosintesis{RESET}\n")

    # ------------------------------------------------------------------
    # Paso 1: Explicacion con ejemplos reales
    # ------------------------------------------------------------------
    print(f"{BOLD}[1/3] EXPLICACION DEL TEMA{RESET}")
    print(f"{GRAY}{'─'*58}{RESET}")
    input(f"{GRAY}  Presiona Enter para que RAPIRO explique el tema...{RESET}")
    print()

    explanation = tutor.explain_with_example("fotosintesis")
    print(explanation)
    print()

    # ------------------------------------------------------------------
    # Paso 2: Generar quiz
    # ------------------------------------------------------------------
    print(f"{BOLD}[2/3] QUIZ - Vamos a ver cuanto entendiste{RESET}")
    print(f"{GRAY}{'─'*58}{RESET}")
    input(f"{GRAY}  Presiona Enter para generar las preguntas...{RESET}")
    print(f"\n{GRAY}  Generando preguntas con Claude...{RESET}\n")

    quiz = tutor.generate_quiz("fotosintesis", n_questions=3)

    if not quiz.questions:
        print(f"{RED}  No se pudieron generar preguntas.{RESET}")
        return

    # ------------------------------------------------------------------
    # Paso 3: Quiz interactivo
    # ------------------------------------------------------------------
    results = []
    for i, q in enumerate(quiz.questions):
        print(f"{BOLD}Pregunta {i+1}/{len(quiz.questions)}:{RESET}")
        print(f"  {q.question}\n")
        for letter, option in q.options.items():
            print(f"  {BOLD}{letter}{RESET}) {option}")
        print()

        while True:
            answer = input(f"  Tu respuesta (A/B/C/D): ").strip().upper()
            if answer in ("A", "B", "C", "D"):
                break
            print(f"  {YELLOW}Ingresa A, B, C o D{RESET}")

        result = tutor.evaluate_answer(q, answer)
        results.append(result)

        if result.is_correct:
            print(f"\n  {GREEN}{BOLD}CORRECTO!{RESET}")
        else:
            print(f"\n  {RED}{BOLD}INCORRECTO. La respuesta era {q.correct}{RESET}")

        print(f"\n  {result.feedback}\n")
        print(f"{GRAY}{'─'*58}{RESET}\n")

    # ------------------------------------------------------------------
    # Paso 4: Puntaje final
    # ------------------------------------------------------------------
    print(f"{BOLD}[3/3] RESULTADO FINAL{RESET}")
    print(f"{GRAY}{'─'*58}{RESET}")
    correct = sum(1 for r in results if r.is_correct)
    total = len(results)
    pct = correct / total * 100

    color = GREEN if pct >= 70 else YELLOW if pct >= 40 else RED
    print(f"\n  Puntaje: {color}{BOLD}{correct}/{total} ({pct:.0f}%){RESET}\n")

    if pct == 100:
        print(f"  {GREEN}Perfecto! Dominaste el tema.{RESET}")
    elif pct >= 70:
        print(f"  {GREEN}Muy bien! Repasa los errores para llegar al 100%.{RESET}")
    elif pct >= 40:
        print(f"  {YELLOW}Buen intento. Te recomiendo releer la explicacion.{RESET}")
    else:
        print(f"  {RED}Necesitas repasar. Usa la opcion de explicacion nuevamente.{RESET}")

    print()


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
    parser.add_argument("--quiz",     action="store_true",
                        help="Modo tutor interactivo: explicacion + quiz")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.quiz:
        run_quiz_mode()
    else:
        run_demo(
            fixed_class=args.fixed_class,
            no_cam=args.no_cam,
            interval=args.interval,
        )
