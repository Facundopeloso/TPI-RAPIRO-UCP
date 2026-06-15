"""
demo.py
Pipeline completo RAPIRO sin hardware real.

Cámara (DroidCam/webcam) → TFLite → ventana GUI con ojos RAPIRO animados
→ DynamoDB → Claude Haiku → Polly TTS (audio)

Uso:
    python demo.py                   # pipeline real (CAMERA_URL del .env)
    python demo.py --no-cam          # frames sintéticos (sin cámara)
    python demo.py --class 1         # fuerza clase fija — modo mock (0/1/2)
    python demo.py --interval 8      # segundos entre clasificaciones (default: 8)
    python demo.py --quiz            # modo tutor interactivo

Requiere .env con: ANTHROPIC_API_KEY, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
                   CAMERA_URL (ej: http://192.168.100.6:4747/video)
"""

import argparse
import math
import random
import signal
import threading
import time
import sys
import os

# Fuerza UTF-8 en terminales Windows que usan cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("MODEL_PATH", "models/mobilenetv2_int8.tflite")

# Cargar .env (ANTHROPIC_API_KEY, AWS keys, CAMERA_URL, etc.)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=False)
except ImportError:
    pass

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
    3: (BLUE,   "[3] CONFUNDIDO    ", "LED azul x2  | Inclina cabeza (think) + re-explica"),
    4: (YELLOW, "[4] ABURRIDO      ", "LED amarillo | Sacude cabeza + genera quiz"),
}

# ---------------------------------------------------------------------------
# Pipeline real — configuración
# ---------------------------------------------------------------------------
try:
    from config.settings import CLASS_STUDYING, CLASS_PHONE, CLASS_ABSENT
except ImportError:
    CLASS_STUDYING, CLASS_PHONE, CLASS_ABSENT = 0, 1, 2

_SPEAK_COOLDOWN = float(os.getenv("SPEAK_COOLDOWN", "90"))
_last_spoken: dict = {}
_current_class: list = [-1]

_STATE_PROMPTS = {
    CLASS_PHONE: (
        "Sos RAPIRO, un robot compañero de estudio. Detectaste que el estudiante está mirando el celular. "
        "Decile algo corto y amigable para que lo deje y vuelva a estudiar. "
        "Máximo 2 oraciones, español rioplatense, sin markdown."
    ),
    CLASS_ABSENT: (
        "Sos RAPIRO, un robot compañero de estudio. El estudiante lleva un rato sin estar en su puesto. "
        "Mandá un mensaje corto para que vuelva. "
        "Máximo 2 oraciones, español rioplatense, sin markdown."
    ),
}

# ---------------------------------------------------------------------------
# Clasificador mock
# ---------------------------------------------------------------------------

class MockClassifier:
    """Devuelve resultados simulados sin modelo real."""

    def __init__(self, fixed_class: int | None = None):
        self._fixed = fixed_class
        # Probabilidades base para simular variabilidad
        self._probs = [0.60, 0.15, 0.10, 0.10, 0.05]

    def predict(self, frame: np.ndarray):
        if self._fixed is not None:
            class_id = self._fixed
            confidence = random.uniform(0.75, 0.98)
        else:
            class_id = random.choices([0, 1, 2, 3, 4], weights=self._probs)[0]
            confidence = random.uniform(0.68, 0.97)

        latency_ms = random.uniform(120, 280)
        return class_id, confidence, latency_ms


# ---------------------------------------------------------------------------
# Robot simulado (sin GPIO ni serial)
# ---------------------------------------------------------------------------

class SimulatedRAPIRO:
    """Reemplaza RAPIROController en demo: sin hardware real."""

    def __init__(self):
        self._last_class_id = CLASS_STUDYING

    def react(self, class_id: int) -> None:
        self._last_class_id = class_id

    def react_speaking(self) -> None:
        pass

    def stop_listening_effect(self) -> None:
        pass

    def cleanup(self) -> None:
        pass


def _speak_for_state_demo(class_id: int) -> None:
    """Llama Claude Haiku → Polly TTS. Corre en daemon thread."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return
    prompt = _STATE_PROMPTS.get(class_id)
    if not prompt:
        return
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        if _current_class[0] != class_id:
            print(f"  {GRAY}[TTS] Estado cambió — skip audio.{RESET}")
            return
        from src.tutoring.polly_tts import speak as polly_speak
        polly_speak(text)
    except Exception as exc:
        print(f"  {RED}[TTS] Error: {exc}{RESET}")


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


# ---------------------------------------------------------------------------
# Panel RAPIRO (robot visual)
# ---------------------------------------------------------------------------

_PANEL_W = 300
_PANEL_H = 480

_LED_COLORS = {
    0: (0, 210, 0),      # verde  — estudiando
    1: (0, 210, 210),    # cian   — celular
    2: (40, 40, 210),    # rojo   — ausente
    3: (210, 100, 0),    # azul   — confundido
    4: (0, 165, 210),    # naranja-amarillo — aburrido
}

_STATE_LABELS = {
    0: "ESTUDIANDO",
    1: "USANDO CELULAR",
    2: "PUESTO VACIO",
    3: "CONFUNDIDO",
    4: "ABURRIDO",
}


def _pulse(frame_n: int, speed: float = 0.12) -> float:
    return 0.55 + 0.45 * abs(math.sin(frame_n * speed))


def _scale(color: tuple, factor: float) -> tuple:
    return tuple(min(255, int(c * factor)) for c in color)


def draw_rapiro_panel(class_id: int, confidence: float, frame_n: int) -> np.ndarray:
    """Devuelve imagen BGR (_PANEL_W x _PANEL_H) con el robot RAPIRO dibujado."""
    panel = np.full((_PANEL_H, _PANEL_W, 3), (22, 22, 32), dtype=np.uint8)

    led = _LED_COLORS.get(class_id, (160, 160, 160))
    p   = _pulse(frame_n)
    lit = _scale(led, p)
    dim = _scale(led, p * 0.35)
    cx  = _PANEL_W // 2   # 150

    # --- HEAD offset por estado ---
    hx = cx
    hy = 108
    if class_id == 3:   # confundido → ladeado
        hx += 8
        hy -= 4
    elif class_id == 4:  # aburrido → caído
        hy += 7

    hw, hh = 82, 68

    # Antena
    ant_base = (hx, hy - hh // 2)
    ant_tip  = (hx, hy - hh // 2 - 20)
    cv2.line(panel, ant_base, ant_tip, (80, 80, 100), 2)
    cv2.circle(panel, ant_tip, 6, lit, -1)
    cv2.circle(panel, ant_tip, 8, dim, 1)

    # Cabeza
    cv2.rectangle(panel,
                  (hx - hw // 2, hy - hh // 2),
                  (hx + hw // 2, hy + hh // 2),
                  (52, 52, 68), -1)
    cv2.rectangle(panel,
                  (hx - hw // 2, hy - hh // 2),
                  (hx + hw // 2, hy + hh // 2),
                  (85, 85, 105), 2)

    # Ojos (LEDs)
    ey = hy - 10
    for ex in (hx - 22, hx + 22):
        cv2.ellipse(panel, (ex, ey), (14, 11), 0, 0, 360, (35, 35, 50), -1)
        cv2.circle(panel, (ex, ey), 9, lit, -1)
        cv2.circle(panel, (ex, ey), 12, dim, 1)

    # Boca según estado
    my = hy + 20
    if class_id == 0:                   # sonrisa
        cv2.ellipse(panel, (hx, my), (15, 7), 0, 0, 180, (85, 85, 105), 2)
    elif class_id in (2, 4):            # triste
        cv2.ellipse(panel, (hx, my + 9), (15, 7), 0, 180, 360, (85, 85, 105), 2)
    elif class_id == 3:                 # zigzag — confundido
        pts = np.array([(hx - 15, my + 4), (hx - 7, my),
                        (hx, my + 4), (hx + 7, my), (hx + 15, my + 4)], np.int32)
        cv2.polylines(panel, [pts], False, (85, 85, 105), 2)
    else:                               # línea neutra
        cv2.line(panel, (hx - 13, my + 4), (hx + 13, my + 4), (85, 85, 105), 2)

    # --- CUELLO ---
    nt = hy + hh // 2
    nb = nt + 14
    cv2.rectangle(panel, (hx - 9, nt), (hx + 9, nb), (52, 52, 68), -1)
    cv2.rectangle(panel, (hx - 9, nt), (hx + 9, nb), (75, 75, 95), 1)

    # --- TORSO ---
    bt = nb
    bb = bt + 88
    bw = 82
    cv2.rectangle(panel, (cx - bw // 2, bt), (cx + bw // 2, bb), (42, 42, 58), -1)
    cv2.rectangle(panel, (cx - bw // 2, bt), (cx + bw // 2, bb), (78, 78, 98), 2)

    # Panel pecho con LEDs
    pr_y = bt + 14
    cv2.rectangle(panel, (cx - 26, pr_y), (cx + 26, pr_y + 28), (28, 28, 42), -1)
    cv2.rectangle(panel, (cx - 26, pr_y), (cx + 26, pr_y + 28), (65, 65, 85), 1)
    for i, dx in enumerate((-16, -8, 0, 8, 16)):
        dot_col = lit if i == 2 else dim
        cv2.circle(panel, (cx + dx, pr_y + 14), 5, dot_col, -1)

    # --- BRAZOS ---
    at = bt + 6
    aw, ah = 17, 62

    def _arm(x1, y1, x2, y2):
        cv2.rectangle(panel, (x1, y1), (x2, y2), (48, 48, 64), -1)
        cv2.rectangle(panel, (x1, y1), (x2, y2), (72, 72, 92), 1)

    lax = cx - bw // 2 - aw   # left arm x1
    rax = cx + bw // 2         # right arm x1

    if class_id == 1:           # celular → brazo derecho arriba con cel
        _arm(lax, at, lax + aw, at + ah)
        _arm(rax, at - 38, rax + aw, at + 22)
        px = rax + aw
        cv2.rectangle(panel, (px, at - 56), (px + 22, at - 26), (25, 25, 25), -1)
        cv2.rectangle(panel, (px, at - 56), (px + 22, at - 26), (90, 90, 90), 1)
        cv2.rectangle(panel, (px + 2, at - 54), (px + 20, at - 28), (0, 70, 150), -1)

    elif class_id == 3:         # confundido → brazo derecho al lado de la cabeza + "?"
        _arm(lax, at, lax + aw, at + ah)
        _arm(rax, at - 22, rax + aw, at + 28)
        cv2.putText(panel, "?", (rax + aw + 4, at - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, lit, 2)

    elif class_id == 4:         # aburrido → brazos caídos
        _arm(lax, at + 18, lax + aw, at + ah + 22)
        _arm(rax, at + 18, rax + aw, at + ah + 22)

    elif class_id == 2:         # ausente → brazos levemente abiertos
        _arm(lax - 5, at, lax + aw - 5, at + ah)
        _arm(rax + 5, at, rax + aw + 5, at + ah)

    else:                       # neutro
        _arm(lax, at, lax + aw, at + ah)
        _arm(rax, at, rax + aw, at + ah)

    # --- PIERNAS ---
    lt  = bb
    lh  = 68
    lw  = 19
    gap = 7
    for lx in (cx - gap // 2 - lw, cx + gap // 2):
        cv2.rectangle(panel, (lx, lt), (lx + lw, lt + lh), (42, 42, 58), -1)
        cv2.rectangle(panel, (lx, lt), (lx + lw, lt + lh), (68, 68, 88), 1)

    # Pies
    ft = lt + lh
    fh = 13
    fp = 7    # protuberancia hacia adelante
    cv2.rectangle(panel, (cx - gap // 2 - lw - 3, ft), (cx - gap // 2 + fp, ft + fh), (38, 38, 54), -1)
    cv2.rectangle(panel, (cx - gap // 2 - lw - 3, ft), (cx - gap // 2 + fp, ft + fh), (62, 62, 82), 1)
    cv2.rectangle(panel, (cx + gap // 2 - fp, ft), (cx + gap // 2 + lw + 3, ft + fh), (38, 38, 54), -1)
    cv2.rectangle(panel, (cx + gap // 2 - fp, ft), (cx + gap // 2 + lw + 3, ft + fh), (62, 62, 82), 1)

    # --- ETIQUETA inferior ---
    font = cv2.FONT_HERSHEY_SIMPLEX
    lbl  = _STATE_LABELS.get(class_id, "")
    lsz  = cv2.getTextSize(lbl, font, 0.50, 1)[0]
    cv2.putText(panel, lbl, ((_PANEL_W - lsz[0]) // 2, _PANEL_H - 38), font, 0.50, lit, 1)

    conf_txt = f"{confidence * 100:.1f}%"
    csz = cv2.getTextSize(conf_txt, font, 0.72, 2)[0]
    cv2.putText(panel, conf_txt, ((_PANEL_W - csz[0]) // 2, _PANEL_H - 12), font, 0.72, (170, 170, 170), 2)

    # Separador izquierdo
    cv2.line(panel, (0, 0), (0, _PANEL_H), (55, 55, 75), 2)

    return panel


def show_opencv_window(frame: np.ndarray, class_id: int, confidence: float, frame_n: int = 0):
    """Muestra frame + panel RAPIRO en ventana OpenCV."""
    if not CV2_AVAILABLE:
        return

    COLOR_MAP = {0: (0, 200, 0), 1: (0, 200, 200), 2: (0, 0, 200), 3: (255, 140, 0), 4: (180, 180, 0)}
    LABEL_MAP = {0: "Estudiando", 1: "Usando celular", 2: "Puesto vacio", 3: "Confundido", 4: "Aburrido"}

    cam = cv2.resize(frame.copy(), (640, _PANEL_H))
    color = COLOR_MAP[class_id]
    label = f"{LABEL_MAP[class_id]}  {confidence*100:.1f}%"
    cv2.rectangle(cam, (0, 0), (cam.shape[1], 58), color, -1)
    cv2.putText(cam, label, (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    rapiro = draw_rapiro_panel(class_id, confidence, frame_n)
    combined = np.hstack([cam, rapiro])
    cv2.imshow("RAPIRO Demo", combined)


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def _get_state_color(class_id: int) -> str:
    return {CLASS_STUDYING: GREEN, CLASS_PHONE: YELLOW, CLASS_ABSENT: RED}.get(class_id, RESET)


def run_demo(fixed_class: int | None, no_cam: bool, interval: float):
    print_header()

    # --- Clasificador ---
    clf_real = None
    if fixed_class is None:
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from src.classification.classifier import StudentClassifier
            clf_real = StudentClassifier()
            print(f"{GREEN}OK Clasificador TFLite{RESET}")
        except Exception as e:
            print(f"{YELLOW}!! TFLite no disponible — mock ({e}){RESET}")
    clf_mock = MockClassifier(fixed_class) if clf_real is None else None
    if clf_mock is not None:
        print(f"{GRAY}  Clasificador: mock (sin modelo real){RESET}")

    # --- Cámara ---
    cap = None
    _latest_frame: list = [None]
    _frame_lock = threading.Lock()
    _drain_running = [True]
    _is_stream = False

    if not no_cam and CV2_AVAILABLE:
        url = os.getenv("CAMERA_URL", "")
        source: object = url if url else 0
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            _is_stream = bool(url)
            if _is_stream:
                def _drain():
                    while _drain_running[0]:
                        ret, f = cap.read()
                        if ret and f is not None:
                            with _frame_lock:
                                _latest_frame[0] = f
                        else:
                            time.sleep(0.05)
                threading.Thread(target=_drain, daemon=True).start()
            print(f"{GREEN}OK Cámara: {source}{RESET}")
        else:
            cap = None
            print(f"{YELLOW}!! Cámara no disponible — frames sintéticos{RESET}")
    elif not CV2_AVAILABLE:
        print(f"{YELLOW}!! opencv-python no instalado{RESET}")
    else:
        print(f"{GRAY}  --no-cam: frames sintéticos{RESET}")

    def _read_frame() -> np.ndarray:
        if cap is None or not CV2_AVAILABLE:
            return np.random.randint(0, 200, (480, 640, 3), dtype=np.uint8)
        if _is_stream:
            deadline = time.perf_counter() + 3.0
            while time.perf_counter() < deadline:
                with _frame_lock:
                    if _latest_frame[0] is not None:
                        return _latest_frame[0].copy()
                time.sleep(0.05)
            return np.zeros((480, 640, 3), dtype=np.uint8)
        ret, f = cap.read()
        return f if (ret and f is not None) else np.zeros((480, 640, 3), dtype=np.uint8)

    # --- Cloud ---
    cloud = None
    try:
        from src.cloud.boto3_publisher import Boto3Publisher
        cloud = Boto3Publisher()
        print(f"{GREEN}OK DynamoDB conectado{RESET}")
    except Exception as e:
        print(f"{YELLOW}!! DynamoDB no disponible: {e}{RESET}")

    print(f"\n{GRAY}Presioná ESC, cerrá la ventana, o Ctrl+C para detener.\n{RESET}")

    if CV2_AVAILABLE:
        cv2.namedWindow("RAPIRO Demo", cv2.WINDOW_NORMAL)

    # Estado pipeline
    last_class_id = -1
    pending_class_id = -1
    pending_count = 0
    CONFIRM_CYCLES = 2
    disp_class = CLASS_STUDYING
    disp_conf = 0.0
    frame_n = 0

    # Frame compartido: main loop captura, classify thread consume (evita lecturas concurrentes)
    _shared_frame: list = [None]
    _shared_frame_lock = threading.Lock()
    _clf_result: list = [None]
    _clf_lock = threading.Lock()
    _running = [True]
    _clf_ready = threading.Event()  # señal: hay frame nuevo para clasificar

    def _classify_loop():
        last_classify = 0.0
        while _running[0]:
            now = time.monotonic()
            if now - last_classify < interval:
                time.sleep(0.05)
                continue
            with _shared_frame_lock:
                frame = _shared_frame[0]
            if frame is None:
                time.sleep(0.05)
                continue
            last_classify = time.monotonic()
            try:
                if clf_real is not None:
                    res = clf_real.predict(frame)
                    if res is not None:
                        with _clf_lock:
                            _clf_result[0] = (res.class_id, res.confidence, res.latency_ms)
                else:
                    with _clf_lock:
                        _clf_result[0] = clf_mock.predict(frame)
            except Exception as e:
                print(f"  {GRAY}[CLF] {e}{RESET}")

    threading.Thread(target=_classify_loop, daemon=True).start()

    def _cleanup():
        _running[0] = False
        _drain_running[0] = False
        if cap is not None:
            cap.release()
        if CV2_AVAILABLE:
            cv2.destroyAllWindows()
        if cloud:
            try:
                cloud.disconnect()
            except Exception:
                pass

    signal.signal(signal.SIGTERM, lambda *_: (_cleanup(), sys.exit(0)))

    try:
        while True:
            frame = _read_frame()

            # Compartir frame con classify thread (único lector de cámara)
            with _shared_frame_lock:
                _shared_frame[0] = frame

            # Procesar última clasificación disponible
            with _clf_lock:
                raw = _clf_result[0]
                _clf_result[0] = None

            if raw is not None:
                raw_class, confidence, latency_ms = raw

                # Mapeo 3 estados: 0/3/4 → estudiar, 1 → celular, 2 → ausente
                if raw_class == CLASS_PHONE:
                    effective = CLASS_PHONE
                elif raw_class == CLASS_ABSENT:
                    effective = CLASS_ABSENT
                else:
                    effective = CLASS_STUDYING

                # Histéresis: confirmar N ciclos antes de cambiar
                if effective == pending_class_id:
                    pending_count += 1
                else:
                    pending_class_id = effective
                    pending_count = 1

                if pending_count >= CONFIRM_CYCLES:
                    _current_class[0] = effective
                    disp_conf = confidence

                    if effective != last_class_id:
                        disp_class = effective
                        last_class_id = effective
                        lbl = _STATE_LABELS.get(effective, "?")
                        col = _get_state_color(effective)
                        print(f"\n{col}{BOLD}[RAPIRO] {lbl}  ({confidence*100:.1f}%){RESET}")

                    if effective != CLASS_STUDYING:
                        now = time.monotonic()
                        if now - _last_spoken.get(effective, 0.0) >= _SPEAK_COOLDOWN:
                            _last_spoken[effective] = now
                            threading.Thread(
                                target=_speak_for_state_demo,
                                args=(effective,),
                                daemon=True,
                            ).start()

                    if cloud:
                        try:
                            from config.settings import CLASS_LABELS
                            cloud.publish_event(
                                class_id=effective,
                                label=CLASS_LABELS[effective],
                                confidence=confidence,
                                latency_ms=latency_ms,
                            )
                        except Exception as e:
                            print(f"  {GRAY}[Cloud] {e}{RESET}")

            if CV2_AVAILABLE:
                show_opencv_window(frame, disp_class, disp_conf, frame_n)
                key = cv2.waitKey(50) & 0xFF
                if key == 27:  # ESC
                    break
                # Cerrar si el usuario clica la X de la ventana
                if cv2.getWindowProperty("RAPIRO Demo", cv2.WND_PROP_VISIBLE) < 1:
                    break
            else:
                time.sleep(0.05)

            frame_n += 1

    except (KeyboardInterrupt, SystemExit):
        print(f"\n{GRAY}Demo detenido. Frames: {frame_n}{RESET}\n")
    finally:
        _cleanup()


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


def rapiro_action(msg: str) -> None:
    print(f"  {BLUE}{BOLD}[RAPIRO]{RESET} {GRAY}{msg}{RESET}")


# ---------------------------------------------------------------------------
# Quiz UI helpers
# ---------------------------------------------------------------------------

def _put_text_wrapped(img: np.ndarray, text: str, x: int, y: int, max_w: int,
                      font, scale: float, color: tuple,
                      thickness: int = 1, line_gap: int = 8) -> int:
    """Draws word-wrapped text. Returns y after last line."""
    if not text:
        return y
    words = text.split()
    lines: list[str] = []
    cur = ""
    for word in words:
        test = (cur + " " + word).strip()
        (tw, _), _ = cv2.getTextSize(test, font, scale, thickness)
        if tw <= max_w:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    (_, lh), _ = cv2.getTextSize("Ag", font, scale, thickness)
    for line in lines:
        cv2.putText(img, line, (x, y), font, scale, color, thickness, cv2.LINE_AA)
        y += lh + line_gap
    return y


def draw_quiz_panel(state: str, data: dict, frame_n: int) -> np.ndarray:
    """Renders left panel (640x480) for each quiz state."""
    panel = np.full((_PANEL_H, 640, 3), (18, 18, 28), dtype=np.uint8)
    F   = cv2.FONT_HERSHEY_SIMPLEX
    W   = 640
    PAD = 26

    if state == "LOADING":
        n_dots = (frame_n // 7) % 4
        txt = "Generando con Claude" + "." * n_dots + " " * (3 - n_dots)
        (tw, _), _ = cv2.getTextSize(txt, F, 0.74, 2)
        cv2.putText(panel, txt, ((W - tw) // 2, _PANEL_H // 2 - 18), F, 0.74, (130, 130, 158), 2, cv2.LINE_AA)
        sub = "Preparando explicacion y preguntas..."
        (sw, _), _ = cv2.getTextSize(sub, F, 0.46, 1)
        cv2.putText(panel, sub, ((W - sw) // 2, _PANEL_H // 2 + 22), F, 0.46, (65, 65, 88), 1, cv2.LINE_AA)

    elif state == "EXPLAINING":
        cv2.rectangle(panel, (0, 0), (W, 56), (25, 50, 25), -1)
        hdr = f"RAPIRO explica:  {data['topic'].upper()}"
        cv2.putText(panel, hdr, (PAD, 37), F, 0.68, (75, 205, 75), 2, cv2.LINE_AA)
        _put_text_wrapped(panel, data["explanation"],
                          PAD, 80, W - PAD * 2, F, 0.49, (192, 192, 210), 1, 10)
        hint = "[ ESPACIO ]  iniciar quiz"
        (hw, _), _ = cv2.getTextSize(hint, F, 0.47, 1)
        cv2.putText(panel, hint, ((W - hw) // 2, _PANEL_H - 18), F, 0.47, (60, 60, 85), 1, cv2.LINE_AA)

    elif state == "QUESTION":
        quiz  = data["quiz"]
        qi    = data["current_q"]
        q     = quiz.questions[qi]
        tot   = len(quiz.questions)
        cv2.rectangle(panel, (0, 0), (W, 56), (20, 38, 60), -1)
        hdr = f"PREGUNTA  {qi + 1}  /  {tot}"
        cv2.putText(panel, hdr, (PAD, 37), F, 0.78, (75, 145, 215), 2, cv2.LINE_AA)
        y = _put_text_wrapped(panel, q.question, PAD, 76, W - PAD * 2, F, 0.58, (212, 212, 228), 1, 9)
        y += 18
        opt_cols = {'A': (0, 170, 210), 'B': (205, 150, 0), 'C': (0, 190, 90), 'D': (165, 70, 195)}
        for letter in ('A', 'B', 'C', 'D'):
            line = f"  [{letter}]  {q.options[letter]}"
            (_, lh2), _ = cv2.getTextSize("Ag", F, 0.50, 1)
            cv2.rectangle(panel, (PAD - 6, y - lh2 - 4), (W - PAD + 6, y + 9), (28, 28, 42), -1)
            y = _put_text_wrapped(panel, line, PAD, y, W - PAD * 2, F, 0.50, opt_cols[letter], 1, 6)
            y += 10

        ms = data.get("mic_state", "idle")
        if ms == "listening":
            pulse_r = 8 + int(4 * abs(math.sin(frame_n * 0.25)))
            cv2.circle(panel, (PAD + 12, _PANEL_H - 40), pulse_r, (0, 50, 215), -1)
            cv2.putText(panel, "Escuchando...  di  A / B / C / D",
                        (PAD + 28, _PANEL_H - 34), F, 0.50, (80, 120, 220), 1, cv2.LINE_AA)
        elif ms == "speaking":
            cv2.putText(panel, "RAPIRO hablando...",
                        (PAD + 10, _PANEL_H - 34), F, 0.50, (60, 180, 60), 1, cv2.LINE_AA)
        else:
            hint = "Presiona  A / B / C / D"
            (hw, _), _ = cv2.getTextSize(hint, F, 0.47, 1)
            cv2.putText(panel, hint, ((W - hw) // 2, _PANEL_H - 18), F, 0.47, (60, 60, 85), 1, cv2.LINE_AA)

    elif state == "FEEDBACK":
        result  = data["last_result"]
        elapsed = data.get("feedback_elapsed", 0.0)
        is_ok   = result.is_correct
        hdr_bg  = (15, 75, 15)  if is_ok else (68, 15, 15)
        lbl     = "CORRECTO!"   if is_ok else "INCORRECTO"
        lbl_col = (50, 220, 50) if is_ok else (80, 80, 220)
        cv2.rectangle(panel, (0, 0), (W, 56), hdr_bg, -1)
        cv2.putText(panel, lbl, (PAD, 38), F, 1.0, lbl_col, 2, cv2.LINE_AA)
        sa_txt = f"Tu respuesta:  [{result.student_answer}]"
        cv2.putText(panel, sa_txt, (W - 228, 38), F, 0.50, lbl_col, 1, cv2.LINE_AA)
        y = 72
        q_short = result.question.question[:70] + ("..." if len(result.question.question) > 70 else "")
        cv2.putText(panel, q_short, (PAD, y), F, 0.45, (115, 115, 135), 1, cv2.LINE_AA)
        y += 28
        cl   = result.question.correct
        corr = f"Correcta:  [{cl}]  {result.question.options[cl]}"
        y = _put_text_wrapped(panel, corr, PAD, y, W - PAD * 2, F, 0.53, (50, 210, 50), 1, 8)
        y += 14
        _put_text_wrapped(panel, result.feedback, PAD, y, W - PAD * 2, F, 0.48, (192, 192, 210), 1, 9)
        # Progress bar (auto-advance)
        FSEC = 4.0
        prog = min(1.0, elapsed / FSEC)
        bw   = W - PAD * 2
        cv2.rectangle(panel, (PAD, _PANEL_H - 38), (PAD + bw, _PANEL_H - 28), (33, 33, 48), -1)
        cv2.rectangle(panel, (PAD, _PANEL_H - 38), (PAD + int(bw * prog), _PANEL_H - 28), lbl_col, -1)
        hint = "[ ESPACIO ] adelantar"
        (hw, _), _ = cv2.getTextSize(hint, F, 0.43, 1)
        cv2.putText(panel, hint, ((W - hw) // 2, _PANEL_H - 10), F, 0.43, (55, 55, 78), 1, cv2.LINE_AA)

    elif state == "SCORE":
        results = data["results"]
        correct = sum(1 for r in results if r.is_correct)
        total   = len(results)
        pct     = correct / total if total else 0
        p_col   = (50, 210, 50) if pct >= 0.7 else ((0, 190, 210) if pct >= 0.4 else (80, 80, 210))
        cv2.rectangle(panel, (0, 0), (W, 56), (20, 20, 36), -1)
        cv2.putText(panel, "RESULTADO FINAL", (PAD, 38), F, 0.82, (165, 165, 190), 2, cv2.LINE_AA)
        sc_txt = f"{correct} / {total}"
        (sw, _), _ = cv2.getTextSize(sc_txt, F, 2.8, 4)
        cv2.putText(panel, sc_txt, ((W - sw) // 2, 168), F, 2.8, p_col, 4, cv2.LINE_AA)
        pct_txt = f"{int(pct * 100)} %"
        (pw, _), _ = cv2.getTextSize(pct_txt, F, 1.1, 2)
        cv2.putText(panel, pct_txt, ((W - pw) // 2, 218), F, 1.1, p_col, 2, cv2.LINE_AA)
        msg = ("Excelente, la tenes clara!" if pct >= 0.7
               else "Bien encaminado, repasa los errores." if pct >= 0.4
               else "No te cuelgues, repasemos juntos.")
        (mw, _), _ = cv2.getTextSize(msg, F, 0.60, 1)
        cv2.putText(panel, msg, ((W - mw) // 2, 258), F, 0.60, (170, 170, 192), 1, cv2.LINE_AA)
        y = 292
        for i, r in enumerate(results):
            ok   = r.is_correct
            col  = (45, 190, 45) if ok else (80, 80, 205)
            mark = "OK" if ok else "XX"
            line = f"P{i + 1}: [{mark}]  respondi {r.student_answer}, correcta {r.question.correct}"
            cv2.putText(panel, line, (PAD, y), F, 0.44, col, 1, cv2.LINE_AA)
            y += 24
        hint = "[ ESPACIO ] o [ Q ] para salir"
        (hw, _), _ = cv2.getTextSize(hint, F, 0.47, 1)
        cv2.putText(panel, hint, ((W - hw) // 2, _PANEL_H - 15), F, 0.47, (55, 55, 78), 1, cv2.LINE_AA)

    return panel


# ---------------------------------------------------------------------------
# Modo tutor visual (OpenCV)
# ---------------------------------------------------------------------------

def run_quiz_mode(audio: bool = False):
    """Quiz completo integrado en ventana OpenCV con robot RAPIRO animado."""
    if not CV2_AVAILABLE:
        print(f"{RED}OpenCV no disponible. Instalar: pip install opencv-python{RESET}")
        return

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    os.environ.setdefault("MODEL_PATH", "models/mobilenetv2_int8.tflite")

    from src.tutoring.document_processor import DocumentProcessor
    from src.tutoring.tutor import IntelligentTutor

    doc = DocumentProcessor()
    doc.load_from_text(SAMPLE_TEXT)
    tutor = IntelligentTutor(doc)

    if not tutor._available():
        print(f"{RED}ANTHROPIC_API_KEY no configurada.{RESET}")
        return

    TOPIC        = "fotosintesis"
    FEEDBACK_SEC = 4.0

    state          = "LOADING"
    explanation    = ""
    quiz           = None
    current_q      = 0
    results: list  = []
    last_result    = None
    feedback_start = 0.0
    frame_n        = 0
    rapiro_vis     = 0

    # Generar contenido en background
    loading_done:       threading.Event = threading.Event()
    explanation_holder: list            = [None]
    quiz_holder:        list            = [None]

    def _generate():
        explanation_holder[0] = tutor.explain_with_example(TOPIC)
        q = tutor.generate_quiz(TOPIC, n_questions=3)
        quiz_holder[0] = q
        if not q or not q.questions:
            ctx = tutor._context(TOPIC, top_k=6)
            print(f"[DEBUG] Quiz vacio. ctx={len(ctx)} chars. API={tutor._available()}")
        else:
            print(f"[DEBUG] Quiz OK: {len(q.questions)} preguntas")
        loading_done.set()

    threading.Thread(target=_generate, daemon=True).start()

    # Estado del micrófono compartido entre threads
    # Valores: "idle" | "speaking" | "listening" | "heard:X" | "failed"
    mic_state:   list = ["idle"]
    question_id: list = [0]   # evita que respuestas stale de preguntas viejas se apliquen

    def _speak_and_listen(q, n: int, tot: int, qid: int):
        """Habla la pregunta y luego escucha la respuesta (en thread aparte)."""
        mic_state[0] = "speaking"
        tutor.speak(f"Pregunta {n} de {tot}. {q.question}")
        tutor.speak(f"A: {q.options['A']}. B: {q.options['B']}. "
                    f"C: {q.options['C']}. D: {q.options['D']}.")
        tutor.speak("¿Cuál te parece?")
        if question_id[0] != qid:
            return
        mic_state[0] = "listening"
        ans = tutor.listen_for_answer(timeout=12)
        if question_id[0] != qid:
            return
        mic_state[0] = f"heard:{ans}" if ans else "failed"

    def _retry_listen(qid: int):
        """Re-escucha después de no entender (sin volver a leer la pregunta)."""
        if question_id[0] != qid:
            return
        mic_state[0] = "speaking"
        tutor.speak("No te escuché. Decime la letra: A, B, C o D.")
        if question_id[0] != qid:
            return
        mic_state[0] = "listening"
        ans = tutor.listen_for_answer(timeout=12)
        if question_id[0] != qid:
            return
        mic_state[0] = f"heard:{ans}" if ans else "failed"

    def _render():
        data = {
            "topic":            TOPIC,
            "explanation":      explanation,
            "quiz":             quiz,
            "current_q":        current_q,
            "results":          results,
            "last_result":      last_result,
            "feedback_elapsed": time.time() - feedback_start if state == "FEEDBACK" else 0.0,
            "mic_state":        mic_state[0],
        }
        left  = draw_quiz_panel(state, data, frame_n)
        right = draw_rapiro_panel(rapiro_vis, 1.0, frame_n)
        cv2.imshow("RAPIRO Demo", np.hstack([left, right]))

    cv2.namedWindow("RAPIRO Demo")

    while True:
        _render()
        key = cv2.waitKey(50) & 0xFF
        frame_n += 1

        if key == 27:
            break

        # ── LOADING ─────────────────────────────────────────────────
        if state == "LOADING":
            if loading_done.is_set():
                explanation = explanation_holder[0] or ""
                quiz        = quiz_holder[0]
                state       = "EXPLAINING"
                rapiro_vis  = 0
                if audio and explanation:
                    threading.Thread(target=tutor.speak, args=(explanation,), daemon=True).start()

        # ── EXPLAINING ──────────────────────────────────────────────
        elif state == "EXPLAINING":
            rapiro_vis = 0
            any_key = 0 < key < 255  # cualquier tecla excepta timeout(255) y ESC(27)
            if any_key and quiz is not None and quiz.questions:
                state          = "QUESTION"
                current_q      = 0
                question_id[0] += 1
                mic_state[0]   = "idle"
                if audio:
                    qid = question_id[0]
                    threading.Thread(
                        target=_speak_and_listen,
                        args=(quiz.questions[0], 1, len(quiz.questions), qid),
                        daemon=True,
                    ).start()
            elif any_key and (quiz is None or not quiz.questions):
                print(f"{RED}Quiz vacio — revisa la API key o el documento.{RESET}")

        # ── QUESTION ────────────────────────────────────────────────
        elif state == "QUESTION":
            rapiro_vis = 0
            ans = None

            # Teclado (siempre disponible como fallback)
            if   key in (ord('a'), ord('A')): ans = 'A'
            elif key in (ord('b'), ord('B')): ans = 'B'
            elif key in (ord('c'), ord('C')): ans = 'C'
            elif key in (ord('d'), ord('D')): ans = 'D'

            # Micrófono
            if audio and ans is None:
                ms = mic_state[0]
                if ms.startswith("heard:"):
                    heard = ms.split(":", 1)[1]
                    if heard in ("A", "B", "C", "D"):
                        ans = heard
                    mic_state[0] = "idle"
                elif ms == "failed":
                    mic_state[0] = "idle"
                    qid = question_id[0]
                    threading.Thread(target=_retry_listen, args=(qid,), daemon=True).start()

            if ans:
                question_id[0] += 1          # invalida thread de escucha anterior
                mic_state[0]   = "idle"
                last_result    = tutor.evaluate_answer(quiz.questions[current_q], ans)
                results.append(last_result)
                rapiro_vis     = 0 if last_result.is_correct else 3
                feedback_start = time.time()
                state          = "FEEDBACK"
                if audio:
                    threading.Thread(target=tutor.speak, args=(last_result.feedback,), daemon=True).start()

        # ── FEEDBACK ────────────────────────────────────────────────
        elif state == "FEEDBACK":
            elapsed = time.time() - feedback_start
            if elapsed >= FEEDBACK_SEC or (0 < key < 255):
                current_q += 1
                if current_q >= len(quiz.questions):
                    pct        = sum(1 for r in results if r.is_correct) / len(results)
                    rapiro_vis = 0 if pct >= 0.7 else (4 if pct >= 0.4 else 3)
                    state      = "SCORE"
                    if audio:
                        c   = sum(1 for r in results if r.is_correct)
                        t   = len(results)
                        msg = (f"Muy bien, {c} de {t}. La tenés clara." if pct >= 0.7
                               else f"{c} de {t}. Bien encaminado." if pct >= 0.4
                               else f"{c} de {t}. Repasemos juntos.")
                        threading.Thread(target=tutor.speak, args=(msg,), daemon=True).start()
                else:
                    state          = "QUESTION"
                    rapiro_vis     = 0
                    question_id[0] += 1
                    mic_state[0]   = "idle"
                    if audio:
                        qid = question_id[0]
                        threading.Thread(
                            target=_speak_and_listen,
                            args=(quiz.questions[current_q], current_q + 1,
                                  len(quiz.questions), qid),
                            daemon=True,
                        ).start()

        # ── SCORE ───────────────────────────────────────────────────
        elif state == "SCORE":
            if 0 < key < 255:
                break

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="RAPIRO demo sin hardware")
    parser.add_argument("--no-cam",   action="store_true", help="No usar webcam")
    parser.add_argument("--class",    dest="fixed_class", type=int, choices=[0, 1, 2, 3, 4],
                        default=None, help="Forzar clase fija (0=estudio 1=celular 2=ausente 3=confundido 4=aburrido)")
    parser.add_argument("--interval", type=float, default=8.0,
                        help="Segundos entre clasificaciones (default: 8)")
    parser.add_argument("--quiz",     action="store_true",
                        help="Modo tutor interactivo: explicacion + quiz")
    parser.add_argument("--audio",    action="store_true",
                        help="Modo audio: RAPIRO habla y escucha por microfono (usar con --quiz)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.quiz:
        run_quiz_mode(audio=args.audio)
    else:
        run_demo(
            fixed_class=args.fixed_class,
            no_cam=args.no_cam,
            interval=args.interval,
        )
