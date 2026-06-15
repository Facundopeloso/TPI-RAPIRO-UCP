"""
main.py
Punto de entrada principal del sistema RAPIRO Guardián de Distracción.

Uso:
    python main.py
    python main.py --no-cloud        # Sin publicación AWS
    python main.py --demo            # Modo demo (no requiere hardware)
"""

import argparse
import logging
import os
import signal
import sys
import threading
import time

from config.settings import (
    LOG_LEVEL, LOG_FORMAT, ABSENCE_ALERT_THRESHOLD_SEC,
    CLASS_ABSENT, CLASS_STUDYING, CLASS_PHONE,
)
from src.perception.camera import CameraCapture
from src.classification.classifier import StudentClassifier
from src.actuation.rapiro import RAPIROController
from src.cloud.boto3_publisher import Boto3Publisher

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

_SPEAK_COOLDOWN = float(os.getenv("SPEAK_COOLDOWN", "90"))
_last_spoken: dict[int, float] = {}
_current_class: list[int] = [-1]  # mutable para que threads lean estado actual

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


def _load_study_notes() -> str:
    """Carga study_notes.txt si existe (contexto opcional para Claude)."""
    try:
        with open("study_notes.txt", encoding="utf-8") as f:
            return f.read(2000)
    except FileNotFoundError:
        return ""
    except Exception as exc:
        logger.debug("No se pudo leer study_notes.txt: %s", exc)
        return ""


def _speak_for_state(class_id: int, rapiro) -> None:
    """Llama a Claude Haiku y habla recomendación según estado. Corre en daemon thread."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return
    base_prompt = _STATE_PROMPTS.get(class_id)
    if not base_prompt:
        return

    notes = _load_study_notes()
    prompt = (
        f"Material de estudio del alumno:\n{notes}\n\n{base_prompt}"
        if notes else base_prompt
    )

    try:
        import anthropic
        rapiro.react_speaking()
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = resp.content[0].text.strip()
        logger.info("Tutor (clase=%d): %s", class_id, answer[:120])
        # Verificar que el estado no cambió mientras Claude respondía
        if _current_class[0] != class_id:
            logger.info("Estado cambió durante respuesta Claude (era=%d, ahora=%d) — skip audio.", class_id, _current_class[0])
            return
        from src.tutoring.polly_tts import speak as polly_speak
        polly_speak(answer)
    except Exception as exc:
        logger.error("Error tutor: %s", exc)
    finally:
        rapiro.stop_listening_effect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAPIRO Guardián de Distracción")
    parser.add_argument("--no-cloud", action="store_true", help="Desactivar publicación AWS")
    parser.add_argument("--demo", action="store_true", help="Modo demo sin hardware real")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("RAPIRO Guardián de Distracción — Iniciando sistema")
    logger.info("=" * 60)

    rapiro = RAPIROController()
    classifier = StudentClassifier()
    cloud = Boto3Publisher() if not args.no_cloud else None

    def shutdown(sig, frame):
        logger.info("Señal de apagado recibida. Cerrando...")
        if cloud:
            try:
                cloud.disconnect()
            except Exception:
                pass
        os._exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    absence_start: float | None = None
    last_class_id: int = -1
    pending_class_id: int = -1
    pending_count: int = 0
    CONFIRM_CYCLES = 2  # detectar mismo estado N veces seguidas antes de cambiar

    with CameraCapture() as camera:
        logger.info("Pipeline activo. Procesando frames...")

        for frame in camera.frames():
            result = classifier.predict(frame)

            time.sleep(8)

            if result is None:
                continue

            absent_threshold = float(os.getenv("MIN_CONFIDENCE_ABSENT", "0.20"))
            if result.class_id == CLASS_ABSENT and result.confidence < absent_threshold:
                logger.debug("Ausente ignorado: conf=%.2f < %.2f", result.confidence, absent_threshold)
                continue

            # 3 estados: clase=0,3,4 → verde (persona presente), clase=1 → amarillo, clase=2 → rojo
            if result.class_id == CLASS_PHONE:
                effective_class = CLASS_PHONE
            elif result.class_id == CLASS_ABSENT:
                effective_class = CLASS_ABSENT
            else:
                effective_class = CLASS_STUDYING  # 0, 3 (confundido), 4 (aburrido) = persona presente
            # Histéresis: confirmar estado N ciclos seguidos antes de cambiar
            if effective_class == pending_class_id:
                pending_count += 1
            else:
                pending_class_id = effective_class
                pending_count = 1

            if pending_count >= CONFIRM_CYCLES:
                _current_class[0] = effective_class
                if effective_class != last_class_id:
                    rapiro.react(effective_class)
                    last_class_id = effective_class

                # Tutor: solo hablar para celular y ausente (no para estudiando)
                if effective_class != CLASS_STUDYING:
                    now = time.monotonic()
                    if now - _last_spoken.get(effective_class, 0.0) >= _SPEAK_COOLDOWN:
                        _last_spoken[effective_class] = now
                        threading.Thread(
                            target=_speak_for_state,
                            args=(effective_class, rapiro),
                            daemon=True,
                        ).start()

            if effective_class == CLASS_ABSENT:
                if absence_start is None:
                    absence_start = time.monotonic()
                elif time.monotonic() - absence_start > ABSENCE_ALERT_THRESHOLD_SEC:
                    logger.warning(
                        "Ausencia prolongada detectada (>%d seg).", ABSENCE_ALERT_THRESHOLD_SEC
                    )
                    absence_start = None
            else:
                absence_start = None

            if cloud:
                from config.settings import CLASS_LABELS
                cloud.publish_event(
                    class_id=effective_class,
                    label=CLASS_LABELS[effective_class],
                    confidence=result.confidence,
                    latency_ms=result.latency_ms,
                )


if __name__ == "__main__":
    main()
