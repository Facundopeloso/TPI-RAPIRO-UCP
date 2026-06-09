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
import signal
import sys
import time

from config.settings import LOG_LEVEL, LOG_FORMAT, ABSENCE_ALERT_THRESHOLD_SEC, CLASS_ABSENT
from src.perception.camera import CameraCapture
from src.classification.classifier import StudentClassifier
from src.actuation.rapiro import RAPIROController
from src.cloud.iot_publisher import IoTPublisher
from src.tutoring.document_processor import DocumentProcessor
from src.tutoring.tutor import IntelligentTutor

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAPIRO Guardián de Distracción")
    parser.add_argument("--no-cloud", action="store_true", help="Desactivar publicación AWS")
    parser.add_argument("--demo", action="store_true", help="Modo demo sin hardware real")
    parser.add_argument("--document", type=str, default=None, help="Documento para el tutor")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("RAPIRO Guardián de Distracción — Iniciando sistema")
    logger.info("=" * 60)

    rapiro = RAPIROController()
    classifier = StudentClassifier()
    cloud = IoTPublisher() if not args.no_cloud else None

    doc_processor = DocumentProcessor()
    if args.document:
        n_chunks = doc_processor.load_from_file(args.document)
        logger.info("Documento cargado: %d chunks indexados.", n_chunks)

    tutor = IntelligentTutor(doc_processor)

    def shutdown(sig, frame):
        logger.info("Señal de apagado recibida. Cerrando...")
        rapiro.cleanup()
        if cloud:
            cloud.disconnect()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    absence_start: float | None = None

    with CameraCapture() as camera:
        logger.info("Pipeline activo. Procesando frames...")

        for frame in camera.frames():
            result = classifier.predict(frame)

            if result is None:
                continue

            rapiro.react(result.class_id)

            if result.class_id == CLASS_ABSENT:
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
                cloud.publish_event(
                    class_id=result.class_id,
                    label=result.label,
                    confidence=result.confidence,
                    latency_ms=result.latency_ms,
                )


if __name__ == "__main__":
    main()
