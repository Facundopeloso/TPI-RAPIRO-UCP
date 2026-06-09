"""
src/actuation/rapiro.py
Facade que unifica el control del robot RAPIRO (LEDs + servos).
"""

import logging
import threading
from config.settings import (
    CLASS_STUDYING, CLASS_PHONE, CLASS_ABSENT,
    DO_NOT_DISTURB_DURATION_SEC
)
from src.actuation.led_controller import LEDController
from src.actuation.servo_controller import ServoController

logger = logging.getLogger(__name__)


class RAPIROController:
    """Controlador de alto nivel del robot RAPIRO."""

    def __init__(self):
        self._leds = LEDController()
        self._servos = ServoController()
        self._do_not_disturb = False
        self._dnd_timer: threading.Timer | None = None

    def react(self, class_id: int) -> None:
        if self._do_not_disturb:
            logger.debug("Modo no molestar activo — sin reacción.")
            return

        if class_id == CLASS_STUDYING:
            self._react_studying()
        elif class_id == CLASS_PHONE:
            self._react_phone()
        elif class_id == CLASS_ABSENT:
            self._react_absent()
        else:
            logger.warning("Clase desconocida: %d", class_id)

    def activate_tutoring(self) -> None:
        self._leds.set_tutoring()
        self._servos.look_at_user()

    def deactivate_tutoring(self) -> None:
        self._leds.set_studying()
        self._servos.neutral()

    def enable_do_not_disturb(self) -> None:
        self._do_not_disturb = True
        self._leds.turn_off()
        self._servos.neutral()
        logger.info("Modo no molestar activado por %d segundos.", DO_NOT_DISTURB_DURATION_SEC)

        if self._dnd_timer:
            self._dnd_timer.cancel()
        self._dnd_timer = threading.Timer(
            DO_NOT_DISTURB_DURATION_SEC, self._disable_do_not_disturb
        )
        self._dnd_timer.daemon = True
        self._dnd_timer.start()

    def cleanup(self) -> None:
        self._leds.cleanup()
        self._servos.close()
        if self._dnd_timer:
            self._dnd_timer.cancel()

    def _react_studying(self) -> None:
        self._leds.set_studying()
        self._servos.neutral()

    def _react_phone(self) -> None:
        self._leds.set_phone()
        self._servos.head_shake()

    def _react_absent(self) -> None:
        self._leds.set_absent()
        self._servos.alert_pose()

    def _disable_do_not_disturb(self) -> None:
        self._do_not_disturb = False
        logger.info("Modo no molestar desactivado.")
