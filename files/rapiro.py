"""
src/actuation/rapiro.py
Facade que unifica el control del robot RAPIRO (LEDs + servos + altavoz).

Expone una interfaz de alto nivel orientada a los estados del clasificador,
ocultando los detalles de GPIO y comunicación serial.
"""

import time
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
    """
    Controlador de alto nivel del robot RAPIRO.

    Método principal: `react(class_id)` — recibe la clase detectada
    y ejecuta la respuesta correspondiente.
    """

    def __init__(self):
        self._leds = LEDController()
        self._servos = ServoController()
        self._do_not_disturb = False
        self._dnd_timer: threading.Timer | None = None

    # ------------------------------------------------------------------
    # Reacción según clase detectada
    # ------------------------------------------------------------------

    def react(self, class_id: int) -> None:
        """
        Ejecuta la respuesta física/lumínica según la clase del clasificador.

        Args:
            class_id: 0 (estudiando), 1 (celular), 2 (ausente)
        """
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
        """Activa la señal visual/física de tutoría LLM."""
        self._leds.set_tutoring()
        self._servos.look_at_user()
        logger.info("Tutoría activada en RAPIRO.")

    def deactivate_tutoring(self) -> None:
        """Vuelve al estado neutral después de la tutoría."""
        self._leds.set_studying()
        self._servos.neutral()

    def enable_do_not_disturb(self) -> None:
        """
        Activa el modo 'no molestar' por DO_NOT_DISTURB_DURATION_SEC segundos.
        Todas las reacciones físicas quedan suspendidas.
        """
        self._do_not_disturb = True
        self._leds.turn_off()
        self._servos.neutral()
        logger.info("Modo no molestar activado por %d segundos.", DO_NOT_DISTURB_DURATION_SEC)

        # Timer para desactivar automáticamente
        if self._dnd_timer:
            self._dnd_timer.cancel()
        self._dnd_timer = threading.Timer(
            DO_NOT_DISTURB_DURATION_SEC, self._disable_do_not_disturb
        )
        self._dnd_timer.daemon = True
        self._dnd_timer.start()

    def cleanup(self) -> None:
        """Libera todos los recursos de hardware."""
        self._leds.cleanup()
        self._servos.close()
        if self._dnd_timer:
            self._dnd_timer.cancel()

    # ------------------------------------------------------------------
    # Reacciones internas
    # ------------------------------------------------------------------

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
