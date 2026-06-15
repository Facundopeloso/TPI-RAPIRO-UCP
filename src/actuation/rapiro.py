"""
src/actuation/rapiro.py
Facade que unifica el control del robot RAPIRO (LEDs + servos).

Reacciones según contexto:
  Clasificador  — react(class_id)
  Tutor         — react_explaining, react_thinking, react_asking_question,
                  react_correct_answer, react_wrong_answer, react_quiz_score
  Voz           — react_listening, react_hotword_detected
"""

import logging
import threading
from config.settings import (
    CLASS_STUDYING, CLASS_PHONE, CLASS_ABSENT, CLASS_CONFUSED, CLASS_BORED,
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
        self._last_class_id: int = CLASS_STUDYING
        self._servos.neutral()

    # ==================================================================
    # CLASIFICADOR — reacciones según estado detectado
    # ==================================================================

    def react(self, class_id: int) -> None:
        """Reacciona físicamente según la clase detectada por la CNN."""
        if self._do_not_disturb:
            logger.debug("Modo no molestar activo — sin reacción.")
            return
        self._last_class_id = class_id
        if class_id == CLASS_STUDYING:
            self._react_studying()
        elif class_id == CLASS_PHONE:
            self._react_phone()
        elif class_id == CLASS_ABSENT:
            self._react_absent()
        elif class_id == CLASS_CONFUSED:
            self.react_confused()
        elif class_id == CLASS_BORED:
            self.react_bored()
        else:
            logger.warning("Clase desconocida: %d", class_id)

    def _react_studying(self) -> None:
        self._leds.set_studying()
        self._servos.react_studying()
        logger.info("Reaccion: estudiando — LED VERDE.")

    def _react_phone(self) -> None:
        self._leds.set_phone()
        self._servos.react_phone()
        logger.info("Reaccion: celular — LED AMARILLO.")

    def _react_absent(self) -> None:
        self._leds.set_absent()
        self._servos.react_absent()
        logger.info("Reaccion: ausente — LED ROJO.")

    def react_confused(self) -> None:
        self._leds.set_tutoring()
        self._servos.react_confused()
        logger.info("Reaccion: confundido — LED AZUL.")

    def react_bored(self) -> None:
        self._leds.set_bored()
        self._servos.react_bored()
        logger.info("Reaccion: aburrido — LED CYAN.")

    # ==================================================================
    # TUTOR — reacciones durante la sesión de tutoría
    # ==================================================================

    def react_explaining(self) -> None:
        self._servos.react_explaining()
        logger.debug("Reaccion tutor: explicando.")

    def react_thinking(self) -> None:
        self._servos.react_confused()
        logger.debug("Reaccion tutor: pensando.")

    def react_asking_question(self) -> None:
        self._servos.react_listening()
        logger.debug("Reaccion tutor: haciendo pregunta.")

    def react_correct_answer(self) -> None:
        self._servos.react_correct()
        logger.info("Reaccion tutor: respuesta CORRECTA.")

    def react_wrong_answer(self) -> None:
        self._servos.react_wrong()
        logger.info("Reaccion tutor: respuesta INCORRECTA.")

    def react_quiz_score(self, score_pct: float) -> None:
        """
        Reacción final según puntaje del quiz.
          >= 70%: celebración completa (verde + brazos arriba)
          40-69%: aliento (amarillo + asentir)
          < 40%:  apoyo (azul + mirar al usuario)
        """
        if score_pct >= 0.70:
            self._leds.flash_green(times=5, interval=0.15)
            self._servos.celebrate()
            logger.info("Quiz score %.0f%% — celebracion.", score_pct * 100)
        elif score_pct >= 0.40:
            self._leds.set_phone()   # amarillo
            self._servos.nod()
            logger.info("Quiz score %.0f%% — aliento.", score_pct * 100)
        else:
            self._leds.set_tutoring()   # azul
            self._servos.look_at_user()
            logger.info("Quiz score %.0f%% — apoyo.", score_pct * 100)

    def react_speaking(self) -> None:
        """No cambia LED ni servo — preserva estado actual."""
        pass

    def react_nod(self) -> None:
        """Asiente: confirmación o acuerdo."""
        self._servos.nod()

    # ==================================================================
    # VOZ — reacciones durante la escucha por hotword
    # ==================================================================

    def react_listening(self) -> None:
        """Azul pulsante: escuchando en background, esperando hotword."""
        self._leds.pulse_blue()
        self._servos.neutral()
        logger.debug("Reaccion voz: escuchando.")

    def react_hotword_detected(self) -> None:
        """Azul brillante + mira usuario: hotword detectado, listo para pregunta."""
        self._leds.set_tutoring()
        self._servos.look_at_user()
        logger.info("Reaccion voz: hotword detectado.")

    def stop_listening_effect(self) -> None:
        """Restaura el color del último estado detectado. No mueve servos (react_speaking es pass)."""
        self._restore_led()

    def _restore_led(self) -> None:
        if self._last_class_id == CLASS_STUDYING:
            self._leds.set_studying()
            self._servos.ojos(0, 200, 0)
        elif self._last_class_id == CLASS_PHONE:
            self._leds.set_phone()
            self._servos.ojos(200, 200, 0)
        elif self._last_class_id == CLASS_ABSENT:
            self._leds.set_absent()
            self._servos.ojos(200, 0, 0)

    # ==================================================================
    # TUTORÍA (compatibilidad con main.py existente)
    # ==================================================================

    def activate_tutoring(self) -> None:
        self.react_explaining()

    def deactivate_tutoring(self) -> None:
        self._leds.set_studying()
        self._servos.neutral()

    # ==================================================================
    # MODO NO MOLESTAR
    # ==================================================================

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

    def _disable_do_not_disturb(self) -> None:
        self._do_not_disturb = False
        logger.info("Modo no molestar desactivado.")

    # ==================================================================
    # LIMPIEZA
    # ==================================================================

    def cleanup(self) -> None:
        self._leds.cleanup()
        self._servos.close()
        if self._dnd_timer:
            self._dnd_timer.cancel()
