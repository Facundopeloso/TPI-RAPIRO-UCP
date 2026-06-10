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

    # ==================================================================
    # CLASIFICADOR — reacciones según estado detectado
    # ==================================================================

    def react(self, class_id: int) -> None:
        """Reacciona físicamente según la clase detectada por la CNN."""
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

    def _react_studying(self) -> None:
        """Verde + neutro: todo bien, el estudiante está concentrado."""
        self._leds.set_studying()
        self._servos.neutral()
        logger.debug("Reaccion: estudiando.")

    def _react_phone(self) -> None:
        """Amarillo + sacude cabeza: detectó uso del celular."""
        self._leds.set_phone()
        self._servos.head_shake()
        logger.debug("Reaccion: celular detectado.")

    def _react_absent(self) -> None:
        """Rojo + pose de alerta: puesto vacío."""
        self._leds.set_absent()
        self._servos.alert_pose()
        logger.debug("Reaccion: ausente.")

    # ==================================================================
    # TUTOR — reacciones durante la sesión de tutoría
    # ==================================================================

    def react_explaining(self) -> None:
        """Azul + mira al usuario: RAPIRO está explicando un tema."""
        self._leds.set_tutoring()
        self._servos.look_at_user()
        logger.debug("Reaccion tutor: explicando.")

    def react_thinking(self) -> None:
        """Azul parpadeante + pose de pensar: generando quiz o procesando."""
        self._leds.flash_blue(times=2, interval=0.4)
        self._servos.think()
        logger.debug("Reaccion tutor: pensando.")

    def react_asking_question(self) -> None:
        """Blanco + escuchando: pregunta activa, esperando respuesta."""
        self._leds.set_white()
        self._servos.listen()
        logger.debug("Reaccion tutor: haciendo pregunta.")

    def react_correct_answer(self) -> None:
        """Verde flash + celebración: respuesta correcta."""
        self._leds.flash_green(times=3, interval=0.2)
        self._servos.celebrate()
        logger.info("Reaccion tutor: respuesta CORRECTA.")

    def react_wrong_answer(self) -> None:
        """Amarillo + empatía: respuesta incorrecta, sin juzgar."""
        self._leds.set_wrong()
        self._servos.empathize()
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
        """Azul + mira usuario: RAPIRO está hablando por TTS."""
        self._leds.set_tutoring()
        self._servos.look_at_user()

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
        """Detiene el pulso azul y vuelve al estado de estudio."""
        self._leds.set_studying()
        self._servos.neutral()

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
