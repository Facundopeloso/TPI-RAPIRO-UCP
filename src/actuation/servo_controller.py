"""
src/actuation/servo_controller.py
Control de los 12 servos del robot RAPIRO via comunicación serial con Arduino.

Comandos disponibles (enviados como byte + newline):
  N — neutral           postura de descanso
  S — head_shake        sacude la cabeza (no, no hagas eso)
  A — alert_pose        pose de alerta (ausencia detectada)
  L — look_at_user      gira cabeza hacia el estudiante
  C — celebrate         levanta brazos y asiente (respuesta correcta)
  E — empathize         inclina cabeza suavemente (respuesta incorrecta)
  T — think             inclina cabeza hacia un lado (procesando)
  V — listen            se inclina levemente hacia adelante (escuchando)
  D — nod               asiente una vez (confirmación)
"""

import logging
import serial
from config.settings import SERIAL_PORT, SERIAL_BAUD_RATE

logger = logging.getLogger(__name__)

CMD_NEUTRAL    = b"N\n"
CMD_HEAD_SHAKE = b"S\n"
CMD_ALERT_POSE = b"A\n"
CMD_LOOK_AT    = b"L\n"
CMD_CELEBRATE  = b"C\n"
CMD_EMPATHIZE  = b"E\n"
CMD_THINK      = b"T\n"
CMD_LISTEN     = b"V\n"
CMD_NOD        = b"D\n"


class ServoController:
    """Envía comandos al Arduino de RAPIRO para mover los servos."""

    def __init__(self):
        self._serial: serial.Serial | None = None
        try:
            self._serial = serial.Serial(SERIAL_PORT, SERIAL_BAUD_RATE, timeout=1)
            logger.info("Serial abierto en %s @ %d baud.", SERIAL_PORT, SERIAL_BAUD_RATE)
        except (serial.SerialException, OSError):
            logger.warning("Puerto serial no disponible — servos en modo simulado.")

    def _send(self, cmd: bytes) -> None:
        if self._serial and self._serial.is_open:
            try:
                self._serial.write(cmd)
            except serial.SerialException as exc:
                logger.error("Error escribiendo serial: %s", exc)
        else:
            logger.debug("Serial simulado — cmd=%s", cmd.strip())

    # ------------------------------------------------------------------
    # Clasificador (comportamiento existente)
    # ------------------------------------------------------------------

    def neutral(self) -> None:
        """Postura de descanso."""
        self._send(CMD_NEUTRAL)

    def head_shake(self) -> None:
        """Sacude cabeza: detectó celular."""
        self._send(CMD_HEAD_SHAKE)

    def alert_pose(self) -> None:
        """Pose de alerta: puesto vacío."""
        self._send(CMD_ALERT_POSE)

    def look_at_user(self) -> None:
        """Gira hacia el estudiante."""
        self._send(CMD_LOOK_AT)

    # ------------------------------------------------------------------
    # Tutor (comportamiento nuevo)
    # ------------------------------------------------------------------

    def celebrate(self) -> None:
        """Levanta brazos y asiente: respuesta correcta o buen puntaje."""
        self._send(CMD_CELEBRATE)

    def empathize(self) -> None:
        """Inclina cabeza suavemente: respuesta incorrecta, con empatía."""
        self._send(CMD_EMPATHIZE)

    def think(self) -> None:
        """Inclina cabeza hacia un lado: generando quiz o procesando."""
        self._send(CMD_THINK)

    def listen(self) -> None:
        """Se inclina levemente hacia adelante: escuchando activamente."""
        self._send(CMD_LISTEN)

    def nod(self) -> None:
        """Asiente una vez: confirmación o acuerdo."""
        self._send(CMD_NOD)

    # ------------------------------------------------------------------
    # Limpieza
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._serial and self._serial.is_open:
            self._serial.close()
            logger.info("Puerto serial cerrado.")
