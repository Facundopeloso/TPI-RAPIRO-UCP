"""
src/actuation/servo_controller.py
Control de los 12 servos del robot RAPIRO via comunicación serial con Arduino.
"""

import logging
import serial
from config.settings import SERIAL_PORT, SERIAL_BAUD_RATE

logger = logging.getLogger(__name__)

CMD_NEUTRAL = b"N\n"
CMD_HEAD_SHAKE = b"S\n"
CMD_ALERT_POSE = b"A\n"
CMD_LOOK_AT_USER = b"L\n"


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
            logger.debug("Serial simulado — cmd=%s", cmd)

    def neutral(self) -> None:
        self._send(CMD_NEUTRAL)

    def head_shake(self) -> None:
        self._send(CMD_HEAD_SHAKE)

    def alert_pose(self) -> None:
        self._send(CMD_ALERT_POSE)

    def look_at_user(self) -> None:
        self._send(CMD_LOOK_AT_USER)

    def close(self) -> None:
        if self._serial and self._serial.is_open:
            self._serial.close()
            logger.info("Puerto serial cerrado.")
