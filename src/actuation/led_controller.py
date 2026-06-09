"""
src/actuation/led_controller.py
Control de LEDs RGB via GPIO (Raspberry Pi).
En entornos sin GPIO disponible, las operaciones se loguean sin error.
"""

import logging
from config.settings import LED_RED_PIN, LED_GREEN_PIN, LED_BLUE_PIN

logger = logging.getLogger(__name__)


class LEDController:
    """Controla un LED RGB mediante tres pines GPIO en modo BCM."""

    def __init__(self):
        self._gpio_available = False
        try:
            import RPi.GPIO as GPIO
            self._GPIO = GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            for pin in (LED_RED_PIN, LED_GREEN_PIN, LED_BLUE_PIN):
                GPIO.setup(pin, GPIO.OUT)
                GPIO.output(pin, GPIO.LOW)
            self._gpio_available = True
            logger.info("GPIO inicializado para LEDs.")
        except (ImportError, RuntimeError):
            logger.warning("RPi.GPIO no disponible — LEDs en modo simulado.")

    def _set(self, r: bool, g: bool, b: bool) -> None:
        if not self._gpio_available:
            logger.debug("LED simulado — R=%s G=%s B=%s", r, g, b)
            return
        GPIO = self._GPIO
        GPIO.output(LED_RED_PIN, GPIO.HIGH if r else GPIO.LOW)
        GPIO.output(LED_GREEN_PIN, GPIO.HIGH if g else GPIO.LOW)
        GPIO.output(LED_BLUE_PIN, GPIO.HIGH if b else GPIO.LOW)

    def set_studying(self) -> None:
        self._set(False, True, False)   # verde

    def set_phone(self) -> None:
        self._set(True, True, False)    # amarillo (R+G)

    def set_absent(self) -> None:
        self._set(True, False, False)   # rojo

    def set_tutoring(self) -> None:
        self._set(False, False, True)   # azul

    def turn_off(self) -> None:
        self._set(False, False, False)

    def cleanup(self) -> None:
        self.turn_off()
        if self._gpio_available:
            self._GPIO.cleanup()
