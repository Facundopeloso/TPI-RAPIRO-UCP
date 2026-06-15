"""
src/actuation/led_controller.py
Control de LEDs RGB via GPIO (Raspberry Pi).
En entornos sin GPIO disponible, las operaciones se loguean sin error.
"""

import logging
import threading
import time
from config.settings import LED_RED_PIN, LED_GREEN_PIN, LED_BLUE_PIN

logger = logging.getLogger(__name__)


class LEDController:
    """Controla un LED RGB mediante tres pines GPIO en modo BCM."""

    def __init__(self):
        self._gpio_available = False
        self._flash_thread: threading.Thread | None = None
        self._flash_stop = threading.Event()

        self._h   = None   # lgpio handle
        self._pins = (LED_RED_PIN, LED_GREEN_PIN, LED_BLUE_PIN)

        # Intentar lgpio (Raspbian Trixie / bookworm)
        try:
            import lgpio
            self._lgpio = lgpio
            self._h = lgpio.gpiochip_open(0)
            for pin in self._pins:
                lgpio.gpio_claim_output(self._h, pin, 0, 0)
                lgpio.gpio_write(self._h, pin, 1)  # HIGH = LED off (common anode)
            self._gpio_available = True
            logger.info("GPIO inicializado para LEDs (lgpio).")
            return
        except Exception as exc:
            logger.debug("lgpio falló (%s), intentando RPi.GPIO.", exc)

        # Fallback: RPi.GPIO
        try:
            import RPi.GPIO as GPIO
            self._GPIO = GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            for pin in self._pins:
                GPIO.setup(pin, GPIO.OUT)
                GPIO.output(pin, GPIO.LOW)
            self._gpio_available = True
            self._h = None
            logger.info("GPIO inicializado para LEDs (RPi.GPIO).")
        except Exception:
            logger.warning("GPIO no disponible — LEDs en modo simulado.")

    def _set(self, r: bool, g: bool, b: bool) -> None:
        if not self._gpio_available:
            logger.debug("LED simulado — R=%s G=%s B=%s", r, g, b)
            return
        vals = (r, g, b)
        if self._h is not None:
            for pin, v in zip(self._pins, vals):
                self._lgpio.gpio_write(self._h, pin, 0 if v else 1)  # common anode: LOW=ON
        else:
            GPIO = self._GPIO
            GPIO.output(LED_RED_PIN,   GPIO.LOW if r else GPIO.HIGH)
            GPIO.output(LED_GREEN_PIN, GPIO.LOW if g else GPIO.HIGH)
            GPIO.output(LED_BLUE_PIN,  GPIO.LOW if b else GPIO.HIGH)

    # ------------------------------------------------------------------
    # Estados fijos
    # ------------------------------------------------------------------

    def set_studying(self) -> None:
        self._stop_flash()
        self._set(False, True, False)    # verde

    def set_phone(self) -> None:
        self._stop_flash()
        self._set(True, True, False)     # amarillo (R+G)

    def set_absent(self) -> None:
        self._stop_flash()
        self._set(True, False, False)    # rojo

    def set_tutoring(self) -> None:
        self._stop_flash()
        self._set(False, False, True)    # azul

    def set_bored(self) -> None:
        self._stop_flash()
        self._set(False, True, True)     # cyan (verde+azul)

    def set_white(self) -> None:
        """Blanco: pregunta activa, esperando respuesta."""
        self._stop_flash()
        self._set(True, True, True)

    def set_correct(self) -> None:
        """Verde fijo: respuesta correcta."""
        self._stop_flash()
        self._set(False, True, False)

    def set_wrong(self) -> None:
        """Amarillo fijo: respuesta incorrecta."""
        self._stop_flash()
        self._set(True, True, False)

    def turn_off(self) -> None:
        self._stop_flash()
        self._set(False, False, False)

    # ------------------------------------------------------------------
    # Efectos
    # ------------------------------------------------------------------

    def flash_green(self, times: int = 3, interval: float = 0.25) -> None:
        """Parpadeo verde N veces en hilo de fondo (celebración)."""
        self._start_flash(r=False, g=True, b=False, times=times, interval=interval)

    def flash_blue(self, times: int = 2, interval: float = 0.4) -> None:
        """Parpadeo azul N veces (hotword detectado / pensando)."""
        self._start_flash(r=False, g=False, b=True, times=times, interval=interval)

    def pulse_blue(self) -> None:
        """Pulso azul continuo hasta llamar a stop_flash() (escuchando)."""
        self._stop_flash()
        self._flash_stop.clear()
        self._flash_thread = threading.Thread(
            target=self._pulse_loop, args=(False, False, True), daemon=True
        )
        self._flash_thread.start()

    def _start_flash(self, r: bool, g: bool, b: bool, times: int, interval: float) -> None:
        self._stop_flash()
        self._flash_stop.clear()
        self._flash_thread = threading.Thread(
            target=self._flash_loop,
            args=(r, g, b, times, interval),
            daemon=True,
        )
        self._flash_thread.start()

    def _flash_loop(self, r: bool, g: bool, b: bool, times: int, interval: float) -> None:
        for _ in range(times):
            if self._flash_stop.is_set():
                break
            self._set(r, g, b)
            time.sleep(interval)
            self._set(False, False, False)
            time.sleep(interval * 0.5)
        # Quedar en estado final encendido
        if not self._flash_stop.is_set():
            self._set(r, g, b)

    def _pulse_loop(self, r: bool, g: bool, b: bool) -> None:
        while not self._flash_stop.is_set():
            self._set(r, g, b)
            time.sleep(0.6)
            self._set(False, False, False)
            time.sleep(0.4)

    def _stop_flash(self) -> None:
        self._flash_stop.set()
        if self._flash_thread and self._flash_thread.is_alive():
            self._flash_thread.join(timeout=1)

    # ------------------------------------------------------------------
    # Limpieza
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        self._stop_flash()
        try:
            self.turn_off()
        except Exception:
            pass
        if self._gpio_available:
            try:
                if self._h is not None:
                    self._lgpio.gpiochip_close(self._h)
                elif hasattr(self, "_GPIO"):
                    self._GPIO.cleanup()
            except Exception:
                pass
