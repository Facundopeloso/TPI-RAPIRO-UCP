"""
src/actuation/servo_controller.py
Control del robot RAPIRO via UART serial con Arduino.

Protocolo verificado contra firmware oficial y rapiro_client.py del grupo:
  #M{n}                    motion 0-9
  #PR{rrr}G{ggg}B{bbb}T{ttt}   ojos RGB (T en decimas de segundo, min 1)
  #PS{id:02d}A{ang:03d}T{ttt}  servo individual

Puerto abierto UNA SOLA VEZ al init, con 2s de espera para que ATmega arranque.
Mantenerlo abierto evita reset por DTR en cada comando.
"""

import logging
import threading
import time
import serial

from config.settings import SERIAL_PORT, SERIAL_BAUD_RATE

logger = logging.getLogger(__name__)

_BOOT_WAIT = 2.0   # segundos para que ATmega arranque tras abrir puerto


_LED_T_MAX = 90   # firmware rechaza T > 90 de forma inconsistente

def _led_cmd(r: int, g: int, b: int, duration_ms: int = 9000) -> str:
    tenths = max(1, min(_LED_T_MAX, int(duration_ms / 100)))
    return f"#PR{r:03d}G{g:03d}B{b:03d}T{tenths:03d}"


class ServoController:
    """Envía comandos al Arduino de RAPIRO con conexión serial persistente."""

    def __init__(self):
        self._port = SERIAL_PORT
        self._baud = SERIAL_BAUD_RATE
        self._lock = threading.Lock()
        self._ser: serial.Serial | None = None
        self._available = self._connect()

    def _connect(self) -> bool:
        try:
            self._ser = serial.Serial(self._port, self._baud, timeout=1.0)
            time.sleep(_BOOT_WAIT)   # ATmega necesita ~2s para inicializar UART
            logger.info("Serial OK: %s @ %d", self._port, self._baud)
            return True
        except (serial.SerialException, OSError) as exc:
            logger.warning("Serial no disponible (%s) — servos simulados.", exc)
            return False

    def _run(self, frames: list[tuple[str, float]]) -> None:
        """Manda secuencia de comandos por el puerto persistente. En hilo daemon."""
        if not self._available or self._ser is None:
            for cmd, _ in frames:
                logger.debug("Servo simulado: %s", cmd)
            return

        def _worker():
            with self._lock:
                try:
                    for cmd, wait in frames:
                        self._ser.write(f"{cmd}\r".encode("ascii"))
                        self._ser.flush()
                        if wait > 0:
                            time.sleep(wait)
                except (serial.SerialException, OSError) as exc:
                    logger.error("Error serial: %s", exc)

        threading.Thread(target=_worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Ojos RGB
    # ------------------------------------------------------------------

    def ojos(self, r: int, g: int, b: int, duration_ms: int = 1000) -> None:
        self._run([(_led_cmd(r, g, b, duration_ms), 0.3)])

    # ------------------------------------------------------------------
    # Acciones combinadas (LED + movimiento en una sola conexión)
    # ------------------------------------------------------------------

    def react_studying(self) -> None:
        """Verde + asiente (cabeza + vuelve a neutral)."""
        self._run([
            (_led_cmd(0, 200, 0), 0.3),
            ("#PS00A070T005", 0.5),
            ("#PS00A090T005", 0.4),
            ("#PS00A070T005", 0.5),
            ("#PS00A090T005", 0.4),
            ("#M0", 1.0),
        ])

    def react_phone(self) -> None:
        """Amarillo + sacude cabeza rapido (no,no,no) + brazo arriba."""
        self._run([
            (_led_cmd(200, 200, 0), 0.3),
            ("#PS00A060T003", 0.35),
            ("#PS00A120T003", 0.35),
            ("#PS00A060T003", 0.35),
            ("#PS00A120T003", 0.35),
            ("#PS00A090T003", 0.3),
            ("#M6", 1.5),
            ("#M0", 1.0),
        ])

    def react_absent(self) -> None:
        """Rojo + brazos arriba + busca lento izquierda-derecha."""
        self._run([
            (_led_cmd(200, 0, 0), 0.3),
            ("#M5", 1.5),
            ("#PS00A055T006", 0.7),
            ("#PS00A125T006", 0.7),
            ("#PS00A090T005", 0.5),
            ("#M0", 1.5),
        ])

    def react_confused(self) -> None:
        """Azul + ladea + brazo derecho arriba."""
        self._run([
            (_led_cmd(0, 100, 200), 0.3),
            ("#M6", 1.5),
            ("#PS00A065T005", 0.6),
            ("#PS00A115T005", 0.6),
            ("#PS00A090T005", 0.4),
            ("#M0", 1.0),
        ])

    def react_bored(self) -> None:
        """Cyan + sacude cabeza + brazo izquierdo."""
        self._run([
            (_led_cmd(0, 200, 200), 0.3),
            ("#M8", 1.5),
            ("#PS00A065T003", 0.4),
            ("#PS00A115T003", 0.4),
            ("#PS00A090T003", 0.4),
            ("#M0", 1.0),
        ])

    def react_explaining(self) -> None:
        """Azul + ambos brazos arriba."""
        self._run([(_led_cmd(0, 80, 220), 0.3), ("#M5", 2.0)])

    def react_correct(self) -> None:
        """Verde brillante + ambos brazos."""
        self._run([(_led_cmd(0, 255, 0), 0.3), ("#M5", 2.0), ("#M0", 1.0)])

    def react_wrong(self) -> None:
        """Amarillo + ladea cabeza."""
        self._run([
            (_led_cmd(180, 120, 0), 0.3),
            ("#PS00A075T005", 0.6),
            ("#PS00A090T005", 0.5),
        ])

    def react_listening(self) -> None:
        """Blanco + neutral."""
        self._run([(_led_cmd(180, 180, 180), 0.3), ("#M0", 1.0)])

    def neutral(self) -> None:
        self._run([("#M0", 2.0)])

    def apagar_ojos(self) -> None:
        self._run([(_led_cmd(0, 0, 0, 200), 0.3)])

    # ------------------------------------------------------------------
    # Limpieza
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._ser and self._ser.is_open:
            try:
                self._ser.close()
            except Exception:
                pass
