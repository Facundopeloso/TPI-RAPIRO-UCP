"""
src/tutoring/voice_listener.py
Escucha continua del hotword y captura de preguntas por micrófono.
"""

import logging
import threading
from config.settings import TUTOR_HOTWORD, SPEECH_LANGUAGE

logger = logging.getLogger(__name__)


class VoiceListener:
    """
    Escucha en segundo plano. Cuando detecta el hotword llama a on_question(pregunta).
    """

    def __init__(self, on_question, hotword: str = TUTOR_HOTWORD, language: str = SPEECH_LANGUAGE):
        self._on_question = on_question
        self._hotword = hotword.lower()
        self._language = language
        self._running = False
        self._thread: threading.Thread | None = None
        self._recognizer = None
        self._mic = None
        self._available = self._init_audio()

    def _init_audio(self) -> bool:
        try:
            import os
            import speech_recognition as sr
            self._recognizer = sr.Recognizer()
            self._recognizer.energy_threshold = 300
            self._recognizer.dynamic_energy_threshold = True
            self._recognizer.pause_threshold = 1.0

            env_idx = os.getenv("VOICE_DEVICE_INDEX")
            if env_idx is not None:
                device_index = int(env_idx)
                logger.info("Micrófono: device_index=%d (VOICE_DEVICE_INDEX)", device_index)
            else:
                import pyaudio
                device_index = None
                try:
                    with sr.Microphone():
                        pass
                except OSError:
                    pa = pyaudio.PyAudio()
                    for i in range(pa.get_device_count()):
                        if pa.get_device_info_by_index(i)["maxInputChannels"] > 0:
                            device_index = i
                            break
                    pa.terminate()
                    if device_index is None:
                        raise OSError("No hay dispositivos de entrada de audio")
                    logger.info("Usando dispositivo de entrada #%d", device_index)

            self._mic = sr.Microphone(device_index=device_index)
            logger.info("Micrófono inicializado. Hotword: '%s'", TUTOR_HOTWORD)
            return True
        except ImportError:
            logger.error("SpeechRecognition no instalado: pip install SpeechRecognition pyaudio")
            return False
        except Exception as exc:
            logger.error("Error inicializando micrófono: %s", exc)
            return False

    def start(self) -> bool:
        if not self._available:
            return False
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        logger.info("Escucha de voz iniciada.")
        return True

    def stop(self) -> None:
        self._running = False
        logger.info("Escucha de voz detenida.")

    def _listen_loop(self) -> None:
        import speech_recognition as sr

        try:
            with self._mic as source:
                self._recognizer.adjust_for_ambient_noise(source, duration=0.5)
                logger.info("Ajuste de ruido ambiental listo.")
                while self._running:
                    try:
                        audio = self._recognizer.listen(source, timeout=5, phrase_time_limit=8)
                        text = self._recognizer.recognize_google(
                            audio, language=self._language
                        ).lower()
                        logger.info("Escuchado: '%s'", text)

                        if self._hotword in text:
                            logger.info("Hotword detectado.")
                            question = self._capture_question(source)
                            if question:
                                self._on_question(question)

                    except sr.WaitTimeoutError:
                        pass
                    except sr.UnknownValueError:
                        pass
                    except sr.RequestError as exc:
                        logger.warning("Error reconocimiento: %s", exc)
                    except Exception as exc:
                        logger.error("Error en loop: %s", exc)
        except Exception as exc:
            logger.error("Error abriendo micrófono: %s", exc)

    def _capture_question(self, source) -> str:
        """Graba la pregunta después de detectar el hotword (micrófono ya abierto)."""
        import speech_recognition as sr

        logger.info("Esperando pregunta...")
        try:
            audio = self._recognizer.listen(source, timeout=5, phrase_time_limit=15)
            question = self._recognizer.recognize_google(audio, language=self._language)
            logger.info("Pregunta capturada: '%s'", question)
            return question
        except (sr.WaitTimeoutError, sr.UnknownValueError):
            logger.warning("No se entendió la pregunta.")
            return ""
        except sr.RequestError as exc:
            logger.warning("Error capturando pregunta: %s", exc)
            return ""
