"""
src/tutoring/polly_tts.py
TTS mediante Amazon Polly. Fallback a pyttsx3 si falla.
Requiere boto3 + polly:SynthesizeSpeech en las credenciales del .env.
"""

import io
import logging
import os

from config.settings import AWS_REGION

logger    = logging.getLogger(__name__)
VOICE_ID  = os.getenv("POLLY_VOICE", "Conchita")   # es-ES female, más cercana al castellano rioplatense
ENGINE    = os.getenv("POLLY_ENGINE", "standard")

_polly = None


def _client():
    global _polly
    if _polly is None:
        import boto3
        _polly = boto3.client("polly", region_name=AWS_REGION)
    return _polly


def speak(text: str) -> None:
    """Sintetiza con Polly y reproduce inline. Fallback a pyttsx3."""
    try:
        resp       = _client().synthesize_speech(
            Text=text, VoiceId=VOICE_ID, OutputFormat="mp3", Engine=ENGINE
        )
        audio_data = resp["AudioStream"].read()
        _play_mp3(audio_data)
    except Exception as exc:
        logger.warning("Polly TTS falló (%s) — usando fallback pyttsx3.", exc)
        _fallback(text)


def _play_mp3(data: bytes) -> None:
    import subprocess
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(data)
        tmp_path = f.name
    try:
        subprocess.run(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", tmp_path],
            check=True,
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _fallback(text: str) -> None:
    try:
        import pyttsx3
        eng = pyttsx3.init()
        eng.say(text)
        eng.runAndWait()
    except Exception as exc:
        logger.error("Fallback TTS también falló: %s", exc)
