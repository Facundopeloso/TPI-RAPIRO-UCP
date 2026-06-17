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
VOICE_ID  = os.getenv("POLLY_VOICE", "Conchita")
ENGINE    = os.getenv("POLLY_ENGINE", "standard")
# Neural voices requieren us-east-1 (sa-east-1 no las soporta)
POLLY_REGION = os.getenv("POLLY_REGION", "us-east-1" if ENGINE == "neural" else AWS_REGION)

_polly = None


def _client():
    global _polly
    if _polly is None:
        import boto3
        _polly = boto3.client("polly", region_name=POLLY_REGION)
    return _polly


def speak(text: str) -> None:
    """Sintetiza con Polly y reproduce inline. Fallback a pyttsx3."""
    logger.info("TTS: sintetizando voz=%s engine=%s region=%s", VOICE_ID, ENGINE, POLLY_REGION)
    try:
        resp       = _client().synthesize_speech(
            Text=text, VoiceId=VOICE_ID, OutputFormat="mp3", Engine=ENGINE
        )
        audio_data = resp["AudioStream"].read()
        logger.info("TTS: Polly OK (%d bytes) — reproduciendo...", len(audio_data))
        _play_mp3(audio_data)
        logger.info("TTS: reproducción completa.")
    except Exception as exc:
        logger.warning("Polly TTS falló (%s) — usando fallback pyttsx3.", exc)
        _fallback(text)


_BT_SINK = "bluez_output.E8_07_BF_00_63_6D.1"

def _play_mp3(data: bytes) -> None:
    import subprocess
    import sys
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(data)
        tmp_path = f.name
    try:
        if sys.platform == "win32":
            # Windows: PowerShell WPF MediaPlayer (sincrónico)
            duration = max(3, len(data) // 8000 + 2)
            fwd = tmp_path.replace("\\", "/")
            subprocess.run(
                [
                    "powershell", "-NoProfile", "-Command",
                    f"Add-Type -AssemblyName presentationCore;"
                    f"$mp=[Windows.Media.MediaPlayer]::new();"
                    f"$mp.Open([Uri]::new('{fwd}'));"
                    f"$mp.Play();"
                    f"Start-Sleep -Seconds {duration}",
                ],
                check=False,
            )
        else:
            # Linux: pw-play → BT speaker, fallback ffplay
            env = os.environ.copy()
            env["XDG_RUNTIME_DIR"] = f"/run/user/{os.getuid()}"
            try:
                subprocess.run(
                    ["pw-play", "--target", _BT_SINK, tmp_path],
                    env=env, check=False,
                )
            except Exception:
                subprocess.run(
                    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", tmp_path],
                    check=False, env=env,
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
