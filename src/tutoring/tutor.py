"""
src/tutoring/tutor.py
Módulo de tutoría inteligente: TF-IDF + Claude (Anthropic).
"""

import logging
from config.settings import (
    ANTHROPIC_API_KEY, LLM_MODEL, LLM_MAX_TOKENS,
    DOCUMENT_TOP_K_CHUNKS, SPEECH_LANGUAGE
)
from src.tutoring.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Eres RAPIRO, un tutor inteligente amigable. "
    "Solo debes responder preguntas basándote ESTRICTAMENTE en el material "
    "de estudio proporcionado en el contexto. "
    "Si la pregunta no puede responderse con el material disponible, indícalo claramente. "
    "Responde en español, de forma concisa y pedagógica."
)


class IntelligentTutor:
    """Tutor adaptativo que responde preguntas sobre el documento cargado."""

    def __init__(self, document_processor: DocumentProcessor):
        self._doc_processor = document_processor
        self._client = self._init_anthropic_client()

    def _init_anthropic_client(self):
        if not ANTHROPIC_API_KEY:
            logger.warning("ANTHROPIC_API_KEY no configurada — tutoría LLM deshabilitada.")
            return None
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            logger.info("Cliente Anthropic inicializado (modelo: %s).", LLM_MODEL)
            return client
        except ImportError:
            logger.error("anthropic no instalado. Ejecutar: pip install anthropic")
            return None

    def answer(self, question: str) -> str:
        if not self._client:
            return "El módulo de tutoría no está disponible. Verificar configuración."

        if not self._doc_processor.chunks:
            return "No hay ningún documento cargado. Por favor, carga tu material de estudio."

        relevant_chunks = self._doc_processor.retrieve(question, top_k=DOCUMENT_TOP_K_CHUNKS)
        context = "\n\n---\n\n".join(relevant_chunks)

        user_message = (
            f"Contexto del material de estudio:\n\n{context}\n\n"
            f"Pregunta del estudiante: {question}"
        )

        try:
            response = self._client.messages.create(
                model=LLM_MODEL,
                max_tokens=LLM_MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            answer_text = response.content[0].text.strip()
            logger.info("Claude respondió correctamente (%d chars).", len(answer_text))
            return answer_text

        except Exception as exc:
            logger.error("Error al llamar a Claude: %s", exc)
            return "Hubo un error al procesar tu consulta. Intenta nuevamente."

    def speak(self, text: str) -> None:
        try:
            import io
            import pygame
            from gtts import gTTS

            tts = gTTS(text=text, lang=SPEECH_LANGUAGE[:2], slow=False)
            audio_bytes = io.BytesIO()
            tts.write_to_fp(audio_bytes)
            audio_bytes.seek(0)

            pygame.mixer.init()
            pygame.mixer.music.load(audio_bytes)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

        except ImportError:
            logger.warning("gTTS o pygame no disponibles — respuesta solo por texto.")
        except Exception as exc:
            logger.error("Error en TTS: %s", exc)

    def answer_and_speak(self, question: str) -> str:
        text = self.answer(question)
        self.speak(text)
        return text
