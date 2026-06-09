"""
src/tutoring/tutor.py
Módulo de tutoría inteligente: integra recuperación de contexto (TF-IDF)
con un LLM (OpenAI GPT-3.5 Turbo o AWS Bedrock) para responder preguntas
sobre el material de estudio del usuario.

Flujo:
    Consulta del usuario
        → Recuperar chunks relevantes (TF-IDF)
        → Construir prompt con contexto
        → Llamar al LLM
        → Devolver respuesta
        → (Opcional) Sintetizar audio con TTS
"""

import logging
from config.settings import (
    OPENAI_API_KEY, LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE,
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
    """
    Tutor adaptativo que responde preguntas sobre el documento cargado.
    """

    def __init__(self, document_processor: DocumentProcessor):
        self._doc_processor = document_processor
        self._client = self._init_openai_client()

    def _init_openai_client(self):
        """Inicializa el cliente OpenAI o retorna None si no está disponible."""
        if not OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY no configurada — tutoría LLM deshabilitada.")
            return None
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            logger.info("Cliente OpenAI inicializado (modelo: %s).", LLM_MODEL)
            return client
        except ImportError:
            logger.error("openai no instalado. Ejecutar: pip install openai")
            return None

    def answer(self, question: str) -> str:
        """
        Responde una pregunta del estudiante usando el material cargado.

        Args:
            question: Pregunta en texto libre.

        Returns:
            Respuesta del LLM o mensaje de error.
        """
        if not self._client:
            return "El módulo de tutoría no está disponible. Verificar configuración."

        if not self._doc_processor.chunks:
            return "No hay ningún documento cargado. Por favor, carga tu material de estudio."

        # Recuperar contexto relevante
        relevant_chunks = self._doc_processor.retrieve(question, top_k=DOCUMENT_TOP_K_CHUNKS)
        context = "\n\n---\n\n".join(relevant_chunks)

        user_message = (
            f"Contexto del material de estudio:\n\n{context}\n\n"
            f"Pregunta del estudiante: {question}"
        )

        try:
            response = self._client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
            )
            answer_text = response.choices[0].message.content.strip()
            logger.info("LLM respondió correctamente (%d chars).", len(answer_text))
            return answer_text

        except Exception as exc:
            logger.error("Error al llamar al LLM: %s", exc)
            return "Hubo un error al procesar tu consulta. Intenta nuevamente."

    def speak(self, text: str) -> None:
        """
        Sintetiza el texto en audio y lo reproduce por el altavoz de RAPIRO.

        Args:
            text: Texto a sintetizar.
        """
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
        """
        Responde la pregunta y reproduce el audio de la respuesta.

        Returns:
            Texto de la respuesta.
        """
        text = self.answer(question)
        self.speak(text)
        return text
