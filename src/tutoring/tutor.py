"""
src/tutoring/tutor.py
Tutor inteligente RAPIRO: respuestas, explicaciones con ejemplos reales y quizzes.
LLM: Claude Haiku via Anthropic API.
"""

import json
import re
import logging
from dataclasses import dataclass, field

from config.settings import (
    ANTHROPIC_API_KEY, LLM_MODEL, LLM_MAX_TOKENS,
    DOCUMENT_TOP_K_CHUNKS, SPEECH_LANGUAGE
)
from src.tutoring.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts del sistema
# ---------------------------------------------------------------------------

SYSTEM_TUTOR = (
    "Eres RAPIRO, un profesor virtual amigable y motivador. "
    "Tu estilo es claro, cercano y pedagógico. "
    "SIEMPRE que expliques un concepto usa al menos una analogía o ejemplo de la vida cotidiana. "
    "Cuando des feedback sé alentador: si el alumno se equivoca, explica sin juzgar. "
    "Responde siempre en español rioplatense."
)

SYSTEM_QUIZ = (
    "Eres RAPIRO, generador de evaluaciones educativas. "
    "Tu tarea es crear preguntas de opción múltiple (A/B/C/D) basadas ESTRICTAMENTE "
    "en el material de estudio provisto. "
    "Devuelve ÚNICAMENTE JSON válido, sin texto adicional antes ni después."
)

QUIZ_SCHEMA = """{
  "questions": [
    {
      "question": "texto de la pregunta",
      "options": {
        "A": "opción A",
        "B": "opción B",
        "C": "opción C",
        "D": "opción D"
      },
      "correct": "A",
      "explanation": "por qué esa es la respuesta correcta",
      "real_world_example": "ejemplo cotidiano que refuerza el concepto"
    }
  ]
}"""

# ---------------------------------------------------------------------------
# Modelos de datos
# ---------------------------------------------------------------------------

@dataclass
class QuizQuestion:
    question: str
    options: dict
    correct: str
    explanation: str
    real_world_example: str


@dataclass
class Quiz:
    topic: str
    questions: list[QuizQuestion] = field(default_factory=list)

    def __len__(self):
        return len(self.questions)


@dataclass
class QuizResult:
    question: QuizQuestion
    student_answer: str
    is_correct: bool
    feedback: str


# ---------------------------------------------------------------------------
# Tutor principal
# ---------------------------------------------------------------------------

class IntelligentTutor:
    """
    Tutor adaptativo con cuatro modos:
      answer()              — responde pregunta libre con contexto del documento
      explain_with_example()— explica un tema con analogías de la vida real
      generate_quiz()       — genera preguntas multiple choice del documento
      evaluate_answer()     — corrige respuesta y da feedback personalizado
      study_session()       — flujo completo: explicación → quiz → puntaje
    """

    def __init__(self, document_processor: DocumentProcessor):
        self._doc = document_processor
        self._client = self._init_client()

    # ------------------------------------------------------------------
    # Inicialización
    # ------------------------------------------------------------------

    def _init_client(self):
        if not ANTHROPIC_API_KEY:
            logger.warning("ANTHROPIC_API_KEY no configurada — tutor LLM deshabilitado.")
            return None
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            logger.info("Cliente Anthropic listo (modelo: %s).", LLM_MODEL)
            return client
        except ImportError:
            logger.error("anthropic no instalado: pip install anthropic")
            return None

    def _available(self) -> bool:
        return self._client is not None

    def _context(self, query: str, top_k: int = DOCUMENT_TOP_K_CHUNKS) -> str:
        chunks = self._doc.retrieve(query, top_k=top_k)
        return "\n\n---\n\n".join(chunks) if chunks else ""

    def _call(self, system: str, user: str, max_tokens: int = LLM_MAX_TOKENS) -> str:
        try:
            resp = self._client.messages.create(
                model=LLM_MODEL,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return resp.content[0].text.strip()
        except Exception as exc:
            logger.error("Error llamando a Claude: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # Modo 1: respuesta libre
    # ------------------------------------------------------------------

    def answer(self, question: str) -> str:
        """Responde pregunta libre basada en el documento cargado."""
        if not self._available():
            return "Tutor no disponible. Verificar ANTHROPIC_API_KEY."
        if not self._doc.chunks:
            return "No hay documento cargado. Por favor, carga tu material de estudio."

        ctx = self._context(question)
        user_msg = (
            f"Material de estudio:\n\n{ctx}\n\n"
            f"Pregunta del estudiante: {question}\n\n"
            "Responde de forma clara y pedagógica. "
            "Incluye al menos un ejemplo de la vida cotidiana si aplica."
        )
        result = self._call(SYSTEM_TUTOR, user_msg)
        return result or "No pude generar una respuesta. Intentá de nuevo."

    # ------------------------------------------------------------------
    # Modo 2: explicación con ejemplos reales
    # ------------------------------------------------------------------

    def explain_with_example(self, topic: str) -> str:
        """
        Explica un tema en profundidad con analogías de la vida real.
        Cubre el panorama completo: qué es, cómo funciona, para qué sirve.
        """
        if not self._available():
            return "Tutor no disponible."

        ctx = self._context(topic, top_k=5)
        user_msg = (
            f"Material de estudio:\n\n{ctx}\n\n"
            f"Tema a explicar: {topic}\n\n"
            "Estructura tu respuesta así:\n"
            "1. QUE ES: definición simple en 2-3 oraciones\n"
            "2. COMO FUNCIONA: mecanismo o proceso principal\n"
            "3. EJEMPLOS REALES: 2 o 3 analogías con situaciones cotidianas "
            "(puede ser cocina, deporte, trabajo, ciudad, etc.)\n"
            "4. PANORAMA COMPLETO: cómo encaja este tema con el resto del material\n"
            "5. DATO CLAVE: una sola oración para recordar el concepto central\n"
        )
        result = self._call(SYSTEM_TUTOR, user_msg, max_tokens=800)
        return result or f"No pude generar explicación sobre '{topic}'."

    # ------------------------------------------------------------------
    # Modo 3: generar quiz
    # ------------------------------------------------------------------

    def generate_quiz(self, topic: str, n_questions: int = 4) -> Quiz:
        """
        Genera un quiz de opción múltiple (A/B/C/D) basado en el documento.

        Args:
            topic: Tema sobre el cual hacer las preguntas.
            n_questions: Cantidad de preguntas (2-6 recomendado).

        Returns:
            Quiz con QuizQuestion[] listo para presentar al estudiante.
        """
        quiz = Quiz(topic=topic)

        if not self._available():
            logger.warning("Tutor no disponible para generar quiz.")
            return quiz

        ctx = self._context(topic, top_k=6)
        if not ctx:
            logger.warning("No hay contexto para el tema '%s'.", topic)
            return quiz

        user_msg = (
            f"Material de estudio:\n\n{ctx}\n\n"
            f"Genera exactamente {n_questions} preguntas de opción múltiple sobre: {topic}\n\n"
            f"Usa este formato JSON exacto:\n{QUIZ_SCHEMA}\n\n"
            "Requisitos:\n"
            "- Las preguntas deben cubrir distintos aspectos del tema\n"
            "- Las opciones incorrectas deben ser plausibles (no obviamente falsas)\n"
            "- El campo real_world_example debe conectar la respuesta con la vida cotidiana\n"
            "- Devuelve SOLO el JSON, sin texto adicional"
        )

        raw = self._call(SYSTEM_QUIZ, user_msg, max_tokens=1200)
        quiz.questions = self._parse_quiz_json(raw)

        if not quiz.questions:
            logger.error("No se pudo parsear el quiz generado.")
        else:
            logger.info("Quiz generado: %d preguntas sobre '%s'.", len(quiz.questions), topic)

        return quiz

    def _parse_quiz_json(self, raw: str) -> list[QuizQuestion]:
        """Extrae y parsea el JSON del quiz desde la respuesta de Claude."""
        # Intentar extraer bloque JSON si Claude agregó texto extra
        match = re.search(r'\{[\s\S]*\}', raw)
        if not match:
            logger.error("No se encontró JSON en la respuesta del quiz.")
            return []

        try:
            data = json.loads(match.group())
            questions = []
            for q in data.get("questions", []):
                questions.append(QuizQuestion(
                    question=q["question"],
                    options=q["options"],
                    correct=q["correct"].upper(),
                    explanation=q["explanation"],
                    real_world_example=q["real_world_example"],
                ))
            return questions
        except (json.JSONDecodeError, KeyError) as exc:
            logger.error("Error parseando JSON del quiz: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Modo 4: evaluar respuesta
    # ------------------------------------------------------------------

    def evaluate_answer(self, question: QuizQuestion, student_answer: str) -> QuizResult:
        """
        Evalúa la respuesta del estudiante y genera feedback personalizado.

        Args:
            question: La pregunta respondida.
            student_answer: "A", "B", "C" o "D".

        Returns:
            QuizResult con corrección y feedback motivador.
        """
        sa = student_answer.upper().strip()
        is_correct = sa == question.correct

        if not self._available():
            feedback = (
                f"{'Correcto!' if is_correct else 'Incorrecto.'} "
                f"La respuesta era {question.correct}: {question.options[question.correct]}. "
                f"{question.explanation}"
            )
            return QuizResult(question, sa, is_correct, feedback)

        if is_correct:
            prompt = (
                f"El estudiante respondió correctamente '{sa}' a esta pregunta:\n"
                f"Pregunta: {question.question}\n"
                f"Respuesta correcta: {question.correct}) {question.options[question.correct]}\n"
                f"Explicación: {question.explanation}\n"
                f"Ejemplo real: {question.real_world_example}\n\n"
                "Felicítalo brevemente (1-2 oraciones), refuerza el concepto clave "
                "y menciona el ejemplo de la vida real para que lo recuerde."
            )
        else:
            chosen_text = question.options.get(sa, "opción inválida")
            prompt = (
                f"El estudiante respondió '{sa}' ({chosen_text}) pero la respuesta correcta "
                f"era '{question.correct}' ({question.options[question.correct]}).\n"
                f"Pregunta: {question.question}\n"
                f"Explicación correcta: {question.explanation}\n"
                f"Ejemplo real: {question.real_world_example}\n\n"
                "Explícale con amabilidad por qué se equivocó, cuál es la respuesta correcta "
                "y usa el ejemplo de la vida real para que lo entienda mejor. "
                "Sé alentador, no condescendiente. Máximo 3 oraciones."
            )

        feedback = self._call(SYSTEM_TUTOR, prompt, max_tokens=300)
        if not feedback:
            feedback = (
                f"{'Correcto!' if is_correct else f'La respuesta era {question.correct}.'} "
                f"{question.explanation}"
            )

        return QuizResult(question, sa, is_correct, feedback)

    # ------------------------------------------------------------------
    # Modo 5: sesión de estudio completa
    # ------------------------------------------------------------------

    def study_session(self, topic: str, n_questions: int = 3) -> dict:
        """
        Flujo de estudio completo:
          1. Explicación con ejemplos reales del tema
          2. Quiz de n_questions preguntas
          3. Retorna diccionario listo para presentar

        Returns:
            {
                "topic": str,
                "explanation": str,
                "quiz": Quiz,
            }
        """
        logger.info("Iniciando sesion de estudio: '%s' (%d preguntas).", topic, n_questions)
        explanation = self.explain_with_example(topic)
        quiz = self.generate_quiz(topic, n_questions)
        return {
            "topic": topic,
            "explanation": explanation,
            "quiz": quiz,
        }

    # ------------------------------------------------------------------
    # TTS
    # ------------------------------------------------------------------

    def speak(self, text: str) -> None:
        """Sintetiza texto en audio y lo reproduce."""
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
            logger.warning("gTTS o pygame no disponibles — solo respuesta por texto.")
        except Exception as exc:
            logger.error("Error en TTS: %s", exc)

    def answer_and_speak(self, question: str) -> str:
        text = self.answer(question)
        self.speak(text)
        return text
