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
    "Sos RAPIRO, un compañero de estudio que habla como una persona real. "
    "Hablás en español rioplatense, de forma corta y directa. "
    "Nunca usás bullets, asteriscos, números de lista ni formato markdown. "
    "Hablás en párrafos cortos, como en una conversación. "
    "Usás analogías simples y cotidianas. "
    "Si el alumno se equivoca, lo alentás sin juzgar. "
    "Máximo 4 oraciones por respuesta salvo que se pida más."
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

    Acepta rapiro: RAPIROController opcional para reacciones físicas del robot.
    Si rapiro es None, el tutor funciona igual pero sin actuación física.
    """

    def __init__(self, document_processor: DocumentProcessor, rapiro=None):
        self._doc = document_processor
        self._rapiro = rapiro
        self._client = self._init_client()

    def _r(self, method: str, *args, **kwargs) -> None:
        """Llama a un método del RAPIROController si está disponible."""
        if self._rapiro is not None:
            try:
                getattr(self._rapiro, method)(*args, **kwargs)
            except Exception as exc:
                logger.debug("Error en reacción RAPIRO '%s': %s", method, exc)

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

        self._r("react_thinking")           # pensando la respuesta
        ctx = self._context(question)
        user_msg = (
            f"Material de estudio:\n\n{ctx}\n\n"
            f"Pregunta del estudiante: {question}\n\n"
            "Responde de forma clara y pedagógica. "
            "Incluye al menos un ejemplo de la vida cotidiana si aplica."
        )
        result = self._call(SYSTEM_TUTOR, user_msg)
        self._r("react_explaining")         # mira al estudiante mientras responde
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

        self._r("react_thinking")           # pose de pensar mientras genera
        ctx = self._context(topic, top_k=5)
        user_msg = (
            f"Material de estudio:\n\n{ctx}\n\n"
            f"Explicame en voz alta el tema '{topic}' como si me lo estuvieras contando "
            "en persona. Arrancá con una oración que engancha, después explicá la idea "
            "central con una analogía de la vida diaria, y cerrá con algo que lo haga "
            "fácil de recordar. Sin listas, sin títulos, sin asteriscos. "
            "Máximo 5 oraciones en total, como una charla."
        )
        result = self._call(SYSTEM_TUTOR, user_msg, max_tokens=400)
        self._r("react_explaining")         # azul + mira usuario al dar la explicación
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

        self._r("react_thinking")           # pose pensar mientras Claude genera
        raw = self._call(SYSTEM_QUIZ, user_msg, max_tokens=1200)
        quiz.questions = self._parse_quiz_json(raw)

        if not quiz.questions:
            logger.error("No se pudo parsear el quiz generado.")
        else:
            logger.info("Quiz generado: %d preguntas sobre '%s'.", len(quiz.questions), topic)
            self._r("react_nod")            # asiente: quiz listo

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

        # Reacción física inmediata (antes de generar feedback con LLM)
        if is_correct:
            self._r("react_correct_answer")
        else:
            self._r("react_wrong_answer")

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
        Flujo de estudio completo con reacciones físicas de RAPIRO:
          1. Azul + mirar usuario: inicio de sesión
          2. Pensar + explicar: explain_with_example
          3. Pensar + asentir: generate_quiz
          4. Retorna diccionario listo para presentar

        Returns:
            {
                "topic": str,
                "explanation": str,
                "quiz": Quiz,
            }
        """
        logger.info("Iniciando sesion de estudio: '%s' (%d preguntas).", topic, n_questions)
        self._r("react_explaining")         # inicio: azul, mira al estudiante
        explanation = self.explain_with_example(topic)
        quiz = self.generate_quiz(topic, n_questions)
        self._r("react_asking_question")    # blanco: quiz listo, esperando
        return {
            "topic": topic,
            "explanation": explanation,
            "quiz": quiz,
        }

    # ------------------------------------------------------------------
    # TTS
    # ------------------------------------------------------------------

    def speak(self, text: str) -> None:
        """Sintetiza texto en audio y lo reproduce. RAPIRO mira al usuario mientras habla."""
        self._r("react_speaking")
        if not text:
            return

        # Intentar pygame + gTTS (Linux / Raspberry Pi)
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
            return

        except ImportError:
            pass
        except Exception as exc:
            logger.warning("pygame/gTTS falló: %s — intentando pyttsx3.", exc)

        # Fallback: pyttsx3 (Windows, sin dependencias nativas)
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 160)
            # Buscar voz en español si está disponible
            for voice in engine.getProperty("voices"):
                if "es" in voice.id.lower() or "spanish" in voice.name.lower():
                    engine.setProperty("voice", voice.id)
                    break
            engine.say(text)
            engine.runAndWait()
            return
        except ImportError:
            pass
        except Exception as exc:
            logger.warning("pyttsx3 falló: %s", exc)

        logger.warning("Sin TTS disponible — solo respuesta por texto.")

    def answer_and_speak(self, question: str) -> str:
        text = self.answer(question)
        self.speak(text)
        return text

    # ------------------------------------------------------------------
    # Modo 6: escuchar respuesta por micrófono
    # ------------------------------------------------------------------

    def listen_for_answer(self, timeout: int = 10) -> str | None:
        """
        Escucha la respuesta del estudiante por micrófono.
        Retorna 'A', 'B', 'C' o 'D', o None si no se entendió.
        """
        try:
            import speech_recognition as sr
        except ImportError:
            logger.warning("SpeechRecognition no disponible.")
            return None

        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300
        recognizer.pause_threshold = 0.8

        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.3)
                logger.info("Escuchando respuesta...")
                audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=5)

            text = recognizer.recognize_google(audio, language=SPEECH_LANGUAGE).upper()
            logger.info("Respuesta escuchada: '%s'", text)

            # Buscar A/B/C/D como palabra suelta o inicio de texto
            for letter in ("A", "B", "C", "D"):
                if letter in text.split() or text.strip().startswith(letter):
                    return letter

            # Manejar "LA A", "LA B", etc. (forma española)
            for letter in ("A", "B", "C", "D"):
                if f"LA {letter}" in text or f"EL {letter}" in text:
                    return letter

            return None

        except sr.WaitTimeoutError:
            logger.warning("Tiempo de espera agotado esperando respuesta.")
            return None
        except sr.UnknownValueError:
            logger.warning("No se entendió la respuesta.")
            return None
        except Exception as exc:
            logger.error("Error escuchando respuesta: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Modo 7: sesión de estudio completa por audio
    # ------------------------------------------------------------------

    def _listen_short(self, timeout: int = 6) -> str:
        """Escucha respuesta corta de voz. Retorna texto en minúsculas o ''."""
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            recognizer.energy_threshold = 300
            recognizer.pause_threshold = 0.7
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.3)
                audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=5)
            return recognizer.recognize_google(audio, language=SPEECH_LANGUAGE).lower()
        except Exception:
            return ""

    def audio_study_session(self, topic: str, n_questions: int = 3) -> dict:
        """
        Sesión de estudio conversacional por voz. Con pausas y preguntas al estudiante.
        """
        logger.info("Sesion audio: '%s'", topic)

        # 1. Introducción
        self._r("react_thinking")
        self.speak(f"Dale, arrancamos con {topic}. Preparate que te cuento la idea principal.")

        explanation = self.explain_with_example(topic)
        self._r("react_explaining")
        self.speak(explanation)

        # Pausa: preguntar si entendió
        self._r("react_asking_question")
        self.speak("¿Quedó claro o querés que explique algo más?")
        resp = self._listen_short(timeout=7)

        if resp and any(w in resp for w in ("no", "explicá", "explica", "otra vez", "repetí", "repeti", "no entendí")):
            self._r("react_thinking")
            self.speak("Bueno, te lo cuento de otra forma.")
            alt = self.answer(f"Explicá {topic} con otras palabras y otro ejemplo")
            self._r("react_explaining")
            self.speak(alt)
        elif resp:
            self.speak("Perfecto, seguimos.")
        else:
            self.speak("Bueno, avancemos.")

        # 2. Transición al quiz
        self._r("react_thinking")
        self.speak("Ahora te hago unas preguntas para ver cómo quedó. ¿Arrancamos?")
        resp = self._listen_short(timeout=6)
        if resp and any(w in resp for w in ("no", "esperá", "espera", "un momento")):
            self.speak("Tranquilo, avisame cuando estés listo.")
            self._listen_short(timeout=15)

        quiz = self.generate_quiz(topic, n_questions)
        if not quiz.questions:
            self.speak("No pude armar las preguntas ahora. Intentamos después.")
            return {"topic": topic, "explanation": explanation, "results": [], "score_pct": 0}

        # 3. Quiz conversacional
        results = []
        for i, q in enumerate(quiz.questions):
            self._r("react_asking_question")

            self.speak(f"Pregunta {i + 1}. {q.question}")
            self.speak(
                f"A: {q.options['A']}. "
                f"B: {q.options['B']}. "
                f"C: {q.options['C']}. "
                f"D: {q.options['D']}."
            )
            self.speak("¿Cuál te parece?")

            answer = None
            for attempt in range(2):
                answer = self.listen_for_answer(timeout=12)
                if answer:
                    break
                if attempt == 0:
                    self.speak("No te escuché, decime la letra: A, B, C o D.")

            if not answer:
                self.speak("No te pude escuchar, pasamos.")
                continue

            result = self.evaluate_answer(q, answer)
            results.append(result)
            self.speak(result.feedback)

            # Pequeña pausa entre preguntas
            if i < len(quiz.questions) - 1:
                self.speak("¿Seguimos con la siguiente?")
                self._listen_short(timeout=4)

        # 4. Cierre
        if results:
            correct = sum(1 for r in results if r.is_correct)
            total = len(results)
            pct = correct / total
            self._r("react_quiz_score", pct)

            if pct >= 0.70:
                self.speak(f"Muy bien, {correct} de {total}. La tenés clara.")
            elif pct >= 0.40:
                self.speak(f"{correct} de {total}. Bien encaminado, repasá los que erraste.")
            else:
                self.speak(f"{correct} de {total}. No te cuelgues, repasamos y la próxima mejor.")
        else:
            pct = 0

        return {
            "topic": topic,
            "explanation": explanation,
            "results": results,
            "score_pct": pct,
        }
