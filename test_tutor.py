"""
test_tutor.py
Prueba rápida del tutor Claude sin hardware ni PDF.
Corre desde la raíz del proyecto con el venv activado.

Uso:
    python test_tutor.py                          # texto en terminal
    python test_tutor.py --pdf ruta/archivo.pdf   # con PDF propio
    python test_tutor.py --voice                  # micrófono + speaker
    python test_tutor.py --voice --pdf doc.pdf    # voz + PDF propio
"""
import argparse
import sys
import time
from dotenv import load_dotenv

load_dotenv()

from src.tutoring.document_processor import DocumentProcessor
from src.tutoring.tutor import IntelligentTutor
from src.tutoring.voice_listener import VoiceListener

TEXTO_DEMO = """
La fotosíntesis es el proceso por el cual las plantas convierten luz solar,
agua y dióxido de carbono en glucosa y oxígeno. Ocurre principalmente en los
cloroplastos, que contienen clorofila. La clorofila absorbe luz roja y azul,
reflejando la verde, lo cual explica el color de las hojas.

El proceso tiene dos etapas: las reacciones de la luz (fase luminosa) que
ocurren en los tilacoides y producen ATP y NADPH, y el ciclo de Calvin
(fase oscura) que ocurre en el estroma y fija el CO2 para producir glucosa.

La ecuación general es: 6CO2 + 6H2O + luz → C6H12O6 + 6O2
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=str, default=None, help="Ruta a un PDF de prueba")
    parser.add_argument("--voice", action="store_true", help="Usar micrófono y speaker")
    args = parser.parse_args()

    doc = DocumentProcessor()

    if args.pdf:
        n = doc.load_from_file(args.pdf)
        print(f"PDF cargado: {n} chunks indexados.\n")
    else:
        n = doc.load_from_text(TEXTO_DEMO)
        print(f"Texto demo cargado: {n} chunks.\n")
        print("(Podés pasar tu propio PDF con --pdf ruta/archivo.pdf)\n")

    tutor = IntelligentTutor(doc)

    if not tutor._client:
        print("ERROR: ANTHROPIC_API_KEY no configurada en el .env")
        sys.exit(1)

    if args.voice:
        print("Modo voz activado. Decí 'RAPIRO ayuda' y luego tu pregunta.")
        print("Ctrl+C para salir.\n")
        print("-" * 50)

        def on_question(q: str) -> None:
            print(f"\nPregunta: {q}")
            print("Pensando...")
            tutor.answer_and_speak(q)

        listener = VoiceListener(on_question=on_question)
        if not listener.start():
            print("ERROR: no se pudo iniciar el micrófono.")
            sys.exit(1)

        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            listener.stop()
            print("\nHasta luego.")
    else:
        print("Tutor listo. Escribí tu pregunta (o 'salir' para terminar).\n")
        print("-" * 50)

        while True:
            try:
                pregunta = input("Pregunta: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nHasta luego.")
                break

            if pregunta.lower() in ("salir", "exit", "q"):
                break
            if not pregunta:
                continue

            print("Pensando...", end="\r")
            respuesta = tutor.answer(pregunta)
            print(f"RAPIRO: {respuesta}\n")


if __name__ == "__main__":
    main()
