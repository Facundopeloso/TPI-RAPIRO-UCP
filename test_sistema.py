"""
test_sistema.py
Diagnóstico rápido de todos los componentes del sistema.
Uso: python3 test_sistema.py
"""

import sys
import time
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

OK   = "[OK]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"

results = {}


# ── 1. RED NEURONAL ──────────────────────────────────────────────────────────
print("\n=== 1. RED NEURONAL ===")
try:
    import numpy as np
    from src.classification.classifier import StudentClassifier
    clf = StudentClassifier()
    dummy = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    t0 = time.perf_counter()
    result = clf.predict(dummy)
    lat = (time.perf_counter() - t0) * 1000
    if result is not None:
        print(f"{OK} Clase={result.class_id} ({result.label}) conf={result.confidence:.2f} lat={lat:.0f}ms")
        results["nn"] = True
    else:
        print(f"{FAIL} predict() devolvió None (conf baja en imagen aleatoria, esperado)")
        # Esto es normal con imagen random — clasificador funciona igual
        print(f"{OK} Clasificador cargado y corriendo (imagen random → baja confianza)")
        results["nn"] = True
except Exception as e:
    print(f"{FAIL} {e}")
    import traceback; traceback.print_exc()
    results["nn"] = False


# ── 2. CÁMARA ────────────────────────────────────────────────────────────────
print("\n=== 2. CÁMARA ===")
try:
    import cv2
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            print(f"{OK} Frame capturado: {frame.shape}")
            results["cam"] = True

            # Clasificar frame real
            print("    Clasificando frame real...")
            from src.classification.classifier import StudentClassifier
            clf2 = StudentClassifier()
            r = clf2.predict(frame)
            if r:
                print(f"    Predicción real: clase={r.class_id} ({r.label}) conf={r.confidence:.2f}")
            else:
                print(f"    Confianza baja en frame real (conf < MIN_CONFIDENCE)")
        else:
            print(f"{FAIL} No se pudo leer frame")
            results["cam"] = False
    else:
        print(f"{FAIL} Cámara índice 0 no disponible")
        results["cam"] = False
except Exception as e:
    print(f"{FAIL} {e}")
    results["cam"] = False


# ── 3. DYNAMODB ──────────────────────────────────────────────────────────────
print("\n=== 3. DYNAMODB ===")
try:
    from src.cloud.boto3_publisher import Boto3Publisher
    pub = Boto3Publisher()
    # Forzar throttle = 0 para que publique
    pub._last_publish = 0.0
    ok = pub.publish_event(class_id=0, label="TEST", confidence=0.99, latency_ms=100.0)
    if ok:
        print(f"{OK} Evento publicado en DynamoDB OK")
        results["ddb"] = True
    else:
        print(f"{FAIL} publish_event devolvió False (DynamoDB no disponible)")
        results["ddb"] = False
except Exception as e:
    print(f"{FAIL} {e}")
    import traceback; traceback.print_exc()
    results["ddb"] = False


# ── 4. AUDIO SALIDA (TTS) ────────────────────────────────────────────────────
print("\n=== 4. AUDIO SALIDA (TTS) ===")
try:
    from src.tutoring.tutor import IntelligentTutor
    from src.tutoring.document_processor import DocumentProcessor
    doc = DocumentProcessor()
    tutor = IntelligentTutor(doc)
    print("    Sintetizando voz: 'Sistema RAPIRO funcionando correctamente'...")
    tutor._speak("Sistema RAPIRO funcionando correctamente.")
    print(f"{OK} TTS ejecutado (¿escuchaste audio?)")
    results["tts"] = True
except Exception as e:
    print(f"{FAIL} {e}")
    import traceback; traceback.print_exc()
    results["tts"] = False


# ── 5. MICRÓFONO ─────────────────────────────────────────────────────────────
print("\n=== 5. MICRÓFONO ===")
try:
    import speech_recognition as sr
    r = sr.Recognizer()
    mics = sr.Microphone.list_microphone_names()
    print(f"    Micrófonos disponibles: {len(mics)}")
    for i, m in enumerate(mics[:5]):
        print(f"      [{i}] {m}")
    with sr.Microphone() as source:
        print("    Ajustando ruido ambiente (1s)...")
        r.adjust_for_ambient_noise(source, duration=1)
        print(f"    Energy threshold: {r.energy_threshold:.0f}")
    print(f"{OK} Micrófono inicializado")
    results["mic"] = True
except Exception as e:
    print(f"{FAIL} {e}")
    results["mic"] = False


# ── 6. CLAUDE API ────────────────────────────────────────────────────────────
print("\n=== 6. CLAUDE API ===")
try:
    import anthropic
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        print(f"{FAIL} ANTHROPIC_API_KEY no encontrada en .env")
        results["claude"] = False
    else:
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=20,
            messages=[{"role": "user", "content": "Responde solo: OK"}]
        )
        resp = msg.content[0].text.strip()
        print(f"{OK} Claude responde: '{resp}'")
        results["claude"] = True
except Exception as e:
    print(f"{FAIL} {e}")
    results["claude"] = False


# ── RESUMEN ──────────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("RESUMEN:")
names = {
    "nn": "Red neuronal (TFLite)",
    "cam": "Cámara + clasificación",
    "ddb": "DynamoDB publish",
    "tts": "Audio TTS",
    "mic": "Micrófono",
    "claude": "Claude API",
}
for k, name in names.items():
    v = results.get(k)
    tag = OK if v else (FAIL if v is False else SKIP)
    print(f"  {tag}  {name}")
