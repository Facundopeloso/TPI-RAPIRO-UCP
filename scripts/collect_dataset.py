"""
scripts/collect_dataset.py
Herramienta de captura de imágenes para armar el dataset de entrenamiento.

Captura frames de la webcam y los guarda en:
    dataset/class_0/  ->  Estudiando / trabajando
    dataset/class_1/  ->  Usando el celular
    dataset/class_2/  ->  Puesto vacío

Controles:
    0  — capturar imagen clase 0 (Estudiando)
    1  — capturar imagen clase 1 (Usando celular)
    2  — capturar imagen clase 2 (Puesto vacío)
    s  — mostrar estadísticas del dataset actual
    q  — salir

Recomendación: al menos 200 imágenes por clase, variando:
    - iluminación (natural, artificial, noche)
    - ángulo de cámara
    - persona (si es posible, múltiples personas)
    - fondo

Uso:
    python scripts/collect_dataset.py
    python scripts/collect_dataset.py --camera 1   # índice de cámara alternativo
"""

import argparse
import os
import sys
import time
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATASET_DIR = os.path.join(ROOT, "dataset")
CLASS_DIRS = {
    0: os.path.join(DATASET_DIR, "class_0"),
    1: os.path.join(DATASET_DIR, "class_1"),
    2: os.path.join(DATASET_DIR, "class_2"),
}
CLASS_NAMES = {0: "Estudiando", 1: "Usando celular", 2: "Puesto vacio"}
CLASS_COLORS = {0: (0, 200, 0), 1: (0, 200, 200), 2: (0, 0, 200)}

for d in CLASS_DIRS.values():
    os.makedirs(d, exist_ok=True)


def count_images() -> dict[int, int]:
    return {
        c: len([f for f in os.listdir(d) if f.endswith(".jpg")])
        for c, d in CLASS_DIRS.items()
    }


def save_image(frame: np.ndarray, class_id: int) -> str:
    ts = int(time.time() * 1000)
    path = os.path.join(CLASS_DIRS[class_id], f"img_{ts}.jpg")
    cv2.imwrite(path, frame)
    return path


def draw_overlay(frame: np.ndarray, counts: dict[int, int], last_saved: str) -> np.ndarray:
    display = frame.copy()
    h, w = display.shape[:2]

    # Panel superior
    cv2.rectangle(display, (0, 0), (w, 110), (30, 30, 30), -1)
    cv2.putText(display, "RAPIRO Dataset Collector", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    for cid, name in CLASS_NAMES.items():
        color = CLASS_COLORS[cid]
        x = 10 + cid * 200
        cv2.putText(display, f"[{cid}] {name}", (x, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        cv2.putText(display, f"    {counts[cid]} imgs", (x, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    cv2.putText(display, "[s] stats  [q] salir", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    if last_saved:
        cv2.putText(display, f"Guardado: {os.path.basename(last_saved)}", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return display


def print_stats(counts: dict[int, int]):
    total = sum(counts.values())
    print("\n--- Dataset actual ---")
    for cid, name in CLASS_NAMES.items():
        bar = "#" * counts[cid] + "-" * max(0, 200 - counts[cid])
        pct = counts[cid] / max(total, 1) * 100
        print(f"  Clase {cid} ({name:15s}): {counts[cid]:4d} imgs  {pct:.1f}%")
    print(f"  TOTAL: {total} imgs")
    print(f"  Meta recomendada: 200 por clase (600 total)\n")


def main(camera_index: int):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: no se pudo abrir cámara {camera_index}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Colector de dataset RAPIRO iniciado.")
    print("Teclas: [0] Estudiando  [1] Celular  [2] Ausente  [s] Stats  [q] Salir")

    counts = count_images()
    last_saved = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = draw_overlay(frame, counts, last_saved)
        cv2.imshow("RAPIRO Dataset Collector", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("s"):
            print_stats(counts)
        elif key in (ord("0"), ord("1"), ord("2")):
            class_id = key - ord("0")
            path = save_image(frame, class_id)
            counts[class_id] += 1
            last_saved = path
            print(f"  [{CLASS_NAMES[class_id]}] -> {os.path.basename(path)}  (total: {counts[class_id]})")

    cap.release()
    cv2.destroyAllWindows()
    print_stats(count_images())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    main(args.camera)
