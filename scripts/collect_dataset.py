"""
scripts/collect_dataset.py
Herramienta de captura de imágenes para armar el dataset de entrenamiento.

Captura frames de la webcam y los guarda en:
    dataset/class_0/  ->  Estudiando / concentrado
    dataset/class_1/  ->  Usando el celular
    dataset/class_2/  ->  Puesto vacío
    dataset/class_3/  ->  Confundido (ceño fruncido, mano en barbilla)
    dataset/class_4/  ->  Aburrido (postura caída, mirada perdida)

Controles:
    0  — capturar imagen clase 0 (Estudiando)
    1  — capturar imagen clase 1 (Usando celular)
    2  — capturar imagen clase 2 (Puesto vacío)
    3  — capturar imagen clase 3 (Confundido)
    4  — capturar imagen clase 4 (Aburrido)
    s  — mostrar estadísticas del dataset actual
    q  — salir

Recomendación: al menos 150 imágenes por clase, variando:
    - iluminación (natural, artificial, noche)
    - ángulo de cámara
    - persona (si es posible, múltiples personas)
    - fondo

Poses sugeridas para clases nuevas:
    Confundido (3): ceño fruncido, cabeza ladeada, mano en barbilla o mejilla
    Aburrido   (4): cabeza apoyada en mano, mirada al costado, postura encorvada

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

# IDs de carpetas en Google Drive (una por clase)
DRIVE_FOLDER_IDS = {
    0: '1xUQ91f9rvzeaTTRA_z9cfmbamdgLscHR',  # Estudiando
    1: '1q3pv1gnDspiZbgpYhcNqU7aZTD8rOXPX',  # Usando el Cel
    2: '1sf1tBbg2w7Rb3cPXvoRaBiM6ieaNe0-V',   # Puesto vacio
    3: '1AsYGZHBS_Zz0JhJtmNH9ZoUl98ri9NI_',  # Confundido
    4: '1AFvJBQpk7l1Gl2cOTNkOtpkv7d4gEpKU',  # Aburrido
}
CREDENTIALS_PATH = os.path.join(ROOT, 'credentials.json')
TOKEN_PATH       = os.path.join(ROOT, '.drive_token.json')
DRIVE_SCOPES     = ['https://www.googleapis.com/auth/drive.file']
sys.path.insert(0, ROOT)

DATASET_DIR = os.path.join(ROOT, "dataset")
CLASS_DIRS = {
    0: os.path.join(DATASET_DIR, "class_0"),
    1: os.path.join(DATASET_DIR, "class_1"),
    2: os.path.join(DATASET_DIR, "class_2"),
    3: os.path.join(DATASET_DIR, "class_3"),
    4: os.path.join(DATASET_DIR, "class_4"),
}
CLASS_NAMES = {
    0: "Estudiando",
    1: "Usando celular",
    2: "Puesto vacio",
    3: "Confundido",
    4: "Aburrido",
}
CLASS_COLORS = {
    0: (0, 200, 0),
    1: (0, 200, 200),
    2: (0, 0, 200),
    3: (0, 140, 255),
    4: (180, 180, 0),
}

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

    cv2.putText(display, "[3] Confundido  [4] Aburrido  [s] stats  [x] limpiar  [q] salir", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    if last_saved:
        cv2.putText(display, f"Guardado: {os.path.basename(last_saved)}", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return display


def clear_dataset() -> int:
    """Elimina todas las imágenes locales de todas las clases. Devuelve total borrado."""
    total = 0
    for class_id, d in CLASS_DIRS.items():
        if not os.path.exists(d):
            continue
        images = [f for f in os.listdir(d) if f.endswith(".jpg")]
        for f in images:
            os.remove(os.path.join(d, f))
        total += len(images)
        print(f"  Clase {class_id} ({CLASS_NAMES[class_id]}): {len(images)} imágenes eliminadas")
    print(f"  TOTAL eliminadas: {total}")
    return total


def print_stats(counts: dict[int, int]):
    total = sum(counts.values())
    print("\n--- Dataset actual ---")
    for cid, name in CLASS_NAMES.items():
        bar = "#" * counts[cid] + "-" * max(0, 200 - counts[cid])
        pct = counts[cid] / max(total, 1) * 100
        print(f"  Clase {cid} ({name:15s}): {counts[cid]:4d} imgs  {pct:.1f}%")
    print(f"  TOTAL: {total} imgs")
    print(f"  Meta recomendada: 150 por clase (750 total)\n")


def main(camera_index: int):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: no se pudo abrir cámara {camera_index}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Colector de dataset RAPIRO iniciado.")
    print("Teclas: [0] Estudiando  [1] Celular  [2] Ausente  [3] Confundido  [4] Aburrido  [s] Stats  [q] Salir")

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
        elif key == ord("x"):
            # Mostrar overlay de confirmacion en la misma ventana
            confirming = True
            while confirming:
                ret2, frame2 = cap.read()
                if not ret2:
                    frame2 = frame.copy()
                overlay = frame2.copy()
                cv2.rectangle(overlay, (0, 0), (frame2.shape[1], frame2.shape[0]), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, frame2, 0.4, 0, frame2)
                cv2.putText(frame2, "!! BORRAR todas las fotos locales?", (30, frame2.shape[0]//2 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame2, "S = Si, borrar     N = No, cancelar", (30, frame2.shape[0]//2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("RAPIRO Dataset Collector", frame2)
                confirm_key = cv2.waitKey(30) & 0xFF
                if confirm_key == ord("s"):
                    clear_dataset()
                    counts = count_images()
                    last_saved = ""
                    print("  Dataset limpio.")
                    confirming = False
                elif confirm_key == ord("n") or confirm_key == 27:  # 27 = Esc
                    print("  Cancelado.")
                    confirming = False
        elif key in (ord("0"), ord("1"), ord("2"), ord("3"), ord("4")):
            class_id = key - ord("0")
            path = save_image(frame, class_id)
            counts[class_id] += 1
            last_saved = path
            print(f"  [{CLASS_NAMES[class_id]}] -> {os.path.basename(path)}  (total: {counts[class_id]})")

    cap.release()
    cv2.destroyAllWindows()
    print_stats(count_images())


# ---------------------------------------------------------------------------
# Sync a Google Drive
# ---------------------------------------------------------------------------

def _get_drive_service():
    """Autentica y devuelve el servicio de Drive. Abre el navegador la primera vez."""
    try:
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
    except ImportError:
        print("ERROR: Instalar dependencias de Drive:")
        print("  pip install google-api-python-client google-auth-oauthlib")
        return None

    if not os.path.exists(CREDENTIALS_PATH):
        print(f"\nERROR: No se encontró '{CREDENTIALS_PATH}'.")
        print("Seguí estos pasos una sola vez:")
        print("  1. Ir a console.cloud.google.com")
        print("  2. Crear proyecto → Habilitar Google Drive API")
        print("  3. Credenciales > OAuth 2.0 > Aplicacion de escritorio")
        print("  4. Descargar JSON → guardar como 'credentials.json' en la raiz del proyecto")
        return None

    creds = None
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, DRIVE_SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, DRIVE_SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_PATH, 'w') as token:
            token.write(creds.to_json())

    return build('drive', 'v3', credentials=creds, cache_discovery=False)


def _existing_filenames_in_drive(service, folder_id: str) -> set:
    """Devuelve set de nombres de archivos ya existentes en la carpeta de Drive."""
    existing, token = set(), None
    while True:
        resp = service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields='nextPageToken, files(name)',
            pageToken=token,
        ).execute()
        for f in resp.get('files', []):
            existing.add(f['name'])
        token = resp.get('nextPageToken')
        if not token:
            break
    return existing


def sync_to_drive():
    """Sube todas las imágenes locales a las carpetas de Drive correspondientes."""
    print("\nConectando con Google Drive...")
    service = _get_drive_service()
    if service is None:
        return

    from googleapiclient.http import MediaFileUpload

    total_uploaded = 0
    total_skipped  = 0

    for class_id, folder_id in DRIVE_FOLDER_IDS.items():
        local_dir = CLASS_DIRS[class_id]
        if not os.path.exists(local_dir):
            continue

        images = [f for f in os.listdir(local_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not images:
            print(f"  Clase {class_id} ({CLASS_NAMES[class_id]}): sin imágenes, salteada.")
            continue

        print(f"\n  Clase {class_id} ({CLASS_NAMES[class_id]}): {len(images)} imágenes locales")
        existing = _existing_filenames_in_drive(service, folder_id)

        uploaded = 0
        for fname in images:
            if fname in existing:
                total_skipped += 1
                continue

            fpath = os.path.join(local_dir, fname)
            media = MediaFileUpload(fpath, mimetype='image/jpeg', resumable=False)
            service.files().create(
                body={'name': fname, 'parents': [folder_id]},
                media_body=media,
                fields='id',
            ).execute()
            uploaded += 1
            total_uploaded += 1
            print(f"    Subida {uploaded}/{len(images) - len(existing)}: {fname}", end='\r')

        print(f"    {uploaded} subidas, {len(images) - uploaded} ya existian.        ")

    print(f"\nSync completado: {total_uploaded} subidas, {total_skipped} saltadas (ya existian).")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--upload", action="store_true",
                        help="Subir imágenes a Google Drive al salir")
    args = parser.parse_args()
    main(args.camera)
    if args.upload:
        sync_to_drive()
