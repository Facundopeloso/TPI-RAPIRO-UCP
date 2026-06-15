"""
lambda/inference_handler.py
Lambda de inferencia RAPIRO.
Recibe JPEG en base64 → devuelve class_id + label + confidence.
Diseñado para ejecutar TFLite cuando se agregue el Layer; por ahora
descarga el modelo de S3 y lo corre con tflite-runtime si está disponible,
sino devuelve 404 con mensaje claro.
"""
import base64
import json
import os


CLASS_LABELS = {
    0: "Estudiando",
    1: "Usando celular",
    2: "Puesto vacío",
    3: "Confundido",
    4: "Aburrido",
}

MODEL_BUCKET = os.environ.get("MODEL_BUCKET", "")
MODEL_KEY    = os.environ.get("MODEL_KEY", "model.tflite")

_interpreter = None


def _load_model():
    global _interpreter
    if _interpreter is not None:
        return _interpreter
    try:
        import tflite_runtime.interpreter as tflite
        import boto3, tempfile, pathlib

        s3   = boto3.client("s3")
        tmp  = pathlib.Path(tempfile.gettempdir()) / "model.tflite"
        if not tmp.exists():
            s3.download_file(MODEL_BUCKET, MODEL_KEY, str(tmp))

        _interpreter = tflite.Interpreter(model_path=str(tmp))
        _interpreter.allocate_tensors()
        print("Modelo TFLite cargado desde S3.")
    except ImportError:
        print("tflite_runtime no disponible en este entorno Lambda.")
        _interpreter = None
    return _interpreter


def lambda_handler(event, context):
    body = event.get("body") or ""
    if event.get("isBase64Encoded"):
        image_bytes = base64.b64decode(body)
    else:
        try:
            payload     = json.loads(body) if isinstance(body, str) else body
            image_b64   = payload.get("image", "")
            image_bytes = base64.b64decode(image_b64)
        except Exception:
            return _resp(400, {"error": "body debe ser JSON con campo 'image' en base64"})

    interp = _load_model()
    if interp is None:
        return _resp(503, {
            "error": "tflite_runtime no instalado en Lambda",
            "hint":  "Agrega un Lambda Layer con tflite-runtime compilado para Amazon Linux 2 / Python 3.11",
        })

    try:
        import numpy as np
        import cv2

        nparr  = np.frombuffer(image_bytes, np.uint8)
        frame  = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frame  = cv2.resize(frame, (224, 224))
        frame  = frame.astype("float32") / 255.0
        tensor = np.expand_dims(frame, axis=0)

        in_det  = interp.get_input_details()
        out_det = interp.get_output_details()
        interp.set_tensor(in_det[0]["index"], tensor)
        interp.invoke()
        probs  = interp.get_tensor(out_det[0]["index"])[0]

        class_id   = int(probs.argmax())
        confidence = float(probs[class_id])
        label      = CLASS_LABELS.get(class_id, "desconocido")

        return _resp(200, {
            "class_id":   class_id,
            "label":      label,
            "confidence": round(confidence, 4),
        })
    except Exception as exc:
        return _resp(500, {"error": str(exc)})


def _resp(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers":    {"Content-Type": "application/json"},
        "body":       json.dumps(body),
    }
