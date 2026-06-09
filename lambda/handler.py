import json
import os
import boto3
from datetime import datetime, timezone

dynamodb = boto3.resource("dynamodb")
sns      = boto3.client("sns")

TABLE             = os.environ["DYNAMODB_TABLE"]
SNS_ARN           = os.environ["SNS_TOPIC_ARN"]
ABSENCE_THRESHOLD = int(os.environ.get("ABSENCE_THRESHOLD", "300"))

CLASS_LABELS = {0: "Estudiando", 1: "Usando celular", 2: "Puesto vacío"}

# TTL: guardar eventos 30 días
_TTL_SECONDS = 30 * 24 * 3600


def lambda_handler(event, context):
    print(f"IoT event: {json.dumps(event)}")

    session_id = event.get("session_id", "unknown")
    timestamp  = event.get("timestamp", datetime.now(timezone.utc).isoformat())
    clase      = event.get("clase_detectada")
    confianza  = event.get("confianza", 0)
    latencia   = event.get("latencia_ms", 0)
    label      = event.get("label", CLASS_LABELS.get(clase, "desconocido"))

    table = dynamodb.Table(TABLE)
    table.put_item(Item={
        "session_id":      session_id,
        "timestamp":       timestamp,
        "clase_detectada": clase,
        "label":           label,
        "confianza":       str(round(float(confianza), 4)),
        "latencia_ms":     int(latencia),
        "expires_at":      int(datetime.now(timezone.utc).timestamp() + _TTL_SECONDS),
    })

    if clase == 2:
        _send_absence_alert(session_id, timestamp, confianza)

    return {"statusCode": 200, "body": "OK"}


def _send_absence_alert(session_id: str, timestamp: str, confianza: float) -> None:
    try:
        sns.publish(
            TopicArn=SNS_ARN,
            Subject=f"RAPIRO: Puesto vacío — sesión {session_id[:8]}",
            Message=(
                f"Se detectó que el puesto está vacío.\n\n"
                f"Sesión:    {session_id}\n"
                f"Timestamp: {timestamp}\n"
                f"Confianza: {float(confianza):.1%}\n\n"
                f"Revisá si el estudiante sigue en el lugar."
            ),
        )
    except Exception as exc:
        print(f"SNS publish error: {exc}")
