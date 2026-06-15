"""
src/cloud/boto3_publisher.py
Publica eventos en DynamoDB directamente via boto3.
También publica en IoT Core para monitoreo MQTT opcional.
"""

import json
import time
import uuid
import logging
from decimal import Decimal
from datetime import datetime, timezone

from config.settings import (
    AWS_REGION, AWS_IOT_ENDPOINT, AWS_IOT_TOPIC,
    AWS_DYNAMODB_TABLE, MQTT_THROTTLE_SECONDS,
)

logger = logging.getLogger(__name__)


class Boto3Publisher:
    """Publica eventos directamente en DynamoDB (sin Lambda)."""

    def __init__(self):
        self.session_id = str(uuid.uuid4())[:8]
        self._last_publish: float = 0.0
        self._ddb_table = self._init_dynamodb()
        self._iot_client = self._init_iot()

    def _init_dynamodb(self):
        try:
            import boto3
            ddb = boto3.resource("dynamodb", region_name=AWS_REGION)
            table = ddb.Table(AWS_DYNAMODB_TABLE)
            logger.info("DynamoDB listo → tabla: %s", AWS_DYNAMODB_TABLE)
            return table
        except Exception as exc:
            logger.error("Error init DynamoDB: %s", exc)
            return None

    def _init_iot(self):
        if not AWS_IOT_ENDPOINT:
            return None
        try:
            import boto3
            client = boto3.client(
                "iot-data",
                region_name=AWS_REGION,
                endpoint_url=f"https://{AWS_IOT_ENDPOINT}",
            )
            logger.info("boto3 IoT Data client listo → %s", AWS_IOT_ENDPOINT)
            return client
        except Exception as exc:
            logger.error("Error init boto3 IoT client: %s", exc)
            return None

    def publish_event(
        self, class_id: int, label: str, confidence: float, latency_ms: float
    ) -> bool:
        now = time.monotonic()
        if now - self._last_publish < MQTT_THROTTLE_SECONDS:
            return False

        ts = datetime.now(timezone.utc).isoformat()
        conf_r = round(confidence, 4)
        lat_r  = round(latency_ms, 2)

        # Payload para DynamoDB (Decimal obligatorio para floats)
        ddb_item = {
            "timestamp":       ts,
            "session_id":      self.session_id,
            "clase_detectada": class_id,
            "label":           label,
            "confianza":       Decimal(str(conf_r)),
            "latencia_ms":     Decimal(str(lat_r)),
        }
        # Payload para IoT/JSON (floats normales)
        json_payload = {
            "timestamp":       ts,
            "session_id":      self.session_id,
            "clase_detectada": class_id,
            "label":           label,
            "confianza":       conf_r,
            "latencia_ms":     lat_r,
        }
        self._last_publish = now

        ok = False

        # Escribir directo a DynamoDB
        if self._ddb_table is not None:
            try:
                self._ddb_table.put_item(Item=ddb_item)
                logger.info("DynamoDB OK: clase=%d (%s) conf=%.2f", class_id, label, confidence)
                ok = True
            except Exception as exc:
                logger.error("Error escribiendo DynamoDB: %s", exc)

        # Publicar en IoT Core también (opcional)
        if self._iot_client is not None:
            try:
                self._iot_client.publish(
                    topic=AWS_IOT_TOPIC,
                    qos=0,
                    payload=json.dumps(json_payload),
                )
            except Exception:
                pass

        if not ok:
            logger.info("[OFFLINE] %s", json.dumps(json_payload))

        return ok

    def disconnect(self) -> None:
        pass
