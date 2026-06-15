"""
src/cloud/iot_publisher.py
Publicación de eventos de sesión en AWS IoT Core via MQTT.
Usa awsiotsdk si está disponible, sino paho-mqtt con TLS mutuo.
"""

import time
import json
import uuid
import logging
from datetime import datetime, timezone
from config.settings import (
    AWS_IOT_ENDPOINT, AWS_IOT_TOPIC,
    AWS_IOT_CERT_PATH, AWS_IOT_KEY_PATH, AWS_IOT_CA_PATH,
    MQTT_THROTTLE_SECONDS,
)

logger = logging.getLogger(__name__)


class IoTPublisher:
    """Publica eventos en AWS IoT Core con throttling integrado."""

    def __init__(self):
        self.session_id = str(uuid.uuid4())[:8]
        self._last_publish_time: float = 0.0
        self._client = self._init_mqtt_client()

    def _init_mqtt_client(self):
        if not AWS_IOT_ENDPOINT:
            logger.warning("AWS_IOT_ENDPOINT no configurado — modo offline.")
            return None

        # Intentar awsiotsdk primero
        try:
            from awsiot import mqtt_connection_builder
            import awscrt.io as io

            event_loop_group = io.EventLoopGroup(1)
            host_resolver = io.DefaultHostResolver(event_loop_group)
            client_bootstrap = io.ClientBootstrap(event_loop_group, host_resolver)

            connection = mqtt_connection_builder.mtls_from_path(
                endpoint=AWS_IOT_ENDPOINT,
                cert_filepath=AWS_IOT_CERT_PATH,
                pri_key_filepath=AWS_IOT_KEY_PATH,
                client_bootstrap=client_bootstrap,
                ca_filepath=AWS_IOT_CA_PATH,
                client_id=f"rapiro-{self.session_id}",
                clean_session=False,
                keep_alive_secs=30,
            )
            connect_future = connection.connect()
            connect_future.result()
            logger.info("Conectado a AWS IoT Core (awsiotsdk): %s", AWS_IOT_ENDPOINT)
            self._backend = "awsiotsdk"
            return connection

        except ImportError:
            logger.info("awsiotsdk no disponible — intentando paho-mqtt.")
        except Exception as exc:
            logger.error("awsiotsdk error: %s — intentando paho-mqtt.", exc)

        # Fallback: paho-mqtt con TLS mutuo
        try:
            import paho.mqtt.client as mqtt
            import ssl

            client = mqtt.Client(client_id=f"rapiro-{self.session_id}", protocol=mqtt.MQTTv311)
            context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=AWS_IOT_CA_PATH)
            context.load_cert_chain(certfile=AWS_IOT_CERT_PATH, keyfile=AWS_IOT_KEY_PATH)
            client.tls_set_context(context)

            client.connect(AWS_IOT_ENDPOINT, port=8883, keepalive=30)
            client.loop_start()
            logger.info("Conectado a AWS IoT Core (paho-mqtt): %s", AWS_IOT_ENDPOINT)
            self._backend = "paho"
            return client

        except ImportError:
            logger.warning("paho-mqtt no instalado — modo offline.")
        except Exception as exc:
            logger.error("paho-mqtt error: %s", exc)

        return None

    def publish_event(self, class_id: int, label: str, confidence: float, latency_ms: float) -> bool:
        now = time.monotonic()
        if now - self._last_publish_time < MQTT_THROTTLE_SECONDS:
            return False

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": self.session_id,
            "clase_detectada": class_id,
            "label": label,
            "confianza": round(confidence, 4),
            "latencia_ms": round(latency_ms, 2),
        }

        self._last_publish_time = now

        if self._client is None:
            logger.info("[OFFLINE] Evento: %s", json.dumps(payload))
            return True

        try:
            if getattr(self, "_backend", "") == "paho":
                self._client.publish(AWS_IOT_TOPIC, json.dumps(payload), qos=1)
            else:
                self._client.publish(topic=AWS_IOT_TOPIC, payload=json.dumps(payload), qos=1)
            return True
        except Exception as exc:
            logger.error("Error publicando en IoT Core: %s", exc)
            return False

    def disconnect(self) -> None:
        if self._client:
            try:
                if getattr(self, "_backend", "") == "paho":
                    self._client.loop_stop()
                    self._client.disconnect()
                else:
                    self._client.disconnect().result()
                logger.info("Desconectado de AWS IoT Core.")
            except Exception as exc:
                logger.warning("Error al desconectar IoT: %s", exc)
