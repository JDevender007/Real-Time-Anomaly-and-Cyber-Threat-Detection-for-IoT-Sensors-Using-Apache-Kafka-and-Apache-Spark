"""Kafka message serializers."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
import uuid


def serialize_message(data: dict[str, Any]) -> bytes:
    envelope = {
        "event_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "iot-security-platform",
        "payload": data,
    }
    return json.dumps(envelope).encode("utf-8")


def deserialize_message(raw: bytes) -> dict[str, Any]:
    try:
        return json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"Failed to deserialize message: {e}") from e


def extract_payload(envelope: dict[str, Any]) -> dict[str, Any]:
    return envelope.get("payload", envelope)
