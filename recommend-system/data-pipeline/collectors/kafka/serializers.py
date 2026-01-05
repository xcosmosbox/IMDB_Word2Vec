import json
from typing import Dict, Any
from ..interfaces import RawEvent, EventType
from datetime import datetime

class EventSerializer:
    def serialize(self, event: RawEvent) -> bytes:
        data = {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "user_id": event.user_id,
            "item_id": event.item_id,
            "timestamp": event.timestamp.isoformat(),
            "context": event.context,
            "properties": event.properties
        }
        return json.dumps(data).encode('utf-8')

class EventDeserializer:
    def deserialize(self, data: bytes) -> Dict[str, Any]:
        return json.loads(data.decode('utf-8'))

