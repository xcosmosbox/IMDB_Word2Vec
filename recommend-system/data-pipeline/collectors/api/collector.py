"""
HTTP API 事件收集器
"""
from typing import List, Optional
from datetime import datetime
import uuid
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel, Field, validator
from ...interfaces import RawEvent, EventType
from ..kafka.producer import KafkaEventPublisher
from ..kafka.config import CollectorConfig

app = FastAPI(title="Event Collector API")

class EventRequest(BaseModel):
    event_type: str
    user_id: str
    item_id: str
    timestamp: Optional[datetime] = None
    context: dict = {}
    properties: dict = {}

class BatchEventRequest(BaseModel):
    events: List[EventRequest]

class EventCollectorAPI:
    def __init__(self, config: CollectorConfig, publisher: KafkaEventPublisher):
        self.config = config
        self.publisher = publisher
        self.setup_routes()

    def setup_routes(self):
        @app.post("/v1/events", status_code=202)
        async def collect_event(event: EventRequest, background_tasks: BackgroundTasks):
            raw_event = self._to_raw_event(event)
            background_tasks.add_task(self.publisher.publish, raw_event)
            return {"event_id": raw_event.event_id, "status": "accepted"}

        @app.post("/v1/events/batch", status_code=202)
        async def collect_batch(batch: BatchEventRequest, background_tasks: BackgroundTasks):
            events = [self._to_raw_event(e) for e in batch.events]
            background_tasks.add_task(self.publisher.publish_batch, events)
            return {"count": len(events), "status": "accepted"}

    def _to_raw_event(self, request: EventRequest) -> RawEvent:
        return RawEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType(request.event_type),
            user_id=request.user_id,
            item_id=request.item_id,
            timestamp=request.timestamp or datetime.now(),
            context=request.context,
            properties=request.properties,
        )

