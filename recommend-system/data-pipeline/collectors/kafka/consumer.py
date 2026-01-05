"""
Kafka 事件消费者
实现 EventCollectorInterface 接口
"""

from typing import Generator, Dict, Any, Optional
from datetime import datetime
import json
import logging
from unittest.mock import MagicMock

# 模拟 confluent_kafka 以避免依赖报错 (在没有真实环境时)
try:
    from confluent_kafka import Consumer, KafkaError, KafkaException
except ImportError:
    Consumer = MagicMock()
    KafkaError = MagicMock()
    KafkaException = Exception

from ...interfaces import EventCollectorInterface, RawEvent, EventType
from .config import KafkaConfig
from .serializers import EventDeserializer

logger = logging.getLogger(__name__)

class KafkaEventCollector(EventCollectorInterface):
    """Kafka 事件采集器"""
    
    def __init__(
        self,
        config: KafkaConfig,
        deserializer: Optional[EventDeserializer] = None,
    ):
        self.config = config
        self.deserializer = deserializer or EventDeserializer()
        self._consumer: Optional[Consumer] = None
        self._current_offset: Dict[str, int] = {}
        self._running = False
        
    def _create_consumer(self) -> Consumer:
        conf = {
            'bootstrap.servers': self.config.bootstrap_servers,
            'group.id': self.config.group_id,
            'auto.offset.reset': self.config.auto_offset_reset,
            'enable.auto.commit': False,
            'max.poll.interval.ms': self.config.max_poll_interval_ms,
            'session.timeout.ms': self.config.session_timeout_ms,
        }
        if self.config.security_protocol:
            conf['security.protocol'] = self.config.security_protocol
            conf['sasl.mechanism'] = self.config.sasl_mechanism
            conf['sasl.username'] = self.config.sasl_username
            conf['sasl.password'] = self.config.sasl_password
        
        return Consumer(conf)
    
    def collect(self) -> Generator[RawEvent, None, None]:
        self._consumer = self._create_consumer()
        if self.config.topics:
            self._consumer.subscribe(self.config.topics)
        self._running = True
        
        logger.info(f"Started collecting from topics: {self.config.topics}")
        
        try:
            while self._running:
                msg = self._consumer.poll(timeout=1.0)
                
                if msg is None:
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        raise KafkaException(msg.error())
                
                try:
                    event_data = self.deserializer.deserialize(msg.value())
                    event = self._parse_event(event_data, msg)
                    
                    if self.validate_event(event):
                        self._current_offset[f"{msg.topic()}-{msg.partition()}"] = msg.offset()
                        yield event
                    else:
                        logger.warning(f"Invalid event: {event_data}")
                        
                except Exception as e:
                    logger.error(f"Failed to process message: {e}")
                    self._send_to_dlq(msg, str(e))
                    
        finally:
            self._consumer.close()
    
    def _parse_event(self, data: Dict[str, Any], msg) -> RawEvent:
        return RawEvent(
            event_id=data.get('event_id', f"{msg.topic()}-{msg.partition()}-{msg.offset()}"),
            event_type=EventType(data['event_type']),
            user_id=data['user_id'],
            item_id=data['item_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            context=data.get('context', {}),
            properties=data.get('properties', {}),
        )
    
    def validate_event(self, event: RawEvent) -> bool:
        if not event.user_id or not event.item_id:
            return False
        if not event.event_type:
            return False
        return True
    
    def get_offset(self) -> str:
        return json.dumps(self._current_offset)
    
    def commit_offset(self, offset: str = None) -> None:
        if self._consumer:
            self._consumer.commit(asynchronous=False)
    
    def stop(self) -> None:
        self._running = False
    
    def _send_to_dlq(self, msg, error: str) -> None:
        logger.error(f"DLQ: {error}")

