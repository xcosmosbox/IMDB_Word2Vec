"""
Kafka 事件生产者
实现 EventPublisherInterface 接口
"""

from typing import List, Tuple, Optional
import json
import logging
from unittest.mock import MagicMock

try:
    from confluent_kafka import Producer
except ImportError:
    Producer = MagicMock()

from ...interfaces import EventPublisherInterface, RawEvent
from .config import KafkaConfig
from .serializers import EventSerializer

logger = logging.getLogger(__name__)

class KafkaEventPublisher(EventPublisherInterface):
    """Kafka 事件发布器"""
    
    def __init__(
        self,
        config: KafkaConfig,
        serializer: Optional[EventSerializer] = None,
    ):
        self.config = config
        self.serializer = serializer or EventSerializer()
        self._producer = self._create_producer()
        self._pending_count = 0
        
    def _create_producer(self) -> Producer:
        conf = {
            'bootstrap.servers': self.config.bootstrap_servers,
            'acks': 'all',
            'retries': 3,
            'retry.backoff.ms': 100,
            'linger.ms': 5,
        }
        if self.config.security_protocol:
            conf['security.protocol'] = self.config.security_protocol
            conf['sasl.mechanism'] = self.config.sasl_mechanism
            conf['sasl.username'] = self.config.sasl_username
            conf['sasl.password'] = self.config.sasl_password
        
        return Producer(conf)
    
    def _delivery_callback(self, err, msg):
        self._pending_count -= 1
        if err:
            logger.error(f"Failed to deliver message: {err}")
    
    def publish(self, event: RawEvent) -> bool:
        try:
            topic = self.config.topics[0] if self.config.topics else "default-events"
            key = event.user_id.encode('utf-8')
            value = self.serializer.serialize(event)
            
            self._producer.produce(
                topic=topic,
                key=key,
                value=value,
                callback=self._delivery_callback,
            )
            self._pending_count += 1
            self._producer.poll(0)
            return True
        except Exception as e:
            logger.error(f"Failed to publish event: {e}")
            return False
    
    def publish_batch(self, events: List[RawEvent]) -> Tuple[int, int]:
        success_count = 0
        fail_count = 0
        for event in events:
            if self.publish(event):
                success_count += 1
            else:
                fail_count += 1
        return success_count, fail_count
    
    def flush(self, timeout: float = 10.0) -> int:
        return self._producer.flush(timeout)
    
    def close(self) -> None:
        self.flush()

