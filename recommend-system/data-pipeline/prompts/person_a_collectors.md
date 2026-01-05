# Person A: 数据采集 (Data Collectors)

## 你的角色
你是一名数据工程师，负责实现生成式推荐系统的 **数据采集模块**，包括 Kafka 消费者、API 事件收集器、实时事件流处理等。

---

## ⚠️ 重要：接口驱动开发

**开始编码前，必须先阅读接口定义文件：**

```
data-pipeline/interfaces.py
```

你需要实现的接口：

```python
class EventCollectorInterface(ABC):
    @abstractmethod
    def collect(self) -> Generator[RawEvent, None, None]:
        """采集事件流"""
        pass
    
    @abstractmethod
    def validate_event(self, event: RawEvent) -> bool:
        """验证事件格式"""
        pass
    
    @abstractmethod
    def get_offset(self) -> str:
        """获取当前消费位置"""
        pass
    
    @abstractmethod
    def commit_offset(self, offset: str) -> None:
        """提交消费位置"""
        pass


class EventPublisherInterface(ABC):
    @abstractmethod
    def publish(self, event: RawEvent) -> bool:
        """发布事件"""
        pass
    
    @abstractmethod
    def publish_batch(self, events: List[RawEvent]) -> Tuple[int, int]:
        """批量发布事件"""
        pass
```

---

## 技术栈

- **消息队列**: Apache Kafka 3.x
- **序列化**: Avro / Protobuf / JSON
- **Python 客户端**: confluent-kafka
- **验证**: Pydantic / JSON Schema
- **监控**: Prometheus metrics

---

## 你的任务

```
data-pipeline/collectors/
├── kafka/
│   ├── __init__.py
│   ├── consumer.py           # Kafka 消费者
│   ├── producer.py           # Kafka 生产者
│   ├── config.py             # Kafka 配置
│   ├── serializers.py        # 序列化器
│   └── schemas/
│       ├── event.avsc        # Avro Schema
│       └── event.proto       # Protobuf Schema
├── api/
│   ├── __init__.py
│   ├── collector.py          # HTTP API 收集器
│   ├── webhook.py            # Webhook 处理器
│   └── batch_importer.py     # 批量导入器
├── validators/
│   ├── __init__.py
│   ├── event_validator.py    # 事件验证器
│   └── schema_validator.py   # Schema 验证器
└── tests/
    ├── test_consumer.py
    ├── test_producer.py
    └── test_validators.py
```

---

## 1. Kafka 消费者 (kafka/consumer.py)

```python
"""
Kafka 事件消费者

实现 EventCollectorInterface 接口
"""

from typing import Generator, Dict, Any, Optional
from datetime import datetime
import json
import logging

from confluent_kafka import Consumer, KafkaError, KafkaException
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroDeserializer

from ..interfaces import EventCollectorInterface, RawEvent, EventType
from .config import KafkaConfig
from .serializers import EventDeserializer

logger = logging.getLogger(__name__)


class KafkaEventCollector(EventCollectorInterface):
    """
    Kafka 事件采集器
    
    从 Kafka Topic 消费用户行为事件
    
    使用示例:
        config = KafkaConfig(
            bootstrap_servers="localhost:9092",
            group_id="recommend-collector",
            topics=["user-events"],
        )
        collector = KafkaEventCollector(config)
        
        for event in collector.collect():
            process(event)
    """
    
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
        """创建 Kafka Consumer"""
        conf = {
            'bootstrap.servers': self.config.bootstrap_servers,
            'group.id': self.config.group_id,
            'auto.offset.reset': self.config.auto_offset_reset,
            'enable.auto.commit': False,  # 手动提交
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
        """
        采集事件流
        
        Yields:
            RawEvent: 原始事件
        """
        self._consumer = self._create_consumer()
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
                        logger.debug(f"Reached end of partition {msg.partition()}")
                        continue
                    else:
                        raise KafkaException(msg.error())
                
                try:
                    # 反序列化消息
                    event_data = self.deserializer.deserialize(msg.value())
                    event = self._parse_event(event_data, msg)
                    
                    if self.validate_event(event):
                        # 更新 offset
                        self._current_offset[f"{msg.topic()}-{msg.partition()}"] = msg.offset()
                        yield event
                    else:
                        logger.warning(f"Invalid event: {event_data}")
                        
                except Exception as e:
                    logger.error(f"Failed to process message: {e}")
                    # 发送到死信队列
                    self._send_to_dlq(msg, str(e))
                    
        finally:
            self._consumer.close()
            logger.info("Consumer closed")
    
    def _parse_event(self, data: Dict[str, Any], msg) -> RawEvent:
        """解析事件数据"""
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
        """验证事件格式"""
        if not event.user_id or not event.item_id:
            return False
        
        if not event.event_type:
            return False
        
        # 验证时间戳（不能太旧或太新）
        now = datetime.now()
        if (now - event.timestamp).days > 30:
            logger.warning(f"Event too old: {event.timestamp}")
            return False
        
        if event.timestamp > now:
            logger.warning(f"Event in future: {event.timestamp}")
            return False
        
        return True
    
    def get_offset(self) -> str:
        """获取当前消费位置"""
        return json.dumps(self._current_offset)
    
    def commit_offset(self, offset: str = None) -> None:
        """提交消费位置"""
        if self._consumer:
            self._consumer.commit(asynchronous=False)
            logger.debug("Offset committed")
    
    def stop(self) -> None:
        """停止消费"""
        self._running = False
    
    def _send_to_dlq(self, msg, error: str) -> None:
        """发送到死信队列"""
        # TODO: 实现死信队列
        logger.error(f"DLQ: {error}")


class KafkaEventCollectorWithMetrics(KafkaEventCollector):
    """带 Prometheus 监控的 Kafka 消费者"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_metrics()
    
    def _init_metrics(self):
        """初始化 Prometheus 指标"""
        from prometheus_client import Counter, Histogram, Gauge
        
        self.events_consumed = Counter(
            'kafka_events_consumed_total',
            'Total events consumed',
            ['topic', 'event_type']
        )
        
        self.events_failed = Counter(
            'kafka_events_failed_total',
            'Total events failed',
            ['topic', 'reason']
        )
        
        self.consume_latency = Histogram(
            'kafka_consume_latency_seconds',
            'Event consumption latency',
            ['topic']
        )
        
        self.consumer_lag = Gauge(
            'kafka_consumer_lag',
            'Consumer lag',
            ['topic', 'partition']
        )
    
    def collect(self) -> Generator[RawEvent, None, None]:
        """带监控的事件采集"""
        for event in super().collect():
            self.events_consumed.labels(
                topic=self.config.topics[0],
                event_type=event.event_type.value
            ).inc()
            yield event
```

---

## 2. Kafka 生产者 (kafka/producer.py)

```python
"""
Kafka 事件生产者

实现 EventPublisherInterface 接口
"""

from typing import List, Tuple, Optional
import json
import logging
from datetime import datetime

from confluent_kafka import Producer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer

from ..interfaces import EventPublisherInterface, RawEvent
from .config import KafkaConfig
from .serializers import EventSerializer

logger = logging.getLogger(__name__)


class KafkaEventPublisher(EventPublisherInterface):
    """
    Kafka 事件发布器
    
    将事件发布到 Kafka Topic
    
    使用示例:
        config = KafkaConfig(
            bootstrap_servers="localhost:9092",
            topics=["user-events"],
        )
        publisher = KafkaEventPublisher(config)
        
        event = RawEvent(...)
        publisher.publish(event)
        publisher.flush()
    """
    
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
        """创建 Kafka Producer"""
        conf = {
            'bootstrap.servers': self.config.bootstrap_servers,
            'acks': 'all',
            'retries': 3,
            'retry.backoff.ms': 100,
            'linger.ms': 5,
            'batch.size': 16384,
            'compression.type': 'snappy',
        }
        
        if self.config.security_protocol:
            conf['security.protocol'] = self.config.security_protocol
            conf['sasl.mechanism'] = self.config.sasl_mechanism
            conf['sasl.username'] = self.config.sasl_username
            conf['sasl.password'] = self.config.sasl_password
        
        return Producer(conf)
    
    def _delivery_callback(self, err, msg):
        """发送回调"""
        self._pending_count -= 1
        if err:
            logger.error(f"Failed to deliver message: {err}")
        else:
            logger.debug(f"Message delivered to {msg.topic()} [{msg.partition()}]")
    
    def publish(self, event: RawEvent) -> bool:
        """发布单个事件"""
        try:
            topic = self.config.topics[0]
            key = event.user_id.encode('utf-8')
            value = self.serializer.serialize(event)
            
            self._producer.produce(
                topic=topic,
                key=key,
                value=value,
                callback=self._delivery_callback,
            )
            self._pending_count += 1
            
            # 触发回调
            self._producer.poll(0)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish event: {e}")
            return False
    
    def publish_batch(self, events: List[RawEvent]) -> Tuple[int, int]:
        """批量发布事件"""
        success_count = 0
        fail_count = 0
        
        for event in events:
            if self.publish(event):
                success_count += 1
            else:
                fail_count += 1
            
            # 定期 poll 避免队列积压
            if (success_count + fail_count) % 100 == 0:
                self._producer.poll(0)
        
        return success_count, fail_count
    
    def flush(self, timeout: float = 10.0) -> int:
        """刷新缓冲区"""
        remaining = self._producer.flush(timeout)
        if remaining > 0:
            logger.warning(f"{remaining} messages still pending after flush")
        return remaining
    
    def close(self) -> None:
        """关闭生产者"""
        self.flush()
        logger.info("Producer closed")
```

---

## 3. HTTP API 收集器 (api/collector.py)

```python
"""
HTTP API 事件收集器

通过 REST API 接收事件
"""

from typing import List, Optional
from datetime import datetime
import uuid

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, validator

from ..interfaces import RawEvent, EventType
from ..kafka.producer import KafkaEventPublisher
from .config import CollectorConfig

app = FastAPI(title="Event Collector API")


class EventRequest(BaseModel):
    """事件请求模型"""
    event_type: str = Field(..., description="事件类型")
    user_id: str = Field(..., min_length=1, max_length=100)
    item_id: str = Field(..., min_length=1, max_length=100)
    timestamp: Optional[datetime] = Field(default=None)
    context: dict = Field(default_factory=dict)
    properties: dict = Field(default_factory=dict)
    
    @validator('event_type')
    def validate_event_type(cls, v):
        valid_types = [e.value for e in EventType]
        if v not in valid_types:
            raise ValueError(f"Invalid event_type. Must be one of: {valid_types}")
        return v


class BatchEventRequest(BaseModel):
    """批量事件请求"""
    events: List[EventRequest] = Field(..., min_items=1, max_items=1000)


class EventCollectorAPI:
    """事件收集 API"""
    
    def __init__(self, config: CollectorConfig, publisher: KafkaEventPublisher):
        self.config = config
        self.publisher = publisher
        self._setup_routes()
    
    def _setup_routes(self):
        """设置路由"""
        
        @app.post("/v1/events", status_code=202)
        async def collect_event(
            event: EventRequest,
            background_tasks: BackgroundTasks,
        ):
            """
            收集单个事件
            
            异步处理，立即返回
            """
            raw_event = self._to_raw_event(event)
            background_tasks.add_task(self._publish_event, raw_event)
            return {"event_id": raw_event.event_id, "status": "accepted"}
        
        @app.post("/v1/events/batch", status_code=202)
        async def collect_batch(
            batch: BatchEventRequest,
            background_tasks: BackgroundTasks,
        ):
            """
            批量收集事件
            """
            events = [self._to_raw_event(e) for e in batch.events]
            background_tasks.add_task(self._publish_batch, events)
            return {
                "count": len(events),
                "status": "accepted",
            }
        
        @app.get("/health")
        async def health_check():
            return {"status": "healthy"}
    
    def _to_raw_event(self, request: EventRequest) -> RawEvent:
        """转换为 RawEvent"""
        return RawEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType(request.event_type),
            user_id=request.user_id,
            item_id=request.item_id,
            timestamp=request.timestamp or datetime.now(),
            context=request.context,
            properties=request.properties,
        )
    
    async def _publish_event(self, event: RawEvent):
        """发布事件到 Kafka"""
        if not self.publisher.publish(event):
            # TODO: 重试或保存到本地
            pass
    
    async def _publish_batch(self, events: List[RawEvent]):
        """批量发布事件"""
        self.publisher.publish_batch(events)
        self.publisher.flush()
```

---

## 4. 事件验证器 (validators/event_validator.py)

```python
"""
事件验证器

验证事件数据的完整性和有效性
"""

from typing import List, Optional, Callable
from datetime import datetime, timedelta
import re

from pydantic import BaseModel, validator, ValidationError

from ..interfaces import RawEvent, EventType


class EventValidationRule:
    """验证规则"""
    
    def __init__(
        self,
        name: str,
        check: Callable[[RawEvent], bool],
        error_message: str,
    ):
        self.name = name
        self.check = check
        self.error_message = error_message


class EventValidator:
    """
    事件验证器
    
    使用示例:
        validator = EventValidator()
        validator.add_rule(
            "user_id_format",
            lambda e: e.user_id.startswith("user_"),
            "user_id must start with 'user_'"
        )
        
        errors = validator.validate(event)
    """
    
    DEFAULT_RULES = [
        EventValidationRule(
            "user_id_required",
            lambda e: bool(e.user_id and len(e.user_id) > 0),
            "user_id is required",
        ),
        EventValidationRule(
            "item_id_required",
            lambda e: bool(e.item_id and len(e.item_id) > 0),
            "item_id is required",
        ),
        EventValidationRule(
            "timestamp_not_future",
            lambda e: e.timestamp <= datetime.now() + timedelta(minutes=5),
            "timestamp cannot be in the future",
        ),
        EventValidationRule(
            "timestamp_not_too_old",
            lambda e: e.timestamp >= datetime.now() - timedelta(days=30),
            "timestamp is too old (> 30 days)",
        ),
        EventValidationRule(
            "event_type_valid",
            lambda e: e.event_type in EventType,
            "event_type is invalid",
        ),
    ]
    
    def __init__(self, rules: Optional[List[EventValidationRule]] = None):
        self.rules = rules or self.DEFAULT_RULES.copy()
    
    def add_rule(
        self,
        name: str,
        check: Callable[[RawEvent], bool],
        error_message: str,
    ) -> None:
        """添加验证规则"""
        self.rules.append(EventValidationRule(name, check, error_message))
    
    def validate(self, event: RawEvent) -> List[str]:
        """
        验证事件
        
        Returns:
            List[str]: 错误消息列表，空列表表示验证通过
        """
        errors = []
        
        for rule in self.rules:
            try:
                if not rule.check(event):
                    errors.append(f"{rule.name}: {rule.error_message}")
            except Exception as e:
                errors.append(f"{rule.name}: validation error - {e}")
        
        return errors
    
    def is_valid(self, event: RawEvent) -> bool:
        """检查事件是否有效"""
        return len(self.validate(event)) == 0
    
    def validate_batch(
        self,
        events: List[RawEvent]
    ) -> List[tuple[RawEvent, List[str]]]:
        """
        批量验证事件
        
        Returns:
            List[tuple]: (event, errors) 对，只包含有错误的事件
        """
        results = []
        for event in events:
            errors = self.validate(event)
            if errors:
                results.append((event, errors))
        return results
```

---

## 注意事项

1. 使用 Avro/Protobuf 进行高效序列化
2. 实现死信队列处理失败消息
3. 添加 Prometheus 监控指标
4. 支持优雅关闭
5. 实现消息去重

## 输出要求

请输出完整的可运行代码，包含：
1. Kafka 消费者/生产者
2. HTTP API 收集器
3. 事件验证器
4. 完整测试

