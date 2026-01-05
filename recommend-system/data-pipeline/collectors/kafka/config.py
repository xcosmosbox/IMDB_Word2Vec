from dataclasses import dataclass
from typing import List, Optional

@dataclass
class KafkaConfig:
    bootstrap_servers: str
    group_id: str = "recommend-collector"
    topics: List[str] = None
    auto_offset_reset: str = "earliest"
    max_poll_interval_ms: int = 300000
    session_timeout_ms: int = 45000
    security_protocol: Optional[str] = None
    sasl_mechanism: Optional[str] = None
    sasl_username: Optional[str] = None
    sasl_password: Optional[str] = None

@dataclass
class CollectorConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    kafka_config: KafkaConfig = None

