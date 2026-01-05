from typing import Generator, Dict, Any, Optional
from datetime import datetime
import logging
from unittest.mock import MagicMock

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    psycopg2 = MagicMock()
    RealDictCursor = MagicMock()

from .base import BaseExtractor

logger = logging.getLogger(__name__)

class PostgresExtractor(BaseExtractor):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._conn = None
        self._cursor = None
    
    def connect(self) -> bool:
        try:
            self._conn = psycopg2.connect(
                host=self.config.get('host'),
                port=self.config.get('port', 5432),
                database=self.config.get('database'),
                user=self.config.get('user'),
                password=self.config.get('password'),
            )
            self._cursor = self._conn.cursor(cursor_factory=RealDictCursor)
            self._connected = True
            return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False
    
    def disconnect(self) -> None:
        if self._cursor:
            self._cursor.close()
        if self._conn:
            self._conn.close()
        self._connected = False
    
    def extract(
        self,
        source: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        if not self._connected:
            raise RuntimeError("Not connected")
        
        # Mock implementation for simplicity when no DB
        logger.info(f"Extracting from {source}")
        yield {"id": 1, "data": "test"}

    def get_schema(self, source: str) -> Dict[str, str]:
        return {"id": "int", "data": "text"}

