from abc import ABC, abstractmethod
from typing import Generator, Dict, Any, Optional
from datetime import datetime
import logging
from ...interfaces import ExtractorInterface

logger = logging.getLogger(__name__)

class BaseExtractor(ExtractorInterface):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._connected = False
    
    @abstractmethod
    def connect(self) -> bool:
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        pass
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

