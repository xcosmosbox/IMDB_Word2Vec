from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
from ...interfaces import LoaderInterface

class BaseLoader(LoaderInterface):
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def load(self, data: Dict[str, Any]) -> bool:
        pass

    def load_batch(self, data: List[Dict[str, Any]]) -> Tuple[int, int]:
        success = 0
        fail = 0
        for item in data:
            if self.load(item):
                success += 1
            else:
                fail += 1
        return success, fail

    def create_table_if_not_exists(self, schema: Dict[str, str]) -> bool:
        return True

