from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from ...interfaces import TransformerInterface

class BaseTransformer(TransformerInterface):
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def transform_batch(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.transform(d) for d in data]

    def get_output_schema(self) -> Dict[str, str]:
        return {}

