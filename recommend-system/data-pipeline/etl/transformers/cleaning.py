from typing import Dict, Any
from .base import BaseTransformer

class DataCleaningTransformer(BaseTransformer):
    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        result = data.copy()
        for k, v in result.items():
            if isinstance(v, str):
                result[k] = v.strip()
        return result

