from typing import Dict, Any, List
from ...interfaces import FeaturePipelineInterface, FeatureTransformerInterface
from ..transformers.numeric import StandardScaler

class FeaturePipeline(FeaturePipelineInterface):
    def __init__(self, name: str):
        self.name = name
        self._transformers = {}
        self._order = []

    def add_transformer(self, name: str, transformer: FeatureTransformerInterface) -> None:
        self._transformers[name] = transformer
        self._order.append(name)

    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        result = data.copy()
        for name in self._order:
            result = self._transformers[name].transform(result)
        return result

    def run_batch(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.run(d) for d in data]

    def get_feature_names(self) -> List[str]:
        names = []
        for name in self._order:
            if hasattr(self._transformers[name], 'output_features'):
                names.extend(self._transformers[name].output_features)
        return names

