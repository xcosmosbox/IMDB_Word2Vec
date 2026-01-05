from typing import Dict, Any, List
import numpy as np
from .base import BaseFeatureTransformer

class StandardScaler(BaseFeatureTransformer):
    def __init__(self, name: str, features: List[str]):
        super().__init__(name, {'features': features})
        self._input_features = features
        self._output_features = [f"{f}_scaled" for f in features]
        self._mean = {}
        self._std = {}

    def fit(self, data: List[Dict[str, Any]]) -> None:
        for feature in self._input_features:
            values = [d.get(feature, 0) for d in data if d.get(feature) is not None]
            if values:
                self._mean[feature] = np.mean(values)
                self._std[feature] = np.std(values) or 1.0
        self._fitted = True

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        result = data.copy()
        for i, feature in enumerate(self._input_features):
            val = data.get(feature)
            if val is not None:
                mean = self._mean.get(feature, 0)
                std = self._std.get(feature, 1)
                result[self._output_features[i]] = (val - mean) / std
        return result

    def _get_state(self) -> Dict[str, Any]:
        return {'mean': self._mean, 'std': self._std}

    def _set_state(self, state: Dict[str, Any]) -> None:
        self._mean = state['mean']
        self._std = state['std']

