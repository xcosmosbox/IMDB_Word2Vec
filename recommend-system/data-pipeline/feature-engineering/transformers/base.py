from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import pickle
import logging
from pathlib import Path
from ...interfaces import FeatureTransformerInterface

logger = logging.getLogger(__name__)

class BaseFeatureTransformer(FeatureTransformerInterface):
    def __init__(self, name: str, config: Optional[Dict] = None):
        self.name = name
        self.config = config or {}
        self._fitted = False
        self._input_features: List[str] = []
        self._output_features: List[str] = []
    
    @abstractmethod
    def fit(self, data: List[Dict[str, Any]]) -> None:
        pass
    
    @abstractmethod
    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    def fit_transform(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.fit(data)
        return [self.transform(d) for d in data]
    
    def save(self, path: str) -> None:
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            'name': self.name,
            'config': self.config,
            'fitted': self._fitted,
            'state': self._get_state(),
        }
        with open(save_path, 'wb') as f:
            pickle.dump(state, f)

    def load(self, path: str) -> None:
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.name = state['name']
        self.config = state['config']
        self._fitted = state['fitted']
        self._set_state(state['state'])

    @abstractmethod
    def _get_state(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _set_state(self, state: Dict[str, Any]) -> None:
        pass
    
    @property
    def output_features(self) -> List[str]:
        return self._output_features

