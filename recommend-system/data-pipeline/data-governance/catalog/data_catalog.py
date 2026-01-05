from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
import uuid
from ...interfaces import DataCatalogInterface

@dataclass
class Dataset:
    id: str
    name: str
    schema: Dict[str, str]
    metadata: Dict[str, Any]
    tags: List[str] = field(default_factory=list)

class DataCatalog(DataCatalogInterface):
    def __init__(self):
        self._datasets = {}

    def register_dataset(self, name: str, schema: Dict[str, str], metadata: Dict[str, Any]) -> str:
        did = str(uuid.uuid4())
        ds = Dataset(did, name, schema, metadata)
        self._datasets[did] = ds
        return did

    def get_dataset(self, name: str) -> Dict[str, Any]:
        for ds in self._datasets.values():
            if ds.name == name:
                return {"id": ds.id, "name": ds.name, "schema": ds.schema, "metadata": ds.metadata}
        return {}

    def search_datasets(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        results = []
        for ds in self._datasets.values():
            if query in ds.name:
                results.append({"id": ds.id, "name": ds.name})
        return results

    def add_tags(self, dataset_name: str, tags: List[str]) -> None:
        for ds in self._datasets.values():
            if ds.name == dataset_name:
                ds.tags.extend(tags)
    
    def update_metadata(self, dataset_name: str, metadata: Dict[str, Any]) -> None:
        pass

