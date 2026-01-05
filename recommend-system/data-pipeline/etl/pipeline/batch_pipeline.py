from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import time
from ...interfaces import ExtractorInterface, TransformerInterface, LoaderInterface, ETLPipelineInterface

logger = logging.getLogger(__name__)

class BatchETLPipeline(ETLPipelineInterface):
    def __init__(
        self,
        extractor: ExtractorInterface,
        loader: LoaderInterface,
        batch_size: int = 10000
    ):
        self.extractor = extractor
        self.loader = loader
        self.batch_size = batch_size
        self._status = "idle"
        self._stats = {}

    def run(
        self,
        source: str,
        target: str,
        transformers: List[TransformerInterface],
    ) -> Dict[str, Any]:
        self._status = "running"
        stats = {"extracted": 0, "loaded": 0, "failed": 0}
        
        try:
            batch = []
            for item in self.extractor.extract(source):
                stats["extracted"] += 1
                
                processed = item
                for t in transformers:
                    processed = t.transform(processed)
                    if processed is None:
                        break
                
                if processed:
                    batch.append(processed)
                
                if len(batch) >= self.batch_size:
                    s, f = self.loader.load_batch(batch)
                    stats["loaded"] += s
                    stats["failed"] += f
                    batch = []
            
            if batch:
                s, f = self.loader.load_batch(batch)
                stats["loaded"] += s
                stats["failed"] += f
                
            self._status = "success"
        except Exception as e:
            self._status = "failed"
            logger.error(f"Pipeline failed: {e}")
            raise e
        finally:
            self._stats = stats
            
        return stats

    def schedule(self, cron_expression: str) -> str:
        return "job_id_123"

    def get_status(self) -> Dict[str, Any]:
        return {"status": self._status, "stats": self._stats}

