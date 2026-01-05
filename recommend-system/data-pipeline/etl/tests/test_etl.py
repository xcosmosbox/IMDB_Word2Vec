import unittest
from typing import Dict, Any
from ..extractors.base import BaseExtractor
from ..transformers.cleaning import DataCleaningTransformer
from ..loaders.base import BaseLoader
from ..pipeline.batch_pipeline import BatchETLPipeline

class MockExtractor(BaseExtractor):
    def connect(self): return True
    def disconnect(self): pass
    def extract(self, source, start_time=None, end_time=None):
        yield {"name": " test "}
    def get_schema(self, source): return {}

class MockLoader(BaseLoader):
    def load(self, data): return True

class TestETL(unittest.TestCase):
    def test_pipeline(self):
        extractor = MockExtractor({})
        loader = MockLoader({})
        transformer = DataCleaningTransformer({})
        pipeline = BatchETLPipeline(extractor, loader)
        
        stats = pipeline.run("source", "target", [transformer])
        self.assertEqual(stats["extracted"], 1)
        self.assertEqual(stats["loaded"], 1)

if __name__ == '__main__':
    unittest.main()

