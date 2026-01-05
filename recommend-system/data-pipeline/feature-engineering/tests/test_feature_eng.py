import unittest
from ..transformers.numeric import StandardScaler
from ..pipelines.user_pipeline import FeaturePipeline

class TestFeatureEngineering(unittest.TestCase):
    def test_scaler(self):
        scaler = StandardScaler("test", ["age"])
        data = [{"age": 10}, {"age": 20}, {"age": 30}]
        scaler.fit(data)
        
        t = scaler.transform({"age": 20})
        # mean=20, std=8.16
        self.assertAlmostEqual(t["age_scaled"], 0)

    def test_pipeline(self):
        pipeline = FeaturePipeline("user")
        scaler = StandardScaler("age_scaler", ["age"])
        pipeline.add_transformer("scaler", scaler)
        
        # Manually fit for test
        scaler.fit([{"age": 10}, {"age": 30}])
        
        res = pipeline.run({"age": 20})
        self.assertIn("age_scaled", res)

if __name__ == '__main__':
    unittest.main()

