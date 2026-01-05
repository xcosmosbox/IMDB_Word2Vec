import unittest
from ..online.redis_store import RedisFeatureStore

class TestFeatureStore(unittest.TestCase):
    def test_online_store(self):
        store = RedisFeatureStore({})
        self.assertTrue(store.set_features("user", "u1", {"age": 25}))
        f = store.get_features("user", "u1", ["age"])
        self.assertIn("age", f)

if __name__ == '__main__':
    unittest.main()

