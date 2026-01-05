import unittest
from unittest.mock import MagicMock
from delete_user_data import GDPRDeletionService

class TestGDPRDeletion(unittest.TestCase):
    def setUp(self):
        self.service = GDPRDeletionService()
        # Mock dependencies
        self.service.db = MagicMock()
        self.service.redis = MagicMock()
        self.service.vector_db = MagicMock()
        self.service.s3 = MagicMock()

    def test_successful_deletion(self):
        result = self.service.execute_deletion("u123")
        self.assertTrue(result)
        
        self.service.db.delete_user.assert_called_with("u123")
        self.service.redis.delete_cache.assert_called_with("u123")
        self.service.vector_db.delete_embeddings.assert_called_with("u123")
        self.service.s3.delete_files.assert_called_with("u123")

    def test_failure_handling(self):
        self.service.db.delete_user.side_effect = Exception("DB Connection Failed")
        result = self.service.execute_deletion("u123")
        self.assertFalse(result)

if __name__ == '__main__':
    unittest.main()

