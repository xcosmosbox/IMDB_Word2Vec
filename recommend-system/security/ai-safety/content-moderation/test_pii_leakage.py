import unittest
from pii_leakage import PIILeakageDetector

class TestPIILeakage(unittest.TestCase):
    def setUp(self):
        self.detector = PIILeakageDetector()

    def test_detect_phone(self):
        text = "My phone is 13800138000."
        types = self.detector.detect(text)
        self.assertIn("phone", types)
        
        sanitized = self.detector.sanitize(text)
        self.assertIn("[PHONE_REDACTED]", sanitized)
        self.assertNotIn("13800138000", sanitized)

    def test_detect_email(self):
        text = "Contact admin@example.com for help."
        types = self.detector.detect(text)
        self.assertIn("email", types)

    def test_clean_text(self):
        text = "Hello world."
        types = self.detector.detect(text)
        self.assertEqual(len(types), 0)

if __name__ == '__main__':
    unittest.main()

