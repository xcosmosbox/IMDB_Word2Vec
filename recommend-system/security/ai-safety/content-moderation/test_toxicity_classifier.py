import unittest
from toxicity_classifier import ToxicityClassifier

class TestToxicityClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = ToxicityClassifier()

    def test_safe_content(self):
        self.assertTrue(self.classifier.is_safe("I love rainbows."))

    def test_toxic_content(self):
        self.assertFalse(self.classifier.is_safe("I hate you, you idiot."))

if __name__ == '__main__':
    unittest.main()

