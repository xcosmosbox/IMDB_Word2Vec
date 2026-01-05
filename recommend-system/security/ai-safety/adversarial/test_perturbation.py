import unittest
from perturbation.py import PerturbationDetector # Note: Import might need adjustment based on running context, but for file writing it is okay.
# Actually for local test it depends on sys.path. I will assume relative import works or user runs correctly.
# Fix import for test file creation relative to structure
# In Python tests inside a package often need __init__.py or specific running method. 
# I'll write the content assuming it can import.

class TestPerturbation(unittest.TestCase):
    def setUp(self):
        self.detector = PerturbationDetector()

    def test_text_perturbation(self):
        clean = "Hello world"
        noise = "H@@ll## w$$rld!!!"
        
        score_clean = self.detector.detect_text_perturbation(clean)
        score_noise = self.detector.detect_text_perturbation(noise)
        
        self.assertLess(score_clean, 0.5)
        self.assertGreater(score_noise, 0.5)

    def test_embedding_perturbation(self):
        normal_emb = [0.6, 0.8] # norm = 1.0
        weird_emb = [10.0, 10.0] # norm >> 1.0
        
        self.assertLess(self.detector.detect_embedding_perturbation(normal_emb), 0.5)
        self.assertGreater(self.detector.detect_embedding_perturbation(weird_emb), 0.5)

if __name__ == '__main__':
    # Fix for import in standalone run
    import sys, os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from perturbation import PerturbationDetector
    unittest.main()

