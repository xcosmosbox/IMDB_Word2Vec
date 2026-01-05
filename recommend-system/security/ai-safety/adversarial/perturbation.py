import numpy as np

class PerturbationDetector:
    """
    Detects adversarial perturbations in input embeddings or text.
    Mock implementation assuming we can analyze character distribution entropy.
    """
    
    def detect_text_perturbation(self, text: str) -> float:
        """
        Simple heuristic: check for too many non-printable or rare characters mixed in.
        Returns anomaly score (0-1).
        """
        if not text:
            return 0.0
            
        weird_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        ratio = weird_chars / len(text)
        
        if ratio > 0.3: # High noise
            return 0.8
        return 0.1

    def detect_embedding_perturbation(self, embedding: list) -> float:
        """
        Check if embedding norm or variance is suspicious.
        """
        arr = np.array(embedding)
        norm = np.linalg.norm(arr)
        
        # Mock logic: assuming normalized embeddings should be close to 1
        if abs(norm - 1.0) > 0.5:
            return 0.9
        return 0.1

