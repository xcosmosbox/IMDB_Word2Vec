import random

class ToxicityClassifier:
    """
    Mock implementation of a Toxicity Classifier.
    In production, this would call an external API (OpenAI/Google) or a local BERT model.
    """
    
    BAD_WORDS = ["hate", "kill", "idiot", "violence"]
    
    def predict(self, text: str) -> float:
        """
        Returns toxicity score between 0.0 and 1.0
        """
        text_lower = text.lower()
        for word in self.BAD_WORDS:
            if word in text_lower:
                return 0.9 + (random.random() * 0.1) # > 0.9
        
        return 0.05 # Low score
    
    def is_safe(self, text: str, threshold: float = 0.7) -> bool:
        return self.predict(text) < threshold

