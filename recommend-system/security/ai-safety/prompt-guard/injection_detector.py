import re
from typing import List, Tuple

class InjectionDetector:
    """
    Based on rules and heuristics to detect Prompt Injection.
    """
    
    PATTERNS = [
        r"ignore previous instructions",
        r"system prompt",
        r"you are now",
        r"roleplay",
        r"override",
        r"jailbreak",
    ]
    
    def __init__(self):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.PATTERNS]
    
    def detect(self, text: str) -> Tuple[bool, float, str]:
        """
        Detects injection attempts.
        Returns: (is_injection, confidence, reason)
        """
        # 1. Rule matching
        for pattern in self.patterns:
            if pattern.search(text):
                return True, 1.0, f"Matched pattern: {pattern.pattern}"
        
        # 2. Heuristics (length + complex structure)
        if len(text) > 1000 and "{" in text and "}" in text:
             return True, 0.6, "Complex structure suspicious"
             
        return False, 0.0, ""

