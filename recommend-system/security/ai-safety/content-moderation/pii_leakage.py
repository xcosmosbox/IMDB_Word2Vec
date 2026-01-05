import re
from typing import List, Optional

class PIILeakageDetector:
    """
    Detects PII in model output to prevent leakage.
    """
    
    def __init__(self):
        self.patterns = {
            "phone": re.compile(r"1[3-9]\d{9}"),
            "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
            "id_card": re.compile(r"\d{17}[\dXx]"),
        }

    def detect(self, text: str) -> List[str]:
        """
        Returns a list of detected PII types.
        """
        detected = []
        for p_type, pattern in self.patterns.items():
            if pattern.search(text):
                detected.append(p_type)
        return detected

    def sanitize(self, text: str) -> str:
        """
        Redacts PII from text.
        """
        for p_type, pattern in self.patterns.items():
            text = pattern.sub(f"[{p_type.upper()}_REDACTED]", text)
        return text

