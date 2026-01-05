"""
事件验证器
"""
from typing import List, Optional, Callable
from datetime import datetime, timedelta
from ...interfaces import RawEvent, EventType

class EventValidationRule:
    def __init__(self, name: str, check: Callable[[RawEvent], bool], error_message: str):
        self.name = name
        self.check = check
        self.error_message = error_message

class EventValidator:
    DEFAULT_RULES = [
        EventValidationRule("user_id_required", lambda e: bool(e.user_id), "user_id is required"),
        EventValidationRule("item_id_required", lambda e: bool(e.item_id), "item_id is required"),
    ]
    
    def __init__(self, rules: Optional[List[EventValidationRule]] = None):
        self.rules = rules or self.DEFAULT_RULES.copy()
    
    def add_rule(self, name: str, check: Callable[[RawEvent], bool], error_message: str) -> None:
        self.rules.append(EventValidationRule(name, check, error_message))
    
    def validate(self, event: RawEvent) -> List[str]:
        errors = []
        for rule in self.rules:
            try:
                if not rule.check(event):
                    errors.append(f"{rule.name}: {rule.error_message}")
            except Exception as e:
                errors.append(f"{rule.name}: validation error - {e}")
        return errors
    
    def is_valid(self, event: RawEvent) -> bool:
        return len(self.validate(event)) == 0

