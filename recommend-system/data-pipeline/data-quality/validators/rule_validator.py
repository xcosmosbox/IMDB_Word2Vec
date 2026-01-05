from typing import Dict, Any, List, Callable, Optional
from datetime import datetime
from dataclasses import dataclass, field
import logging
import re
from ...interfaces import DataValidatorInterface, DataQualityReport, DataQualityLevel

logger = logging.getLogger(__name__)

@dataclass
class ValidationRule:
    name: str
    check: Callable[[Dict[str, Any]], bool]
    level: DataQualityLevel
    description: str = ""

class RuleBasedValidator(DataValidatorInterface):
    def __init__(self):
        self._rules: List[ValidationRule] = []

    def add_rule(self, rule_name: str, rule_func: Callable, level: DataQualityLevel, description: str = "") -> None:
        self._rules.append(ValidationRule(rule_name, rule_func, level, description))

    def validate(self, data: Dict[str, Any]) -> DataQualityReport:
        failed = []
        highest = DataQualityLevel.LOW
        
        for rule in self._rules:
            if not rule.check(data):
                failed.append(rule.name)
                # Simply set critical if failed for now
                if rule.level == DataQualityLevel.CRITICAL:
                    highest = DataQualityLevel.CRITICAL
        
        return DataQualityReport(
            check_name="rule_validation",
            level=highest if failed else DataQualityLevel.LOW,
            passed=len(failed) == 0,
            message=f"Failed: {failed}" if failed else "Passed",
            timestamp=datetime.now()
        )

    def validate_batch(self, data: List[Dict[str, Any]]) -> List[DataQualityReport]:
        return [self.validate(d) for d in data]
    
    def get_rules(self) -> List[str]:
        return [r.name for r in self._rules]

