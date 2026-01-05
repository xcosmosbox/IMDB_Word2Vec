import unittest
from ...interfaces import DataQualityLevel
from ..validators.rule_validator import RuleBasedValidator

class TestDataQuality(unittest.TestCase):
    def test_validator(self):
        v = RuleBasedValidator()
        v.add_rule("not_empty", lambda d: bool(d), DataQualityLevel.CRITICAL)
        
        report = v.validate({})
        self.assertFalse(report.passed)
        self.assertEqual(report.level, DataQualityLevel.CRITICAL)

        report = v.validate({"a": 1})
        self.assertTrue(report.passed)

if __name__ == '__main__':
    unittest.main()

