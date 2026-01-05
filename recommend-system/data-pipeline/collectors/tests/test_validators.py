import unittest
from datetime import datetime
from ...interfaces import RawEvent, EventType
from ..validators.event_validator import EventValidator

class TestEventValidator(unittest.TestCase):
    def setUp(self):
        self.validator = EventValidator()

    def test_valid_event(self):
        event = RawEvent(
            event_id="1",
            event_type=EventType.VIEW,
            user_id="u1",
            item_id="i1",
            timestamp=datetime.now()
        )
        self.assertTrue(self.validator.is_valid(event))

    def test_missing_user_id(self):
        event = RawEvent(
            event_id="1",
            event_type=EventType.VIEW,
            user_id="",
            item_id="i1",
            timestamp=datetime.now()
        )
        self.assertFalse(self.validator.is_valid(event))
        errors = self.validator.validate(event)
        self.assertTrue(any("user_id" in e for e in errors))

if __name__ == '__main__':
    unittest.main()

