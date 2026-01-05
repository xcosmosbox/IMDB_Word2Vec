import unittest
from ..catalog.data_catalog import DataCatalog
from ..access.access_control import DataAccessControl

class TestGovernance(unittest.TestCase):
    def test_catalog(self):
        catalog = DataCatalog()
        catalog.register_dataset("ds1", {}, {})
        res = catalog.search_datasets("ds1")
        self.assertEqual(len(res), 1)

    def test_access(self):
        ac = DataAccessControl()
        ac.grant_access("u1", "ds1", ["read"])
        self.assertTrue(ac.check_access("u1", "ds1", "read"))
        self.assertFalse(ac.check_access("u1", "ds1", "write"))

if __name__ == '__main__':
    unittest.main()

