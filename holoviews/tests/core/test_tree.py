import pytest

from holoviews.core.element import AttrTree
from holoviews.element.comparison import ComparisonTestCase


class AttrTreeTest(ComparisonTestCase):
    "For testing of AttrTree"

    def setUp(self):
        self.tree = AttrTree([(('A', 'I'), 1), (('B', 'II'), 2)])

    def test_access_nodes(self):
        self.assertEqual(self.tree.A.I, 1)
        self.assertEqual(self.tree.B.II, 2)

    def test_uppercase_attribute_create_node(self):
        self.assertIsInstance(self.tree.C, AttrTree)

    def test_uppercase_setattr(self):
        self.tree.C = 3
        self.assertEqual(self.tree.C, 3)

    def test_deep_setattr(self):
        self.tree.C.I = 3
        self.assertEqual(self.tree.C.I, 3)

    def test_lowercase_attribute_error(self):
        msg = r"'AttrTree' object has no attribute c\."
        with pytest.raises(AttributeError, match=msg):
            self.tree.c  # noqa: B018

    def test_number_getitem_key_error(self):
        with self.assertRaises(KeyError):
            self.tree['2']

    def test_lowercase_getitem_key_error(self):
        with self.assertRaises(KeyError):
            self.tree['c']

    def test_uppercase_getitem(self):
        self.assertEqual(self.tree['A']['I'], 1)
        self.assertEqual(self.tree['B']['II'], 2)

    def test_uppercase_setitem(self):
        self.tree['C'] = 1
        self.assertEqual(self.tree.C, 1)

    def test_deep_getitem(self):
        self.assertEqual(self.tree[('A', 'I')], 1)
        self.assertEqual(self.tree[('B', 'II')], 2)

    def test_deep_getitem_str(self):
        self.assertEqual(self.tree['A.I'], 1)
        self.assertEqual(self.tree['B.II'], 2)

    def test_deep_setitem(self):
        self.tree[('C', 'I')] = 3
        self.assertEqual(self.tree.C.I, 3)

    def test_deep_setitem_str(self):
        self.tree['C.I'] = 3
        self.assertEqual(self.tree.C.I, 3)

    def test_delitem(self):
        Btree = self.tree.B
        del self.tree['B']
        self.assertIsNot(self.tree.B, Btree)
        self.assertNotIn(('B', 'II'), self.tree.data)

    def test_delitem_on_node(self):
        del self.tree.B['II']
        self.assertNotEqual(self.tree.B.II, 2)
        self.assertNotIn(('B', 'II'), self.tree.data)
        self.assertNotIn(('II',), self.tree.B)

    def test_delitem_keyerror(self):
        with self.assertRaises(KeyError):
            del self.tree['C']

    def test_deep_delitem(self):
        BTree = self.tree.B
        del self.tree[('B', 'II')]
        self.assertIsInstance(self.tree.B.II, AttrTree)
        self.assertIs(self.tree.B, BTree)
        self.assertNotIn(('B', 'II'), self.tree.data)

    def test_deep_delitem_str(self):
        BTree = self.tree.B
        del self.tree['B.II']
        self.assertIsInstance(self.tree.B.II, AttrTree)
        self.assertIs(self.tree.B, BTree)
        self.assertNotIn(('B', 'II'), self.tree.data)
