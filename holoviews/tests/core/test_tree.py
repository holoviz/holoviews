import pytest

from holoviews.core.element import AttrTree
from holoviews.element.comparison import ComparisonTestCase


class AttrTreeTest(ComparisonTestCase):
    "For testing of AttrTree"

    def setUp(self):
        self.tree = AttrTree([(('A', 'I'), 1), (('B', 'II'), 2)])

    def test_access_nodes(self):
        assert self.tree.A.I == 1
        assert self.tree.B.II == 2

    def test_uppercase_attribute_create_node(self):
        assert isinstance(self.tree.C, AttrTree)

    def test_uppercase_setattr(self):
        self.tree.C = 3
        assert self.tree.C == 3

    def test_deep_setattr(self):
        self.tree.C.I = 3
        assert self.tree.C.I == 3

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
        assert self.tree['A']['I'] == 1
        assert self.tree['B']['II'] == 2

    def test_uppercase_setitem(self):
        self.tree['C'] = 1
        assert self.tree.C == 1

    def test_deep_getitem(self):
        assert self.tree[('A', 'I')] == 1
        assert self.tree[('B', 'II')] == 2

    def test_deep_getitem_str(self):
        assert self.tree['A.I'] == 1
        assert self.tree['B.II'] == 2

    def test_deep_setitem(self):
        self.tree[('C', 'I')] = 3
        assert self.tree.C.I == 3

    def test_deep_setitem_str(self):
        self.tree['C.I'] = 3
        assert self.tree.C.I == 3

    def test_delitem(self):
        Btree = self.tree.B
        del self.tree['B']
        assert self.tree.B is not Btree
        assert ('B', 'II') not in self.tree.data

    def test_delitem_on_node(self):
        del self.tree.B['II']
        self.assertNotEqual(self.tree.B.II, 2)
        assert ('B', 'II') not in self.tree.data
        assert ('II',) not in self.tree.B

    def test_delitem_keyerror(self):
        with self.assertRaises(KeyError):
            del self.tree['C']

    def test_deep_delitem(self):
        BTree = self.tree.B
        del self.tree[('B', 'II')]
        assert isinstance(self.tree.B.II, AttrTree)
        assert self.tree.B is BTree
        assert ('B', 'II') not in self.tree.data

    def test_deep_delitem_str(self):
        BTree = self.tree.B
        del self.tree['B.II']
        assert isinstance(self.tree.B.II, AttrTree)
        assert self.tree.B is BTree
        assert ('B', 'II') not in self.tree.data
