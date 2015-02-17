import unittest
from holoviews import AdjointLayout, NdLayout, GridSpace, Layout, Element


class CompositeTest(unittest.TestCase):
    "For testing of basic composite element types"

    def setUp(self):
        self.data1 ='An example of arbitrary data'
        self.data2 = 'Another example...'
        self.data3 = 'A third example.'

        self.view1 = Element(self.data1, label='View1')
        self.view2 = Element(self.data2, label='View2')
        self.view3 = Element(self.data3, label='View3')

    def test_add_operator(self):
        self.assertEqual(type(self.view1 + self.view2), Layout)


class LayoutTest(CompositeTest):

    def test_layout_single(self):
        AdjointLayout([self.view1])

    def test_layout_double(self):
        layout = self.view1 << self.view2
        self.assertEqual(layout.main.data, self.data1)
        self.assertEqual(layout.right.data, self.data2)

    def test_layout_triple(self):
        layout = self.view3 << self.view2 << self.view1
        self.assertEqual(layout.main.data, self.data3)
        self.assertEqual(layout.right.data, self.data2)
        self.assertEqual(layout.top.data, self.data1)

    def test_layout_iter(self):
        layout = self.view3 << self.view2 << self.view1
        for el, data in zip(layout, [self.data3, self.data2, self.data1]):
            self.assertEqual(el.data, data)

    def test_layout_add_operator(self):
        layout1 = self.view3 << self.view2
        layout2 = self.view2 << self.view1
        self.assertEqual(type(layout1 + layout2), Layout)


class NdLayoutTest(CompositeTest):

    def test_ndlayout_init(self):
        grid = NdLayout([(0, self.view1), (1, self.view2), (2, self.view3), (3, self.view2)])
        self.assertEqual(grid.shape, (1,4))


class GridTest(CompositeTest):

    def test_grid_init(self):
        vals = [self.view1, self.view2, self.view3, self.view2]
        keys = [(0,0), (0,1), (1,0), (1,1)]
        grid = GridSpace(zip(keys, vals))
        self.assertEqual(grid.shape, (2,2))
