from holoviews import AdjointLayout, NdLayout, GridSpace, Layout, Element, HoloMap
from holoviews.element.comparison import ComparisonTestCase


class CompositeTest(ComparisonTestCase):
    "For testing of basic composite element types"

    def setUp(self):
        self.data1 ='An example of arbitrary data'
        self.data2 = 'Another example...'
        self.data3 = 'A third example.'

        self.view1 = Element(self.data1, label='View1')
        self.view2 = Element(self.data2, label='View2')
        self.view3 = Element(self.data3, label='View3')
        self.hmap = HoloMap({0: self.view1, 1: self.view2, 2: self.view3})

    def test_add_operator(self):
        self.assertEqual(type(self.view1 + self.view2), Layout)


class AdjointLayoutTest(CompositeTest):

    def test_adjointlayout_single(self):
        layout = AdjointLayout([self.view1])
        self.assertEqual(layout.main, self.view1)

    def test_adjointlayout_double(self):
        layout = self.view1 << self.view2
        self.assertEqual(layout.main, self.view1)
        self.assertEqual(layout.right, self.view2)

    def test_adjointlayout_triple(self):
        layout = self.view3 << self.view2 << self.view1
        self.assertEqual(layout.main, self.view3)
        self.assertEqual(layout.right, self.view2)
        self.assertEqual(layout.top, self.view1)

    def test_adjointlayout_iter(self):
        layout = self.view3 << self.view2 << self.view1
        for el, view in zip(layout, [self.view3, self.view2, self.view1]):
            self.assertEqual(el, view)

    def test_adjointlayout_add_operator(self):
        layout1 = self.view3 << self.view2
        layout2 = self.view2 << self.view1
        self.assertEqual(type(layout1 + layout2), Layout)

    def test_adjointlayout_overlay_main(self):
        layout = (self.view1 << self.view2) * self.view3
        self.assertEqual(layout.main, self.view1 * self.view3)
        self.assertEqual(layout.right, self.view2)

    def test_adjointlayout_overlay_main_reverse(self):
        layout = self.view3 * (self.view1 << self.view2)
        self.assertEqual(layout.main, self.view3 * self.view1)
        self.assertEqual(layout.right, self.view2)

    def test_adjointlayout_overlay_main_and_right_v1(self):
        layout = (self.view1 << self.view2) * (self.view1 << self.view3)
        self.assertEqual(layout.main, self.view1 * self.view1)
        self.assertEqual(layout.right, self.view2 * self.view3)

    def test_adjointlayout_overlay_main_and_right_v2(self):
        layout = (self.view1 << self.view3) * (self.view1 << self.view2)
        self.assertEqual(layout.main, self.view1 * self.view1)
        self.assertEqual(layout.right, self.view3 * self.view2)

    def test_adjointlayout_overlay_holomap(self):
        layout = self.hmap * (self.view1 << self.view3)
        self.assertEqual(layout.main, self.hmap * self.view1)
        self.assertEqual(layout.right, self.view3)

    def test_adjointlayout_overlay_holomap_reverse(self):
        layout = (self.view1 << self.view3) * self.hmap
        self.assertEqual(layout.main, self.view1 * self.hmap)
        self.assertEqual(layout.right, self.view3)

    def test_adjointlayout_overlay_adjoined_holomap(self):
        layout = (self.view1 << self.view3) * (self.hmap << self.view3)
        self.assertEqual(layout.main, self.view1 * self.hmap)
        self.assertEqual(layout.right, self.view3 * self.view3)

    def test_adjointlayout_overlay_adjoined_holomap_reverse(self):
        layout = (self.hmap << self.view3) * (self.view1 << self.view3)
        self.assertEqual(layout.main, self.hmap * self.view1)
        self.assertEqual(layout.right, self.view3 * self.view3)

    def test_adjointlayout_overlay_adjoined_holomap_nomatch(self):
        dim_view = self.view3.clone(kdims=['x', 'y'])
        layout = (self.view1 << self.view3) * (self.hmap << dim_view)
        self.assertEqual(layout.main, self.view1 * self.hmap)
        self.assertEqual(layout.right, self.view3)
        self.assertEqual(layout.top, dim_view)

    def test_adjointlayout_overlay_adjoined_holomap_nomatch_reverse(self):
        dim_view = self.view3.clone(kdims=['x', 'y'])
        layout = (self.hmap << dim_view) * (self.view1 << self.view3)
        self.assertEqual(layout.main, self.hmap * self.view1)
        self.assertEqual(layout.right, dim_view)
        self.assertEqual(layout.top, self.view3)

    def test_adjointlayout_overlay_adjoined_holomap_nomatch_too_many(self):
        dim_view = self.view3.clone(kdims=['x', 'y'])
        with self.assertRaises(ValueError):
            (self.view1 << self.view2 << self.view3) * (self.hmap << dim_view)



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
