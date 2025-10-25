"""
Tests of Layout and related classes
"""
import numpy as np
import pytest

from holoviews import (
    AdjointLayout,
    Element,
    GridSpace,
    HoloMap,
    Layout,
    NdLayout,
    Overlay,
)
from holoviews.element import Curve, HLine, Image
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

    def test_add_unicode(self):
        "Test to avoid regression of #3403 where unicode characters don't capitalize"
        layout = Curve([-1,-2,-3]) + Curve([1,2,3]) .relabel('𝜗_1 vs th_2')
        elements = list(layout)
        self.assertEqual(len(elements), 2)

    def test_layout_overlay_ncols_preserved(self):
        assert ((self.view1 + self.view2).cols(1) * self.view3)._max_cols == 1

    def test_layout_rmul_overlay_ncols_preserved(self):
        assert (self.view3 * (self.view1 + self.view2).cols(1))._max_cols == 1

    def test_or_operator(self):
        """Test that | operator creates a Layout like +"""
        layout = self.view1 | self.view2
        self.assertEqual(type(layout), Layout)
        self.assertEqual(len(layout), 2)

    def test_or_operator_equivalence_to_add(self):
        """Test that | and + operators are equivalent"""
        layout_add = self.view1 + self.view2 + self.view3
        layout_or = self.view1 | self.view2 | self.view3
        self.assertEqual(len(layout_add), len(layout_or))
        self.assertEqual(layout_add.shape, layout_or.shape)
        self.assertEqual(layout_add._max_cols, layout_or._max_cols)

    def test_truediv_operator_single_row(self):
        """Test that / operator creates row breaks"""
        layout = self.view1 / self.view2
        self.assertEqual(type(layout), Layout)
        self.assertEqual(len(layout), 2)
        self.assertEqual(layout._max_cols, 1)
        self.assertEqual(layout.shape, (2, 1))

    def test_truediv_operator_multiple_columns(self):
        """Test (p1 | p2 | p3) / p4 creates 2 rows with p4 spanning full width"""
        layout = (self.view1 | self.view2 | self.view3) / Element('data4')
        # p4 should have colspan=3 in the span_info
        self.assertEqual(len(layout), 4)  # 3 items in row 1 + 1 in row 2
        self.assertEqual(layout._max_cols, 3)
        self.assertEqual(layout.shape, (2, 3))
        # Verify that p4 has colspan of 3
        p4_path = [k for k, v in layout.data.items() if v.data == 'data4'][0]
        self.assertIn(p4_path, layout._span_info)
        self.assertEqual(layout._span_info[p4_path], (3, 1))  # colspan=3, rowspan=1

    def test_truediv_operator_two_rows(self):
        """Test (p1 | p2) / (p3 | p4) creates 2x2 grid with no repetition"""
        layout = (self.view1 | self.view2) / (self.view3 | Element('data4'))
        # No repetition needed - both rows have same width
        self.assertEqual(len(layout), 4)
        self.assertEqual(layout._max_cols, 2)
        self.assertEqual(layout.shape, (2, 2))

    def test_truediv_operator_reverse_spanning(self):
        """Test p1 / (p2 | p3 | p4) where first row narrower than second"""
        layout = self.view1 / (self.view2 | self.view3 | Element('data4'))
        # p1 should have colspan=3 in the span_info
        self.assertEqual(len(layout), 4)  # 1 item in row 1 + 3 items in row 2
        self.assertEqual(layout._max_cols, 3)
        self.assertEqual(layout.shape, (2, 3))
        # Verify that p1 has colspan of 3
        p1_path = [k for k, v in layout.data.items() if v.data == self.view1.data][0]
        self.assertIn(p1_path, layout._span_info)
        self.assertEqual(layout._span_info[p1_path], (3, 1))  # colspan=3, rowspan=1

    def test_truediv_operator_chained(self):
        """Test chaining / operators: (p1 | p2) / (p3 | p4 | p5) / (p6 | p7) creates 3 rows"""
        p6 = Element('data6')
        p7 = Element('data7')
        layout = (self.view1 | self.view2) / (self.view3 | Element('data4') | Element('data5')) / (p6 | p7)
        # Should create 3 distinct rows
        self.assertEqual(len(layout), 7)  # 2 + 3 + 2 elements
        # Check that we have 3 unique row indices
        unique_rows = set(layout._row_indices.values())
        self.assertEqual(len(unique_rows), 3)
        # Grid should use LCM(2, 3, 2) = 6 columns
        self.assertEqual(layout._max_cols, 6)
        # Verify row indices are 0, 1, 2
        self.assertEqual(unique_rows, {0, 1, 2})

    def test_layout_or_overlay_ncols_preserved(self):
        """Test that _max_cols is preserved when using | with overlay"""
        assert ((self.view1 | self.view2).cols(1) * self.view3)._max_cols == 1


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
        for el, view in zip(layout, [self.view3, self.view2, self.view1], strict=None):
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

    @pytest.mark.usefixtures("mpl_backend")
    def test_histogram_image_hline_overlay(self):
        image = Image(np.arange(100).reshape(10, 10))
        overlay = image * HLine(y=0)
        element = overlay.hist()

        assert isinstance(element, AdjointLayout)
        assert element.main == overlay

class NdLayoutTest(CompositeTest):

    def test_ndlayout_init(self):
        grid = NdLayout([(0, self.view1), (1, self.view2), (2, self.view3), (3, self.view2)])
        self.assertEqual(grid.shape, (1,4))

    def test_ndlayout_overlay_element(self):
        items = [(0, self.view1), (1, self.view2), (2, self.view3), (3, self.view2)]
        grid = NdLayout(items)
        hline = HLine(0)
        overlaid_grid = grid * hline
        expected = NdLayout([(k, v*hline) for k, v in items])
        self.assertEqual(overlaid_grid, expected)

    def test_ndlayout_overlay_reverse(self):
        items = [(0, self.view1), (1, self.view2), (2, self.view3), (3, self.view2)]
        grid = NdLayout(items)
        hline = HLine(0)
        overlaid_grid = hline * grid
        expected = NdLayout([(k, hline*v) for k, v in items])
        self.assertEqual(overlaid_grid, expected)

    def test_ndlayout_overlay_ndlayout(self):
        items = [(0, self.view1), (1, self.view2), (2, self.view3), (3, self.view2)]
        grid = NdLayout(items)
        items2 = [(0, self.view2), (1, self.view1), (2, self.view2), (3, self.view3)]
        grid2 = NdLayout(items2)

        expected_items = [(0, self.view1*self.view2), (1, self.view2*self.view1),
                          (2, self.view3*self.view2), (3, self.view2*self.view3)]
        expected = NdLayout(expected_items)
        self.assertEqual(grid*grid2, expected)

    def test_ndlayout_overlay_ndlayout_reverse(self):
        items = [(0, self.view1), (1, self.view2), (2, self.view3), (3, self.view2)]
        grid = NdLayout(items)
        items2 = [(0, self.view2), (1, self.view1), (2, self.view2), (3, self.view3)]
        grid2 = NdLayout(items2)

        expected_items = [(0, self.view2*self.view1), (1, self.view1*self.view2),
                          (2, self.view2*self.view3), (3, self.view3*self.view2)]
        expected = NdLayout(expected_items)
        self.assertEqual(grid2*grid, expected)

    def test_ndlayout_overlay_ndlayout_partial(self):
        items = [(0, self.view1), (1, self.view2), (3, self.view2)]
        grid = NdLayout(items, 'X')
        items2 = [(0, self.view2), (2, self.view2), (3, self.view3)]
        grid2 = NdLayout(items2, 'X')

        expected_items = [(0, Overlay([self.view1, self.view2])),
                          (1, Overlay([self.view2])),
                          (2, Overlay([self.view2])),
                          (3, Overlay([self.view2, self.view3]))]
        expected = NdLayout(expected_items, 'X')
        self.assertEqual(grid*grid2, expected)

    def test_ndlayout_overlay_ndlayout_partial_reverse(self):
        items = [(0, self.view1), (1, self.view2), (3, self.view2)]
        grid = NdLayout(items, 'X')
        items2 = [(0, self.view2), (2, self.view2), (3, self.view3)]
        grid2 = NdLayout(items2, 'X')

        expected_items = [(0, Overlay([self.view2, self.view1])),
                          (1, Overlay([self.view2])),
                          (2, Overlay([self.view2])),
                          (3, Overlay([self.view3, self.view2]))]
        expected = NdLayout(expected_items, 'X')
        self.assertEqual(grid2*grid, expected)


class GridTest(CompositeTest):

    def test_grid_init(self):
        vals = [self.view1, self.view2, self.view3, self.view2]
        keys = [(0,0), (0,1), (1,0), (1,1)]
        grid = GridSpace(zip(keys, vals, strict=None))
        self.assertEqual(grid.shape, (2,2))

    def test_grid_index_snap(self):
        vals = [self.view1, self.view2, self.view3, self.view2]
        keys = [(0,0), (0,1), (1,0), (1,1)]
        grid = GridSpace(zip(keys, vals, strict=None))
        self.assertEqual(grid[0.1, 0.1], self.view1)

    def test_grid_index_strings(self):
        vals = [self.view1, self.view2, self.view3, self.view2]
        keys = [('A', 0), ('B', 1), ('C', 0), ('D', 1)]
        grid = GridSpace(zip(keys, vals, strict=None))
        self.assertEqual(grid['B', 1], self.view2)

    def test_grid_index_one_axis(self):
        vals = [self.view1, self.view2, self.view3, self.view2]
        keys = [('A', 0), ('B', 1), ('C', 0), ('D', 1)]
        grid = GridSpace(zip(keys, vals, strict=None))
        self.assertEqual(grid[:, 0], GridSpace([(('A', 0), self.view1), (('C', 0), self.view3)]))

    def test_gridspace_overlay_element(self):
        items = [(0, self.view1), (1, self.view2), (2, self.view3), (3, self.view2)]
        grid = GridSpace(items, 'X')
        hline = HLine(0)
        overlaid_grid = grid * hline
        expected = GridSpace([(k, v*hline) for k, v in items], 'X')
        self.assertEqual(overlaid_grid, expected)

    def test_gridspace_overlay_reverse(self):
        items = [(0, self.view1), (1, self.view2), (2, self.view3), (3, self.view2)]
        grid = GridSpace(items, 'X')
        hline = HLine(0)
        overlaid_grid = hline * grid
        expected = GridSpace([(k, hline*v) for k, v in items], 'X')
        self.assertEqual(overlaid_grid, expected)

    def test_gridspace_overlay_gridspace(self):
        items = [(0, self.view1), (1, self.view2), (2, self.view3), (3, self.view2)]
        grid = GridSpace(items, 'X')
        items2 = [(0, self.view2), (1, self.view1), (2, self.view2), (3, self.view3)]
        grid2 = GridSpace(items2, 'X')

        expected_items = [(0, self.view1*self.view2), (1, self.view2*self.view1),
                          (2, self.view3*self.view2), (3, self.view2*self.view3)]
        expected = GridSpace(expected_items, 'X')
        self.assertEqual(grid*grid2, expected)

    def test_gridspace_overlay_gridspace_reverse(self):
        items = [(0, self.view1), (1, self.view2), (2, self.view3), (3, self.view2)]
        grid = GridSpace(items, 'X')
        items2 = [(0, self.view2), (1, self.view1), (2, self.view2), (3, self.view3)]
        grid2 = GridSpace(items2, 'X')

        expected_items = [(0, self.view2*self.view1), (1, self.view1*self.view2),
                          (2, self.view2*self.view3), (3, self.view3*self.view2)]
        expected = GridSpace(expected_items, 'X')
        self.assertEqual(grid2*grid, expected)

    def test_gridspace_overlay_gridspace_partial(self):
        items = [(0, self.view1), (1, self.view2), (3, self.view2)]
        grid = GridSpace(items, 'X')
        items2 = [(0, self.view2), (2, self.view2), (3, self.view3)]
        grid2 = GridSpace(items2, 'X')

        expected_items = [(0, Overlay([self.view1, self.view2])),
                          (1, Overlay([self.view2])),
                          (2, Overlay([self.view2])),
                          (3, Overlay([self.view2, self.view3]))]
        expected = GridSpace(expected_items, 'X')
        self.assertEqual(grid*grid2, expected)

    def test_gridspace_overlay_gridspace_partial_reverse(self):
        items = [(0, self.view1), (1, self.view2), (3, self.view2)]
        grid = GridSpace(items, 'X')
        items2 = [(0, self.view2), (2, self.view2), (3, self.view3)]
        grid2 = GridSpace(items2, 'X')

        expected_items = [(0, Overlay([self.view2, self.view1])),
                          (1, Overlay([self.view2])),
                          (2, Overlay([self.view2])),
                          (3, Overlay([self.view3, self.view2]))]
        expected = GridSpace(expected_items, 'X')
        self.assertEqual(grid2*grid, expected)
