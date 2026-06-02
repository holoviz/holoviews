"""
Tests of Layout and related classes
"""

from __future__ import annotations

import numpy as np
import pytest

import holoviews as hv
from holoviews.testing import assert_element_equal


class CompositeTest:
    "For testing of basic composite element types"

    def setup_method(self):
        self.data1 = "An example of arbitrary data"
        self.data2 = "Another example..."
        self.data3 = "A third example."

        self.view1 = hv.Element(self.data1, label="View1")
        self.view2 = hv.Element(self.data2, label="View2")
        self.view3 = hv.Element(self.data3, label="View3")
        self.hmap = hv.HoloMap({0: self.view1, 1: self.view2, 2: self.view3})

    def test_add_operator(self):
        assert type(self.view1 + self.view2) is hv.Layout

    def test_add_unicode(self):
        "Test to avoid regression of #3403 where unicode characters don't capitalize"
        layout = hv.Curve([-1, -2, -3]) + hv.Curve([1, 2, 3]).relabel("𝜗_1 vs th_2")
        elements = list(layout)
        assert len(elements) == 2

    def test_layout_overlay_ncols_preserved(self):
        assert ((self.view1 + self.view2).cols(1) * self.view3)._max_cols == 1

    def test_layout_rmul_overlay_ncols_preserved(self):
        assert (self.view3 * (self.view1 + self.view2).cols(1))._max_cols == 1


class AdjointLayoutTest(CompositeTest):
    def test_adjointlayout_single(self):
        layout = hv.AdjointLayout([self.view1])
        assert_element_equal(layout.main, self.view1)

    def test_adjointlayout_double(self):
        layout = self.view1 << self.view2
        assert_element_equal(layout.main, self.view1)
        assert_element_equal(layout.right, self.view2)

    def test_adjointlayout_triple(self):
        layout = self.view3 << self.view2 << self.view1
        assert_element_equal(layout.main, self.view3)
        assert_element_equal(layout.right, self.view2)
        assert_element_equal(layout.top, self.view1)

    def test_adjointlayout_iter(self):
        layout = self.view3 << self.view2 << self.view1
        for el, view in zip(layout, [self.view3, self.view2, self.view1], strict=True):
            assert_element_equal(el, view)

    def test_adjointlayout_add_operator(self):
        layout1 = self.view3 << self.view2
        layout2 = self.view2 << self.view1
        assert type(layout1 + layout2) is hv.Layout

    def test_adjointlayout_overlay_main(self):
        layout = (self.view1 << self.view2) * self.view3
        assert_element_equal(layout.main, self.view1 * self.view3)
        assert_element_equal(layout.right, self.view2)

    def test_adjointlayout_overlay_main_reverse(self):
        layout = self.view3 * (self.view1 << self.view2)
        assert_element_equal(layout.main, self.view3 * self.view1)
        assert_element_equal(layout.right, self.view2)

    def test_adjointlayout_overlay_main_and_right_v1(self):
        layout = (self.view1 << self.view2) * (self.view1 << self.view3)
        assert_element_equal(layout.main, self.view1 * self.view1)
        assert_element_equal(layout.right, self.view2 * self.view3)

    def test_adjointlayout_overlay_main_and_right_v2(self):
        layout = (self.view1 << self.view3) * (self.view1 << self.view2)
        assert_element_equal(layout.main, self.view1 * self.view1)
        assert_element_equal(layout.right, self.view3 * self.view2)

    def test_adjointlayout_overlay_holomap(self):
        layout = self.hmap * (self.view1 << self.view3)
        assert_element_equal(layout.main, self.hmap * self.view1)
        assert_element_equal(layout.right, self.view3)

    def test_adjointlayout_overlay_holomap_reverse(self):
        layout = (self.view1 << self.view3) * self.hmap
        assert_element_equal(layout.main, self.view1 * self.hmap)
        assert_element_equal(layout.right, self.view3)

    def test_adjointlayout_overlay_adjoined_holomap(self):
        layout = (self.view1 << self.view3) * (self.hmap << self.view3)
        assert_element_equal(layout.main, self.view1 * self.hmap)
        assert_element_equal(layout.right, self.view3 * self.view3)

    def test_adjointlayout_overlay_adjoined_holomap_reverse(self):
        layout = (self.hmap << self.view3) * (self.view1 << self.view3)
        assert_element_equal(layout.main, self.hmap * self.view1)
        assert_element_equal(layout.right, self.view3 * self.view3)

    def test_adjointlayout_overlay_adjoined_holomap_nomatch(self):
        dim_view = self.view3.clone(kdims=["x", "y"])
        layout = (self.view1 << self.view3) * (self.hmap << dim_view)
        assert_element_equal(layout.main, self.view1 * self.hmap)
        assert_element_equal(layout.right, self.view3)
        assert_element_equal(layout.top, dim_view)

    def test_adjointlayout_overlay_adjoined_holomap_nomatch_reverse(self):
        dim_view = self.view3.clone(kdims=["x", "y"])
        layout = (self.hmap << dim_view) * (self.view1 << self.view3)
        assert_element_equal(layout.main, self.hmap * self.view1)
        assert_element_equal(layout.right, dim_view)
        assert_element_equal(layout.top, self.view3)

    def test_adjointlayout_overlay_adjoined_holomap_nomatch_too_many(self):
        dim_view = self.view3.clone(kdims=["x", "y"])
        with pytest.raises(ValueError):  # noqa: PT011
            (self.view1 << self.view2 << self.view3) * (self.hmap << dim_view)

    @pytest.mark.usefixtures("mpl_backend")
    def test_histogram_image_hline_overlay(self):
        image = hv.Image(np.arange(100).reshape(10, 10))
        overlay = image * hv.HLine(y=0)
        element = overlay.hist()

        assert isinstance(element, hv.AdjointLayout)
        assert element.main == overlay


class NdLayoutTest(CompositeTest):
    def test_ndlayout_init(self):
        grid = hv.NdLayout([(0, self.view1), (1, self.view2), (2, self.view3), (3, self.view2)])
        assert grid.shape == (1, 4)

    def test_ndlayout_overlay_element(self):
        items = [(0, self.view1), (1, self.view2), (2, self.view3), (3, self.view2)]
        grid = hv.NdLayout(items)
        hline = hv.HLine(0)
        overlaid_grid = grid * hline
        expected = hv.NdLayout([(k, v * hline) for k, v in items])
        assert_element_equal(overlaid_grid, expected)

    def test_ndlayout_overlay_reverse(self):
        items = [(0, self.view1), (1, self.view2), (2, self.view3), (3, self.view2)]
        grid = hv.NdLayout(items)
        hline = hv.HLine(0)
        overlaid_grid = hline * grid
        expected = hv.NdLayout([(k, hline * v) for k, v in items])
        assert_element_equal(overlaid_grid, expected)

    def test_ndlayout_overlay_ndlayout(self):
        items = [(0, self.view1), (1, self.view2), (2, self.view3), (3, self.view2)]
        grid = hv.NdLayout(items)
        items2 = [(0, self.view2), (1, self.view1), (2, self.view2), (3, self.view3)]
        grid2 = hv.NdLayout(items2)

        expected_items = [
            (0, self.view1 * self.view2),
            (1, self.view2 * self.view1),
            (2, self.view3 * self.view2),
            (3, self.view2 * self.view3),
        ]
        expected = hv.NdLayout(expected_items)
        assert_element_equal(grid * grid2, expected)

    def test_ndlayout_overlay_ndlayout_reverse(self):
        items = [(0, self.view1), (1, self.view2), (2, self.view3), (3, self.view2)]
        grid = hv.NdLayout(items)
        items2 = [(0, self.view2), (1, self.view1), (2, self.view2), (3, self.view3)]
        grid2 = hv.NdLayout(items2)

        expected_items = [
            (0, self.view2 * self.view1),
            (1, self.view1 * self.view2),
            (2, self.view2 * self.view3),
            (3, self.view3 * self.view2),
        ]
        expected = hv.NdLayout(expected_items)
        assert_element_equal(grid2 * grid, expected)

    def test_ndlayout_overlay_ndlayout_partial(self):
        items = [(0, self.view1), (1, self.view2), (3, self.view2)]
        grid = hv.NdLayout(items, "X")
        items2 = [(0, self.view2), (2, self.view2), (3, self.view3)]
        grid2 = hv.NdLayout(items2, "X")

        expected_items = [
            (0, hv.Overlay([self.view1, self.view2])),
            (1, hv.Overlay([self.view2])),
            (2, hv.Overlay([self.view2])),
            (3, hv.Overlay([self.view2, self.view3])),
        ]
        expected = hv.NdLayout(expected_items, "X")
        assert_element_equal(grid * grid2, expected)

    def test_ndlayout_overlay_ndlayout_partial_reverse(self):
        items = [(0, self.view1), (1, self.view2), (3, self.view2)]
        grid = hv.NdLayout(items, "X")
        items2 = [(0, self.view2), (2, self.view2), (3, self.view3)]
        grid2 = hv.NdLayout(items2, "X")

        expected_items = [
            (0, hv.Overlay([self.view2, self.view1])),
            (1, hv.Overlay([self.view2])),
            (2, hv.Overlay([self.view2])),
            (3, hv.Overlay([self.view3, self.view2])),
        ]
        expected = hv.NdLayout(expected_items, "X")
        assert_element_equal(grid2 * grid, expected)


class GridTest(CompositeTest):
    def test_grid_init(self):
        vals = [self.view1, self.view2, self.view3, self.view2]
        keys = [(0, 0), (0, 1), (1, 0), (1, 1)]
        grid = hv.GridSpace(zip(keys, vals, strict=True))
        assert grid.shape == (2, 2)

    def test_grid_index_snap(self):
        vals = [self.view1, self.view2, self.view3, self.view2]
        keys = [(0, 0), (0, 1), (1, 0), (1, 1)]
        grid = hv.GridSpace(zip(keys, vals, strict=True))
        assert_element_equal(grid[0.1, 0.1], self.view1)

    def test_grid_index_strings(self):
        vals = [self.view1, self.view2, self.view3, self.view2]
        keys = [("A", 0), ("B", 1), ("C", 0), ("D", 1)]
        grid = hv.GridSpace(zip(keys, vals, strict=True))
        assert_element_equal(grid["B", 1], self.view2)

    def test_grid_index_one_axis(self):
        vals = [self.view1, self.view2, self.view3, self.view2]
        keys = [("A", 0), ("B", 1), ("C", 0), ("D", 1)]
        grid = hv.GridSpace(zip(keys, vals, strict=True))
        assert_element_equal(
            grid[:, 0], hv.GridSpace([(("A", 0), self.view1), (("C", 0), self.view3)])
        )

    def test_gridspace_overlay_element(self):
        items = [(0, self.view1), (1, self.view2), (2, self.view3), (3, self.view2)]
        grid = hv.GridSpace(items, "X")
        hline = hv.HLine(0)
        overlaid_grid = grid * hline
        expected = hv.GridSpace([(k, v * hline) for k, v in items], "X")
        assert_element_equal(overlaid_grid, expected)

    def test_gridspace_overlay_reverse(self):
        items = [(0, self.view1), (1, self.view2), (2, self.view3), (3, self.view2)]
        grid = hv.GridSpace(items, "X")
        hline = hv.HLine(0)
        overlaid_grid = hline * grid
        expected = hv.GridSpace([(k, hline * v) for k, v in items], "X")
        assert_element_equal(overlaid_grid, expected)

    def test_gridspace_overlay_gridspace(self):
        items = [(0, self.view1), (1, self.view2), (2, self.view3), (3, self.view2)]
        grid = hv.GridSpace(items, "X")
        items2 = [(0, self.view2), (1, self.view1), (2, self.view2), (3, self.view3)]
        grid2 = hv.GridSpace(items2, "X")

        expected_items = [
            (0, self.view1 * self.view2),
            (1, self.view2 * self.view1),
            (2, self.view3 * self.view2),
            (3, self.view2 * self.view3),
        ]
        expected = hv.GridSpace(expected_items, "X")
        assert_element_equal(grid * grid2, expected)

    def test_gridspace_overlay_gridspace_reverse(self):
        items = [(0, self.view1), (1, self.view2), (2, self.view3), (3, self.view2)]
        grid = hv.GridSpace(items, "X")
        items2 = [(0, self.view2), (1, self.view1), (2, self.view2), (3, self.view3)]
        grid2 = hv.GridSpace(items2, "X")

        expected_items = [
            (0, self.view2 * self.view1),
            (1, self.view1 * self.view2),
            (2, self.view2 * self.view3),
            (3, self.view3 * self.view2),
        ]
        expected = hv.GridSpace(expected_items, "X")
        assert_element_equal(grid2 * grid, expected)

    def test_gridspace_overlay_gridspace_partial(self):
        items = [(0, self.view1), (1, self.view2), (3, self.view2)]
        grid = hv.GridSpace(items, "X")
        items2 = [(0, self.view2), (2, self.view2), (3, self.view3)]
        grid2 = hv.GridSpace(items2, "X")

        expected_items = [
            (0, hv.Overlay([self.view1, self.view2])),
            (1, hv.Overlay([self.view2])),
            (2, hv.Overlay([self.view2])),
            (3, hv.Overlay([self.view2, self.view3])),
        ]
        expected = hv.GridSpace(expected_items, "X")
        assert_element_equal(grid * grid2, expected)

    def test_gridspace_overlay_gridspace_partial_reverse(self):
        items = [(0, self.view1), (1, self.view2), (3, self.view2)]
        grid = hv.GridSpace(items, "X")
        items2 = [(0, self.view2), (2, self.view2), (3, self.view3)]
        grid2 = hv.GridSpace(items2, "X")

        expected_items = [
            (0, hv.Overlay([self.view2, self.view1])),
            (1, hv.Overlay([self.view2])),
            (2, hv.Overlay([self.view2])),
            (3, hv.Overlay([self.view3, self.view2])),
        ]
        expected = hv.GridSpace(expected_items, "X")
        assert_element_equal(grid2 * grid, expected)
