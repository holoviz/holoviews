"""
Test cases for Collator
"""
import itertools
import numpy as np

from holoviews.core import Collator, HoloMap, NdOverlay, Overlay, GridSpace
from holoviews.element import Curve
from holoviews.element.comparison import ComparisonTestCase


class TestCollation(ComparisonTestCase):
    def setUp(self):
        alphas, betas, deltas = 2, 2, 2
        Bs = list(range(100))
        coords = itertools.product(*(range(n) for n in [alphas, betas, deltas]))
        mus=np.random.rand(alphas, betas, 100, 10)
        self.phase_boundaries = {(a, b, d): Curve(zip(Bs, mus[a, b, :, i]*a+b))
                                 for i in range(10) for a, b, d in coords}
        self.dimensions = ['alpha', 'beta', 'delta']
        self.nesting_hmap = HoloMap(self.phase_boundaries, kdims=self.dimensions)
        self.nested_hmap = self.nesting_hmap.groupby(['alpha'])
        self.nested_overlay = self.nesting_hmap.overlay(['delta'])
        self.nested_grid = self.nested_overlay.grid(['alpha', 'beta'])
        self.nested_layout = self.nested_overlay.layout(['alpha', 'beta'])

    def test_collate_hmap(self):
        collated = self.nested_hmap.collate()
        self.assertEqual(collated.kdims, self.nesting_hmap.kdims)
        self.assertEqual(collated.keys(), self.nesting_hmap.keys())
        self.assertEqual(collated.type, self.nesting_hmap.type)
        self.assertEqual(repr(collated), repr(self.nesting_hmap))

    def test_collate_ndoverlay(self):
        collated = self.nested_overlay.collate(NdOverlay)
        ndoverlay = NdOverlay(self.phase_boundaries, kdims=self.dimensions)
        self.assertEqual(collated.kdims, ndoverlay.kdims)
        self.assertEqual(collated.keys(), ndoverlay.keys())
        self.assertEqual(repr(collated), repr(ndoverlay))

    def test_collate_gridspace_ndoverlay(self):
        grid = self.nesting_hmap.groupby(['delta']).collate(NdOverlay).grid(['alpha', 'beta'])
        self.assertEqual(grid.dimensions(), self.nested_grid.dimensions())
        self.assertEqual(grid.keys(), self.nested_grid.keys())
        self.assertEqual(repr(grid), repr(self.nested_grid))

    def test_collate_ndlayout_ndoverlay(self):
        layout = self.nesting_hmap.groupby(['delta']).collate(NdOverlay).layout(['alpha', 'beta'])
        self.assertEqual(layout.dimensions(), self.nested_layout.dimensions())
        self.assertEqual(layout.keys(), self.nested_layout.keys())
        self.assertEqual(repr(layout), repr(self.nested_layout))

    def test_collate_layout_overlay(self):
        layout = self.nested_overlay + self.nested_overlay
        collated = Collator(kdims=['alpha', 'beta'])
        for k, v in self.nested_overlay.items():
            collated[k] = v + v
        collated = collated()
        self.assertEqual(collated.dimensions(), layout.dimensions())

    def test_collate_layout_hmap(self):
        layout = self.nested_overlay + self.nested_overlay
        collated = Collator(kdims=['delta'], merge_type=NdOverlay)
        for k, v in self.nesting_hmap.groupby(['delta']).items():
            collated[k] = v + v
        collated = collated()
        self.assertEqual(repr(collated), repr(layout))
        self.assertEqual(collated.dimensions(), layout.dimensions())

    def test_overlay_hmap_collate(self):
        hmap = HoloMap({i: Curve(np.arange(10)*i) for i in range(3)})
        overlaid = Overlay([hmap, hmap, hmap]).collate()
        self.assertEqual(overlaid, hmap*hmap*hmap)

    def test_overlay_gridspace_collate(self):
        grid = GridSpace({(i,j): Curve(np.arange(10)*i) for i in range(3)
                          for j in range(3)})
        overlaid = Overlay([grid, grid, grid]).collate()
        self.assertEqual(overlaid, grid*grid*grid)
