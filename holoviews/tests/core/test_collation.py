"""
Test cases for Collator
"""
import itertools

import numpy as np

from holoviews.core import Collator, GridSpace, HoloMap, NdOverlay, Overlay
from holoviews.element import Curve
from holoviews.testing import assert_element_equal


class TestCollation:
    def setup_method(self):
        alphas, betas, deltas = 2, 2, 2
        Bs = list(range(100))
        coords = itertools.product(*(range(n) for n in [alphas, betas, deltas]))
        mus=np.random.rand(alphas, betas, 100, 10)
        self.phase_boundaries = {(a, b, d): Curve(zip(Bs, mus[a, b, :, i]*a+b, strict=None))
                                 for i in range(10) for a, b, d in coords}
        self.dimensions = ['alpha', 'beta', 'delta']
        self.nesting_hmap = HoloMap(self.phase_boundaries, kdims=self.dimensions)
        self.nested_hmap = self.nesting_hmap.groupby(['alpha'])
        self.nested_overlay = self.nesting_hmap.overlay(['delta'])
        self.nested_grid = self.nested_overlay.grid(['alpha', 'beta'])
        self.nested_layout = self.nested_overlay.layout(['alpha', 'beta'])

    def test_collate_hmap(self):
        collated = self.nested_hmap.collate()
        assert collated.kdims == self.nesting_hmap.kdims
        assert collated.keys() == self.nesting_hmap.keys()
        assert collated.type == self.nesting_hmap.type
        assert repr(collated) == repr(self.nesting_hmap)

    def test_collate_ndoverlay(self):
        collated = self.nested_overlay.collate(NdOverlay)
        ndoverlay = NdOverlay(self.phase_boundaries, kdims=self.dimensions)
        assert collated.kdims == ndoverlay.kdims
        assert collated.keys() == ndoverlay.keys()
        assert repr(collated) == repr(ndoverlay)

    def test_collate_gridspace_ndoverlay(self):
        grid = self.nesting_hmap.groupby(['delta']).collate(NdOverlay).grid(['alpha', 'beta'])
        assert grid.dimensions() == self.nested_grid.dimensions()
        assert grid.keys() == self.nested_grid.keys()
        assert repr(grid) == repr(self.nested_grid)

    def test_collate_ndlayout_ndoverlay(self):
        layout = self.nesting_hmap.groupby(['delta']).collate(NdOverlay).layout(['alpha', 'beta'])
        assert layout.dimensions() == self.nested_layout.dimensions()
        assert layout.keys() == self.nested_layout.keys()
        assert repr(layout) == repr(self.nested_layout)

    def test_collate_layout_overlay(self):
        layout = self.nested_overlay + self.nested_overlay
        collated = Collator(kdims=['alpha', 'beta'])
        for k, v in self.nested_overlay.items():
            collated[k] = v + v
        collated = collated()
        assert collated.dimensions() == layout.dimensions()

    def test_collate_layout_hmap(self):
        layout = self.nested_overlay + self.nested_overlay
        collated = Collator(kdims=['delta'], merge_type=NdOverlay)
        for k, v in self.nesting_hmap.groupby(['delta']).items():
            collated[k] = v + v
        collated = collated()
        assert repr(collated) == repr(layout)
        assert collated.dimensions() == layout.dimensions()

    def test_overlay_hmap_collate(self):
        hmap = HoloMap({i: Curve(np.arange(10)*i) for i in range(3)})
        overlaid = Overlay([hmap, hmap, hmap]).collate()
        assert_element_equal(overlaid, hmap*hmap*hmap)

    def test_overlay_gridspace_collate(self):
        grid = GridSpace({(i,j): Curve(np.arange(10)*i) for i in range(3)
                          for j in range(3)})
        overlaid = Overlay([grid, grid, grid]).collate()
        assert_element_equal(overlaid, grid*grid*grid)
