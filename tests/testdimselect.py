from itertools import product
import numpy as np

from holoviews.core import HoloMap
from holoviews.element import Image, Contours
from holoviews.element.comparison import ComparisonTestCase

class DimensionedSelectionTest(ComparisonTestCase):

    def setUp(self):
        self.img_fn = lambda: Image(np.random.rand(10, 10))
        self.contour_fn = lambda: Contours([np.random.rand(10, 2)
                                            for i in range(2)])
        params = [list(range(3)) for i in range(2)]
        self.img_map = HoloMap({(i, j): self.img_fn()
                                for i, j in product(*params)},
                               key_dimensions=['a', 'b'])
        self.contour_map = HoloMap({(i, j): self.contour_fn()
                                    for i, j in product(*params)},
                                   key_dimensions=['a', 'b'])
        self.ndoverlay_map = self.img_map.overlay('b')
        self.overlay_map = self.img_map * self.contour_map
        self.layout_map = self.ndoverlay_map + self.contour_map
        self.duplicate_map = self.img_map.clone(key_dimensions=['x', 'y'])


    def test_simple_holoselect(self):
        self.assertEqual(self.img_map.select(a=0, b=1),
                         self.img_map[0, 1])


    def test_simple_holoslice(self):
        self.assertEqual(self.img_map.select(a=(1, 3), b=(1, 3)),
                         self.img_map[1:3, 1:3])


    def test_simple_holo_ndoverlay_slice(self):
        self.assertEqual(self.ndoverlay_map.select(a=(1, 3), b=(1, 3)),
                         self.ndoverlay_map[1:3, 1:3])


    def test_deep_holoslice(self):
        selection = self.img_map.select(a=(1,3), b=(1, 3), x=(None, 0), y=(None, 0))
        self.assertEqual(selection, self.img_map[1:3, 1:3, :0, :0])


    def test_deep_holooverlay_slice(self):
        map_slc = self.overlay_map[1:3, 1:3]
        img_slc = map_slc.map(lambda x: x[0:0.5, 0:0.5], [Image, Contours])
        selection = self.overlay_map.select(a=(1,3), b=(1, 3), x=(0, 0.5), y=(0, 0.5))
        self.assertEqual(selection, img_slc)


    def test_deep_layout_nesting_slice(self):
        hmap1 = self.layout_map.HoloMap.I[1:3, 1:3, 0:0.5, 0:0.5]
        hmap2 = self.layout_map.HoloMap.II[1:3, 1:3, 0:0.5, 0:0.5]
        selection = self.layout_map.select(a=(1,3), b=(1, 3), x=(0, 0.5), y=(0, 0.5))
        self.assertEqual(selection, hmap1 + hmap2)

    def test_spec_duplicate_dim_select(self):
        selection = self.duplicate_map.select((HoloMap,), x=(0, 1), y=(1, 3))
        self.assertEqual(selection, self.duplicate_map[0:1, 1:3])

    def test_duplicate_dim_select(self):
        selection = self.duplicate_map.select(x=(None, 0.5), y=(None, 0.5))
        self.assertEqual(selection, self.duplicate_map[:.5, :.5, :.5, :.5])
