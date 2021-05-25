from itertools import product
import datetime as dt
import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

from holoviews.core import HoloMap
from holoviews.element import Image, Contours, Curve
from holoviews.element.comparison import ComparisonTestCase

class DimensionedSelectionTest(ComparisonTestCase):

    def setUp(self):
        self.img_fn = lambda: Image(np.random.rand(10, 10))
        self.contour_fn = lambda: Contours([np.random.rand(10, 2)
                                            for i in range(2)])
        self.datetime_fn = lambda: Curve((
            [dt.datetime(2000,1,1), dt.datetime(2000,1,2),
             dt.datetime(2000,1,3)],
            np.random.rand(3)
            ), 'time', 'x')
        params = [list(range(3)) for i in range(2)]
        self.sanitized_map = HoloMap({i: Image(i*np.random.rand(10,10))
                                      for i in range(1,10)}, kdims=['A B'])
        self.img_map = HoloMap({(i, j): self.img_fn()
                                for i, j in product(*params)},
                               kdims=['a', 'b'])
        self.contour_map = HoloMap({(i, j): self.contour_fn()
                                    for i, j in product(*params)},
                                   kdims=['a', 'b'])
        self.ndoverlay_map = self.img_map.overlay('b')
        self.overlay_map = self.img_map * self.contour_map
        self.layout_map = self.ndoverlay_map + self.contour_map
        self.duplicate_map = self.img_map.clone(kdims=['x', 'y'])

        self.overlap1 = HoloMap({i: self.img_fn() for i in range(5)})
        self.overlap2 = HoloMap({i: self.img_fn() for i in range(10)})
        self.overlap_layout = self.overlap1 + self.overlap2


    def test_simple_holoselect(self):
        self.assertEqual(self.img_map.select(a=0, b=1),
                         self.img_map[0, 1])


    def test_simple_holoslice(self):
        self.assertEqual(self.img_map.select(a=(1, 3), b=(1, 3)),
                         self.img_map[1:3, 1:3])


    def test_function_holoslice(self):
        self.assertEqual(self.img_map.select(a=lambda x: 1 <= x < 3,
                                             b=lambda x: 1 <= x < 3),
                         self.img_map[1:3, 1:3])


    def test_sanitized_holoslice(self):
        self.assertEqual(self.sanitized_map.select(A_B=(1,3)),
                         self.sanitized_map[1:3])


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
        selection = self.duplicate_map.select(
            selection_specs=(HoloMap,), x=(0, 1), y=(1, 3)
        )
        self.assertEqual(selection, self.duplicate_map[0:1, 1:3])

    def test_duplicate_dim_select(self):
        selection = self.duplicate_map.select(x=(None, 0.5), y=(None, 0.5))
        self.assertEqual(selection, self.duplicate_map[:.5, :.5, :.5, :.5])

    def test_overlap_select(self):
        selection = self.overlap_layout.select(Default=(6, None))
        self.assertEqual(selection, self.overlap1.clone(shared_data=False) + self.overlap2[6:])

    def test_datetime_select(self):
        s, e = '1999-12-31', '2000-1-2'
        curve = self.datetime_fn()
        overlay = curve * self.datetime_fn()
        for el in [curve, overlay]:
            v = el.select(time=(s, e))
            self.assertEqual(v, el[s:e])
            self.assertEqual(el.select(time=
                (dt.datetime(1999, 12, 31), dt.datetime(2000, 1, 2))), el[s:e]
            )
            if pd:
                self.assertEqual(el.select(
                    time=(pd.Timestamp(s), pd.Timestamp(e))
                ), el[pd.Timestamp(s):pd.Timestamp(e)])

    def test_selection_spec_positional_error_message(self):
        s, e = '1999-12-31', '2000-1-2'
        curve = self.datetime_fn()
        with self.assertRaisesRegexp(
                ValueError, "Use the selection_specs keyword"
        ):
            curve.select((Curve,), time=(s, e))
