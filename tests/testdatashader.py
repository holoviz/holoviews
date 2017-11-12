from unittest import SkipTest
from nose.plugins.attrib import attr

import numpy as np
from holoviews import Curve, Points, Image, Dataset, RGB, Path, Graph
from holoviews.element.comparison import ComparisonTestCase

try:
    from holoviews.operation.datashader import (
        aggregate, regrid, ds_version, stack, directly_connect_edges
    )
except:
    ds_version = None


@attr(optional=1)
class DatashaderAggregateTests(ComparisonTestCase):
    """
    Tests for datashader aggregation
    """

    def test_aggregate_points(self):
        points = Points([(0.2, 0.3), (0.4, 0.7), (0, 0.99)])
        img = aggregate(points, dynamic=False,  x_range=(0, 1), y_range=(0, 1),
                        width=2, height=2)
        expected = Image(([0.25, 0.75], [0.25, 0.75], [[1, 0], [2, 0]]),
                         vdims=['Count'])
        self.assertEqual(img, expected)

    def test_aggregate_points_target(self):
        points = Points([(0.2, 0.3), (0.4, 0.7), (0, 0.99)])
        expected = Image(([0.25, 0.75], [0.25, 0.75], [[1, 0], [2, 0]]),
                         vdims=['Count'])
        img = aggregate(points, dynamic=False,  target=expected)
        self.assertEqual(img, expected)

    def test_aggregate_points_sampling(self):
        points = Points([(0.2, 0.3), (0.4, 0.7), (0, 0.99)])
        expected = Image(([0.25, 0.75], [0.25, 0.75], [[1, 0], [2, 0]]),
                         vdims=['Count'])
        img = aggregate(points, dynamic=False,  x_range=(0, 1), y_range=(0, 1),
                        x_sampling=0.5, y_sampling=0.5)
        self.assertEqual(img, expected)

    def test_aggregate_curve(self):
        curve = Curve([(0.2, 0.3), (0.4, 0.7), (0.8, 0.99)])
        expected = Image(([0.25, 0.75], [0.25, 0.75], [[1, 0], [1, 1]]),
                         vdims=['Count'])
        img = aggregate(curve, dynamic=False,  x_range=(0, 1), y_range=(0, 1),
                        width=2, height=2)
        self.assertEqual(img, expected)

    def test_aggregate_ndoverlay(self):
        ds = Dataset([(0.2, 0.3, 0), (0.4, 0.7, 1), (0, 0.99, 2)], kdims=['x', 'y', 'z'])
        ndoverlay = ds.to(Points, ['x', 'y'], [], 'z').overlay()
        expected = Image(([0.25, 0.75], [0.25, 0.75], [[1, 0], [2, 0]]),
                         vdims=['Count'])
        img = aggregate(ndoverlay, dynamic=False,  x_range=(0, 1), y_range=(0, 1),
                        width=2, height=2)
        self.assertEqual(img, expected)

    def test_aggregate_path(self):
        path = Path([[(0.2, 0.3), (0.4, 0.7)], [(0.4, 0.7), (0.8, 0.99)]])
        expected = Image(([0.25, 0.75], [0.25, 0.75], [[1, 0], [2, 1]]),
                         vdims=['Count'])
        img = aggregate(path, dynamic=False,  x_range=(0, 1), y_range=(0, 1),
                        width=2, height=2)
        self.assertEqual(img, expected)

    def test_aggregate_dframe_nan_path(self):
        path = Path([Path([[(0.2, 0.3), (0.4, 0.7)], [(0.4, 0.7), (0.8, 0.99)]]).dframe()])
        expected = Image(([0.25, 0.75], [0.25, 0.75], [[1, 0], [2, 1]]),
                         vdims=['Count'])
        img = aggregate(path, dynamic=False,  x_range=(0, 1), y_range=(0, 1),
                        width=2, height=2)
        self.assertEqual(img, expected)




@attr(optional=1)
class DatashaderRegridTests(ComparisonTestCase):
    """
    Tests for datashader aggregation
    """

    def setUp(self):
        if ds_version is None or ds_version <= '0.5.0':
            raise SkipTest('Regridding operations require datashader>=0.6.0')

    def test_regrid_mean(self):
        img = Image((range(10), range(5), np.arange(10) * np.arange(5)[np.newaxis].T))
        regridded = regrid(img, width=2, height=2, dynamic=False)
        expected = Image(([2., 7.], [0.75, 3.25], [[1, 5], [6, 22]]))
        self.assertEqual(regridded, expected)

    def test_regrid_mean_xarray_transposed(self):
        img = Image((range(10), range(5), np.arange(10) * np.arange(5)[np.newaxis].T),
                    datatype=['xarray'])
        img.data = img.data.T
        regridded = regrid(img, width=2, height=2, dynamic=False)
        expected = Image(([2., 7.], [0.75, 3.25], [[1, 5], [6, 22]]))
        self.assertEqual(regridded, expected)

    def test_regrid_rgb_mean(self):
        arr = (np.arange(10) * np.arange(5)[np.newaxis].T).astype('f')
        rgb = RGB((range(10), range(5), arr, arr*2, arr*2))
        regridded = regrid(rgb, width=2, height=2, dynamic=False)
        new_arr = np.array([[1.6, 5.6], [6.4, 22.4]])
        expected = RGB(([2., 7.], [0.75, 3.25], new_arr, new_arr*2, new_arr*2), datatype=['xarray'])
        self.assertEqual(regridded, expected)

    def test_regrid_max(self):
        img = Image((range(10), range(5), np.arange(10) * np.arange(5)[np.newaxis].T))
        regridded = regrid(img, aggregator='max', width=2, height=2, dynamic=False)
        expected = Image(([2., 7.], [0.75, 3.25], [[8, 18], [16, 36]]))
        self.assertEqual(regridded, expected)

    def test_regrid_upsampling(self):
        img = Image(([0.5, 1.5], [0.5, 1.5], [[0, 1], [2, 3]]))
        regridded = regrid(img, width=4, height=4, upsample=True, dynamic=False)
        expected = Image(([0.25, 0.75, 1.25, 1.75], [0.25, 0.75, 1.25, 1.75],
                          [[0, 0, 1, 1],
                           [0, 0, 1, 1],
                           [2, 2, 3, 3],
                           [2, 2, 3, 3]]))
        self.assertEqual(regridded, expected)

    def test_regrid_upsampling_linear(self):
        ### This test causes a numba error using 0.35.0 - temporarily disabled ###
        return
        img = Image(([0.5, 1.5], [0.5, 1.5], [[0, 1], [2, 3]]))
        regridded = regrid(img, width=4, height=4, upsample=True, interpolation='linear', dynamic=False)
        expected = Image(([0.25, 0.75, 1.25, 1.75], [0.25, 0.75, 1.25, 1.75],
                          [[0, 0, 0, 1],
                           [0, 1, 1, 1],
                           [1, 1, 2, 2],
                           [2, 2, 2, 3]]))
        self.assertEqual(regridded, expected)

    def test_regrid_disabled_upsampling(self):
        img = Image(([0.5, 1.5], [0.5, 1.5], [[0, 1], [2, 3]]))
        regridded = regrid(img, width=3, height=3, dynamic=False, upsample=False)
        self.assertEqual(regridded, img)

    def test_regrid_disabled_expand(self):
        img = Image(([0.5, 1.5], [0.5, 1.5], [[0., 1.], [2., 3.]]))
        regridded = regrid(img, width=2, height=2, x_range=(-2, 4), y_range=(-2, 4), expand=False,
                           dynamic=False)
        self.assertEqual(regridded, img)



@attr(optional=1)
class DatashaderStackTests(ComparisonTestCase):

    def setUp(self):
        self.rgb1_arr = np.array([[[0, 1], [1, 0]],
                                  [[1, 0], [0, 1]],
                                  [[0, 0], [0, 0]]], dtype=np.uint8).T*255
        self.rgb2_arr = np.array([[[0, 0], [0, 0]],
                                  [[0, 0], [0, 0]],
                                  [[1, 0], [0, 1]]], dtype=np.uint8).T*255
        self.rgb1 = RGB(self.rgb1_arr)
        self.rgb2 = RGB(self.rgb2_arr)


    def test_stack_add_compositor(self):
        combined = stack(self.rgb1*self.rgb2, compositor='add')
        arr = np.array([[[0, 255, 255], [255,0, 0]], [[255, 0, 0], [0, 255, 255]]], dtype=np.uint8)
        expected = RGB(arr)
        self.assertEqual(combined, expected)

    def test_stack_over_compositor(self):
        combined = stack(self.rgb1*self.rgb2, compositor='over')
        self.assertEqual(combined, self.rgb2)
    
    def test_stack_over_compositor_reverse(self):
        combined = stack(self.rgb2*self.rgb1, compositor='over')
        self.assertEqual(combined, self.rgb1)

    def test_stack_saturate_compositor(self):
        combined = stack(self.rgb1*self.rgb2, compositor='saturate')
        self.assertEqual(combined, self.rgb1)

    def test_stack_saturate_compositor_reverse(self):
        combined = stack(self.rgb2*self.rgb1, compositor='saturate')
        self.assertEqual(combined, self.rgb2)


@attr(optional=1)
class GraphBundlingTests(ComparisonTestCase):

    def setUp(self):
        if ds_version is None or ds_version <= '0.7.0':
            raise SkipTest('Regridding operations require datashader>=0.7.0')
        self.source = np.arange(8)
        self.target = np.zeros(8)
        self.graph = Graph(((self.source, self.target),))

    def test_directly_connect_paths(self):
        direct = directly_connect_edges(self.graph)._split_edgepaths
        self.assertEqual(direct, self.graph.edgepaths)
