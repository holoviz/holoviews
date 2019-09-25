"""
Unit tests of Ellipsis (...) in __getitem__
"""
import numpy as np
import holoviews as hv
from holoviews.element.comparison import ComparisonTestCase



class TestEllipsisCharts(ComparisonTestCase):

    def test_curve_ellipsis_slice_x(self):
        sliced = hv.Curve([(i,2*i) for i in range(10)])[2:7,...]
        self.assertEqual(sliced.range('x'), (2,6))

    def test_curve_ellipsis_slice_y(self):
        sliced = hv.Curve([(i,2*i) for i in range(10)])[..., 3:9]
        self.assertEqual(sliced.range('y'), (4,8))

    def test_points_ellipsis_slice_x(self):
         sliced = hv.Points([(i,2*i) for i in range(10)])[2:7,...]
         self.assertEqual(sliced.range('x'), (2,6))

    def test_scatter_ellipsis_value(self):
        hv.Scatter(range(10))[...,'y']

    def test_scatter_ellipsis_value_missing(self):
        try:
            hv.Scatter(range(10))[...,'Non-existent']
        except Exception as e:
            if str(e) != "'Non-existent' is not an available value dimension":
                raise AssertionError("Incorrect exception raised.")

    def test_points_ellipsis_slice_y(self):
        sliced = hv.Points([(i,2*i) for i in range(10)])[..., 3:9]
        self.assertEqual(sliced.range('y'), (4,8))

    def test_histogram_ellipsis_slice_value(self):
        frequencies, edges = np.histogram(range(20), 20)
        sliced = hv.Histogram((frequencies, edges))[..., 'Frequency']
        self.assertEqual(len(sliced.dimension_values(0)), 20)

    def test_histogram_ellipsis_slice_range(self):
        frequencies, edges = np.histogram(range(20), 20)
        sliced = hv.Histogram((edges, frequencies))[0:5, ...]
        self.assertEqual(len(sliced.dimension_values(0)), 5)


    def test_histogram_ellipsis_slice_value_missing(self):
        frequencies, edges = np.histogram(range(20), 20)
        with self.assertRaises(IndexError):
            hv.Histogram((frequencies, edges))[..., 'Non-existent']


class TestEllipsisTable(ComparisonTestCase):

    def setUp(self):
        keys =   [('M',10), ('M',16), ('F',12)]
        values = [(15, 0.8), (18, 0.6), (10, 0.8)]
        self.table =hv.Table(zip(keys,values),
                             kdims = ['Gender', 'Age'],
                             vdims=['Weight', 'Height'])
        super(TestEllipsisTable, self).setUp()

    def test_table_ellipsis_slice_value_weight(self):
        sliced = self.table[..., 'Weight']
        assert sliced.vdims==['Weight']

    def test_table_ellipsis_slice_value_height(self):
        sliced = self.table[..., 'Height']
        assert sliced.vdims==['Height']

    def test_table_ellipsis_slice_key_gender(self):
        sliced = self.table['M',...]
        if not all(el=='M' for el in sliced.dimension_values('Gender')):
            raise AssertionError("Table key slicing on 'Gender' failed.")



class TestEllipsisRaster(ComparisonTestCase):

    def test_raster_ellipsis_slice_value(self):
        data = np.random.rand(10,10)
        sliced = hv.Raster(data)[...,'z']
        self.assertEqual(sliced.data, data)

    def test_raster_ellipsis_slice_value_missing(self):
        data = np.random.rand(10,10)
        try:
            hv.Raster(data)[...,'Non-existent']
        except Exception as e:
            if "\'z\' is the only selectable value dimension" not in str(e):
                raise AssertionError("Unexpected exception.")

    def test_image_ellipsis_slice_value(self):
        data = np.random.rand(10,10)
        sliced = hv.Image(data)[...,'z']
        self.assertEqual(sliced.data, data)

    def test_image_ellipsis_slice_value_missing(self):
        data = np.random.rand(10,10)
        try:
            hv.Image(data)[...,'Non-existent']
        except Exception as e:
            if str(e) != "'Non-existent' is not an available value dimension":
                raise AssertionError("Unexpected exception.")

    def test_rgb_ellipsis_slice_value(self):
        data = np.random.rand(10,10,3)
        sliced = hv.RGB(data)[:,:,'R']
        self. assertEqual(sliced.data, data[:,:,0])


    def test_rgb_ellipsis_slice_value_missing(self):
        rgb = hv.RGB(np.random.rand(10,10,3))
        try:
            rgb[...,'Non-existent']
        except Exception as e:
            if str(e) != repr("'Non-existent' is not an available value dimension"):
                raise AssertionError("Incorrect exception raised.")



class TestEllipsisDeepIndexing(ComparisonTestCase):

    def test_deep_ellipsis_curve_slicing_1(self):
        hmap = hv.HoloMap({i:hv.Curve([(j,j) for j in range(10)])
                   for i in range(10)})
        sliced = hmap[2:5,...]
        self.assertEqual(sliced.keys(), [2, 3, 4])


    def test_deep_ellipsis_curve_slicing_2(self):
        hmap = hv.HoloMap({i:hv.Curve([(j,j) for j in range(10)])
                   for i in range(10)})
        sliced = hmap[2:5,1:8,...]
        self.assertEqual(sliced.last.range('x'), (1,7))


    def test_deep_ellipsis_curve_slicing_3(self):
        hmap = hv.HoloMap({i:hv.Curve([(j,2*j) for j in range(10)])
                   for i in range(10)})
        sliced = hmap[...,2:5]
        self.assertEqual(sliced.last.range('y'), (2, 4))
