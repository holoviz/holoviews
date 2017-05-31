import numpy as np

from holoviews import Dataset, Curve, Path, Histogram, HeatMap
from holoviews.element.comparison import ComparisonTestCase

class ElementConstructorTest(ComparisonTestCase):
    """
    Tests allowable data formats when constructing
    the basic Element types.
    """

    def setUp(self):
        self.xs = np.linspace(0, 2*np.pi, 11)
        self.hxs = np.arange(len(self.xs))
        self.sin = np.sin(self.xs)
        self.cos = np.cos(self.xs)
        sine_data = np.column_stack((self.xs, self.sin))
        cos_data = np.column_stack((self.xs, self.cos))
        self.curve = Curve(sine_data)
        self.path = Path([sine_data, cos_data])
        self.histogram = Histogram(self.sin, self.hxs)
        super(ElementConstructorTest, self).setUp()

    def test_chart_zipconstruct(self):
        self.assertEqual(Curve(zip(self.xs, self.sin)), self.curve)

    def test_chart_tuple_construct(self):
        self.assertEqual(Curve((self.xs, self.sin)), self.curve)

    def test_path_tuple_construct(self):
        self.assertEqual(Path((self.xs, np.column_stack((self.sin, self.cos)))), self.path)

    def test_path_tuplelist_construct(self):
        self.assertEqual(Path([(self.xs, self.sin), (self.xs, self.cos)]), self.path)

    def test_path_ziplist_construct(self):
        self.assertEqual(Path([list(zip(self.xs, self.sin)), list(zip(self.xs, self.cos))]), self.path)

    def test_hist_zip_construct(self):
        self.assertEqual(Histogram(list(zip(self.hxs, self.sin))), self.histogram)

    def test_hist_array_construct(self):
        self.assertEqual(Histogram(np.column_stack((self.hxs, self.sin))), self.histogram)

    def test_hist_yvalues_construct(self):
        self.assertEqual(Histogram(self.sin), self.histogram)

    def test_hist_curve_construct(self):
        hist = Histogram(Curve(([0.1, 0.3, 0.5], [2.1, 2.2, 3.3])))
        self.assertEqual(hist.data[0], np.array([2.1, 2.2, 3.3]))
        self.assertEqual(hist.data[1], np.array([0, 0.2, 0.4, 0.6]))

    def test_hist_curve_int_edges_construct(self):
        hist = Histogram(Curve(range(3)))
        self.assertEqual(hist.data[0], np.arange(3))
        self.assertEqual(hist.data[1], np.array([-.5, .5, 1.5, 2.5]))

    def test_heatmap_construct(self):
        hmap = HeatMap([('A', 'a', 1), ('B', 'b', 2)])
        dataset = Dataset({'x': ['A', 'B'], 'y': ['a', 'b'], 'z': [[1, np.NaN], [np.NaN, 2]]},
                          kdims=['x', 'y'], vdims=['z'])
        self.assertEqual(hmap.gridded, dataset)

    def test_heatmap_construct_unsorted(self):
        hmap = HeatMap([('B', 'b', 2), ('A', 'a', 1)])
        dataset = Dataset({'x': ['B', 'A'], 'y': ['b', 'a'], 'z': [[2, np.NaN], [np.NaN, 1]]},
                          kdims=['x', 'y'], vdims=['z'])
        self.assertEqual(hmap.gridded, dataset)

    def test_heatmap_construct_partial_sorted(self):
        data = [(chr(65+i),chr(97+j), i*j) for i in range(3) for j in [2, 0, 1] if i!=j]
        hmap = HeatMap(data)
        dataset = Dataset({'x': ['A', 'B', 'C'], 'y': ['c', 'b', 'a'],
                           'z': [[0, 2, np.NaN], [np.NaN, 0, 0], [0, np.NaN, 2]]},
                          kdims=['x', 'y'], vdims=['z'])
        self.assertEqual(hmap.gridded, dataset)

    def test_heatmap_construct_and_sort(self):
        data = [(chr(65+i),chr(97+j), i*j) for i in range(3) for j in [2, 0, 1] if i!=j]
        hmap = HeatMap(data).sort()
        dataset = Dataset({'x': ['A', 'B', 'C'], 'y': ['a', 'b', 'c'],
                           'z': [[np.NaN, 0, 0], [0, np.NaN, 2], [0, 2, np.NaN]]},
                          kdims=['x', 'y'], vdims=['z'])
        self.assertEqual(hmap.gridded, dataset)
