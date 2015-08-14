import numpy as np

from holoviews import Curve, Path, Histogram
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

    def test_chart_zip_construct(self):
        self.assertEqual(Histogram(list(zip(self.hxs, self.sin))), self.histogram)

    def test_chart_array_construct(self):
        self.assertEqual(Histogram(np.column_stack((self.hxs, self.sin))), self.histogram)

    def test_chart_yvalues_construct(self):
        self.assertEqual(Histogram(self.sin), self.histogram)
