"""
Test cases for the Comparisons class over the Chart elements
"""

import numpy as np

from holoviews import Bars, Curve, Dimension, Histogram, Points, Scatter, VectorField
from holoviews.element.comparison import ComparisonTestCase


class CurveComparisonTest(ComparisonTestCase):

    def setUp(self):
        "Variations on the constructors in the Elements notebook"

        self.curve1 = Curve([(0.1*i, np.sin(0.1*i)) for i in range(100)])
        self.curve2 = Curve([(0.1*i, np.sin(0.1*i)) for i in range(101)])

    def test_curves_equal(self):
        self.assertEqual(self.curve1, self.curve1)

    def test_curves_unequal(self):
        try:
            self.assertEqual(self.curve1, self.curve2)
        except AssertionError as e:
            if not str(e).startswith("Curve not of matching length, 100 vs. 101"):
                raise self.failureException("Curve mismatch error not raised.")



class BarsComparisonTest(ComparisonTestCase):

    def setUp(self):
        "Variations on the constructors in the Elements notebook"

        key_dims1=[Dimension('Car occupants')]
        key_dims2=[Dimension('Cyclists')]
        value_dims1=['Count']
        self.bars1 = Bars([('one',8),('two', 10), ('three', 16)],
                          kdims=key_dims1, vdims=value_dims1)
        self.bars2 = Bars([('one',8),('two', 10), ('three', 17)],
                          kdims=key_dims1, vdims=value_dims1)
        self.bars3 = Bars([('one',8),('two', 10), ('three', 16)],
                          kdims=key_dims2, vdims=value_dims1)

    def test_bars_equal_1(self):
        self.assertEqual(self.bars1, self.bars1)

    def test_bars_equal_2(self):
        self.assertEqual(self.bars2, self.bars2)

    def test_bars_equal_3(self):
        self.assertEqual(self.bars3, self.bars3)

    def test_bars_unequal_1(self):
        try:
            self.assertEqual(self.bars1, self.bars2)
        except AssertionError as e:
            if "not almost equal" not in str(e):
                raise Exception(f'Bars mismatched data error not raised. {e}')

    def test_bars_unequal_keydims(self):
        try:
            self.assertEqual(self.bars1, self.bars3)
        except AssertionError as e:
            if not str(e) == 'Dimension names mismatched: Car occupants != Cyclists':
                raise Exception('Bars key dimension mismatch error not raised.')



class HistogramComparisonTest(ComparisonTestCase):

    def setUp(self):
        "Variations on the constructors in the Elements notebook"

        np.random.seed(1)
        frequencies1, edges1 = np.histogram([np.random.normal() for i in range(1000)], 20)
        self.hist1 = Histogram((edges1, frequencies1))
        np.random.seed(2)
        frequencies2, edges2 =  np.histogram([np.random.normal() for i in range(1000)], 20)
        self.hist2 = Histogram((edges2, frequencies2))
        self.hist3 = Histogram((edges2, frequencies1))
        self.hist4 = Histogram((edges1, frequencies2))

    def test_histograms_equal_1(self):
        self.assertEqual(self.hist1, self.hist1)

    def test_histograms_equal_2(self):
        self.assertEqual(self.hist2, self.hist2)

    def test_histograms_unequal_1(self):
        with self.assertRaises(AssertionError):
            self.assertEqual(self.hist1, self.hist2)

    def test_histograms_unequal_2(self):
        with self.assertRaises(AssertionError):
            self.assertEqual(self.hist1, self.hist3)

    def test_histograms_unequal_3(self):
        with self.assertRaises(AssertionError):
            self.assertEqual(self.hist1, self.hist4)



class ScatterComparisonTest(ComparisonTestCase):

    def setUp(self):
        "Variations on the constructors in the Elements notebook"

        self.scatter1 = Scatter([(1, i) for i in range(20)])
        self.scatter2 = Scatter([(1, i) for i in range(21)])
        self.scatter3 = Scatter([(1, i*2) for i in range(20)])


    def test_scatter_equal_1(self):
        self.assertEqual(self.scatter1, self.scatter1)

    def test_scatter_equal_2(self):
        self.assertEqual(self.scatter2, self.scatter2)

    def test_scatter_equal_3(self):
        self.assertEqual(self.scatter3, self.scatter3)

    def test_scatter_unequal_data_shape(self):
        try:
            self.assertEqual(self.scatter1, self.scatter2)
        except  AssertionError as e:
            if not str(e).startswith("Scatter not of matching length, 20 vs. 21."):
                raise self.failureException("Scatter data mismatch error not raised.")

    def test_scatter_unequal_data_values(self):
        try:
            self.assertEqual(self.scatter1, self.scatter3)
        except  AssertionError as e:
            if not str(e).startswith("Scatter not almost equal to 6 decimals"):
                raise self.failureException("Scatter data mismatch error not raised.")



class PointsComparisonTest(ComparisonTestCase):

    def setUp(self):
        "Variations on the constructors in the Elements notebook"

        self.points1 = Points([(1, i) for i in range(20)])
        self.points2 = Points([(1, i) for i in range(21)])
        self.points3 = Points([(1, i*2) for i in range(20)])


    def test_points_equal_1(self):
        self.assertEqual(self.points1, self.points1)

    def test_points_equal_2(self):
        self.assertEqual(self.points2, self.points2)

    def test_points_equal_3(self):
        self.assertEqual(self.points3, self.points3)

    def test_points_unequal_data_shape(self):
        try:
            self.assertEqual(self.points1, self.points2)
        except  AssertionError as e:
            if not str(e).startswith("Points not of matching length, 20 vs. 21."):
                raise self.failureException("Points count mismatch error not raised.")

    def test_points_unequal_data_values(self):
        try:
            self.assertEqual(self.points1, self.points3)
        except  AssertionError as e:
            if not str(e).startswith("Points not almost equal to 6 decimals"):
                raise self.failureException("Points data mismatch error not raised.")



class VectorFieldComparisonTest(ComparisonTestCase):

    def setUp(self):
        "Variations on the constructors in the Elements notebook"

        x,y  = np.mgrid[-10:10,-10:10] * 0.25
        sine_rings  = np.sin(x**2+y**2)*np.pi+np.pi
        exp_falloff1 = 1/np.exp((x**2+y**2)/8)
        exp_falloff2 = 1/np.exp((x**2+y**2)/9)

        self.vfield1 = VectorField([x,y,sine_rings, exp_falloff1])
        self.vfield2 = VectorField([x,y,sine_rings, exp_falloff2])


    def test_vfield_equal_1(self):
        self.assertEqual(self.vfield1, self.vfield1)

    def test_vfield_equal_2(self):
        self.assertEqual(self.vfield2, self.vfield2)

    def test_vfield_unequal_1(self):
        try:
            self.assertEqual(self.vfield1, self.vfield2)
        except  AssertionError as e:
            if not str(e).startswith("VectorField not almost equal to 6 decimals"):
                raise self.failureException("VectorField  data mismatch error not raised.")
