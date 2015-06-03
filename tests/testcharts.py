"""
Tests for the Chart Element types.
"""

import numpy as np
from holoviews import OrderedDict, Dimension, Chart, Curve, ItemTable
from holoviews.element.comparison import ComparisonTestCase

class ChartTest(ComparisonTestCase):
    """
    Test for the Chart baseclass methods.
    """

    def setUp(self):
        self.xs = range(11)
        self.ys = np.linspace(0, 1, 11)
        self.chart = Chart(zip(self.xs, self.ys))
        self.curve = Curve(zip(self.xs, self.ys))

    def test_yvalue_constructor(self):
        ys = np.linspace(0, 1, 11)
        Chart(ys)

    def test_chart_index(self):
        self.assertEqual(self.chart[5], self.ys[5])

    def test_chart_slice(self):
        chart_slice = Curve(zip(range(5, 9), np.linspace(0.5,0.8, 4)))
        self.assertEqual(self.curve[5:9], chart_slice)

    def test_chart_closest(self):
        closest = self.chart.closest([0.51, 1, 9.9])
        self.assertEqual(closest, [1., 1., 10.])

    def test_chart_reduce(self):
        mean = self.chart.reduce(x=np.mean)
        itable = ItemTable(OrderedDict([('y', np.mean(self.ys))]))
        self.assertEqual(mean, itable)

    def test_chart_sample(self):
        samples = self.chart.sample([0, 5, 10]).values()
        self.assertEqual(samples, [(0,), (0.5,), (1,)])
