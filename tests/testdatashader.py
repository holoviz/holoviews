from nose.plugins.attrib import attr

import pandas as pd
import numpy as np

from holoviews import Curve, Scatter, Points, Image, Dataset
from holoviews.element.comparison import ComparisonTestCase

try:
    from holoviews.operation.datashader import aggregate
except:
    aggregate = None


@attr(optional=1)
class DatashaderTests(ComparisonTestCase):
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
