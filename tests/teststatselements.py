import numpy as np

import holoviews
from holoviews.core.dimension import Dimension
from holoviews.core.options import Compositor, Store
from holoviews.element import (Distribution, Bivariate, Points, Image,
                               Curve, Area, Contours, Polygons)
from holoviews.element.comparison import ComparisonTestCase


class StatisticalElementTest(ComparisonTestCase):

    def test_distribution_array_constructor(self):
        dist = Distribution(np.array([0, 1, 2]))
        self.assertEqual(dist.kdims, [Dimension('Value')])
        self.assertEqual(dist.vdims, [Dimension('Density')])

    def test_distribution_array_constructor_custom_vdim(self):
        dist = Distribution(np.array([0, 1, 2]), vdims=['Test'])
        self.assertEqual(dist.kdims, [Dimension('Value')])
        self.assertEqual(dist.vdims, [Dimension('Test')])

    def test_bivariate_array_constructor(self):
        dist = Bivariate(np.array([[0, 1, 2], [0, 1, 2]]))
        self.assertEqual(dist.kdims, [Dimension('x'), Dimension('y')])
        self.assertEqual(dist.vdims, [Dimension('Density')])

    def test_bivariate_array_constructor_custom_vdim(self):
        dist = Bivariate(np.array([[0, 1, 2], [0, 1, 2]]), vdims=['Test'])
        self.assertEqual(dist.kdims, [Dimension('x'), Dimension('y')])
        self.assertEqual(dist.vdims, [Dimension('Test')])

    def test_distribution_array_range_kdims(self):
        dist = Distribution(np.array([0, 1, 2]))
        self.assertEqual(dist.range(0), (0, 2))

    def test_bivariate_array_range_kdims(self):
        dist = Bivariate(np.array([[0, 1], [1, 2], [2, 3]]))
        self.assertEqual(dist.range(0), (0, 2))
        self.assertEqual(dist.range(1), (1, 3))

    def test_distribution_array_range_vdims(self):
        dist = Distribution(np.array([0, 1, 2]))
        dmin, dmax = dist.range(1)
        self.assertFalse(np.isfinite(dmin))
        self.assertFalse(np.isfinite(dmax))

    def test_bivariate_array_range_vdims(self):
        dist = Bivariate(np.array([[0, 1, 2], [0, 1, 3]]))
        dmin, dmax = dist.range(2)
        self.assertFalse(np.isfinite(dmin))
        self.assertFalse(np.isfinite(dmax))

    def test_distribution_array_kdim_type(self):
        dist = Distribution(np.array([0, 1, 2]))
        self.assertEqual(dist.get_dimension_type(0), np.int64)

    def test_bivariate_array_kdim_type(self):
        dist = Bivariate(np.array([[0, 1], [1, 2], [2, 3]]))
        self.assertEqual(dist.get_dimension_type(0), np.int64)
        self.assertEqual(dist.get_dimension_type(1), np.int64)
        
    def test_distribution_array_vdim_type(self):
        dist = Distribution(np.array([0, 1, 2]))
        self.assertEqual(dist.get_dimension_type(1), np.float64)

    def test_bivariate_array_vdim_type(self):
        dist = Bivariate(np.array([[0, 1], [1, 2], [2, 3]]))
        self.assertEqual(dist.get_dimension_type(2), np.float64)

    def test_distribution_from_image(self):
        dist = Distribution(Image(np.arange(5)*np.arange(5)[:, np.newaxis]), 'z')
        self.assertEqual(dist.range(0), (0, 16))

    def test_bivariate_from_points(self):
        points = Points(np.array([[0, 1], [1, 2], [2, 3]]))
        dist = Bivariate(points)
        self.assertEqual(dist.kdims, points.kdims)



class StatisticalCompositorTest(ComparisonTestCase):

    def setUp(self):
        self.renderer = holoviews.renderer('matplotlib')
        np.random.seed(42)

    def test_distribution_composite(self):
        dist = Distribution(np.array([0, 1, 2]))
        area = Compositor.collapse_element(dist)
        self.assertIsInstance(area, Area)
        self.assertEqual(area.vdims, [Dimension(('Value_density', 'Value Density'))])

    def test_distribution_composite_transfer_opts(self):
        dist = Distribution(np.array([0, 1, 2])).opts(style=dict(color='red'))
        area = Compositor.collapse_element(dist)
        opts = Store.lookup_options('matplotlib', area, 'style').kwargs
        self.assertEqual(opts.get('color', None), 'red')

    def test_distribution_composite_transfer_opts_with_group(self):
        dist = Distribution(np.array([0, 1, 2]), group='Test').opts(style=dict(color='red'))
        area = Compositor.collapse_element(dist)
        opts = Store.lookup_options('matplotlib', area, 'style').kwargs
        self.assertEqual(opts.get('color', None), 'red')
        
    def test_distribution_composite_custom_vdim(self):
        dist = Distribution(np.array([0, 1, 2]), vdims=['Test'])
        area = Compositor.collapse_element(dist)
        self.assertIsInstance(area, Area)
        self.assertEqual(area.vdims, [Dimension('Test')])
        
    def test_distribution_composite_not_filled(self):
        dist = Distribution(np.array([0, 1, 2])).opts(plot=dict(filled=False))
        curve = Compositor.collapse_element(dist)
        self.assertIsInstance(curve, Curve)
        self.assertEqual(curve.vdims, [Dimension(('Value_density', 'Value Density'))])

    def test_bivariate_composite(self):
        dist = Bivariate(np.random.rand(10, 2))
        contours = Compositor.collapse_element(dist)
        self.assertIsInstance(contours, Contours)
        self.assertEqual(contours.vdims, [Dimension('Density')])

    def test_bivariate_composite_transfer_opts(self):
        dist = Bivariate(np.random.rand(10, 2)).opts(style=dict(cmap='Blues'))
        contours = Compositor.collapse_element(dist)
        opts = Store.lookup_options('matplotlib', contours, 'style').kwargs
        self.assertEqual(opts.get('cmap', None), 'Blues')

    def test_bivariate_composite_transfer_opts_with_group(self):
        dist = Bivariate(np.random.rand(10, 2), group='Test').opts(style=dict(cmap='Blues'))
        contours = Compositor.collapse_element(dist)
        opts = Store.lookup_options('matplotlib', contours, 'style').kwargs
        self.assertEqual(opts.get('cmap', None), 'Blues')

    def test_bivariate_composite_custom_vdim(self):
        dist = Bivariate(np.random.rand(10, 2), vdims=['Test'])
        contours = Compositor.collapse_element(dist)
        self.assertIsInstance(contours, Contours)
        self.assertEqual(contours.vdims, [Dimension('Test')])

    def test_bivariate_composite_filled(self):
        dist = Bivariate(np.random.rand(10, 2)).opts(plot=dict(filled=True))
        contours = Compositor.collapse_element(dist)
        self.assertIsInstance(contours, Polygons)
        self.assertEqual(contours.vdims, [Dimension('Density')])
