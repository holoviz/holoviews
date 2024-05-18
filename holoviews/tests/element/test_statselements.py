import numpy as np
import pandas as pd
import pytest

from holoviews.core.dimension import Dimension
from holoviews.core.options import Compositor, Store
from holoviews.element import (
    Area,
    Bivariate,
    Contours,
    Curve,
    Distribution,
    Image,
    Points,
    Polygons,
)


class TestStatisticalElement:

    def test_distribution_array_constructor(self):
        dist = Distribution(np.array([0, 1, 2]))
        assert dist.kdims == [Dimension('Value')]
        assert dist.vdims == [Dimension('Density')]

    def test_distribution_dframe_constructor(self):
        dist = Distribution(pd.DataFrame({'Value': [0, 1, 2]}))
        assert dist.kdims == [Dimension('Value')]
        assert dist.vdims == [Dimension('Density')]

    def test_distribution_series_constructor(self):
        dist = Distribution(pd.Series([0, 1, 2], name='Value'))
        assert dist.kdims == [Dimension('Value')]
        assert dist.vdims == [Dimension('Density')]

    def test_distribution_dict_constructor(self):
        dist = Distribution({'Value': [0, 1, 2]})
        assert dist.kdims == [Dimension('Value')]
        assert dist.vdims == [Dimension('Density')]

    def test_distribution_array_constructor_custom_vdim(self):
        dist = Distribution(np.array([0, 1, 2]), vdims=['Test'])
        assert dist.kdims == [Dimension('Value')]
        assert dist.vdims == [Dimension('Test')]

    def test_bivariate_array_constructor(self):
        dist = Bivariate(np.array([[0, 1, 2], [0, 1, 2]]))
        assert dist.kdims == [Dimension('x'), Dimension('y')]
        assert dist.vdims == [Dimension('Density')]

    def test_bivariate_dframe_constructor(self):
        dist = Bivariate(pd.DataFrame({'x': [0, 1, 2], 'y': [0, 1, 2]}, columns=['x', 'y']))
        assert dist.kdims == [Dimension('x'), Dimension('y')]
        assert dist.vdims == [Dimension('Density')]

    def test_bivariate_dict_constructor(self):
        dist = Bivariate({'x': [0, 1, 2], 'y': [0, 1, 2]}, ['x', 'y'])
        assert dist.kdims == [Dimension('x'), Dimension('y')]
        assert dist.vdims == [Dimension('Density')]

    def test_bivariate_array_constructor_custom_vdim(self):
        dist = Bivariate(np.array([[0, 1, 2], [0, 1, 2]]), vdims=['Test'])
        assert dist.kdims == [Dimension('x'), Dimension('y')]
        assert dist.vdims == [Dimension('Test')]

    def test_distribution_array_range_kdims(self):
        dist = Distribution(np.array([0, 1, 2]))
        assert dist.range(0) == (0, 2)

    def test_bivariate_array_range_kdims(self):
        dist = Bivariate(np.array([[0, 1], [1, 2], [2, 3]]))
        assert dist.range(0) == (0, 2)
        assert dist.range(1) == (1, 3)

    def test_distribution_array_range_vdims(self):
        dist = Distribution(np.array([0, 1, 2]))
        dmin, dmax = dist.range(1)
        assert not np.isfinite(dmin)
        assert not np.isfinite(dmax)

    def test_bivariate_array_range_vdims(self):
        dist = Bivariate(np.array([[0, 1, 2], [0, 1, 3]]))
        dmin, dmax = dist.range(2)
        assert not np.isfinite(dmin)
        assert not np.isfinite(dmax)

    def test_distribution_array_kdim_type(self):
        dist = Distribution(np.array([0, 1, 2]))
        assert np.issubdtype(dist.get_dimension_type(0), np.int_)

    def test_bivariate_array_kdim_type(self):
        dist = Bivariate(np.array([[0, 1], [1, 2], [2, 3]]))
        assert np.issubdtype(dist.get_dimension_type(0), np.int_)
        assert np.issubdtype(dist.get_dimension_type(1), np.int_)

    def test_distribution_array_vdim_type(self):
        dist = Distribution(np.array([0, 1, 2]))
        assert dist.get_dimension_type(1) == np.float64

    def test_bivariate_array_vdim_type(self):
        dist = Bivariate(np.array([[0, 1], [1, 2], [2, 3]]))
        assert dist.get_dimension_type(2) == np.float64

    def test_distribution_from_image(self):
        dist = Distribution(Image(np.arange(5)*np.arange(5)[:, np.newaxis]), 'z')
        assert dist.range(0) == (0, 16)

    def test_bivariate_from_points(self):
        points = Points(np.array([[0, 1], [1, 2], [2, 3]]))
        dist = Bivariate(points)
        assert dist.kdims == points.kdims


@pytest.mark.usefixtures("mpl_backend")
class TestStatisticalCompositor:

    def setup_method(self):
        pytest.importorskip('scipy')

    def test_distribution_composite(self):
        dist = Distribution(np.array([0, 1, 2]))
        area = Compositor.collapse_element(dist, backend='matplotlib')
        assert isinstance(area, Area)
        assert area.vdims == [Dimension(('Value_density', 'Density'))]

    def test_distribution_composite_transfer_opts(self):
        dist = Distribution(np.array([0, 1, 2])).opts(color='red')
        area = Compositor.collapse_element(dist, backend='matplotlib')
        opts = Store.lookup_options('matplotlib', area, 'style').kwargs
        assert opts.get('color', None) == 'red'

    def test_distribution_composite_transfer_opts_with_group(self):
        dist = Distribution(np.array([0, 1, 2]), group='Test').opts(color='red')
        area = Compositor.collapse_element(dist, backend='matplotlib')
        opts = Store.lookup_options('matplotlib', area, 'style').kwargs
        assert opts.get('color', None) == 'red'

    def test_distribution_composite_custom_vdim(self):
        dist = Distribution(np.array([0, 1, 2]), vdims=['Test'])
        area = Compositor.collapse_element(dist, backend='matplotlib')
        assert isinstance(area, Area)
        assert area.vdims == [Dimension('Test')]

    def test_distribution_composite_not_filled(self):
        dist = Distribution(np.array([0, 1, 2]), ).opts(filled=False)
        curve = Compositor.collapse_element(dist, backend='matplotlib')
        assert isinstance(curve, Curve)
        assert curve.vdims == [Dimension(('Value_density', 'Density'))]

    def test_distribution_composite_empty_not_filled(self):
        dist = Distribution([]).opts(filled=False)
        curve = Compositor.collapse_element(dist, backend='matplotlib')
        assert isinstance(curve, Curve)
        assert curve.vdims == [Dimension(('Value_density', 'Density'))]

    def test_bivariate_composite(self):
        dist = Bivariate(np.random.rand(10, 2))
        contours = Compositor.collapse_element(dist, backend='matplotlib')
        assert isinstance(contours, Contours)
        assert contours.vdims == [Dimension('Density')]

    def test_bivariate_composite_transfer_opts(self):
        dist = Bivariate(np.random.rand(10, 2)).opts(cmap='Blues')
        contours = Compositor.collapse_element(dist, backend='matplotlib')
        opts = Store.lookup_options('matplotlib', contours, 'style').kwargs
        assert opts.get('cmap', None) == 'Blues'

    def test_bivariate_composite_transfer_opts_with_group(self):
        dist = Bivariate(np.random.rand(10, 2), group='Test').opts(cmap='Blues')
        contours = Compositor.collapse_element(dist, backend='matplotlib')
        opts = Store.lookup_options('matplotlib', contours, 'style').kwargs
        assert opts.get('cmap', None) == 'Blues'

    def test_bivariate_composite_custom_vdim(self):
        dist = Bivariate(np.random.rand(10, 2), vdims=['Test'])
        contours = Compositor.collapse_element(dist, backend='matplotlib')
        assert isinstance(contours, Contours)
        assert contours.vdims == [Dimension('Test')]

    def test_bivariate_composite_filled(self):
        dist = Bivariate(np.random.rand(10, 2)).opts(filled=True)
        contours = Compositor.collapse_element(dist, backend='matplotlib')
        assert isinstance(contours, Polygons)
        assert contours.vdims[0].name == 'Density'

    def test_bivariate_composite_empty_filled(self):
        dist = Bivariate([]).opts(filled=True)
        contours = Compositor.collapse_element(dist, backend='matplotlib')
        assert isinstance(contours, Polygons)
        assert contours.vdims == [Dimension('Density')]
        assert len(contours) == 0

    def test_bivariate_composite_empty_not_filled(self):
        dist = Bivariate([]).opts(filled=True)
        contours = Compositor.collapse_element(dist, backend='matplotlib')
        assert isinstance(contours, Contours)
        assert contours.vdims == [Dimension('Density')]
        assert len(contours) == 0
