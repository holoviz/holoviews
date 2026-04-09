from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import holoviews as hv
from holoviews.core.options import Compositor


class TestStatisticalElement:
    def test_distribution_array_constructor(self):
        dist = hv.Distribution(np.array([0, 1, 2]))
        assert dist.kdims == [hv.Dimension("Value")]
        assert dist.vdims == [hv.Dimension("Density")]

    def test_distribution_dframe_constructor(self):
        dist = hv.Distribution(pd.DataFrame({"Value": [0, 1, 2]}))
        assert dist.kdims == [hv.Dimension("Value")]
        assert dist.vdims == [hv.Dimension("Density")]

    def test_distribution_series_constructor(self):
        dist = hv.Distribution(pd.Series([0, 1, 2], name="Value"))
        assert dist.kdims == [hv.Dimension("Value")]
        assert dist.vdims == [hv.Dimension("Density")]

    def test_distribution_dict_constructor(self):
        dist = hv.Distribution({"Value": [0, 1, 2]})
        assert dist.kdims == [hv.Dimension("Value")]
        assert dist.vdims == [hv.Dimension("Density")]

    def test_distribution_array_constructor_custom_vdim(self):
        dist = hv.Distribution(np.array([0, 1, 2]), vdims=["Test"])
        assert dist.kdims == [hv.Dimension("Value")]
        assert dist.vdims == [hv.Dimension("Test")]

    def test_bivariate_array_constructor(self):
        dist = hv.Bivariate(np.array([[0, 1, 2], [0, 1, 2]]))
        assert dist.kdims == [hv.Dimension("x"), hv.Dimension("y")]
        assert dist.vdims == [hv.Dimension("Density")]

    def test_bivariate_dframe_constructor(self):
        dist = hv.Bivariate(pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 2]}, columns=["x", "y"]))
        assert dist.kdims == [hv.Dimension("x"), hv.Dimension("y")]
        assert dist.vdims == [hv.Dimension("Density")]

    def test_bivariate_dict_constructor(self):
        dist = hv.Bivariate({"x": [0, 1, 2], "y": [0, 1, 2]}, ["x", "y"])
        assert dist.kdims == [hv.Dimension("x"), hv.Dimension("y")]
        assert dist.vdims == [hv.Dimension("Density")]

    def test_bivariate_array_constructor_custom_vdim(self):
        dist = hv.Bivariate(np.array([[0, 1, 2], [0, 1, 2]]), vdims=["Test"])
        assert dist.kdims == [hv.Dimension("x"), hv.Dimension("y")]
        assert dist.vdims == [hv.Dimension("Test")]

    def test_distribution_array_range_kdims(self):
        dist = hv.Distribution(np.array([0, 1, 2]))
        assert dist.range(0) == (0, 2)

    def test_bivariate_array_range_kdims(self):
        dist = hv.Bivariate(np.array([[0, 1], [1, 2], [2, 3]]))
        assert dist.range(0) == (0, 2)
        assert dist.range(1) == (1, 3)

    def test_distribution_array_range_vdims(self):
        dist = hv.Distribution(np.array([0, 1, 2]))
        dmin, dmax = dist.range(1)
        assert not np.isfinite(dmin)
        assert not np.isfinite(dmax)

    def test_bivariate_array_range_vdims(self):
        dist = hv.Bivariate(np.array([[0, 1, 2], [0, 1, 3]]))
        dmin, dmax = dist.range(2)
        assert not np.isfinite(dmin)
        assert not np.isfinite(dmax)

    def test_distribution_array_kdim_type(self):
        dist = hv.Distribution(np.array([0, 1, 2]))
        assert np.issubdtype(dist.get_dimension_type(0), np.int_)

    def test_bivariate_array_kdim_type(self):
        dist = hv.Bivariate(np.array([[0, 1], [1, 2], [2, 3]]))
        assert np.issubdtype(dist.get_dimension_type(0), np.int_)
        assert np.issubdtype(dist.get_dimension_type(1), np.int_)

    def test_distribution_array_vdim_type(self):
        dist = hv.Distribution(np.array([0, 1, 2]))
        assert dist.get_dimension_type(1) == np.float64

    def test_bivariate_array_vdim_type(self):
        dist = hv.Bivariate(np.array([[0, 1], [1, 2], [2, 3]]))
        assert dist.get_dimension_type(2) == np.float64

    def test_distribution_from_image(self):
        dist = hv.Distribution(hv.Image(np.arange(5) * np.arange(5)[:, np.newaxis]), "z")
        assert dist.range(0) == (0, 16)

    def test_bivariate_from_points(self):
        points = hv.Points(np.array([[0, 1], [1, 2], [2, 3]]))
        dist = hv.Bivariate(points)
        assert dist.kdims == points.kdims


@pytest.mark.usefixtures("mpl_backend")
class TestStatisticalCompositor:
    def setup_method(self):
        pytest.importorskip("scipy")

    def test_distribution_composite(self):
        dist = hv.Distribution(np.array([0, 1, 2]))
        area = Compositor.collapse_element(dist, backend="matplotlib")
        assert isinstance(area, hv.Area)
        assert area.vdims == [hv.Dimension(("Value_density", "Density"))]

    def test_distribution_composite_transfer_opts(self):
        dist = hv.Distribution(np.array([0, 1, 2])).opts(color="red")
        area = Compositor.collapse_element(dist, backend="matplotlib")
        opts = hv.Store.lookup_options("matplotlib", area, "style").kwargs
        assert opts.get("color", None) == "red"

    def test_distribution_composite_transfer_opts_with_group(self):
        dist = hv.Distribution(np.array([0, 1, 2]), group="Test").opts(color="red")
        area = Compositor.collapse_element(dist, backend="matplotlib")
        opts = hv.Store.lookup_options("matplotlib", area, "style").kwargs
        assert opts.get("color", None) == "red"

    def test_distribution_composite_custom_vdim(self):
        dist = hv.Distribution(np.array([0, 1, 2]), vdims=["Test"])
        area = Compositor.collapse_element(dist, backend="matplotlib")
        assert isinstance(area, hv.Area)
        assert area.vdims == [hv.Dimension("Test")]

    def test_distribution_composite_not_filled(self):
        dist = hv.Distribution(
            np.array([0, 1, 2]),
        ).opts(filled=False)
        curve = Compositor.collapse_element(dist, backend="matplotlib")
        assert isinstance(curve, hv.Curve)
        assert curve.vdims == [hv.Dimension(("Value_density", "Density"))]

    def test_distribution_composite_empty_not_filled(self):
        dist = hv.Distribution([]).opts(filled=False)
        curve = Compositor.collapse_element(dist, backend="matplotlib")
        assert isinstance(curve, hv.Curve)
        assert curve.vdims == [hv.Dimension(("Value_density", "Density"))]

    def test_bivariate_composite(self):
        dist = hv.Bivariate(np.random.rand(10, 2))
        contours = Compositor.collapse_element(dist, backend="matplotlib")
        assert isinstance(contours, hv.Contours)
        assert contours.vdims == [hv.Dimension("Density")]

    def test_bivariate_composite_transfer_opts(self):
        dist = hv.Bivariate(np.random.rand(10, 2)).opts(cmap="Blues")
        contours = Compositor.collapse_element(dist, backend="matplotlib")
        opts = hv.Store.lookup_options("matplotlib", contours, "style").kwargs
        assert opts.get("cmap", None) == "Blues"

    def test_bivariate_composite_transfer_opts_with_group(self):
        dist = hv.Bivariate(np.random.rand(10, 2), group="Test").opts(cmap="Blues")
        contours = Compositor.collapse_element(dist, backend="matplotlib")
        opts = hv.Store.lookup_options("matplotlib", contours, "style").kwargs
        assert opts.get("cmap", None) == "Blues"

    def test_bivariate_composite_custom_vdim(self):
        dist = hv.Bivariate(np.random.rand(10, 2), vdims=["Test"])
        contours = Compositor.collapse_element(dist, backend="matplotlib")
        assert isinstance(contours, hv.Contours)
        assert contours.vdims == [hv.Dimension("Test")]

    def test_bivariate_composite_filled(self):
        dist = hv.Bivariate(np.random.rand(10, 2)).opts(filled=True)
        contours = Compositor.collapse_element(dist, backend="matplotlib")
        assert isinstance(contours, hv.Polygons)
        assert contours.vdims[0].name == "Density"

    def test_bivariate_composite_empty_filled(self):
        dist = hv.Bivariate([]).opts(filled=True)
        contours = Compositor.collapse_element(dist, backend="matplotlib")
        assert isinstance(contours, hv.Polygons)
        assert contours.vdims == [hv.Dimension("Density")]
        assert len(contours) == 0

    def test_bivariate_composite_empty_not_filled(self):
        dist = hv.Bivariate([]).opts(filled=True)
        contours = Compositor.collapse_element(dist, backend="matplotlib")
        assert isinstance(contours, hv.Contours)
        assert contours.vdims == [hv.Dimension("Density")]
        assert len(contours) == 0
