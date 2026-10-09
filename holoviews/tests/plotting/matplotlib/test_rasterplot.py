from __future__ import annotations

import numpy as np
import pytest
from matplotlib.colors import ListedColormap

import holoviews as hv
from holoviews.core.options import AbbreviatedException
from holoviews.plotting.mpl.raster import RGBPlot

from ..._deps import ds, ds_skip
from .test_plot import TestMPLPlot, mpl_renderer


class TestRasterPlot(TestMPLPlot):
    def test_raster_invert_axes(self):
        arr = np.array([[0, 1, 2], [3, 4, 5]])
        raster = hv.Raster(arr).opts(invert_axes=True)
        plot = mpl_renderer.get_plot(raster)
        artist = plot.handles["artist"]
        np.testing.assert_equal(artist.get_array().data, arr.T[::-1])
        assert artist.get_extent() == [0, 2, 0, 3]

    def test_raster_nodata(self):
        arr = np.array([[0, 1, 2], [3, 4, 5]])
        expected = np.array([[3, 4, 5], [np.nan, 1, 2]])

        raster = hv.Raster(arr).opts(nodata=0)
        plot = mpl_renderer.get_plot(raster)
        artist = plot.handles["artist"]
        np.testing.assert_equal(artist.get_array().data, expected)

    def test_raster_nodata_uint(self):
        arr = np.array([[0, 1, 2], [3, 4, 5]], dtype="uint32")
        expected = np.array([[3, 4, 5], [np.nan, 1, 2]])

        raster = hv.Raster(arr).opts(nodata=0)
        plot = mpl_renderer.get_plot(raster)
        artist = plot.handles["artist"]
        np.testing.assert_equal(artist.get_array().data, expected)

    def test_image_invert_axes(self):
        arr = np.array([[0, 1, 2], [3, 4, 5]])
        raster = hv.Image(arr).opts(invert_axes=True)
        plot = mpl_renderer.get_plot(raster)
        artist = plot.handles["artist"]
        np.testing.assert_equal(artist.get_array().data, arr.T[::-1, ::-1])
        assert artist.get_extent() == [-0.5, 0.5, -0.5, 0.5]

    def test_image_listed_cmap(self):
        colors = ["#ffffff", "#000000"]
        img = hv.Image(np.array([[0, 1, 2], [3, 4, 5]])).opts(cmap=colors)
        plot = mpl_renderer.get_plot(img)
        artist = plot.handles["artist"]
        cmap = artist.get_cmap()
        assert isinstance(cmap, ListedColormap)
        assert cmap.colors == colors

    def test_image_cbar_extend_both(self):
        img = hv.Image(np.array([[0, 1], [2, 3]])).redim(z=dict(range=(1, 2)))
        plot = mpl_renderer.get_plot(img.opts(colorbar=True))
        assert plot.handles["cbar"].extend == "both"

    def test_image_cbar_extend_min(self):
        img = hv.Image(np.array([[0, 1], [2, 3]])).redim(z=dict(range=(1, None)))
        plot = mpl_renderer.get_plot(img.opts(colorbar=True))
        assert plot.handles["cbar"].extend == "min"

    def test_image_cbar_extend_max(self):
        img = hv.Image(np.array([[0, 1], [2, 3]])).redim(z=dict(range=(None, 2)))
        plot = mpl_renderer.get_plot(img.opts(colorbar=True))
        assert plot.handles["cbar"].extend == "max"

    def test_image_cbar_extend_clim(self):
        img = hv.Image(np.array([[0, 1], [2, 3]])).opts(clim=(np.nan, np.nan), colorbar=True)
        plot = mpl_renderer.get_plot(img)
        assert plot.handles["cbar"].extend == "neither"

    @ds_skip
    def test_image_stack(self):
        x = np.arange(0, 3)
        y = np.arange(5, 8)
        a = np.array([[np.nan, np.nan, 1], [np.nan] * 3, [np.nan] * 3])
        b = np.array([[np.nan] * 3, [1, 1, np.nan], [np.nan] * 3])
        c = np.array([[np.nan] * 3, [np.nan] * 3, [1, 1, 1]])

        img_stack = hv.ImageStack((x, y, a, b, c), kdims=["x", "y"], vdims=["a", "b", "c"])
        plot = mpl_renderer.get_plot(img_stack)
        artist = plot.handles["artist"]
        array = artist.get_array().data
        assert array.shape == (3, 3, 4)
        assert artist.get_extent() == [-0.5, 2.5, 4.5, 7.5]
        assert isinstance(plot, RGBPlot)


@ds_skip
class TestSyntheticLegendPlot(TestMPLPlot):
    __test__ = True

    def setup_method(self):
        super().setup_method()

        from holoviews.operation.datashader import datashade, rasterize

        points = hv.Points([(0, 0, "A"), (1, 1, "B"), (2, 2, "C")], vdims=["Label"])
        kwargs = dict(aggregator=ds.by("Label"), dynamic=False, width=10, height=10)
        self.img_stack = rasterize(points, **kwargs).opts(show_legend=True)
        self.rgb = datashade(points, **kwargs).opts(show_legend=True)

    def test_rgb_legend(self):
        plot = mpl_renderer.get_plot(self.rgb)
        legend_labels = [t.get_text() for t in plot.handles["axis"].get_legend().texts]
        assert legend_labels == ["A", "B", "C"]
        assert plot._legend_plot is not None

    def test_image_stack_legend(self):
        plot = mpl_renderer.get_plot(self.img_stack)
        legend_labels = [t.get_text() for t in plot.handles["axis"].get_legend().texts]
        assert legend_labels == ["A", "B", "C"]
        assert plot._legend_plot is not None

    def test_rgb_legend_does_not_swallow_exception(self, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("legend failed")

        monkeypatch.setattr("holoviews.plotting.mpl.raster.categorical_legend", boom)
        with pytest.raises(AbbreviatedException, match="legend failed"):
            mpl_renderer.get_plot(self.rgb)

    def test_image_stack_legend_does_not_swallow_exception(self, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("legend failed")

        monkeypatch.setattr("holoviews.plotting.mpl.raster.categorical_legend", boom)
        with pytest.raises(AbbreviatedException, match="legend failed"):
            mpl_renderer.get_plot(self.img_stack)

    def test_non_categorical_datashade_with_legend_still_renders(self):
        from holoviews.operation.datashader import datashade

        rgb = datashade(hv.Points([(0, 0), (1, 1)]), dynamic=False, width=10, height=10).opts(
            show_legend=True
        )
        plot = mpl_renderer.get_plot(rgb)
        assert getattr(plot, "_legend_plot", None) is None
        assert plot.handles["axis"].get_legend() is None
