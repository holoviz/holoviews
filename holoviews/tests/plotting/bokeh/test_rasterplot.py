from unittest import SkipTest

import numpy as np
import pandas as pd
import pytest
from bokeh.models import CustomJSHover, HoverTool

from holoviews.element import RGB, Image, ImageStack, Raster
from holoviews.plotting.bokeh.raster import ImageStackPlot
from holoviews.plotting.bokeh.util import BOKEH_GE_3_4_0

from .test_plot import TestBokehPlot, bokeh_renderer


class TestRasterPlot(TestBokehPlot):
    def test_image_colormapping(self):
        img = Image(np.random.rand(10, 10)).opts(logz=True)
        self._test_colormapping(img, 2, True)

    def test_image_boolean_array(self):
        img = Image(np.array([[True, False], [False, True]]))
        plot = bokeh_renderer.get_plot(img)
        cmapper = plot.handles["color_mapper"]
        source = plot.handles["source"]
        assert cmapper.low == 0
        assert cmapper.high == 1
        np.testing.assert_equal(source.data["image"][0], np.array([[0, 1], [1, 0]]))

    def test_nodata_array(self):
        img = Image(np.array([[0, 1], [2, 0]])).opts(nodata=0)
        plot = bokeh_renderer.get_plot(img)
        cmapper = plot.handles["color_mapper"]
        source = plot.handles["source"]
        assert cmapper.low == 1
        assert cmapper.high == 2
        np.testing.assert_equal(
            source.data["image"][0], np.array([[2, np.nan], [np.nan, 1]])
        )

    def test_nodata_array_uint(self):
        img = Image(np.array([[0, 1], [2, 0]], dtype="uint32")).opts(nodata=0)
        plot = bokeh_renderer.get_plot(img)
        cmapper = plot.handles["color_mapper"]
        source = plot.handles["source"]
        assert cmapper.low == 1
        assert cmapper.high == 2
        np.testing.assert_equal(
            source.data["image"][0], np.array([[2, np.nan], [np.nan, 1]])
        )

    def test_nodata_rgb(self):
        N = 2
        rgb_d = np.linspace(0, 1, N * N * 3).reshape(N, N, 3)
        rgb = RGB(rgb_d).redim.nodata(R=0)
        plot = bokeh_renderer.get_plot(rgb)
        image_data = plot.handles["source"].data["image"][0]
        assert (image_data == 0).sum() == 1

    def test_raster_invert_axes(self):
        arr = np.array([[0, 1, 2], [3, 4, 5]])
        raster = Raster(arr).opts(invert_axes=True)
        plot = bokeh_renderer.get_plot(raster)
        source = plot.handles["source"]

        np.testing.assert_equal(source.data["image"][0], arr.T)
        assert source.data["x"][0] == 0
        assert source.data["y"][0] == 0
        assert source.data["dw"][0] == 2
        assert source.data["dh"][0] == 3

    def test_image_invert_axes(self):
        arr = np.array([[0, 1, 2], [3, 4, 5]])
        raster = Image(arr).opts(invert_axes=True)
        plot = bokeh_renderer.get_plot(raster)
        source = plot.handles["source"]
        np.testing.assert_equal(source.data["image"][0], np.rot90(arr)[::-1, ::-1])
        assert source.data["x"][0] == -0.5
        assert source.data["y"][0] == -0.5
        assert source.data["dw"][0] == 1
        assert source.data["dh"][0] == 1

    def test_image_invert_xaxis(self):
        arr = np.random.rand(10, 10)
        img = Image(arr).opts(invert_xaxis=True)
        plot = bokeh_renderer.get_plot(img)
        x_range = plot.handles["x_range"]
        assert x_range.start == 0.5
        assert x_range.end == -0.5
        cdata = plot.handles["source"].data
        assert cdata["y"] == [-0.5]
        assert cdata["dh"] == [1.0]
        assert cdata["dw"] == [1.0]

        assert cdata["x"] == [-0.5]
        np.testing.assert_equal(cdata["image"][0], arr[::-1])

    def test_image_invert_yaxis(self):
        arr = np.random.rand(10, 10)
        img = Image(arr).opts(invert_yaxis=True)
        plot = bokeh_renderer.get_plot(img)
        y_range = plot.handles["y_range"]
        assert y_range.start == 0.5
        assert y_range.end == -0.5
        cdata = plot.handles["source"].data
        assert cdata["x"] == [-0.5]
        assert cdata["dh"] == [1.0]
        assert cdata["dw"] == [1.0]

        assert cdata["y"] == [-0.5]
        np.testing.assert_equal(cdata["image"][0], arr[::-1])

    def test_rgb_invert_xaxis(self):
        rgb = RGB(np.random.rand(10, 10, 3)).opts(invert_xaxis=True)
        plot = bokeh_renderer.get_plot(rgb)
        x_range = plot.handles["x_range"]
        assert x_range.start == 0.5
        assert x_range.end == -0.5
        cdata = plot.handles["source"].data
        assert cdata["y"] == [-0.5]
        assert cdata["dh"] == [1.0]
        assert cdata["dw"] == [1.0]
        assert cdata["x"] == [-0.5]

    def test_rgb_invert_yaxis(self):
        rgb = RGB(np.random.rand(10, 10, 3)).opts(invert_yaxis=True)
        plot = bokeh_renderer.get_plot(rgb)
        y_range = plot.handles["y_range"]
        assert y_range.start == 0.5
        assert y_range.end == -0.5
        cdata = plot.handles["source"].data
        assert cdata["x"] == [-0.5]
        assert cdata["dh"] == [1.0]
        assert cdata["dw"] == [1.0]
        assert cdata["y"] == [-0.5]

    def test_image_datetime_hover(self):
        xr = pytest.importorskip("xarray")
        ts = pd.Timestamp("2020-01-01")
        data = xr.Dataset(
            coords={"x": [-0.5, 0.5], "y": [-0.5, 0.5]},
            data_vars={
                "Count": (["y", "x"], [[0, 1], [2, 3]]),
                "Timestamp": (["y", "x"], [[ts, pd.NaT], [ts, ts]]),
            },
        )
        img = Image(data).opts(tools=["hover"])
        plot = bokeh_renderer.get_plot(img)

        hover = plot.handles["hover"]
        assert hover.tooltips[-1] == ("Timestamp", "@{Timestamp}{%F %T}")
        assert "@{Timestamp}" in hover.formatters

        if BOKEH_GE_3_4_0:  # https://github.com/bokeh/bokeh/issues/13598
            assert hover.formatters["@{Timestamp}"] == "datetime"
        else:
            assert isinstance(hover.formatters["@{Timestamp}"], CustomJSHover)

    def test_image_hover_with_custom_js(self):
        # Regression for https://github.com/holoviz/holoviews/issues/6101
        hover_tool = HoverTool(
            tooltips=[("x", "$x{custom}")], formatters={"x": CustomJSHover(code="return value + '2'")}
        )
        img = Image(np.ones(100).reshape(10, 10)).opts(tools=[hover_tool])
        plot = bokeh_renderer.get_plot(img)

        hover = plot.handles["hover"]
        assert hover.formatters == hover_tool.formatters

class _ImageStackBase(TestRasterPlot):
    __test__ = False

    def test_image_stack_tuple(self):
        x = np.arange(self.xsize)
        y = np.arange(self.ysize) + 5
        a, b, c = self.a, self.b, self.c
        img_stack = ImageStack((x, y, a, b, c), kdims=["x", "y"], vdims=["a", "b", "c"])
        plot = bokeh_renderer.get_plot(img_stack)
        source = plot.handles["source"]
        np.testing.assert_equal(source.data["image"][0][:, :, 0], a)
        np.testing.assert_equal(source.data["image"][0][:, :, 1], b)
        np.testing.assert_equal(source.data["image"][0][:, :, 2], c)
        assert source.data["x"][0] == -0.5
        assert source.data["y"][0] == 4.5
        assert source.data["dw"][0] == self.xsize
        assert source.data["dh"][0] == self.ysize
        assert isinstance(plot, ImageStackPlot)

    def test_image_stack_tuple_unspecified_dims(self):
        x = np.arange(self.xsize)
        y = np.arange(self.ysize) + 5
        a, b, c = self.a, self.b, self.c

        img_stack = ImageStack((x, y, a, b, c), kdims=["x", "y"])
        assert img_stack.vdims == ["level_0", "level_1", "level_2"]
        plot = bokeh_renderer.get_plot(img_stack)
        source = plot.handles["source"]
        np.testing.assert_equal(source.data["image"][0][:, :, 0], a)
        np.testing.assert_equal(source.data["image"][0][:, :, 1], b)
        np.testing.assert_equal(source.data["image"][0][:, :, 2], c)
        assert source.data["x"][0] == -0.5
        assert source.data["y"][0] == 4.5
        assert source.data["dw"][0] == self.xsize
        assert source.data["dh"][0] == self.ysize
        assert isinstance(plot, ImageStackPlot)

    def test_image_stack_dict(self):
        x = np.arange(self.xsize)
        y = np.arange(self.ysize) + 5
        a, b, c = self.a, self.b, self.c

        ds = {"x": x, "y": y, "a": a, "b": b, "c": c}
        img_stack = ImageStack(ds, kdims=["x", "y"], vdims=["a", "b", "c"])
        plot = bokeh_renderer.get_plot(img_stack)
        source = plot.handles["source"]
        np.testing.assert_equal(source.data["image"][0][:, :, 0], a)
        np.testing.assert_equal(source.data["image"][0][:, :, 1], b)
        np.testing.assert_equal(source.data["image"][0][:, :, 2], c)
        assert source.data["x"][0] == -0.5
        assert source.data["y"][0] == 4.5
        assert source.data["dw"][0] == self.xsize
        assert source.data["dh"][0] == self.ysize
        assert isinstance(plot, ImageStackPlot)

    def test_image_stack_dict_unspecified_dims(self):
        x = np.arange(self.xsize)
        y = np.arange(self.ysize) + 5
        a, b, c = self.a, self.b, self.c

        ds = {"x": x, "y": y, "a": a, "b": b, "c": c}
        img_stack = ImageStack(ds)
        plot = bokeh_renderer.get_plot(img_stack)
        source = plot.handles["source"]
        np.testing.assert_equal(source.data["image"][0][:, :, 0], a)
        np.testing.assert_equal(source.data["image"][0][:, :, 1], b)
        np.testing.assert_equal(source.data["image"][0][:, :, 2], c)
        assert source.data["x"][0] == -0.5
        assert source.data["y"][0] == 4.5
        assert source.data["dw"][0] == self.xsize
        assert source.data["dh"][0] == self.ysize
        assert isinstance(plot, ImageStackPlot)

    def test_image_stack_dict_no_kdims(self):
        a, b, c = self.a, self.b, self.c

        ds = {"a": a, "b": b, "c": c}
        img_stack = ImageStack(ds)
        plot = bokeh_renderer.get_plot(img_stack)
        source = plot.handles["source"]
        np.testing.assert_equal(source.data["image"][0][:, :, 0], a)
        np.testing.assert_equal(source.data["image"][0][:, :, 1], b)
        np.testing.assert_equal(source.data["image"][0][:, :, 2], c)
        assert source.data["x"][0] == -0.5
        assert source.data["y"][0] == -0.5
        assert source.data["dw"][0] == self.xsize
        assert source.data["dh"][0] == self.ysize
        assert isinstance(plot, ImageStackPlot)

    def test_image_stack_list(self):
        a, b, c = self.a, self.b, self.c

        ds = [a, b, c]
        img_stack = ImageStack(ds)
        plot = bokeh_renderer.get_plot(img_stack)
        source = plot.handles["source"]
        np.testing.assert_equal(source.data["image"][0][:, :, 0], a)
        np.testing.assert_equal(source.data["image"][0][:, :, 1], b)
        np.testing.assert_equal(source.data["image"][0][:, :, 2], c)
        assert source.data["x"][0] == -0.5
        assert source.data["y"][0] == -0.5
        assert source.data["dw"][0] == self.xsize
        assert source.data["dh"][0] == self.ysize
        assert isinstance(plot, ImageStackPlot)

    def test_image_stack_xarray_dataset(self):
        try:
            import xarray as xr
        except ImportError:
            raise SkipTest("xarray not available for core tests")

        x = np.arange(self.xsize)
        y = np.arange(self.ysize) + 5
        a, b, c = self.a, self.b, self.c

        ds = xr.Dataset(
            {"a": (["y", "x"], a), "b": (["y", "x"], b), "c": (["y", "x"], c)},
            coords={"x": x, "y": y},
        )
        img_stack = ImageStack(ds, kdims=["x", "y"])
        plot = bokeh_renderer.get_plot(img_stack)
        source = plot.handles["source"]
        np.testing.assert_equal(source.data["image"][0][:, :, 0], a)
        np.testing.assert_equal(source.data["image"][0][:, :, 1], b)
        np.testing.assert_equal(source.data["image"][0][:, :, 2], c)
        assert source.data["x"][0] == -0.5
        assert source.data["y"][0] == 4.5
        assert source.data["dw"][0] == self.xsize
        assert source.data["dh"][0] == self.ysize
        assert isinstance(plot, ImageStackPlot)

    def test_image_stack_xarray_dataarray(self):
        try:
            import xarray as xr
        except ImportError:
            raise SkipTest("xarray not available for core tests")

        x = np.arange(self.xsize)
        y = np.arange(self.ysize) + 5
        a, b, c = self.a, self.b, self.c

        da = xr.Dataset(
            {"a": (["y", "x"], a), "b": (["y", "x"], b), "c": (["y", "x"], c)},
            coords={"x": x, "y": y},
        ).to_array("level")
        img_stack = ImageStack(da, vdims=["level"])
        plot = bokeh_renderer.get_plot(img_stack)
        source = plot.handles["source"]
        np.testing.assert_equal(source.data["image"][0][:, :, 0], a)
        np.testing.assert_equal(source.data["image"][0][:, :, 1], b)
        np.testing.assert_equal(source.data["image"][0][:, :, 2], c)
        assert source.data["x"][0] == -0.5
        assert source.data["y"][0] == 4.5
        assert source.data["dw"][0] == self.xsize
        assert source.data["dh"][0] == self.ysize
        assert isinstance(plot, ImageStackPlot)

    def test_image_stack_invert_xaxis(self):
        x = np.arange(self.xsize)
        y = np.arange(self.ysize) + 5
        a, b, c = self.a, self.b, self.c

        img_stack = ImageStack((x, y, a, b, c), kdims=["x", "y"], vdims=["a", "b", "c"])
        plot = bokeh_renderer.get_plot(img_stack.opts(invert_xaxis=True))
        source = plot.handles["source"]
        np.testing.assert_equal(source.data["image"][0][:, :, 0], a)
        np.testing.assert_equal(source.data["image"][0][:, :, 1], b)
        np.testing.assert_equal(source.data["image"][0][:, :, 2], c)
        assert source.data["x"][0] == -0.5
        assert source.data["y"][0] == 4.5
        assert source.data["dw"][0] == self.xsize
        assert source.data["dh"][0] == self.ysize

    def test_image_stack_invert_yaxis(self):
        x = np.arange(self.xsize)
        y = np.arange(self.ysize) + 5
        a, b, c = self.a, self.b, self.c

        img_stack = ImageStack((x, y, a, b, c), kdims=["x", "y"], vdims=["a", "b", "c"])
        plot = bokeh_renderer.get_plot(img_stack.opts(invert_yaxis=True))
        source = plot.handles["source"]
        np.testing.assert_equal(source.data["image"][0][:, :, 0], a)
        np.testing.assert_equal(source.data["image"][0][:, :, 1], b)
        np.testing.assert_equal(source.data["image"][0][:, :, 2], c)
        assert source.data["x"][0] == -0.5
        assert source.data["y"][0] == 4.5
        assert source.data["dw"][0] == self.xsize
        assert source.data["dh"][0] == self.ysize

    def test_image_stack_invert_axes(self):
        x = np.arange(self.xsize)
        y = np.arange(self.ysize) + 5
        a, b, c = self.a, self.b, self.c

        img_stack = ImageStack((x, y, a, b, c), kdims=["x", "y"], vdims=["a", "b", "c"])
        plot = bokeh_renderer.get_plot(img_stack.opts(invert_axes=True))
        source = plot.handles["source"]

        np.testing.assert_equal(source.data["image"][0][:, :, 0].T, a)
        np.testing.assert_equal(source.data["image"][0][:, :, 1].T, b)
        np.testing.assert_equal(source.data["image"][0][:, :, 2].T, c)
        assert source.data["x"][0] == 4.5
        assert source.data["y"][0] == -0.5
        assert source.data["dw"][0] == self.ysize
        assert source.data["dh"][0] == self.xsize

    def test_image_stack_array(self):
        a, b, c = self.a, self.b, self.c

        data = np.dstack((a, b, c))

        img_stack = ImageStack(data, kdims=["x", "y"], vdims=["a", "b", "c"])
        plot = bokeh_renderer.get_plot(img_stack)
        source = plot.handles["source"]
        np.testing.assert_equal(source.data["image"][0][:, :, 0], a)
        np.testing.assert_equal(source.data["image"][0][:, :, 1], b)
        np.testing.assert_equal(source.data["image"][0][:, :, 2], c)
        assert source.data["x"][0] == -0.5
        assert source.data["y"][0] == -0.5
        assert source.data["dw"][0] == self.xsize
        assert source.data["dh"][0] == self.ysize
        assert isinstance(plot, ImageStackPlot)

    def test_image_stack_tuple_single_3darray(self):
        x = np.arange(self.xsize)
        y = np.arange(self.ysize) + 5
        a, b, c = self.a, self.b, self.c

        data = (x, y, np.dstack((a, b, c)))

        img_stack = ImageStack(data, kdims=["x", "y"], vdims=["a", "b", "c"])
        plot = bokeh_renderer.get_plot(img_stack)
        source = plot.handles["source"]
        np.testing.assert_equal(source.data["image"][0][:, :, 0], a)
        np.testing.assert_equal(source.data["image"][0][:, :, 1], b)
        np.testing.assert_equal(source.data["image"][0][:, :, 2], c)
        assert source.data["x"][0] == -0.5
        assert source.data["y"][0] == 4.5
        assert source.data["dw"][0] == self.xsize
        assert source.data["dh"][0] == self.ysize
        assert isinstance(plot, ImageStackPlot)

    def test_image_stack_dict_cmap(self):
        x = np.arange(0, 3)
        y = np.arange(5, 8)
        a = np.array([[np.nan, np.nan, 1], [np.nan] * 3, [np.nan] * 3])
        b = np.array([[np.nan] * 3, [1, 1, np.nan], [np.nan] * 3])
        c = np.array([[np.nan] * 3, [np.nan] * 3, [1, 1, 1]])

        img_stack = ImageStack((x, y, a, b, c), kdims=["x", "y"], vdims=["b", "a", "c"])
        img_stack.opts(cmap={"c": "yellow", "a": "red", "b": "green"})
        plot = bokeh_renderer.get_plot(img_stack)
        source = plot.handles["source"]
        np.testing.assert_equal(source.data["image"][0][:, :, 0], a)
        np.testing.assert_equal(source.data["image"][0][:, :, 1], b)
        np.testing.assert_equal(source.data["image"][0][:, :, 2], c)
        assert plot.handles["color_mapper"].palette == ["green", "red", "yellow"]

    def test_image_stack_dict_cmap_missing(self):
        x = np.arange(0, 3)
        y = np.arange(5, 8)
        a = np.array([[np.nan, np.nan, 1], [np.nan] * 3, [np.nan] * 3])
        b = np.array([[np.nan] * 3, [1, 1, np.nan], [np.nan] * 3])
        c = np.array([[np.nan] * 3, [np.nan] * 3, [1, 1, 1]])

        img_stack = ImageStack((x, y, a, b, c), kdims=["x", "y"], vdims=["b", "a", "c"])
        with pytest.raises(ValueError, match="must have the same value dimensions"):
            img_stack.opts(cmap={"c": "yellow", "a": "red"})
            bokeh_renderer.get_plot(img_stack)


class TestImageStackEven(_ImageStackBase):
    __test__ = True

    def setUp(self):
        self.a = np.array([[np.nan, np.nan, 1], [np.nan] * 3, [np.nan] * 3])
        self.b = np.array([[np.nan] * 3, [1, 1, np.nan], [np.nan] * 3])
        self.c = np.array([[np.nan] * 3, [np.nan] * 3, [1, 1, 1]])
        self.ysize = 3
        self.xsize = 3
        super().setUp()


class TestImageStackUneven1(_ImageStackBase):
    __test__ = True

    def setUp(self):
        self.a = np.array(
            [[np.nan, np.nan, 1], [np.nan] * 3, [np.nan] * 3, [np.nan] * 3]
        )
        self.b = np.array([[np.nan] * 3, [1, 1, np.nan], [np.nan] * 3, [np.nan] * 3])
        self.c = np.array([[np.nan] * 3, [np.nan] * 3, [1, 1, 1], [np.nan] * 3])
        self.ysize = 4
        self.xsize = 3
        super().setUp()


class TestImageStackUneven2(_ImageStackBase):
    __test__ = True

    def setUp(self):
        self.a = np.array([[np.nan, np.nan, 1, np.nan], [np.nan] * 4, [np.nan] * 4])
        self.b = np.array([[np.nan] * 4, [1, 1, np.nan, np.nan], [np.nan] * 4])
        self.c = np.array([[np.nan] * 4, [np.nan] * 4, [1, 1, 1, 1]])
        self.ysize = 3
        self.xsize = 4
        super().setUp()
