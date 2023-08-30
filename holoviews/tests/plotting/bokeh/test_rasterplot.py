import pytest
import numpy as np
import xarray as xr

from holoviews.element import Raster, Image, RGB, ImageStack
from holoviews.plotting.bokeh.util import bokeh3
from holoviews.plotting.bokeh.raster import ImageStackPlot

from .test_plot import TestBokehPlot, bokeh_renderer


class TestRasterPlot(TestBokehPlot):

    def test_image_colormapping(self):
        img = Image(np.random.rand(10, 10)).opts(logz=True)
        self._test_colormapping(img, 2, True)

    def test_image_boolean_array(self):
        img = Image(np.array([[True, False], [False, True]]))
        plot = bokeh_renderer.get_plot(img)
        cmapper = plot.handles['color_mapper']
        source = plot.handles['source']
        self.assertEqual(cmapper.low, 0)
        self.assertEqual(cmapper.high, 1)
        self.assertEqual(source.data['image'][0],
                         np.array([[0, 1], [1, 0]]))

    def test_nodata_array(self):
        img = Image(np.array([[0, 1], [2, 0]])).opts(nodata=0)
        plot = bokeh_renderer.get_plot(img)
        cmapper = plot.handles['color_mapper']
        source = plot.handles['source']
        self.assertEqual(cmapper.low, 1)
        self.assertEqual(cmapper.high, 2)
        self.assertEqual(source.data['image'][0],
                         np.array([[2, np.NaN], [np.NaN, 1]]))

    def test_nodata_array_uint(self):
        img = Image(np.array([[0, 1], [2, 0]], dtype='uint32')).opts(nodata=0)
        plot = bokeh_renderer.get_plot(img)
        cmapper = plot.handles['color_mapper']
        source = plot.handles['source']
        self.assertEqual(cmapper.low, 1)
        self.assertEqual(cmapper.high, 2)
        self.assertEqual(source.data['image'][0],
                         np.array([[2, np.NaN], [np.NaN, 1]]))

    def test_nodata_rgb(self):
        N = 2
        rgb_d = np.linspace(0, 1, N * N * 3).reshape(N, N, 3)
        rgb = RGB(rgb_d).redim.nodata(R=0)
        plot = bokeh_renderer.get_plot(rgb)
        image_data = plot.handles["source"].data["image"][0]
        # Image sets nan-values to 0
        assert (image_data == 0).sum() == 1

    def test_raster_invert_axes(self):
        arr = np.array([[0, 1, 2], [3, 4,  5]])
        raster = Raster(arr).opts(invert_axes=True)
        plot = bokeh_renderer.get_plot(raster)
        source = plot.handles['source']

        if bokeh3:
            self.assertEqual(source.data['image'][0], arr.T)
            self.assertEqual(source.data['x'][0], 0)
            self.assertEqual(source.data['y'][0], 0)
            self.assertEqual(source.data['dw'][0], 2)
            self.assertEqual(source.data['dh'][0], 3)
        else:
            self.assertEqual(source.data['image'][0], np.rot90(arr))
            self.assertEqual(source.data['x'][0], 0)
            self.assertEqual(source.data['y'][0], 3)
            self.assertEqual(source.data['dw'][0], 2)
            self.assertEqual(source.data['dh'][0], 3)

    def test_image_invert_axes(self):
        arr = np.array([[0, 1, 2], [3, 4,  5]])
        raster = Image(arr).opts(invert_axes=True)
        plot = bokeh_renderer.get_plot(raster)
        source = plot.handles['source']
        self.assertEqual(source.data['image'][0], np.rot90(arr)[::-1, ::-1])
        self.assertEqual(source.data['x'][0], -.5)
        self.assertEqual(source.data['y'][0], -.5)
        self.assertEqual(source.data['dw'][0], 1)
        self.assertEqual(source.data['dh'][0], 1)

    def test_image_invert_xaxis(self):
        arr = np.random.rand(10, 10)
        img = Image(arr).opts(invert_xaxis=True)
        plot = bokeh_renderer.get_plot(img)
        x_range = plot.handles['x_range']
        self.assertEqual(x_range.start, 0.5)
        self.assertEqual(x_range.end, -0.5)
        cdata = plot.handles['source'].data
        self.assertEqual(cdata['y'], [-0.5])
        self.assertEqual(cdata['dh'], [1.0])
        self.assertEqual(cdata['dw'], [1.0])

        if bokeh3:
            self.assertEqual(cdata['x'], [-0.5])
            self.assertEqual(cdata['image'][0], arr[::-1])
        else:
            self.assertEqual(cdata['x'], [0.5])
            self.assertEqual(cdata['image'][0], arr[::-1, ::-1])

    def test_image_invert_yaxis(self):
        arr = np.random.rand(10, 10)
        img = Image(arr).opts(invert_yaxis=True)
        plot = bokeh_renderer.get_plot(img)
        y_range = plot.handles['y_range']
        self.assertEqual(y_range.start, 0.5)
        self.assertEqual(y_range.end, -0.5)
        cdata = plot.handles['source'].data
        self.assertEqual(cdata['x'], [-0.5])
        self.assertEqual(cdata['dh'], [1.0])
        self.assertEqual(cdata['dw'], [1.0])

        if bokeh3:
            self.assertEqual(cdata['y'], [-0.5])
            self.assertEqual(cdata['image'][0], arr[::-1])
        else:
            self.assertEqual(cdata['y'], [0.5])
            self.assertEqual(cdata['image'][0], arr)

    def test_rgb_invert_xaxis(self):
        rgb = RGB(np.random.rand(10, 10, 3)).opts(invert_xaxis=True)
        plot = bokeh_renderer.get_plot(rgb)
        x_range = plot.handles['x_range']
        self.assertEqual(x_range.start, 0.5)
        self.assertEqual(x_range.end, -0.5)
        cdata = plot.handles['source'].data
        self.assertEqual(cdata['y'], [-0.5])
        self.assertEqual(cdata['dh'], [1.0])
        self.assertEqual(cdata['dw'], [1.0])

        if bokeh3:
            self.assertEqual(cdata['x'], [-0.5])
        else:
            self.assertEqual(cdata['x'], [0.5])

    def test_rgb_invert_yaxis(self):
        rgb = RGB(np.random.rand(10, 10, 3)).opts(invert_yaxis=True)
        plot = bokeh_renderer.get_plot(rgb)
        y_range = plot.handles['y_range']
        self.assertEqual(y_range.start, 0.5)
        self.assertEqual(y_range.end, -0.5)
        cdata = plot.handles['source'].data
        self.assertEqual(cdata['x'], [-0.5])
        self.assertEqual(cdata['dh'], [1.0])
        self.assertEqual(cdata['dw'], [1.0])

        if bokeh3:
            self.assertEqual(cdata['y'], [-0.5])
        else:
            self.assertEqual(cdata['y'], [0.5])

    def test_image_stack(self):
        x = np.arange(0, 3)
        y = np.arange(5, 8)
        a = np.array([[np.nan, np.nan, 1], [np.nan] * 3, [np.nan] * 3])
        b = np.array([[np.nan] * 3, [1, 1, np.nan], [np.nan] * 3])
        c = np.array([[np.nan] * 3, [np.nan] * 3, [1, 1, 1]])

        img_stack = ImageStack((x, y, a, b, c), kdims=["x", "y"], vdims=["a", "b", "c"])
        plot = bokeh_renderer.get_plot(img_stack)
        source = plot.handles['source']
        self.assertEqual(source.data['image'][0][0], a.T)
        self.assertEqual(source.data['image'][0][1], b.T)
        self.assertEqual(source.data['image'][0][2], c.T)
        self.assertEqual(source.data['x'][0], -0.5)
        self.assertEqual(source.data['y'][0], 4.5)
        self.assertEqual(source.data['dw'][0], 3)
        self.assertEqual(source.data['dh'][0], 3)
        assert isinstance(plot, ImageStackPlot)

    def test_image_stack_xarray(self):
        x = np.arange(0, 3)
        y = np.arange(5, 8)
        a = np.array([[np.nan, np.nan, 1], [np.nan] * 3, [np.nan] * 3])
        b = np.array([[np.nan] * 3, [1, 1, np.nan], [np.nan] * 3])
        c = np.array([[np.nan] * 3, [np.nan] * 3, [1, 1, 1]])

        ds = xr.Dataset(
            {"a": (["x", "y"], a), "b": (["x", "y"], b), "c": (["x", "y"], c)},
            coords={"x": x, "y": y},
        )
        at, bt, ct = np.dstack([a, b, c]).reshape(3, 9).T.reshape(3, 3, 3)
        img_stack = ImageStack(ds, kdims=["x", "y"])
        plot = bokeh_renderer.get_plot(img_stack)
        source = plot.handles['source']
        self.assertEqual(source.data['image'][0][0], at)
        self.assertEqual(source.data['image'][0][1], bt)
        self.assertEqual(source.data['image'][0][2], ct)
        self.assertEqual(source.data['x'][0], -0.5)
        self.assertEqual(source.data['y'][0], 4.5)
        self.assertEqual(source.data['dw'][0], 3)
        self.assertEqual(source.data['dh'][0], 3)
        assert isinstance(plot, ImageStackPlot)

    def test_image_stack_dict(self):
        x = np.arange(0, 3)
        y = np.arange(5, 8)
        a = np.array([[np.nan, np.nan, 1], [np.nan] * 3, [np.nan] * 3])
        b = np.array([[np.nan] * 3, [1, 1, np.nan], [np.nan] * 3])
        c = np.array([[np.nan] * 3, [np.nan] * 3, [1, 1, 1]])

        ds = {"x": x, "y": y, "a": a, "b": b, "c": c}
        img_stack = ImageStack(ds, kdims=["x", "y"], vdims=["a", "b", "c"])
        plot = bokeh_renderer.get_plot(img_stack)
        source = plot.handles['source']
        self.assertEqual(source.data['image'][0][0], a.T)
        self.assertEqual(source.data['image'][0][1], b.T)
        self.assertEqual(source.data['image'][0][2], c.T)
        self.assertEqual(source.data['x'][0], -0.5)
        self.assertEqual(source.data['y'][0], 4.5)
        self.assertEqual(source.data['dw'][0], 3)
        self.assertEqual(source.data['dh'][0], 3)
        assert isinstance(plot, ImageStackPlot)

    def test_image_stack_extra_dims(self):
        x = np.arange(0, 3)
        y = np.arange(5, 8)
        a = np.array([[np.nan, np.nan, 1], [np.nan] * 3, [np.nan] * 3])
        b = np.array([[np.nan] * 3, [1, 1, np.nan], [np.nan] * 3])
        c = np.array([[np.nan] * 3, [np.nan] * 3, [1, 1, 1]])

        with pytest.raises(ValueError, match="Detected 2 unused dimensions"):
            ImageStack((x, y, a, b, c), kdims=["x", "y"])

    def test_image_stack_invert_xaxis(self):
        x = np.arange(0, 3)
        y = np.arange(5, 8)
        a = np.array([[np.nan, np.nan, 1], [np.nan] * 3, [np.nan] * 3])
        b = np.array([[np.nan] * 3, [1, 1, np.nan], [np.nan] * 3])
        c = np.array([[np.nan] * 3, [np.nan] * 3, [1, 1, 1]])

        img_stack = ImageStack((x, y, a, b, c), kdims=["x", "y"], vdims=["a", "b", "c"])
        plot = bokeh_renderer.get_plot(img_stack.opts(invert_xaxis=True))
        source = plot.handles['source']
        self.assertEqual(source.data['image'][0][0], a.T)
        self.assertEqual(source.data['image'][0][1], b.T)
        self.assertEqual(source.data['image'][0][2], c.T)
        self.assertEqual(source.data['x'][0], -0.5)
        self.assertEqual(source.data['y'][0], 4.5)
        self.assertEqual(source.data['dw'][0], 3)
        self.assertEqual(source.data['dh'][0], 3)

    def test_image_stack_invert_yaxis(self):
        x = np.arange(0, 3)
        y = np.arange(5, 8)
        a = np.array([[np.nan, np.nan, 1], [np.nan] * 3, [np.nan] * 3])
        b = np.array([[np.nan] * 3, [1, 1, np.nan], [np.nan] * 3])
        c = np.array([[np.nan] * 3, [np.nan] * 3, [1, 1, 1]])

        img_stack = ImageStack((x, y, a, b, c), kdims=["x", "y"], vdims=["a", "b", "c"])
        plot = bokeh_renderer.get_plot(img_stack.opts(invert_yaxis=True))
        source = plot.handles['source']
        self.assertEqual(source.data['image'][0][0], a.T)
        self.assertEqual(source.data['image'][0][1], b.T)
        self.assertEqual(source.data['image'][0][2], c.T)
        self.assertEqual(source.data['x'][0], -0.5)
        self.assertEqual(source.data['y'][0], 4.5)
        self.assertEqual(source.data['dw'][0], 3)
        self.assertEqual(source.data['dh'][0], 3)

    def test_image_stack_invert_axes(self):
        x = np.arange(0, 3)
        y = np.arange(5, 8)
        a = np.array([[np.nan, np.nan, 1], [np.nan] * 3, [np.nan] * 3])
        b = np.array([[np.nan] * 3, [1, 1, np.nan], [np.nan] * 3])
        c = np.array([[np.nan] * 3, [np.nan] * 3, [1, 1, 1]])

        img_stack = ImageStack((x, y, a, b, c), kdims=["x", "y"], vdims=["a", "b", "c"])
        plot = bokeh_renderer.get_plot(img_stack.opts(invert_axes=True))
        source = plot.handles['source']

        at, bt, ct = np.dstack([a, b, c]).reshape(3, 9).T.reshape(3, 3, 3)
        self.assertEqual(source.data['image'][0][0], at)
        self.assertEqual(source.data['image'][0][1], bt)
        self.assertEqual(source.data['image'][0][2], ct)
        self.assertEqual(source.data['x'][0], 4.5)
        self.assertEqual(source.data['y'][0], -0.5)
        self.assertEqual(source.data['dw'][0], 3)
        self.assertEqual(source.data['dh'][0], 3)
