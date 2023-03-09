import numpy as np

from holoviews.element import Raster, Image, RGB

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

    def test_raster_invert_axes(self):
        arr = np.array([[0, 1, 2], [3, 4,  5]])
        raster = Raster(arr).opts(invert_axes=True)
        plot = bokeh_renderer.get_plot(raster)
        source = plot.handles['source']
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
        self.assertEqual(cdata['x'], [0.5])
        self.assertEqual(cdata['y'], [-0.5])
        self.assertEqual(cdata['dh'], [1.0])
        self.assertEqual(cdata['dw'], [1.0])
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
        self.assertEqual(cdata['y'], [0.5])
        self.assertEqual(cdata['dh'], [1.0])
        self.assertEqual(cdata['dw'], [1.0])
        self.assertEqual(cdata['image'][0], arr)

    def test_rgb_invert_xaxis(self):
        rgb = RGB(np.random.rand(10, 10, 3)).opts(invert_xaxis=True)
        plot = bokeh_renderer.get_plot(rgb)
        x_range = plot.handles['x_range']
        self.assertEqual(x_range.start, 0.5)
        self.assertEqual(x_range.end, -0.5)
        cdata = plot.handles['source'].data
        self.assertEqual(cdata['x'], [0.5])
        self.assertEqual(cdata['y'], [-0.5])
        self.assertEqual(cdata['dh'], [1.0])
        self.assertEqual(cdata['dw'], [1.0])

    def test_rgb_invert_yaxis(self):
        rgb = RGB(np.random.rand(10, 10, 3)).opts(invert_yaxis=True)
        plot = bokeh_renderer.get_plot(rgb)
        y_range = plot.handles['y_range']
        self.assertEqual(y_range.start, 0.5)
        self.assertEqual(y_range.end, -0.5)
        cdata = plot.handles['source'].data
        self.assertEqual(cdata['x'], [-0.5])
        self.assertEqual(cdata['y'], [0.5])
        self.assertEqual(cdata['dh'], [1.0])
        self.assertEqual(cdata['dw'], [1.0])
