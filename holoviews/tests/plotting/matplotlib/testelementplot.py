import numpy as np

from holoviews.core.spaces import DynamicMap
from holoviews.element import Image, Curve, Scatter, Scatter3D
from holoviews.streams import Stream

from .testplot import TestMPLPlot, mpl_renderer

try:
    from matplotlib.ticker import FormatStrFormatter, FuncFormatter, PercentFormatter
except:
    pass

class TestElementPlot(TestMPLPlot):

    def test_stream_cleanup(self):
        stream = Stream.define(str('Test'), test=1)()
        dmap = DynamicMap(lambda test: Curve([]), streams=[stream])
        plot = mpl_renderer.get_plot(dmap)
        self.assertTrue(bool(stream._subscribers))
        plot.cleanup()
        self.assertFalse(bool(stream._subscribers))

    def test_element_xlabel(self):
        element = Curve(range(10)).options(xlabel='custom x-label')
        axes = mpl_renderer.get_plot(element).handles['axis']
        self.assertEqual(axes.get_xlabel(), 'custom x-label')

    def test_element_ylabel(self):
        element = Curve(range(10)).options(ylabel='custom y-label')
        axes = mpl_renderer.get_plot(element).handles['axis']
        self.assertEqual(axes.get_ylabel(), 'custom y-label')

    def test_element_xformatter_string(self):
        curve = Curve(range(10)).options(xformatter='%d')
        plot = mpl_renderer.get_plot(curve)
        xaxis = plot.handles['axis'].xaxis
        xformatter = xaxis.get_major_formatter()
        self.assertIsInstance(xformatter, FormatStrFormatter)
        self.assertEqual(xformatter.fmt, '%d')

    def test_element_yformatter_string(self):
        curve = Curve(range(10)).options(yformatter='%d')
        plot = mpl_renderer.get_plot(curve)
        yaxis = plot.handles['axis'].yaxis
        yformatter = yaxis.get_major_formatter()
        self.assertIsInstance(yformatter, FormatStrFormatter)
        self.assertEqual(yformatter.fmt, '%d')

    def test_element_zformatter_string(self):
        curve = Scatter3D([]).options(zformatter='%d')
        plot = mpl_renderer.get_plot(curve)
        zaxis = plot.handles['axis'].zaxis
        zformatter = zaxis.get_major_formatter()
        self.assertIsInstance(zformatter, FormatStrFormatter)
        self.assertEqual(zformatter.fmt, '%d')

    def test_element_xformatter_function(self):
        def formatter(value):
            return str(value) + ' %'
        curve = Curve(range(10)).options(xformatter=formatter)
        plot = mpl_renderer.get_plot(curve)
        xaxis = plot.handles['axis'].xaxis
        xformatter = xaxis.get_major_formatter()
        self.assertIsInstance(xformatter, FuncFormatter)

    def test_element_yformatter_function(self):
        def formatter(value):
            return str(value) + ' %'
        curve = Curve(range(10)).options(yformatter=formatter)
        plot = mpl_renderer.get_plot(curve)
        yaxis = plot.handles['axis'].yaxis
        yformatter = yaxis.get_major_formatter()
        self.assertIsInstance(yformatter, FuncFormatter)

    def test_element_zformatter_function(self):
        def formatter(value):
            return str(value) + ' %'
        curve = Scatter3D([]).options(zformatter=formatter)
        plot = mpl_renderer.get_plot(curve)
        zaxis = plot.handles['axis'].zaxis
        zformatter = zaxis.get_major_formatter()
        self.assertIsInstance(zformatter, FuncFormatter)

    def test_element_xformatter_instance(self):
        formatter = PercentFormatter()
        curve = Curve(range(10)).options(xformatter=formatter)
        plot = mpl_renderer.get_plot(curve)
        xaxis = plot.handles['axis'].xaxis
        xformatter = xaxis.get_major_formatter()
        self.assertIs(xformatter, formatter)

    def test_element_yformatter_instance(self):
        formatter = PercentFormatter()
        curve = Curve(range(10)).options(yformatter=formatter)
        plot = mpl_renderer.get_plot(curve)
        yaxis = plot.handles['axis'].yaxis
        yformatter = yaxis.get_major_formatter()
        self.assertIs(yformatter, formatter)

    def test_element_zformatter_instance(self):
        formatter = PercentFormatter()
        curve = Scatter3D([]).options(zformatter=formatter)
        plot = mpl_renderer.get_plot(curve)
        zaxis = plot.handles['axis'].zaxis
        zformatter = zaxis.get_major_formatter()
        self.assertIs(zformatter, formatter)

        

class TestColorbarPlot(TestMPLPlot):

    def test_colormapper_unsigned_int(self):
        img = Image(np.array([[1, 1, 1, 2], [2, 2, 3, 4]]).astype('uint16'))
        plot = mpl_renderer.get_plot(img)
        artist = plot.handles['artist']
        self.assertEqual(artist.get_clim(), (1, 4))

    def test_colormapper_symmetric(self):
        img = Image(np.array([[0, 1], [2, 3]])).options(symmetric=True)
        plot = mpl_renderer.get_plot(img)
        artist = plot.handles['artist']
        self.assertEqual(artist.get_clim(), (-3, 3))

    def test_colormapper_clims(self):
        img = Image(np.array([[0, 1], [2, 3]])).options(clims=(0, 4))
        plot = mpl_renderer.get_plot(img)
        artist = plot.handles['artist']
        self.assertEqual(artist.get_clim(), (0, 4))

    def test_colormapper_color_levels(self):
        img = Image(np.array([[0, 1], [2, 3]])).options(color_levels=5)
        plot = mpl_renderer.get_plot(img)
        artist = plot.handles['artist']
        self.assertEqual(len(artist.cmap.colors), 5)

    def test_colormapper_transparent_nan(self):
        img = Image(np.array([[0, 1], [2, 3]])).options(clipping_colors={'NaN': 'transparent'})
        plot = mpl_renderer.get_plot(img)
        cmap = plot.handles['artist'].cmap
        self.assertEqual(cmap._rgba_bad, (1.0, 1.0, 1.0, 0))

    def test_colormapper_min_max_colors(self):
        img = Image(np.array([[0, 1], [2, 3]])).options(clipping_colors={'min': 'red', 'max': 'blue'})
        plot = mpl_renderer.get_plot(img)
        cmap = plot.handles['artist'].cmap
        self.assertEqual(cmap._rgba_under, (1.0, 0, 0, 1))
        self.assertEqual(cmap._rgba_over, (0, 0, 1.0, 1))

    def test_colorbar_label(self):
        scatter = Scatter(np.random.rand(100, 3), vdims=["y", "c"]).options(color_index=2, colorbar=True)
        plot = mpl_renderer.get_plot(scatter)
        cbar_ax = plot.handles['cax']
        self.assertEqual(cbar_ax.get_ylabel(), 'c')

    def test_colorbar_label_style_mapping(self):
        scatter = Scatter(np.random.rand(100, 3), vdims=["y", "color"]).options(color='color', colorbar=True)
        plot = mpl_renderer.get_plot(scatter)
        cbar_ax = plot.handles['cax']
        self.assertEqual(cbar_ax.get_ylabel(), 'color')
