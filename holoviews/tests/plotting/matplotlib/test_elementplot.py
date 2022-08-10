import numpy as np

from holoviews.core.spaces import DynamicMap
from holoviews.element import Image, Curve, Scatter, Scatter3D
from holoviews.streams import Stream

from .test_plot import TestMPLPlot, mpl_renderer

from matplotlib.ticker import FormatStrFormatter, FuncFormatter, PercentFormatter

class TestElementPlot(TestMPLPlot):

    def test_stream_cleanup(self):
        stream = Stream.define(str('Test'), test=1)()
        dmap = DynamicMap(lambda test: Curve([]), streams=[stream])
        plot = mpl_renderer.get_plot(dmap)
        self.assertTrue(bool(stream._subscribers))
        plot.cleanup()
        self.assertFalse(bool(stream._subscribers))

    def test_element_hooks(self):
        def hook(plot, element):
            plot.handles['title'].set_text('Called')
        curve = Curve(range(10), label='Not Called').opts(hooks=[hook])
        plot = mpl_renderer.get_plot(curve)
        self.assertEqual(plot.handles['title'].get_text(), 'Called')

    def test_element_font_scaling(self):
        curve = Curve(range(10)).opts(fontscale=2, title='A title')
        plot = mpl_renderer.get_plot(curve)
        ax = plot.handles['axis']
        self.assertEqual(ax.title.get_fontsize(), 24)
        self.assertEqual(ax.xaxis.label.get_fontsize(), 20)
        self.assertEqual(ax.yaxis.label.get_fontsize(), 20)
        self.assertEqual(ax.xaxis._major_tick_kw['labelsize'], 20)
        self.assertEqual(ax.yaxis._major_tick_kw['labelsize'], 20)

    def test_element_font_scaling_fontsize_override_common(self):
        curve = Curve(range(10)).opts(fontscale=2, fontsize=14, title='A title')
        plot = mpl_renderer.get_plot(curve)
        ax = plot.handles['axis']
        self.assertEqual(ax.title.get_fontsize(), 28)
        self.assertEqual(ax.xaxis.label.get_fontsize(), 28)
        self.assertEqual(ax.yaxis.label.get_fontsize(), 28)
        self.assertEqual(ax.xaxis._major_tick_kw['labelsize'], 20)
        self.assertEqual(ax.yaxis._major_tick_kw['labelsize'], 20)

    def test_element_font_scaling_fontsize_override_specific(self):
        curve = Curve(range(10)).opts(
            fontscale=2, fontsize={'title': 16, 'xticks': 12, 'xlabel': 6}, title='A title')
        plot = mpl_renderer.get_plot(curve)
        ax = plot.handles['axis']
        self.assertEqual(ax.title.get_fontsize(), 32)
        self.assertEqual(ax.xaxis.label.get_fontsize(), 12)
        self.assertEqual(ax.yaxis.label.get_fontsize(), 20)
        self.assertEqual(ax.xaxis._major_tick_kw['labelsize'], 24)
        self.assertEqual(ax.yaxis._major_tick_kw['labelsize'], 20)

    def test_element_no_xaxis_yaxis(self):
        element = Curve(range(10)).opts(xaxis=None, yaxis=None)
        axes = mpl_renderer.get_plot(element).handles['axis']
        xaxis = axes.get_xaxis()
        yaxis = axes.get_yaxis()
        self.assertEqual(xaxis.get_visible(), False)
        self.assertEqual(yaxis.get_visible(), False)

    def test_element_xlabel(self):
        element = Curve(range(10)).opts(xlabel='custom x-label')
        axes = mpl_renderer.get_plot(element).handles['axis']
        self.assertEqual(axes.get_xlabel(), 'custom x-label')

    def test_element_ylabel(self):
        element = Curve(range(10)).opts(ylabel='custom y-label')
        axes = mpl_renderer.get_plot(element).handles['axis']
        self.assertEqual(axes.get_ylabel(), 'custom y-label')

    def test_element_xformatter_string(self):
        curve = Curve(range(10)).opts(xformatter='%d')
        plot = mpl_renderer.get_plot(curve)
        xaxis = plot.handles['axis'].xaxis
        xformatter = xaxis.get_major_formatter()
        self.assertIsInstance(xformatter, FormatStrFormatter)
        self.assertEqual(xformatter.fmt, '%d')

    def test_element_yformatter_string(self):
        curve = Curve(range(10)).opts(yformatter='%d')
        plot = mpl_renderer.get_plot(curve)
        yaxis = plot.handles['axis'].yaxis
        yformatter = yaxis.get_major_formatter()
        self.assertIsInstance(yformatter, FormatStrFormatter)
        self.assertEqual(yformatter.fmt, '%d')

    def test_element_zformatter_string(self):
        curve = Scatter3D([]).opts(zformatter='%d')
        plot = mpl_renderer.get_plot(curve)
        zaxis = plot.handles['axis'].zaxis
        zformatter = zaxis.get_major_formatter()
        self.assertIsInstance(zformatter, FormatStrFormatter)
        self.assertEqual(zformatter.fmt, '%d')

    def test_element_xformatter_function(self):
        def formatter(value):
            return str(value) + ' %'
        curve = Curve(range(10)).opts(xformatter=formatter)
        plot = mpl_renderer.get_plot(curve)
        xaxis = plot.handles['axis'].xaxis
        xformatter = xaxis.get_major_formatter()
        self.assertIsInstance(xformatter, FuncFormatter)

    def test_element_yformatter_function(self):
        def formatter(value):
            return str(value) + ' %'
        curve = Curve(range(10)).opts(yformatter=formatter)
        plot = mpl_renderer.get_plot(curve)
        yaxis = plot.handles['axis'].yaxis
        yformatter = yaxis.get_major_formatter()
        self.assertIsInstance(yformatter, FuncFormatter)

    def test_element_zformatter_function(self):
        def formatter(value):
            return str(value) + ' %'
        curve = Scatter3D([]).opts(zformatter=formatter)
        plot = mpl_renderer.get_plot(curve)
        zaxis = plot.handles['axis'].zaxis
        zformatter = zaxis.get_major_formatter()
        self.assertIsInstance(zformatter, FuncFormatter)

    def test_element_xformatter_instance(self):
        formatter = PercentFormatter()
        curve = Curve(range(10)).opts(xformatter=formatter)
        plot = mpl_renderer.get_plot(curve)
        xaxis = plot.handles['axis'].xaxis
        xformatter = xaxis.get_major_formatter()
        self.assertIs(xformatter, formatter)

    def test_element_yformatter_instance(self):
        formatter = PercentFormatter()
        curve = Curve(range(10)).opts(yformatter=formatter)
        plot = mpl_renderer.get_plot(curve)
        yaxis = plot.handles['axis'].yaxis
        yformatter = yaxis.get_major_formatter()
        self.assertIs(yformatter, formatter)

    def test_element_zformatter_instance(self):
        formatter = PercentFormatter()
        curve = Scatter3D([]).opts(zformatter=formatter)
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
        img = Image(np.array([[0, 1], [2, 3]])).opts(symmetric=True)
        plot = mpl_renderer.get_plot(img)
        artist = plot.handles['artist']
        self.assertEqual(artist.get_clim(), (-3, 3))

    def test_colormapper_clims(self):
        img = Image(np.array([[0, 1], [2, 3]])).opts(clims=(0, 4))
        plot = mpl_renderer.get_plot(img)
        artist = plot.handles['artist']
        self.assertEqual(artist.get_clim(), (0, 4))

    def test_colormapper_color_levels(self):
        img = Image(np.array([[0, 1], [2, 3]])).opts(color_levels=5)
        plot = mpl_renderer.get_plot(img)
        artist = plot.handles['artist']
        self.assertEqual(len(artist.cmap.colors), 5)

    def test_colormapper_transparent_nan(self):
        img = Image(np.array([[0, 1], [2, 3]])).opts(clipping_colors={'NaN': 'transparent'})
        plot = mpl_renderer.get_plot(img)
        cmap = plot.handles['artist'].cmap
        self.assertEqual(cmap._rgba_bad, (1.0, 1.0, 1.0, 0))

    def test_colormapper_min_max_colors(self):
        img = Image(np.array([[0, 1], [2, 3]])).opts(clipping_colors={'min': 'red', 'max': 'blue'})
        plot = mpl_renderer.get_plot(img)
        cmap = plot.handles['artist'].cmap
        self.assertEqual(cmap._rgba_under, (1.0, 0, 0, 1))
        self.assertEqual(cmap._rgba_over, (0, 0, 1.0, 1))

    def test_colorbar_label(self):
        scatter = Scatter(np.random.rand(100, 3), vdims=["y", "color"]).opts(color_index=2, colorbar=True)
        plot = mpl_renderer.get_plot(scatter)
        cbar_ax = plot.handles['cax']
        self.assertEqual(cbar_ax.get_ylabel(), 'color')

    def test_colorbar_empty_clabel(self):
        img = Image(np.array([[1, 1, 1, 2], [2, 2, 3, 4]])).opts(clabel='', colorbar=True)
        plot = mpl_renderer.get_plot(img)
        colorbar = plot.handles['cax']
        self.assertEqual(colorbar.get_label(), '')

    def test_colorbar_label_style_mapping(self):
        scatter = Scatter(np.random.rand(100, 3), vdims=["y", "color"]).opts(color='color', colorbar=True)
        plot = mpl_renderer.get_plot(scatter)
        cbar_ax = plot.handles['cax']
        self.assertEqual(cbar_ax.get_ylabel(), 'color')


class TestOverlayPlot(TestMPLPlot):

    def test_overlay_legend_opts(self):
        overlay = (
            Curve(np.random.randn(10).cumsum(), label='A') *
            Curve(np.random.randn(10).cumsum(), label='B')
        ).opts(legend_opts={'framealpha': 0.5, 'facecolor': 'red'})
        plot = mpl_renderer.get_plot(overlay)
        legend_frame = plot.handles['legend'].get_frame()
        self.assertEqual(legend_frame.get_alpha(), 0.5)
        self.assertEqual(legend_frame.get_facecolor(), (1.0, 0.0, 0.0, 0.5))
