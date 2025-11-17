import re

import numpy as np
import pandas as pd
import pytest
from matplotlib import style
from matplotlib.projections import PolarAxes
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, PercentFormatter

from holoviews.core.dimension import Dimension
from holoviews.core.spaces import DynamicMap
from holoviews.element import Curve, HeatMap, Image, QuadMesh, Scatter, Scatter3D
from holoviews.streams import Stream

from ...utils import LoggingComparisonTestCase
from .test_plot import TestMPLPlot, mpl_renderer


class TestElementPlot(LoggingComparisonTestCase, TestMPLPlot):

    def test_stream_cleanup(self):
        stream = Stream.define('Test', test=1)()
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
        curve = Curve(range(10)).options(fontscale=2, title='A title')
        with style.context(
            {
                "axes.labelsize": 10,
                "axes.titlesize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
            }
        ):
            plot = mpl_renderer.get_plot(curve)
        ax = plot.handles['axis']
        self.assertEqual(ax.title.get_fontsize(), 24)
        self.assertEqual(ax.xaxis.label.get_fontsize(), 20)
        self.assertEqual(ax.yaxis.label.get_fontsize(), 20)
        self.assertEqual(ax.xaxis._major_tick_kw['labelsize'], 20)
        self.assertEqual(ax.yaxis._major_tick_kw['labelsize'], 20)

    def test_element_font_scaling_fontsize_override_common(self):
        curve = Curve(range(10)).options(fontscale=2, fontsize=14, title='A title')
        with style.context(
            {
                "axes.labelsize": 10,
                "axes.titlesize": 12,
                "font.size": 10,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
            }
        ):
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
        with style.context(
            {
                "axes.labelsize": 10,
                "axes.titlesize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
            }
        ):
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

    def test_element_polar_xlimits(self):
        theta = np.arange(0, 5.4, 0.1)
        r = np.ones(len(theta))
        scatter = Scatter((theta, r), 'theta', 'r').opts(projection='polar')
        plot = mpl_renderer.get_plot(scatter)
        ax = plot.handles['axis']
        self.assertIsInstance(ax, PolarAxes)
        self.assertEqual(ax.get_xlim(), (0, 2 * np.pi))

    #################################################################
    # Custom opts tests
    #################################################################

    def test_element_backend_opts(self):
        heat_map = HeatMap([(1, 2, 3), (2, 3, 4), (3, 4, 5)]).opts(
            colorbar=True,
            backend_opts={
                "colorbar.set_label": "Testing",
                "colorbar.set_ticks": [3.5, 5],
                "colorbar.ax.yticklabels": ["A", "B"],
            },
        )
        plot = mpl_renderer.get_plot(heat_map)
        colorbar = plot.handles['cbar']
        self.assertEqual(colorbar.ax.yaxis.get_label().get_text(), "Testing")
        self.assertEqual(colorbar.get_ticks(), (3.5, 5))
        ticklabels = [ticklabel.get_text() for ticklabel in colorbar.ax.get_yticklabels()]
        self.assertEqual(ticklabels, ["A", "B"])

    def test_element_backend_opts_alias(self):
        heat_map = HeatMap([(1, 2, 3), (2, 3, 4), (3, 4, 5)]).opts(
            colorbar=True,
            backend_opts={
                "cbar.set_label": "Testing",
                "cbar.set_ticks": [3.5, 5],
                "cbar.ax.yticklabels": ["A", "B"]
            },
        )
        plot = mpl_renderer.get_plot(heat_map)
        colorbar = plot.handles['cbar']
        self.assertEqual(colorbar.ax.yaxis.get_label().get_text(), "Testing")
        self.assertEqual(colorbar.get_ticks(), (3.5, 5))
        ticklabels = [ticklabel.get_text() for ticklabel in colorbar.ax.get_yticklabels()]
        self.assertEqual(ticklabels, ["A", "B"])

    def test_element_backend_opts_method(self):
        a = Curve([1, 2, 3], label="a")
        b = Curve([1, 4, 9], label="b")
        curve = (a * b).opts(
            show_legend=True,
            backend_opts={
                "legend.frame_on": False,
            }
        )
        plot = mpl_renderer.get_plot(curve)
        legend = plot.handles['legend']
        self.assertFalse(legend.get_frame_on())

    def test_element_backend_opts_sequential_method(self):
        a = Curve([1, 2, 3], label="a")
        b = Curve([1, 4, 9], label="b")
        curve = (a * b).opts(
            show_legend=True,
            backend_opts={
                "legend.get_title().set_fontsize": 188,
            }
        )
        plot = mpl_renderer.get_plot(curve)
        legend = plot.handles['legend']
        self.assertEqual(legend.get_title().get_fontsize(), 188)

    def test_element_backend_opts_getitem(self):
        a = Curve([1, 2, 3], label="a")
        b = Curve([1, 4, 9], label="b")
        c = Curve([1, 4, 18], label="c")
        d = Curve([1, 4, 36], label="d")
        e = Curve([1, 4, 36], label="e")
        curve = (a * b * c * d * e).opts(
            show_legend=True,
            backend_opts={
                "legend.get_texts()[0].fontsize": 188,
                "legend.get_texts()[1:3].fontsize": 288,
                "legend.get_texts()[3,4].fontsize": 388,
            }
        )
        plot = mpl_renderer.get_plot(curve)
        legend = plot.handles['legend']
        self.assertEqual(legend.get_texts()[0].get_fontsize(), 188)
        self.assertEqual(legend.get_texts()[1].get_fontsize(), 288)
        self.assertEqual(legend.get_texts()[2].get_fontsize(), 288)
        self.assertEqual(legend.get_texts()[3].get_fontsize(), 388)
        self.assertEqual(legend.get_texts()[4].get_fontsize(), 388)

    def test_element_backend_opts_two_accessors(self):
        heat_map = HeatMap([(1, 2, 3), (2, 3, 4), (3, 4, 5)]).opts(
            colorbar=True, backend_opts={"colorbar": "Testing"},
        )
        mpl_renderer.get_plot(heat_map)
        self.log_handler.assertContains(
            "WARNING", "Custom option 'colorbar' expects at least two"
        )

    def test_element_backend_opts_model_not_resolved(self):
        heat_map = HeatMap([(1, 2, 3), (2, 3, 4), (3, 4, 5)]).opts(
            colorbar=True, backend_opts={"cb.title": "Testing"},
        )
        mpl_renderer.get_plot(heat_map)
        self.log_handler.assertContains(
            "WARNING", "cb model could not be"
        )

    def test_element_backend_opts_model_invalid_method(self):
        a = Curve([1, 2, 3], label="a")
        b = Curve([1, 4, 9], label="b")
        curve = (a * b).opts(
            show_legend=True,
            backend_opts={
                "legend.get_texts()[0,1].f0ntzise": 811,
            }
        )
        mpl_renderer.get_plot(curve)
        self.log_handler.assertContains(
            "WARNING", "valid method on the specified model"
        )

    ### Aspect ratio ###
    def test_aspect_non_matching_types(self):
        X = pd.date_range(start="1/1/2018", end="1/08/2018", periods=100)
        Y = np.linspace(1, 100, 100)
        Z = np.random.randn(100, 100)
        qm = QuadMesh((X, Y, Z)).opts(aspect='equal')
        msg = (
            "The aspect is set to 'equal', but the axes does not have the same type: "
            "x-axis timedelta64 and y-axis float64. "
            "Either have the axes be the same type or or set '.opts(aspect=)' "
            "to either a number or 'square'."
        )
        with pytest.raises(TypeError, match=re.escape(msg)):
            mpl_renderer.get_plot(qm)

    ### Grid ###
    def test_grid_both(self):
        curve = Curve(range(10)).opts(gridstyle={'grid_color': 'red', 'grid_linestyle': '--', 'grid_alpha': 0.5, 'grid_linewidth': 2}, show_grid=True)
        plot = mpl_renderer.get_plot(curve)
        ax = plot.handles['axis']
        gridlines = ax.get_xgridlines() + ax.get_ygridlines()
        for line in gridlines:
            self.assertEqual(line.get_color(), 'red')
            self.assertEqual(line.get_linestyle(), '--')
            self.assertEqual(line.get_alpha(), 0.5)
            self.assertEqual(line.get_linewidth(), 2)

    def test_grid_both_show_grid_False(self):
        curve = Curve(range(10)).opts(gridstyle={'grid_color': 'red', 'grid_linestyle': '--', 'grid_alpha': 0.5, 'grid_linewidth': 2}, show_grid=False)
        plot = mpl_renderer.get_plot(curve)
        ax = plot.handles['axis']
        assert not any(line.get_visible() for line in ax.get_xgridlines() + ax.get_ygridlines())

    def test_grid_both_no_grid_prefix(self):
        curve = Curve(range(10)).opts(gridstyle={'color': 'red', 'linestyle': '--', 'alpha': 0.5, 'linewidth': 2}, show_grid=True)
        plot = mpl_renderer.get_plot(curve)
        ax = plot.handles['axis']
        gridlines = ax.get_xgridlines() + ax.get_ygridlines()
        for line in gridlines:
            self.assertEqual(line.get_color(), 'red')
            self.assertEqual(line.get_linestyle(), '--')
            self.assertEqual(line.get_alpha(), 0.5)
            self.assertEqual(line.get_linewidth(), 2)

    def test_grid_x(self):
        curve = Curve(range(10)).opts(gridstyle={'xgrid_color': 'blue', 'xgrid_linestyle': '--', 'xgrid_alpha': 0.5, 'xgrid_linewidth': 2}, show_grid=True)
        plot = mpl_renderer.get_plot(curve)
        ax = plot.handles['axis']
        xgridlines = ax.get_xgridlines()
        for line in xgridlines:
            self.assertEqual(line.get_color(), 'blue')
            self.assertEqual(line.get_linestyle(), '--')
            self.assertEqual(line.get_alpha(), 0.5)
            self.assertEqual(line.get_linewidth(), 2)
        ygridlines = ax.get_ygridlines()
        for line in ygridlines:
            self.assertNotEqual(line.get_color(), 'blue')
            self.assertNotEqual(line.get_linestyle(), '--')
            self.assertNotEqual(line.get_alpha(), 0.5)
            self.assertNotEqual(line.get_linewidth(), 2)

    def test_grid_x_no_grid_prefix(self):
        curve = Curve(range(10)).opts(gridstyle={'color': 'blue', 'linestyle': '--', 'alpha': 0.5, 'linewidth': 2}, show_grid=True)
        plot = mpl_renderer.get_plot(curve)
        ax = plot.handles['axis']
        xgridlines = ax.get_xgridlines()
        for line in xgridlines:
            self.assertEqual(line.get_color(), 'blue')
            self.assertEqual(line.get_linestyle(), '--')
            self.assertEqual(line.get_alpha(), 0.5)
            self.assertEqual(line.get_linewidth(), 2)
        ygridlines = ax.get_ygridlines()
        for line in ygridlines:
            self.assertEqual(line.get_color(), 'blue')
            self.assertEqual(line.get_linestyle(), '--')
            self.assertEqual(line.get_alpha(), 0.5)
            self.assertEqual(line.get_linewidth(), 2)

    def test_grid_y(self):
        curve = Curve(range(10)).opts(gridstyle={'ygrid_color': 'green', 'ygrid_linestyle': '--', 'ygrid_alpha': 0.5, 'ygrid_linewidth': 2}, show_grid=True)
        plot = mpl_renderer.get_plot(curve)
        ax = plot.handles['axis']
        ygridlines = ax.get_ygridlines()
        for line in ygridlines:
            self.assertEqual(line.get_color(), 'green')
            self.assertEqual(line.get_linestyle(), '--')
            self.assertEqual(line.get_alpha(), 0.5)
            self.assertEqual(line.get_linewidth(), 2)
        xgridlines = ax.get_xgridlines()
        for line in xgridlines:
            self.assertNotEqual(line.get_color(), 'green')
            self.assertNotEqual(line.get_linestyle(), '--')
            self.assertNotEqual(line.get_alpha(), 0.5)
            self.assertNotEqual(line.get_linewidth(), 2)

    def test_grid_y_no_grid_prefix(self):
        curve = Curve(range(10)).opts(gridstyle={'color': 'green', 'linestyle': '--', 'alpha': 0.5, 'linewidth': 2}, show_grid=True)
        plot = mpl_renderer.get_plot(curve)
        ax = plot.handles['axis']
        ygridlines = ax.get_ygridlines()
        for line in ygridlines:
            self.assertEqual(line.get_color(), 'green')
            self.assertEqual(line.get_linestyle(), '--')
            self.assertEqual(line.get_alpha(), 0.5)
            self.assertEqual(line.get_linewidth(), 2)
        xgridlines = ax.get_xgridlines()
        for line in xgridlines:
            self.assertEqual(line.get_color(), 'green')
            self.assertEqual(line.get_linestyle(), '--')
            self.assertEqual(line.get_alpha(), 0.5)
            self.assertEqual(line.get_linewidth(), 2)

    def test_grid_mix(self):
        curve = Curve(range(10)).opts(gridstyle={'grid_color': 'red', 'ygrid_linestyle': '--', 'ygrid_alpha': 0.5, 'ygrid_linewidth': 2}, show_grid=True)
        plot = mpl_renderer.get_plot(curve)
        ax = plot.handles['axis']
        xgridlines = ax.get_xgridlines()
        for line in xgridlines:
            self.assertEqual(line.get_color(), 'red')
            self.assertNotEqual(line.get_linestyle(), '--')
            self.assertNotEqual(line.get_alpha(), 0.5)
            self.assertNotEqual(line.get_linewidth(), 2)
        ygridlines = ax.get_ygridlines()
        for line in ygridlines:
            self.assertEqual(line.get_color(), 'red')
            self.assertEqual(line.get_linestyle(), '--')
            self.assertEqual(line.get_alpha(), 0.5)
            self.assertEqual(line.get_linewidth(), 2)

    def test_grid_mix_no_grid_prefix(self):
        curve = Curve(range(10)).opts(gridstyle={'color': 'red', 'y_linestyle': '--', 'y_alpha': 0.5, 'y_linewidth': 2}, show_grid=True)
        plot = mpl_renderer.get_plot(curve)
        ax = plot.handles['axis']
        xgridlines = ax.get_xgridlines()
        for line in xgridlines:
            self.assertEqual(line.get_color(), 'red')
            self.assertNotEqual(line.get_linestyle(), '--')
            self.assertNotEqual(line.get_alpha(), 0.5)
            self.assertNotEqual(line.get_linewidth(), 2)
        ygridlines = ax.get_ygridlines()
        for line in ygridlines:
            self.assertEqual(line.get_color(), 'red')
            self.assertEqual(line.get_linestyle(), '--')
            self.assertEqual(line.get_alpha(), 0.5)
            self.assertEqual(line.get_linewidth(), 2)


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

    def test_style_map_dimension_object(self):
        x = Dimension('x')
        y = Dimension('y')
        scatter = Scatter([1, 2, 3], kdims=[x], vdims=[y]).opts(color=x)
        plot = mpl_renderer.get_plot(scatter)
        artist = plot.handles['artist']
        self.assertEqual(artist.get_clim(), (0, 2))


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
