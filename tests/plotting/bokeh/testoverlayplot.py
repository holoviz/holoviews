import numpy as np

from holoviews.core import NdOverlay, HoloMap, DynamicMap
from holoviews.core.options import Cycle
from holoviews.element import Curve, Points, ErrorBars, Text

from .testplot import TestBokehPlot, bokeh_renderer

try:
    from bokeh.models import FixedTicker, HoverTool, FactorRange, Range1d
except:
    pass


class TestOverlayPlot(TestBokehPlot):

    def test_overlay_legend(self):
        overlay = Curve(range(10), label='A') * Curve(range(10), label='B')
        plot = bokeh_renderer.get_plot(overlay)
        legend_labels = [l.label['value'] for l in plot.state.legend[0].items]
        self.assertEqual(legend_labels, ['A', 'B'])

    def test_overlay_update_sources(self):
        hmap = HoloMap({i: (Curve(np.arange(i), label='A') *
                            Curve(np.arange(i)*2, label='B'))
                        for i in range(10, 13)})
        plot = bokeh_renderer.get_plot(hmap)
        plot.update((12,))
        subplot1, subplot2 = plot.subplots.values()
        self.assertEqual(subplot1.handles['source'].data['y'], np.arange(12))
        self.assertEqual(subplot2.handles['source'].data['y'], np.arange(12)*2)

    def test_overlay_update_visible(self):
        hmap = HoloMap({i: Curve(np.arange(i), label='A') for i in range(1, 3)})
        hmap2 = HoloMap({i: Curve(np.arange(i), label='B') for i in range(3, 5)})
        plot = bokeh_renderer.get_plot(hmap*hmap2)
        subplot1, subplot2 = plot.subplots.values()
        self.assertTrue(subplot1.handles['glyph_renderer'].visible)
        self.assertFalse(subplot2.handles['glyph_renderer'].visible)
        plot.update((4,))
        self.assertFalse(subplot1.handles['glyph_renderer'].visible)
        self.assertTrue(subplot2.handles['glyph_renderer'].visible)

    def test_hover_tool_instance_renderer_association(self):
        tooltips = [("index", "$index")]
        hover = HoverTool(tooltips=tooltips)
        opts = dict(tools=[hover])
        overlay = Curve(np.random.rand(10,2)).opts(plot=opts) * Points(np.random.rand(10,2))
        plot = bokeh_renderer.get_plot(overlay)
        curve_plot = plot.subplots[('Curve', 'I')]
        self.assertEqual(len(curve_plot.handles['hover'].renderers), 1)
        self.assertIn(curve_plot.handles['glyph_renderer'], curve_plot.handles['hover'].renderers)
        self.assertEqual(plot.handles['hover'].tooltips, tooltips)

    def test_hover_tool_nested_overlay_renderers(self):
        overlay1 = NdOverlay({0: Curve(range(2)), 1: Curve(range(3))}, kdims=['Test'])
        overlay2 = NdOverlay({0: Curve(range(4)), 1: Curve(range(5))}, kdims=['Test'])
        nested_overlay = (overlay1 * overlay2).opts(plot={'Curve': dict(tools=['hover'])})
        plot = bokeh_renderer.get_plot(nested_overlay)
        self.assertEqual(len(plot.handles['hover'].renderers), 4)
        self.assertEqual(plot.handles['hover'].tooltips,
                         [('Test', '@{Test}'), ('x', '@{x}'), ('y', '@{y}')])

    def test_overlay_empty_layers(self):
        overlay = Curve(range(10)) * NdOverlay()
        plot = bokeh_renderer.get_plot(overlay)
        self.assertEqual(len(plot.subplots), 1)

    def test_overlay_show_frame_disabled(self):
        overlay = (Curve(range(10)) * Curve(range(10))).opts(plot=dict(show_frame=False))
        plot = bokeh_renderer.get_plot(overlay).state
        self.assertEqual(plot.outline_line_alpha, 0)

    def test_overlay_no_xaxis(self):
        overlay = (Curve(range(10)) * Curve(range(10))).opts(plot=dict(xaxis=None))
        plot = bokeh_renderer.get_plot(overlay).state
        self.assertFalse(plot.xaxis[0].visible)

    def test_overlay_no_yaxis(self):
        overlay = (Curve(range(10)) * Curve(range(10))).opts(plot=dict(yaxis=None))
        plot = bokeh_renderer.get_plot(overlay).state
        self.assertFalse(plot.yaxis[0].visible)

    def test_overlay_xrotation(self):
        overlay = (Curve(range(10)) * Curve(range(10))).opts(plot=dict(xrotation=90))
        plot = bokeh_renderer.get_plot(overlay).state
        self.assertEqual(plot.xaxis[0].major_label_orientation, np.pi/2)

    def test_overlay_yrotation(self):
        overlay = (Curve(range(10)) * Curve(range(10))).opts(plot=dict(yrotation=90))
        plot = bokeh_renderer.get_plot(overlay).state
        self.assertEqual(plot.yaxis[0].major_label_orientation, np.pi/2)

    def test_overlay_xticks_list(self):
        overlay = (Curve(range(10)) * Curve(range(10))).opts(plot=dict(xticks=[0, 5, 10]))
        plot = bokeh_renderer.get_plot(overlay).state
        self.assertIsInstance(plot.xaxis[0].ticker, FixedTicker)
        self.assertEqual(plot.xaxis[0].ticker.ticks, [0, 5, 10])

    def test_overlay_yticks_list(self):
        overlay = (Curve(range(10)) * Curve(range(10))).opts(plot=dict(yticks=[0, 5, 10]))
        plot = bokeh_renderer.get_plot(overlay).state
        self.assertIsInstance(plot.yaxis[0].ticker, FixedTicker)
        self.assertEqual(plot.yaxis[0].ticker.ticks, [0, 5, 10])

    def test_points_errorbars_text_ndoverlay_categorical_xaxis(self):
        overlay = NdOverlay({i: Points(([chr(65+i)]*10,np.random.randn(10)))
                             for i in range(5)})
        error = ErrorBars([(el['x'][0], np.mean(el['y']), np.std(el['y']))
                           for el in overlay])
        text = Text('C', 0, 'Test')
        plot = bokeh_renderer.get_plot(overlay*error*text)
        x_range = plot.handles['x_range']
        y_range = plot.handles['y_range']
        self.assertIsInstance(x_range, FactorRange)
        factors = ['A', 'B', 'C', 'D', 'E']
        self.assertEqual(x_range.factors, ['A', 'B', 'C', 'D', 'E'])
        self.assertIsInstance(y_range, Range1d)
        error_plot = plot.subplots[('ErrorBars', 'I')]
        for xs, factor in zip(error_plot.handles['source'].data['base'], factors):
            self.assertEqual(factor, xs)

    def test_points_errorbars_text_ndoverlay_categorical_xaxis_invert_axes(self):
        overlay = NdOverlay({i: Points(([chr(65+i)]*10,np.random.randn(10)))
                             for i in range(5)})
        error = ErrorBars([(el['x'][0], np.mean(el['y']), np.std(el['y']))
                           for el in overlay]).opts(plot=dict(invert_axes=True))
        text = Text('C', 0, 'Test')
        plot = bokeh_renderer.get_plot(overlay*error*text)
        x_range = plot.handles['x_range']
        y_range = plot.handles['y_range']
        self.assertIsInstance(x_range, Range1d)
        self.assertIsInstance(y_range, FactorRange)
        self.assertEqual(y_range.factors, ['A', 'B', 'C', 'D', 'E'])

    def test_overlay_empty_element_extent(self):
        overlay = Curve([]).redim.range(x=(-10, 10)) * Points([]).redim.range(y=(-20, 20))
        plot = bokeh_renderer.get_plot(overlay)
        extents = plot.get_extents(overlay, {})
        self.assertEqual(extents, (-10, -20, 10, 20))

    def test_dynamic_subplot_remapping(self):
        # Checks that a plot is appropriately updated when reused
        def cb(X):
            return NdOverlay({i: Curve(np.arange(10)+i) for i in range(X-2, X)})
        dmap = DynamicMap(cb, kdims=['X']).redim.range(X=(1, 10))
        plot = bokeh_renderer.get_plot(dmap)
        plot.update((3,))
        legend_labels = [item.label for item in plot.state.legend[0].items]
        self.assertEqual(legend_labels, [{'value': '1'}, {'value': '2'}])
        colors = Cycle().values
        for i, (subplot, color) in enumerate(zip(plot.subplots.values(), colors[3:])):
            self.assertEqual(subplot.handles['glyph'].line_color, color)
            self.assertEqual(subplot.cyclic_index, i+3)
            self.assertEqual(list(subplot.overlay_dims.values()), [i+1])

    def test_dynamic_subplot_creation(self):
        def cb(X):
            return NdOverlay({i: Curve(np.arange(10)+i) for i in range(X)})
        dmap = DynamicMap(cb, kdims=['X']).redim.range(X=(1, 10))
        plot = bokeh_renderer.get_plot(dmap)
        self.assertEqual(len(plot.subplots), 1)
        plot.update((3,))
        self.assertEqual(len(plot.subplots), 3)
        for i, subplot in enumerate(plot.subplots.values()):
            self.assertEqual(subplot.cyclic_index, i)
