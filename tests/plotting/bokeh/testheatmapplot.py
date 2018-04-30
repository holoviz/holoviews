import numpy as np

from holoviews.element import HeatMap, Points, Image

try:
    from bokeh.models import FactorRange, HoverTool
except:
    pass

from .testplot import TestBokehPlot, bokeh_renderer


class TestHeatMapPlot(TestBokehPlot):

    def test_heatmap_hover_ensure_kdims_sanitized(self):
        hm = HeatMap([(1,1,1), (2,2,0)], kdims=['x with space', 'y with $pecial symbol'])
        hm = hm(plot={'tools': ['hover']})
        self._test_hover_info(hm, [('x with space', '@{x_with_space}'),
                                   ('y with $pecial symbol', '@{y_with_pecial_symbol}'),
                                   ('z', '@{z}')])

    def test_heatmap_custom_string_tooltip_hover(self):
        tooltips = "<div><h1>Test</h1></div>"
        custom_hover = HoverTool(tooltips=tooltips)
        hm = HeatMap([(1,1,1), (2,2,0)], kdims=['x with space', 'y with $pecial symbol'])
        hm = hm.options(tools=[custom_hover])
        plot = bokeh_renderer.get_plot(hm)
        hover = plot.handles['hover']
        self.assertEqual(hover.tooltips, tooltips)
        self.assertEqual(hover.renderers, [plot.handles['glyph_renderer']])

    def test_heatmap_hover_ensure_vdims_sanitized(self):
        hm = HeatMap([(1,1,1), (2,2,0)], vdims=['z with $pace'])
        hm = hm(plot={'tools': ['hover']})
        self._test_hover_info(hm, [('x', '@{x}'), ('y', '@{y}'),
                                   ('z with $pace', '@{z_with_pace}')])

    def test_heatmap_colormapping(self):
        hm = HeatMap([(1,1,1), (2,2,0)])
        self._test_colormapping(hm, 2)

    def test_heatmap_categorical_axes_string_int(self):
        hmap = HeatMap([('A',1, 1), ('B', 2, 2)])
        plot = bokeh_renderer.get_plot(hmap)
        x_range = plot.handles['x_range']
        y_range = plot.handles['y_range']
        self.assertIsInstance(x_range, FactorRange)
        self.assertEqual(x_range.factors, ['A', 'B'])
        self.assertIsInstance(y_range, FactorRange)
        self.assertEqual(y_range.factors, ['1', '2'])

    def test_heatmap_categorical_axes_string_int_invert_xyaxis(self):
        opts = dict(invert_xaxis=True, invert_yaxis=True)
        hmap = HeatMap([('A',1, 1), ('B', 2, 2)]).opts(plot=opts)
        plot = bokeh_renderer.get_plot(hmap)
        x_range = plot.handles['x_range']
        y_range = plot.handles['y_range']
        self.assertIsInstance(x_range, FactorRange)
        self.assertEqual(x_range.factors, ['A', 'B'][::-1])
        self.assertIsInstance(y_range, FactorRange)
        self.assertEqual(y_range.factors, ['1', '2'][::-1])

    def test_heatmap_categorical_axes_string_int_inverted(self):
        hmap = HeatMap([('A',1, 1), ('B', 2, 2)]).opts(plot=dict(invert_axes=True))
        plot = bokeh_renderer.get_plot(hmap)
        x_range = plot.handles['x_range']
        y_range = plot.handles['y_range']
        self.assertIsInstance(x_range, FactorRange)
        self.assertEqual(x_range.factors, ['1', '2'])
        self.assertIsInstance(y_range, FactorRange)
        self.assertEqual(y_range.factors, ['A', 'B'])

    def test_heatmap_points_categorical_axes_string_int(self):
        hmap = HeatMap([('A',1, 1), ('B', 2, 2)])
        points = Points([('A', 2), ('B', 1),  ('C', 3)])
        plot = bokeh_renderer.get_plot(hmap*points)
        x_range = plot.handles['x_range']
        y_range = plot.handles['y_range']
        self.assertIsInstance(x_range, FactorRange)
        self.assertEqual(x_range.factors, ['A', 'B', 'C'])
        self.assertIsInstance(y_range, FactorRange)
        self.assertEqual(y_range.factors, ['1', '2', '3'])

    def test_heatmap_points_categorical_axes_string_int_inverted(self):
        hmap = HeatMap([('A',1, 1), ('B', 2, 2)]).opts(plot=dict(invert_axes=True))
        points = Points([('A', 2), ('B', 1),  ('C', 3)])
        plot = bokeh_renderer.get_plot(hmap*points)
        x_range = plot.handles['x_range']
        y_range = plot.handles['y_range']
        self.assertIsInstance(x_range, FactorRange)
        self.assertEqual(x_range.factors, ['1', '2', '3'])
        self.assertIsInstance(y_range, FactorRange)
        self.assertEqual(y_range.factors, ['A', 'B', 'C'])

    def test_heatmap_invert_axes(self):
        arr = np.array([[0, 1, 2], [3, 4,  5]])
        hm = HeatMap(Image(arr)).opts(plot=dict(invert_axes=True))
        plot = bokeh_renderer.get_plot(hm)
        xdim, ydim = hm.kdims
        source = plot.handles['source']
        self.assertEqual(source.data['zvalues'], hm.dimension_values(2, flat=False).T.flatten())
        self.assertEqual(source.data['x'], [xdim.pprint_value(v) for v in hm.dimension_values(0)])
        self.assertEqual(source.data['y'], [ydim.pprint_value(v) for v in hm.dimension_values(1)])

    def test_heatmap_xmarks_int(self):
        hmap = HeatMap([('A',1, 1), ('B', 2, 2)]).options(xmarks=2)
        plot = bokeh_renderer.get_plot(hmap)
        for marker, pos in zip(plot.handles['xmarks'], (0, 1)):
            self.assertEqual(marker.location, pos)
            self.assertEqual(marker.dimension, 'height')

    def test_heatmap_xmarks_tuple(self):
        hmap = HeatMap([('A',1, 1), ('B', 2, 2)]).options(xmarks=('A', 'B'))
        plot = bokeh_renderer.get_plot(hmap)
        for marker, pos in zip(plot.handles['xmarks'], (0, 1)):
            self.assertEqual(marker.location, pos)
            self.assertEqual(marker.dimension, 'height')

    def test_heatmap_xmarks_list(self):
        hmap = HeatMap([('A',1, 1), ('B', 2, 2)]).options(xmarks=[0, 1])
        plot = bokeh_renderer.get_plot(hmap)
        for marker, pos in zip(plot.handles['xmarks'], (0, 1)):
            self.assertEqual(marker.location, pos)
            self.assertEqual(marker.dimension, 'height')

    def test_heatmap_ymarks_int(self):
        hmap = HeatMap([('A',1, 1), ('B', 2, 2)]).options(ymarks=2)
        plot = bokeh_renderer.get_plot(hmap)
        for marker, pos in zip(plot.handles['ymarks'], (2, 1)):
            self.assertEqual(marker.location, pos)
            self.assertEqual(marker.dimension, 'width')

    def test_heatmap_ymarks_tuple(self):
        hmap = HeatMap([('A',1, 1), ('B', 2, 2)]).options(ymarks=('A', 'B'))
        plot = bokeh_renderer.get_plot(hmap)
        for marker, pos in zip(plot.handles['ymarks'], (0, 1)):
            self.assertEqual(marker.location, pos)
            self.assertEqual(marker.dimension, 'width')

    def test_heatmap_ymarks_list(self):
        hmap = HeatMap([('A',1, 1), ('B', 2, 2)]).options(ymarks=[0, 1])
        plot = bokeh_renderer.get_plot(hmap)
        for marker, pos in zip(plot.handles['ymarks'], (2, 1)):
            self.assertEqual(marker.location, pos)
            self.assertEqual(marker.dimension, 'width')
