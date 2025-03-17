import numpy as np
import pandas as pd
from bokeh.models import FactorRange, HoverTool, Range1d

from holoviews.element import HeatMap, Image, Points

from .test_plot import TestBokehPlot, bokeh_renderer


class TestHeatMapPlot(TestBokehPlot):

    def test_heatmap_hover_ensure_kdims_sanitized(self):
        hm = HeatMap([(1,1,1), (2,2,0)], kdims=['x with space', 'y with $pecial symbol'])
        hm = hm.opts(tools=['hover'])
        self._test_hover_info(hm, [('x with space', '@{x_with_space}'),
                                   ('y with $pecial symbol', '@{y_with_pecial_symbol}'),
                                   ('z', '@{z}')])

    def test_heatmap_custom_string_tooltip_hover(self):
        tooltips = "<div><h1>Test</h1></div>"
        custom_hover = HoverTool(tooltips=tooltips)
        hm = HeatMap([(1,1,1), (2,2,0)], kdims=['x with space', 'y with $pecial symbol'])
        hm = hm.opts(tools=[custom_hover])
        plot = bokeh_renderer.get_plot(hm)
        hover = plot.handles['hover']
        self.assertEqual(hover.tooltips, tooltips)
        self.assertEqual(hover.renderers, [plot.handles['glyph_renderer']])

    def test_heatmap_hover_ensure_vdims_sanitized(self):
        hm = HeatMap([(1,1,1), (2,2,0)], vdims=['z with $pace']).opts(tools=['hover'])
        self._test_hover_info(hm, [('x', '@{x}'), ('y', '@{y}'),
                                   ('z with $pace', '@{z_with_pace}')])

    def test_heatmap_colormapping(self):
        hm = HeatMap([(1,1,1), (2,2,0)])
        self._test_colormapping(hm, 2)

    def test_heatmap_categorical_axes_string_int(self):
        hmap = HeatMap([('A', 1, 1), ('B', 2, 2)])
        plot = bokeh_renderer.get_plot(hmap)
        x_range = plot.handles['x_range']
        y_range = plot.handles['y_range']
        self.assertIsInstance(x_range, FactorRange)
        self.assertEqual(x_range.factors, ['A', 'B'])
        self.assertIsInstance(y_range, Range1d)
        self.assertEqual(y_range.start, 0.5)
        self.assertEqual(y_range.end, 2.5)

    def test_heatmap_categorical_axes_string_int_invert_xyaxis(self):
        opts = dict(invert_xaxis=True, invert_yaxis=True)
        hmap = HeatMap([('A', 1, 1), ('B', 2, 2)]).opts(**opts)
        plot = bokeh_renderer.get_plot(hmap)
        x_range = plot.handles['x_range']
        y_range = plot.handles['y_range']
        self.assertIsInstance(x_range, FactorRange)
        self.assertEqual(x_range.factors, ['A', 'B'][::-1])
        self.assertIsInstance(y_range, Range1d)
        self.assertEqual(y_range.start, 2.5)
        self.assertEqual(y_range.end, 0.5)

    def test_heatmap_categorical_axes_string_int_inverted(self):
        hmap = HeatMap([('A',1, 1), ('B', 2, 2)]).opts(invert_axes=True)
        plot = bokeh_renderer.get_plot(hmap)
        x_range = plot.handles['x_range']
        y_range = plot.handles['y_range']
        self.assertIsInstance(x_range, Range1d)
        self.assertEqual(x_range.start, 0.5)
        self.assertEqual(x_range.end, 2.5)
        self.assertIsInstance(y_range, FactorRange)
        self.assertEqual(y_range.factors, ['A', 'B'])

    def test_heatmap_points_categorical_axes_string_int(self):
        hmap = HeatMap([('A', 1, 1), ('B', 2, 2)])
        points = Points([('A', 2), ('B', 1),  ('C', 3)])
        plot = bokeh_renderer.get_plot(hmap*points)
        x_range = plot.handles['x_range']
        y_range = plot.handles['y_range']
        self.assertIsInstance(x_range, FactorRange)
        self.assertEqual(x_range.factors, ['A', 'B', 'C'])
        self.assertIsInstance(y_range, Range1d)
        self.assertEqual(y_range.start, 0.5)
        self.assertEqual(y_range.end, 3)

    def test_heatmap_points_categorical_axes_string_int_inverted(self):
        hmap = HeatMap([('A',1, 1), ('B', 2, 2)]).opts(invert_axes=True)
        points = Points([('A', 2), ('B', 1),  ('C', 3)])
        plot = bokeh_renderer.get_plot(hmap*points)
        x_range = plot.handles['x_range']
        y_range = plot.handles['y_range']
        self.assertIsInstance(x_range, Range1d)
        self.assertEqual(x_range.start, 0.5)
        self.assertEqual(x_range.end, 3)
        self.assertIsInstance(y_range, FactorRange)
        self.assertEqual(y_range.factors, ['A', 'B', 'C'])

    def test_heatmap_invert_axes(self):
        arr = np.array([[0, 1, 2], [3, 4,  5]])
        hm = HeatMap(Image(arr)).opts(invert_axes=True)
        plot = bokeh_renderer.get_plot(hm)
        xdim, ydim = hm.kdims
        source = plot.handles['source']
        self.assertEqual(source.data['zvalues'], hm.dimension_values(2, flat=False).T.flatten())
        self.assertEqual(source.data['x'], hm.dimension_values(1))
        self.assertEqual(source.data['y'], hm.dimension_values(0))

    def test_heatmap_dilate(self):
        hmap = HeatMap([('A',1, 1), ('B', 2, 2)]).opts(dilate=True)
        plot = bokeh_renderer.get_plot(hmap)
        glyph = plot.handles['glyph']
        self.assertTrue(glyph.dilate)

    def test_heatmap_single_x_value(self):
        hmap = HeatMap(([1], ['A', 'B'], np.array([[1], [2]])))
        plot = bokeh_renderer.get_plot(hmap)
        cds = plot.handles['cds']
        self.assertEqual(cds.data['x'], np.array([1, 1]))
        self.assertEqual(cds.data['y'], np.array(['A', 'B']))
        self.assertEqual(cds.data['width'], [2.0, 2.0])
        self.assertEqual(plot.handles['glyph'].height, 1)

    def test_heatmap_single_y_value(self):
        hmap = HeatMap((['A', 'B'], [1], np.array([[1, 2]])))
        plot = bokeh_renderer.get_plot(hmap)
        cds = plot.handles['cds']
        self.assertEqual(cds.data['y'], np.array([1, 1]))
        self.assertEqual(cds.data['x'], np.array(['A', 'B']))
        self.assertEqual(cds.data['height'], [2.0, 2.0])
        self.assertEqual(plot.handles['glyph'].width, 1)

    def test_heatmap_alpha_dim(self):
        data = {
            "row": [1, 2, 1, 2],
            "col": [1, 2, 2, 1],
            "alpha": [0, 0, 0, 1],
            "val": [.5, .6, .2, .1]
        }
        hm = HeatMap(data, kdims=["col", "row"], vdims=["val", "alpha"]).opts(alpha="alpha")
        plot = bokeh_renderer.get_plot(hm)
        cds = plot.handles['cds']
        self.assertEqual(cds.data['row'], np.array([1, 2, 1, 2]))
        self.assertEqual(cds.data['col'], np.array([1, 1, 2, 2]))
        self.assertEqual(cds.data['alpha'], np.array([0, 1, 0, 0]))
        self.assertEqual(cds.data['zvalues'], np.array([0.5, 0.1, 0.2, 0.6]))

    def test_heatmap_pandas_categorial(self):
        # Test for https://github.com/holoviz/holoviews/issues/6313
        df = pd.DataFrame({
            'X': pd.Series(['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c',], dtype='category'),
            'Y': pd.Series(['O', 'P', 'Q', 'O', 'P', 'Q', 'O', 'P', 'Q',], dtype='category'),
            'Z': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        })

        hm = HeatMap(df, ['X', 'Y'], 'Z').aggregate(function=np.mean)
        plot = bokeh_renderer.get_plot(hm)
        data = plot.handles["cds"].data
        np.testing.assert_array_equal(data["X"], df["X"])
        np.testing.assert_array_equal(data["Y"], df["Y"])
        np.testing.assert_array_equal(data["zvalues"], df["Z"])

    def test_heatmap_pandas_multiindex(self):
        df = pd.DataFrame(
            data={'C': [5, 2, -1, 5]},
            index=pd.MultiIndex.from_product([(0, 1), (0, 1)], names=['A', 'B']),
        )
        hm = HeatMap(df, ['A', 'B'], 'C')
        plot = bokeh_renderer.get_plot(hm)
        data = plot.handles["cds"].data
        np.testing.assert_array_equal(data['A'], df.index.get_level_values('A'))
        np.testing.assert_array_equal(data['B'], df.index.get_level_values('B'))
        np.testing.assert_array_equal(data['zvalues'], df['C'])
