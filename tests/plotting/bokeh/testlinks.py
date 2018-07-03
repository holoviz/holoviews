from unittest import SkipTest

import numpy as np

from holoviews.element import Curve, Polygons, Table
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting.links import (RangeToolLink, DataLink)

try:
    from holoviews.plotting.bokeh.renderer import BokehRenderer
    from holoviews.plotting.bokeh.util import bokeh_version
    from bokeh.models import ColumnDataSource
    bokeh_renderer = BokehRenderer.instance()
except:
    bokeh_renderer = None


class TestLinkCallbacks(ComparisonTestCase):

    def test_range_tool_link_callback_single_axis(self):
        if bokeh_version < '0.13':
            raise SkipTest('RangeTool requires bokeh version >= 0.13')
        from bokeh.models import RangeTool
        array = np.random.rand(100, 2)
        src = Curve(array)
        target = Curve(array)
        RangeToolLink(src, target)
        layout = target + src
        plot = bokeh_renderer.get_plot(layout)
        tgt_plot = plot.subplots[(0, 0)].subplots['main']
        src_plot = plot.subplots[(0, 1)].subplots['main']
        range_tool = src_plot.state.select_one({'type': RangeTool})
        self.assertEqual(range_tool.x_range, tgt_plot.handles['x_range'])
        self.assertIs(range_tool.y_range, None)

    def test_range_tool_link_callback_both_axes(self):
        if bokeh_version < '0.13':
            raise SkipTest('RangeTool requires bokeh version >= 0.13')
        from bokeh.models import RangeTool
        array = np.random.rand(100, 2)
        src = Curve(array)
        target = Curve(array)
        RangeToolLink(src, target, axes=['x', 'y'])
        layout = target + src
        plot = bokeh_renderer.get_plot(layout)
        tgt_plot = plot.subplots[(0, 0)].subplots['main']
        src_plot = plot.subplots[(0, 1)].subplots['main']
        range_tool = src_plot.state.select_one({'type': RangeTool})
        self.assertEqual(range_tool.x_range, tgt_plot.handles['x_range'])
        self.assertEqual(range_tool.y_range, tgt_plot.handles['y_range'])

    def test_data_link_poly_table(self):
        arr1 = np.random.rand(10, 2)
        arr2 = np.random.rand(10, 2)
        polys = Polygons([arr1, arr2])
        table = Table([('A', 1), ('B', 2)], 'A', 'B')
        DataLink(polys, table)
        layout = polys + table
        plot = bokeh_renderer.get_plot(layout)
        cds = list(plot.state.select({'type': ColumnDataSource}))
        self.assertEqual(len(cds), 1)
        merged_data = {'xs': [arr1[:, 0], arr2[:, 0]],
                       'ys': [arr1[:, 1], arr2[:, 1]],
                       'A': np.array(['A', 'B']), 'B': np.array([1, 2])}
        for k, v in cds[0].data.items():
            self.assertEqual(v, merged_data[k])
    
    def test_data_link_mismatch(self):
        polys = Polygons([np.random.rand(10, 2)])
        table = Table([('A', 1), ('B', 2)], 'A', 'B')
        DataLink(polys, table)
        layout = polys + table
        with self.assertRaises(Exception):
            bokeh_renderer.get_plot(layout)
