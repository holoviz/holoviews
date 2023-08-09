import numpy as np
import pytest

from holoviews.core.spaces import DynamicMap
from holoviews.element import Curve, Polygons, Table, Scatter, Path, Points
from holoviews.plotting.links import (Link, RangeToolLink, DataLink)

from bokeh.models import ColumnDataSource

from .test_plot import TestBokehPlot, bokeh_renderer


class TestLinkCallbacks(TestBokehPlot):

    def test_range_tool_link_callback_single_axis(self):
        from bokeh.models import RangeTool
        array = np.random.rand(100, 2)
        src = Curve(array)
        target = Scatter(array)
        RangeToolLink(src, target)
        layout = target + src
        plot = bokeh_renderer.get_plot(layout)
        tgt_plot = plot.subplots[(0, 0)].subplots['main']
        src_plot = plot.subplots[(0, 1)].subplots['main']
        range_tool = src_plot.state.select_one({'type': RangeTool})
        self.assertEqual(range_tool.x_range, tgt_plot.handles['x_range'])
        self.assertIs(range_tool.y_range, None)

    def test_range_tool_link_callback_both_axes(self):
        from bokeh.models import RangeTool
        array = np.random.rand(100, 2)
        src = Curve(array)
        target = Scatter(array)
        RangeToolLink(src, target, axes=['x', 'y'])
        layout = target + src
        plot = bokeh_renderer.get_plot(layout)
        tgt_plot = plot.subplots[(0, 0)].subplots['main']
        src_plot = plot.subplots[(0, 1)].subplots['main']
        range_tool = src_plot.state.select_one({'type': RangeTool})
        self.assertEqual(range_tool.x_range, tgt_plot.handles['x_range'])
        self.assertEqual(range_tool.y_range, tgt_plot.handles['y_range'])

    def test_range_tool_link_callback_boundsx_arg(self):
        array = np.random.rand(100, 2)
        src = Curve(array)
        target = Scatter(array)
        x_start = 0.2
        x_end = 0.3
        RangeToolLink(src, target, axes=['x', 'y'], boundsx=(x_start, x_end))
        layout = target + src
        plot = bokeh_renderer.get_plot(layout)
        tgt_plot = plot.subplots[(0, 0)].subplots['main']
        self.assertEqual(tgt_plot.handles['x_range'].start, x_start)
        self.assertEqual(tgt_plot.handles['x_range'].end, x_end)
        self.assertEqual(tgt_plot.handles['x_range'].reset_start, x_start)
        self.assertEqual(tgt_plot.handles['x_range'].reset_end, x_end)

    def test_range_tool_link_callback_boundsy_arg(self):
        array = np.random.rand(100, 2)
        src = Curve(array)
        target = Scatter(array)
        y_start = 0.8
        y_end = 0.9
        RangeToolLink(src, target, axes=['x', 'y'], boundsy=(y_start, y_end))
        layout = target + src
        plot = bokeh_renderer.get_plot(layout)
        tgt_plot = plot.subplots[(0, 0)].subplots['main']
        self.assertEqual(tgt_plot.handles['y_range'].start, y_start)
        self.assertEqual(tgt_plot.handles['y_range'].end, y_end)
        self.assertEqual(tgt_plot.handles['y_range'].reset_start, y_start)
        self.assertEqual(tgt_plot.handles['y_range'].reset_end, y_end)

    def test_data_link_dynamicmap_table(self):
        dmap = DynamicMap(lambda X: Points([(0, X)]), kdims='X').redim.range(X=(-1, 1))
        table = Table([(-1,)], vdims='y')
        DataLink(dmap, table)
        layout = dmap + table
        plot = bokeh_renderer.get_plot(layout)
        cds = list(plot.state.select({'type': ColumnDataSource}))
        self.assertEqual(len(cds), 1)
        data = {'x': np.array([0]), 'y': np.array([-1])}
        for k, v in cds[0].data.items():
            self.assertEqual(v, data[k])

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
        merged_data = {'xs': [[[np.concatenate([arr1[:, 0], arr1[:1, 0]])]],
                              [[np.concatenate([arr2[:, 0], arr2[:1, 0]])]]],
                       'ys': [[[np.concatenate([arr1[:, 1], arr1[:1, 1]])]],
                              [[np.concatenate([arr2[:, 1], arr2[:1, 1]])]]],
                       'A': np.array(['A', 'B']), 'B': np.array([1, 2])}
        for k, v in cds[0].data.items():
            self.assertEqual(v, merged_data[k])

    def test_data_link_poly_table_on_clone(self):
        arr1 = np.random.rand(10, 2)
        arr2 = np.random.rand(10, 2)
        polys = Polygons([arr1, arr2])
        table = Table([('A', 1), ('B', 2)], 'A', 'B')
        DataLink(polys, table)
        layout = polys.clone() + table.clone()
        plot = bokeh_renderer.get_plot(layout)
        cds = list(plot.state.select({'type': ColumnDataSource}))
        self.assertEqual(len(cds), 1)

    def test_data_link_poly_table_on_unlinked_clone(self):
        arr1 = np.random.rand(10, 2)
        arr2 = np.random.rand(10, 2)
        polys = Polygons([arr1, arr2])
        table = Table([('A', 1), ('B', 2)], 'A', 'B')
        DataLink(polys, table)
        layout = polys.clone() + table.clone(link=False)
        plot = bokeh_renderer.get_plot(layout)
        cds = list(plot.state.select({'type': ColumnDataSource}))
        self.assertEqual(len(cds), 2)

    def test_data_link_mismatch(self):
        polys = Polygons([np.random.rand(10, 2)])
        table = Table([('A', 1), ('B', 2)], 'A', 'B')
        DataLink(polys, table)
        layout = polys + table
        msg = "DataLink source data length must match target"
        with pytest.raises(ValueError, match=msg):
            bokeh_renderer.get_plot(layout)

    def test_data_link_list(self):
        path = Path([[(0, 0, 0), (1, 1, 1), (2, 2, 2)]], vdims='color').opts(color='color')
        table = Table([('A', 1), ('B', 2)], 'A', 'B')
        DataLink(path, table)
        layout = path + table
        plot = bokeh_renderer.get_plot(layout)
        path_plot, table_plot = (sp.subplots['main'] for sp in plot.subplots.values())
        self.assertIs(path_plot.handles['source'], table_plot.handles['source'])

    def test_data_link_idempotent(self):
        table1 = Table([], 'A', 'B')
        table2 = Table([], 'C', 'D')
        link1 = DataLink(table1, table2)
        DataLink(table1, table2)
        self.assertEqual(len(Link.registry[table1]), 1)
        self.assertIn(link1, Link.registry[table1])

    def test_data_link_nan(self):
        arr = np.random.rand(3, 5)
        arr[0, 0] = np.nan
        data = {k: v for k, v in zip(['x', 'y', 'z'], arr)}
        a = Scatter(data, 'x', 'z')
        b = Scatter(data, 'x', 'y')
        DataLink(a, b)
        try:
            bokeh_renderer.get_plot(a+b)
        except Exception:
            self.fail()
