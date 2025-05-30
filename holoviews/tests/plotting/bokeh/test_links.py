import numpy as np
import pytest
from bokeh.models import ColumnDataSource, RangeTool

import holoviews as hv
from holoviews.core.spaces import DynamicMap
from holoviews.element import Curve, Image, Path, Points, Polygons, Scatter, Table
from holoviews.plotting.links import DataLink, Link, RangeToolLink

from .test_plot import TestBokehPlot, bokeh_renderer


class TestLinkCallbacks(TestBokehPlot):

    def test_range_tool_link_callback_single_axis(self):
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

    def test_range_tool_link_callback_single_axis_overlay_target(self):
        array = np.random.rand(100, 2)
        src = Curve(array)
        target = Scatter(array, label='a') * Scatter(array, label='b')
        RangeToolLink(src, target)
        layout = target + src
        plot = bokeh_renderer.get_plot(layout)
        tgt_plot = plot.subplots[(0, 0)].subplots['main']
        src_plot = plot.subplots[(0, 1)].subplots['main']
        range_tool = src_plot.state.select_one({'type': RangeTool})
        self.assertEqual(range_tool.x_range, tgt_plot.handles['x_range'])
        self.assertIs(range_tool.y_range, None)

    def test_range_tool_link_callback_single_axis_overlay_target_image_source(self):
        data = np.random.rand(50, 50)
        target = Curve(data) * Curve(data)
        source = Image(np.random.rand(50, 50), bounds=(0, 0, 1, 1))
        RangeToolLink(source, target)
        layout = target + source
        plot = bokeh_renderer.get_plot(layout)
        tgt_plot = plot.subplots[(0, 0)].subplots['main']
        src_plot = plot.subplots[(0, 1)].subplots['main']
        range_tool = src_plot.state.select_one({'type': RangeTool})
        self.assertEqual(range_tool.x_range, tgt_plot.handles['x_range'])
        self.assertIs(range_tool.y_range, None)

    def test_range_tool_link_callback_single_axis_curve_target_image_dmap_source(self):
        # Choosing Image to exert the apply_nodata compositor
        src = DynamicMap(
            lambda a: Image(a*np.random.random((20, 20)), bounds=[0, 0, 9, 9]),
            kdims=['a']
        ).redim.range(a=(0.1,1))
        target = Curve(np.arange(10))
        RangeToolLink(src, target)
        layout = target + src
        plot = bokeh_renderer.get_plot(layout)
        tgt_plot = plot.subplots[(0, 0)].subplots['main']
        src_plot = plot.subplots[(0, 1)].subplots['main']
        range_tool = src_plot.state.select_one({'type': RangeTool})
        assert range_tool.x_range == tgt_plot.handles['x_range']
        assert range_tool.y_range is None

    def test_range_tool_link_callback_single_axis_overlay_target_image_dmap_source(self):
        # Choosing Image to exert the apply_nodata compositor
        src = DynamicMap(
            lambda a: Image(a*np.random.random((20, 20)), bounds=[0, 0, 9, 9]),
            kdims=['a']
        ).redim.range(a=(0.1,1))
        data = np.random.rand(50, 50)
        target = Curve(data) * Curve(data)
        RangeToolLink(src, target)
        layout = target + src
        plot = bokeh_renderer.get_plot(layout)
        tgt_plot = plot.subplots[(0, 0)].subplots['main']
        src_plot = plot.subplots[(0, 1)].subplots['main']
        range_tool = src_plot.state.select_one({'type': RangeTool})
        assert range_tool.x_range == tgt_plot.handles['x_range']
        assert range_tool.y_range is None

    def test_range_tool_link_callback_both_axes(self):
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
        data = {k: v for k, v in zip(['x', 'y', 'z'], arr, strict=None)}
        a = Scatter(data, 'x', 'z')
        b = Scatter(data, 'x', 'y')
        DataLink(a, b)
        try:
            bokeh_renderer.get_plot(a+b)
        except Exception:
            self.fail()

def test_range_tool_link_clones_axis():
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    c0 = Curve((x, y)).opts(title="Source")
    c1 = Curve((x, y)).opts(title="Target")

    bk_grid = hv.render(c0 + c1)
    bk_plot0 = bk_grid.children[0][0]
    bk_plot1 = bk_grid.children[1][0]
    assert bk_plot0.x_range is bk_plot1.x_range
    assert bk_plot0.y_range is bk_plot1.y_range

    # Will clone the source axis
    RangeToolLink(c0, c1, axes=["x", "y"])

    bk_grid = hv.render(c0 + c1)
    bk_plot0 = bk_grid.children[0][0]
    bk_plot1 = bk_grid.children[1][0]
    assert bk_plot0.x_range is not bk_plot1.x_range
    assert bk_plot0.y_range is not bk_plot1.y_range
