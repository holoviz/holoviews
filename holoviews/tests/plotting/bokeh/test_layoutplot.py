import datetime as dt
import re

import pytest
import numpy as np

from holoviews.core import (HoloMap, GridSpace, Layout, Empty, Dataset,
                            NdOverlay, NdLayout, DynamicMap, Dimension)
from holoviews.element import Curve, Image, Points, Histogram, Scatter
from holoviews.streams import Stream
from holoviews.util import render, opts
from holoviews.util.transform import dim
from holoviews.plotting.bokeh.util import bokeh3

from bokeh.models import Div, GlyphRenderer, Tabs, Spacer, Title, Row, Column

from ...utils import LoggingComparisonTestCase
from .test_plot import TestBokehPlot, bokeh_renderer

if bokeh3:
    from bokeh.models.layouts import TabPanel
    from bokeh.plotting import figure
    from bokeh.models import GridPlot
    from bokeh.models import Toolbar
else:
    from bokeh.models.layouts import Panel as TabPanel
    from bokeh.plotting import Figure as figure
    from bokeh.models import ToolbarBox as Toolbar  # Not completely correct
    from bokeh.models import GridBox as GridPlot  # Not completely correct


class TestLayoutPlot(LoggingComparisonTestCase, TestBokehPlot):

    def test_layout_update_visible(self):
        hmap = HoloMap({i: Curve(np.arange(i), label='A') for i in range(1, 3)})
        hmap2 = HoloMap({i: Curve(np.arange(i), label='B') for i in range(3, 5)})
        plot = bokeh_renderer.get_plot(hmap+hmap2)
        subplot1, subplot2 = (p for k, p in sorted(plot.subplots.items()))
        subplot1 = subplot1.subplots['main']
        subplot2 = subplot2.subplots['main']
        self.assertTrue(subplot1.handles['glyph_renderer'].visible)
        self.assertFalse(subplot2.handles['glyph_renderer'].visible)
        plot.update((4,))
        self.assertFalse(subplot1.handles['glyph_renderer'].visible)
        self.assertTrue(subplot2.handles['glyph_renderer'].visible)

    def test_layout_framewise_norm(self):
        img1 = Image(np.mgrid[0:5, 0:5][0]).opts(framewise=True)
        img2 = Image(np.mgrid[0:5, 0:5][0]*10).opts(framewise=True)
        plot = bokeh_renderer.get_plot(img1+img2)
        img1_plot, img2_plot = (sp.subplots['main'] for sp in plot.subplots.values())
        img1_cmapper = img1_plot.handles['color_mapper']
        img2_cmapper = img2_plot.handles['color_mapper']
        self.assertEqual(img1_cmapper.low, 0)
        self.assertEqual(img2_cmapper.low, 0)
        self.assertEqual(img1_cmapper.high, 40)
        self.assertEqual(img2_cmapper.high, 40)

    def test_layout_framewise_matching_norm_update(self):
        img1 = Image(np.mgrid[0:5, 0:5][0], vdims='z').opts(framewise=True, axiswise=True)
        stream = Stream.define('zscale', value=1)()
        transform = dim('z')*stream.param.value
        img2 = Image(np.mgrid[0:5, 0:5][0], vdims='z').apply.transform(
            z=transform).opts(framewise=True, axiswise=True)
        plot = bokeh_renderer.get_plot(img1+img2)
        img1_plot = plot.subplots[(0, 0)].subplots['main']
        img2_plot = plot.subplots[(0, 1)].subplots['main']
        img1_cmapper = img1_plot.handles['color_mapper']
        img2_cmapper = img2_plot.handles['color_mapper']
        self.assertEqual(img1_cmapper.low, 0)
        self.assertEqual(img2_cmapper.low, 0)
        self.assertEqual(img1_cmapper.high, 4)
        self.assertEqual(img2_cmapper.high, 4)
        stream.update(value=10)
        self.assertEqual(img1_cmapper.high, 4)
        self.assertEqual(img2_cmapper.high, 40)
        stream.update(value=2)
        self.assertEqual(img1_cmapper.high, 4)
        self.assertEqual(img2_cmapper.high, 8)

    def test_layout_framewise_nonmatching_norm_update(self):
        img1 = Image(np.mgrid[0:5, 0:5][0], vdims='z').opts(framewise=True)
        stream = Stream.define('zscale', value=1)()
        transform = dim('z2')*stream.param.value
        img2 = Image(np.mgrid[0:5, 0:5][0], vdims='z2').apply.transform(
            z2=transform).opts(framewise=True)
        plot = bokeh_renderer.get_plot(img1+img2)
        img1_plot = plot.subplots[(0, 0)].subplots['main']
        img2_plot = plot.subplots[(0, 1)].subplots['main']
        img1_cmapper = img1_plot.handles['color_mapper']
        img2_cmapper = img2_plot.handles['color_mapper']
        self.assertEqual(img1_cmapper.low, 0)
        self.assertEqual(img2_cmapper.low, 0)
        self.assertEqual(img1_cmapper.high, 4)
        self.assertEqual(img2_cmapper.high, 4)
        stream.update(value=10)
        self.assertEqual(img1_cmapper.high, 4)
        self.assertEqual(img2_cmapper.high, 40)
        stream.update(value=2)
        self.assertEqual(img1_cmapper.high, 4)
        self.assertEqual(img2_cmapper.high, 8)

    def test_layout_title(self):
        hmap1 = HoloMap({a: Image(np.random.rand(10,10)) for a in range(3)})
        hmap2 = HoloMap({a: Image(np.random.rand(10,10)) for a in range(3)})
        plot = bokeh_renderer.get_plot(hmap1+hmap2)
        title = plot.handles['title']
        self.assertIsInstance(title, Div)
        text = ('<span style="color:black;font-family:Arial;font-style:bold;'
                'font-weight:bold;font-size:12pt">Default: 0</span>')
        self.assertEqual(title.text, text)

    def test_layout_title_format(self):
        title_str = ('Label: {label}, group: {group}, '
                     'dims: {dimensions}, type: {type}')
        layout = NdLayout(
            {'Element 1': Scatter(
                [],
                label='ONE',
                group='first',
            ), 'Element 2': Scatter(
                [],
                label='TWO',
                group='second',
            )},
            kdims='MYDIM',
            label='the_label',
            group='the_group',
        ).opts(opts.NdLayout(title=title_str), opts.Scatter(title=title_str))
        # Title of NdLayout
        title = bokeh_renderer.get_plot(layout).handles['title']
        self.assertIsInstance(title, Div)
        text = 'Label: the_label, group: the_group, dims: , type: NdLayout'
        self.assertEqual(re.split('>|</', title.text)[1], text)
        # Titles of subplots
        plot = render(layout)
        titles = {
            title.text for title in list(plot.select({'type': Title}))
        }
        titles_correct = {
            'Label: ONE, group: first, dims: MYDIM: Element 1, type: Scatter',
            'Label: TWO, group: second, dims: MYDIM: Element 2, type: Scatter',
        }
        self.assertEqual(titles_correct, titles)

    def test_layout_title_fontsize(self):
        hmap1 = HoloMap({a: Image(np.random.rand(10,10)) for a in range(3)})
        hmap2 = HoloMap({a: Image(np.random.rand(10,10)) for a in range(3)})
        layout = Layout([hmap1, hmap2]).opts(fontsize={'title': '12pt'})
        plot = bokeh_renderer.get_plot(layout)
        title = plot.handles['title']
        self.assertIsInstance(title, Div)
        text = ('<span style="color:black;font-family:Arial;font-style:bold;'
                'font-weight:bold;font-size:12pt">Default: 0</span>')
        self.assertEqual(title.text, text)

    def test_layout_title_show_title_false(self):
        hmap1 = HoloMap({a: Image(np.random.rand(10,10)) for a in range(3)})
        hmap2 = HoloMap({a: Image(np.random.rand(10,10)) for a in range(3)})
        layout = Layout([hmap1, hmap2]).opts(show_title=False)
        plot = bokeh_renderer.get_plot(layout)
        self.assertTrue('title' not in plot.handles)

    def test_layout_title_update(self):
        hmap1 = HoloMap({a: Image(np.random.rand(10,10)) for a in range(3)})
        hmap2 = HoloMap({a: Image(np.random.rand(10,10)) for a in range(3)})
        plot = bokeh_renderer.get_plot(hmap1+hmap2)
        plot.update(1)
        title = plot.handles['title']
        self.assertIsInstance(title, Div)
        text = ('<span style="color:black;font-family:Arial;font-style:bold;'
                'font-weight:bold;font-size:12pt">Default: 1</span>')
        self.assertEqual(title.text, text)

    @pytest.mark.skipif(not bokeh3, reason="Only work for Bokeh 3")
    def test_layout_gridspaces_bokeh3(self):
        layout = (GridSpace({(i, j): Curve(range(i+j)) for i in range(1, 3)
                             for j in range(2,4)}) +
                  GridSpace({(i, j): Curve(range(i+j)) for i in range(1, 3)
                             for j in range(2,4)}) +
                  Curve(range(10))).cols(2)
        layout_plot = bokeh_renderer.get_plot(layout)
        plot = layout_plot.state

        self.assertIsInstance(plot, GridPlot)
        self.assertEqual(len(plot.children), 3)
        self.assertIsInstance(plot.toolbar, Toolbar)

        (grid1, *_), (grid2, *_), (fig, *_) = plot.children
        self.assertIsInstance(grid1, GridPlot)
        self.assertIsInstance(grid2, GridPlot)
        self.assertIsInstance(fig, figure)

        self.assertEqual(len(grid1.children), 3)
        _, (inner_grid1, *_), _ = grid1.children
        self.assertIsInstance(inner_grid1, GridPlot)

        self.assertEqual(len(grid2.children), 3)
        _, (inner_grid2, *_), _ = grid2.children
        self.assertIsInstance(inner_grid2, GridPlot)

        for grid in [inner_grid1, inner_grid2]:
            self.assertEqual(len(grid.children), 4)
            for gfig, *_ in grid.children:
                self.assertIsInstance(gfig, figure)

    @pytest.mark.skipif(bokeh3, reason="Only work for Bokeh 2")
    def test_layout_gridspaces_bokeh2(self):
        layout = (GridSpace({(i, j): Curve(range(i+j)) for i in range(1, 3)
                             for j in range(2,4)}) +
                  GridSpace({(i, j): Curve(range(i+j)) for i in range(1, 3)
                             for j in range(2,4)}) +
                  Curve(range(10))).cols(2)
        layout_plot = bokeh_renderer.get_plot(layout)
        plot = layout_plot.state

        # Unpack until getting down to two rows
        self.assertIsInstance(plot, Column)
        self.assertEqual(len(plot.children), 2)
        toolbar, grid = plot.children
        self.assertIsInstance(toolbar, Toolbar)
        self.assertIsInstance(grid, GridPlot)
        self.assertEqual(len(grid.children), 3)
        (col1, *_), (col2, *_), _ = grid.children
        self.assertIsInstance(col1, Column)
        self.assertIsInstance(col2, Column)
        grid1 = col1.children[0]
        grid2 = col2.children[0]

        # Check the row of GridSpaces
        self.assertEqual(len(grid1.children), 3)
        _, (col1, *_), _ = grid1.children
        self.assertIsInstance(col1, Column)
        inner_grid1 = col1.children[0]

        self.assertEqual(len(grid2.children), 3)
        _, (col2, *_), _ = grid2.children
        self.assertIsInstance(col2, Column)
        inner_grid2 = col2.children[0]
        for grid in [inner_grid1, inner_grid2]:
            self.assertEqual(len(grid.children), 4)
            for gfig, *_ in grid.children:
                self.assertIsInstance(gfig, figure)


    def test_layout_instantiate_subplots(self):
        layout = (Curve(range(10)) + Curve(range(10)) + Image(np.random.rand(10,10)) +
                  Curve(range(10)) + Curve(range(10)))
        plot = bokeh_renderer.get_plot(layout)
        positions = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0)]
        self.assertEqual(sorted(plot.subplots.keys()), positions)

    def test_layout_instantiate_subplots_transposed(self):
        layout = (Curve(range(10)) + Curve(range(10)) + Image(np.random.rand(10,10)) +
                  Curve(range(10)) + Curve(range(10)))
        plot = bokeh_renderer.get_plot(layout.opts(transpose=True))
        positions = [(0, 0), (0, 1), (1, 0), (2, 0), (3, 0)]
        self.assertEqual(sorted(plot.subplots.keys()), positions)

    def test_empty_adjoint_plot(self):
        adjoint = Curve([0,1,1,2,3]) << Empty() << Curve([0,1,1,0,1])
        plot = bokeh_renderer.get_plot(adjoint)
        adjoint_plot = plot.subplots[(0, 0)]
        self.assertEqual(len(adjoint_plot.subplots), 3)
        if bokeh3:
            grid = plot.state
        else:
            grid = plot.state.children[1]
        (f1, *_), (f2, *_), (s1, *_) = grid.children
        self.assertIsInstance(grid, GridPlot)
        self.assertIsInstance(s1, Spacer)
        self.assertEqual(s1.width, 0)
        self.assertEqual(s1.height, 0)
        self.assertEqual(f1.height, f2.height)
        self.assertEqual(f1.height, 300)

    def test_empty_adjoint_plot_with_renderer(self):
        # https://github.com/holoviz/holoviews/pull/5584
        scatter = Scatter(range(10))
        adjoin_layout_plot = scatter << Empty() << scatter.hist(adjoin=False)

        # To render the plot
        bokeh_renderer(adjoin_layout_plot)

    def test_layout_plot_with_adjoints(self):
        layout = (Curve([]) + Curve([]).hist()).cols(1)
        plot = bokeh_renderer.get_plot(layout)
        if bokeh3:
            grid = plot.state
            toolbar = grid.toolbar
        else:
            toolbar, grid = plot.state.children
        self.assertIsInstance(toolbar, Toolbar)
        self.assertIsInstance(grid, GridPlot)
        for (fig, _, _) in grid.children:
            self.assertIsInstance(fig, figure)
        self.assertTrue([len([r for r in f.renderers if isinstance(r, GlyphRenderer)])
                         for (f, _, _) in grid.children], [1, 1, 1])

    def test_layout_plot_tabs_with_adjoints(self):
        layout = (Curve([]) + Curve([]).hist()).opts(tabs=True)
        plot = bokeh_renderer.get_plot(layout)
        self.assertIsInstance(plot.state, Tabs)
        panel1, panel2 = plot.state.tabs
        self.assertIsInstance(panel1, TabPanel)
        self.assertIsInstance(panel2, TabPanel)
        self.assertEqual(panel1.title, 'Curve I')
        self.assertEqual(panel2.title, 'AdjointLayout I')

    def test_layout_shared_source_synced_update(self):
        hmap = HoloMap({i: Dataset({chr(65+j): np.random.rand(i+2)
                                    for j in range(4)}, kdims=['A', 'B', 'C', 'D'])
                        for i in range(3)})

        # Create two holomaps of points sharing the same data source
        hmap1=  hmap.map(lambda x: Points(x.clone(kdims=['A', 'B'])), Dataset)
        hmap2 = hmap.map(lambda x: Points(x.clone(kdims=['D', 'C'])), Dataset)

        # Pop key (1,) for one of the HoloMaps and make Layout
        hmap2.pop((1,))
        layout = (hmap1 + hmap2).opts(shared_datasource=True)

        # Get plot
        plot = bokeh_renderer.get_plot(layout)

        # Check plot created shared data source and recorded expected columns
        sources = plot.handles.get('shared_sources', [])
        source_cols = plot.handles.get('source_cols', {})
        self.assertEqual(len(sources), 1)
        source = sources[0]
        data = source.data
        cols = source_cols[id(source)]
        self.assertEqual(set(cols), {'A', 'B', 'C', 'D'})

        # Ensure the source contains the expected columns
        self.assertEqual(set(data.keys()), {'A', 'B', 'C', 'D'})

        # Update to key (1,) and check the source contains data
        # corresponding to hmap1 and filled in NaNs for hmap2,
        # which was popped above
        plot.update((1,))
        self.assertEqual(data['A'], hmap1[1].dimension_values(0))
        self.assertEqual(data['B'], hmap1[1].dimension_values(1))
        self.assertEqual(data['C'], np.full_like(hmap1[1].dimension_values(0), np.NaN))
        self.assertEqual(data['D'], np.full_like(hmap1[1].dimension_values(0), np.NaN))

    def test_shared_axes(self):
        curve = Curve(range(10))
        img = Image(np.random.rand(10,10))
        plot = bokeh_renderer.get_plot(curve+img)
        plot = plot.subplots[(0, 1)].subplots['main']
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual((x_range.start, x_range.end), (-.5, 9))
        self.assertEqual((y_range.start, y_range.end), (-.5, 9))

    def test_shared_axes_disable(self):
        curve = Curve(range(10))
        img = Image(np.random.rand(10,10)).opts(shared_axes=False)
        plot = bokeh_renderer.get_plot(curve+img)
        plot = plot.subplots[(0, 1)].subplots['main']
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual((x_range.start, x_range.end), (-.5, .5))
        self.assertEqual((y_range.start, y_range.end), (-.5, .5))

    def test_layout_empty_subplots(self):
        layout = Curve(range(10)) + NdOverlay() + HoloMap() + HoloMap({1: Image(np.random.rand(10,10))})
        plot = bokeh_renderer.get_plot(layout)
        self.assertEqual(len(plot.subplots.values()), 2)
        self.log_handler.assertContains('WARNING', 'skipping subplot')
        self.log_handler.assertContains('WARNING', 'skipping subplot')

    def test_layout_set_toolbar_location(self):
        layout = (Curve([]) + Points([])).opts(toolbar='left')
        plot = bokeh_renderer.get_plot(layout)
        if bokeh3:
            self.assertIsInstance(plot.state, GridPlot)
            self.assertIsInstance(plot.state.toolbar, Toolbar)
        else:
            self.assertIsInstance(plot.state, Row)
            self.assertIsInstance(plot.state.children[0], Toolbar)

    def test_layout_disable_toolbar(self):
        layout = (Curve([]) + Points([])).opts(toolbar=None)
        plot = bokeh_renderer.get_plot(layout)
        self.assertIsInstance(plot.state, GridPlot)
        self.assertEqual(len(plot.state.children), 2)

    def test_layout_shared_inverted_yaxis(self):
        layout = (Curve([]) + Curve([])).opts('Curve', invert_yaxis=True)
        plot = bokeh_renderer.get_plot(layout)
        subplot = next(iter(plot.subplots.values())).subplots['main']
        self.assertEqual(subplot.handles['y_range'].start, 1)
        self.assertEqual(subplot.handles['y_range'].end, 0)

    def test_layout_dimensioned_stream_title_update(self):
        stream = Stream.define('Test', test=0)()
        dmap = DynamicMap(lambda test: Curve([]), kdims=['test'], streams=[stream])
        layout = dmap + Curve([])
        plot = bokeh_renderer.get_plot(layout)
        self.assertIn('test: 0', plot.handles['title'].text)
        stream.event(test=1)
        self.assertIn('test: 1', plot.handles['title'].text)
        plot.cleanup()
        self.assertEqual(stream._subscribers, [])

    def test_layout_axis_link_matching_name_label(self):
        layout = Curve([1, 2, 3], vdims=('a', 'A')) + Curve([1, 2, 3], vdims=('a', 'A'))
        plot = bokeh_renderer.get_plot(layout)
        p1, p2 = (sp.subplots['main'] for sp in plot.subplots.values())
        self.assertIs(p1.handles['y_range'], p2.handles['y_range'])

    def test_layout_axis_not_linked_mismatching_name(self):
        layout = Curve([1, 2, 3], vdims=('b', 'A')) + Curve([1, 2, 3], vdims=('a', 'A'))
        plot = bokeh_renderer.get_plot(layout)
        p1, p2 = (sp.subplots['main'] for sp in plot.subplots.values())
        self.assertIsNot(p1.handles['y_range'], p2.handles['y_range'])

    def test_layout_axis_linked_unit_and_no_unit(self):
        layout = (Curve([1, 2, 3], vdims=Dimension('length', unit='m')) +
                  Curve([1, 2, 3], vdims='length'))
        plot = bokeh_renderer.get_plot(layout)
        p1, p2 = (sp.subplots['main'] for sp in plot.subplots.values())
        self.assertIs(p1.handles['y_range'], p2.handles['y_range'])

    def test_layout_axis_not_linked_mismatching_unit(self):
        layout = (Curve([1, 2, 3], vdims=Dimension('length', unit='m')) +
                  Curve([1, 2, 3], vdims=Dimension('length', unit='cm')))
        plot = bokeh_renderer.get_plot(layout)
        p1, p2 = (sp.subplots['main'] for sp in plot.subplots.values())
        self.assertIsNot(p1.handles['y_range'], p2.handles['y_range'])

    def test_dimensioned_streams_with_dynamic_callback_returns_layout(self):
        stream = Stream.define('aname', aname='a')()
        def cb(aname):
            x = np.linspace(0, 1, 10)
            y = np.random.randn(10)
            curve = Curve((x, y), group=aname)
            hist = Histogram(y)
            return (curve + hist).opts(shared_axes=False)
        m = DynamicMap(cb, kdims=['aname'], streams=[stream])
        p = bokeh_renderer.get_plot(m)
        T = 'XYZT'
        stream.event(aname=T)
        self.assertIn('aname: ' + T, p.handles['title'].text, p.handles['title'].text)
        p.cleanup()
        self.assertEqual(stream._subscribers, [])

    def test_layout_shared_axes_disabled(self):
        layout = (Curve([1, 2, 3]) + Curve([10, 20, 30])).opts(shared_axes=False)
        plot = bokeh_renderer.get_plot(layout)
        cp1, cp2 = plot.subplots[(0, 0)].subplots['main'], plot.subplots[(0, 1)].subplots['main']
        self.assertFalse(cp1.handles['y_range'] is cp2.handles['y_range'])
        self.assertEqual(cp1.handles['y_range'].start, 1)
        self.assertEqual(cp1.handles['y_range'].end, 3)
        self.assertEqual(cp2.handles['y_range'].start, 10)
        self.assertEqual(cp2.handles['y_range'].end, 30)

    def test_layout_categorical_numeric_type_axes_not_linked(self):
        curve1 = Curve([1, 2, 3])
        curve2 = Curve([('A', 0), ('B', 1), ('C', 2)])
        layout = curve1 + curve2
        plot = bokeh_renderer.get_plot(layout)
        cp1, cp2 = plot.subplots[(0, 0)].subplots['main'], plot.subplots[(0, 1)].subplots['main']
        self.assertIsNot(cp1.handles['x_range'], cp2.handles['x_range'])
        self.assertIs(cp1.handles['y_range'], cp2.handles['y_range'])

    def test_layout_datetime_numeric_type_axes_not_linked(self):
        curve1 = Curve([1, 2, 3])
        curve2 = Curve([(dt.datetime(2020, 1, 1), 0), (dt.datetime(2020, 1, 2), 1), (dt.datetime(2020, 1, 3), 2)])
        layout = curve1 + curve2
        plot = bokeh_renderer.get_plot(layout)
        cp1, cp2 = plot.subplots[(0, 0)].subplots['main'], plot.subplots[(0, 1)].subplots['main']
        self.assertIsNot(cp1.handles['x_range'], cp2.handles['x_range'])
        self.assertIs(cp1.handles['y_range'], cp2.handles['y_range'])
