import numpy as np

from holoviews.core import HoloMap, GridSpace, Layout, Empty, Dataset, NdOverlay
from holoviews.element import Curve, Image, Points

from .testplot import TestBokehPlot, bokeh_renderer

try:
    from bokeh.layouts import Column, Row
    from bokeh.models import Div, ToolbarBox
    from bokeh.plotting import Figure
except:
    pass


class TestLayoutPlot(TestBokehPlot):

    def test_layout_update_visible(self):
        hmap = HoloMap({i: Curve(np.arange(i), label='A') for i in range(1, 3)})
        hmap2 = HoloMap({i: Curve(np.arange(i), label='B') for i in range(3, 5)})
        plot = bokeh_renderer.get_plot(hmap+hmap2)
        subplot1, subplot2 = [p for k, p in sorted(plot.subplots.items())]
        subplot1 = subplot1.subplots['main']
        subplot2 = subplot2.subplots['main']
        self.assertTrue(subplot1.handles['glyph_renderer'].visible)
        self.assertFalse(subplot2.handles['glyph_renderer'].visible)
        plot.update((4,))
        self.assertFalse(subplot1.handles['glyph_renderer'].visible)
        self.assertTrue(subplot2.handles['glyph_renderer'].visible)

    def test_layout_title(self):
        hmap1 = HoloMap({a: Image(np.random.rand(10,10)) for a in range(3)})
        hmap2 = HoloMap({a: Image(np.random.rand(10,10)) for a in range(3)})
        plot = bokeh_renderer.get_plot(hmap1+hmap2)
        title = plot.handles['title']
        self.assertIsInstance(title, Div)
        text = "<span style='font-size: 16pt'><b>Default: 0</b></font>"
        self.assertEqual(title.text, text)

    def test_layout_title_fontsize(self):
        hmap1 = HoloMap({a: Image(np.random.rand(10,10)) for a in range(3)})
        hmap2 = HoloMap({a: Image(np.random.rand(10,10)) for a in range(3)})
        layout = Layout([hmap1, hmap2]).opts(plot=dict(fontsize={'title': '12pt'}))
        plot = bokeh_renderer.get_plot(layout)
        title = plot.handles['title']
        self.assertIsInstance(title, Div)
        text = "<span style='font-size: 12pt'><b>Default: 0</b></font>"
        self.assertEqual(title.text, text)

    def test_layout_title_show_title_false(self):
        hmap1 = HoloMap({a: Image(np.random.rand(10,10)) for a in range(3)})
        hmap2 = HoloMap({a: Image(np.random.rand(10,10)) for a in range(3)})
        layout = Layout([hmap1, hmap2]).opts(plot=dict(show_title=False))
        plot = bokeh_renderer.get_plot(layout)
        self.assertTrue('title' not in plot.handles)

    def test_layout_title_update(self):
        hmap1 = HoloMap({a: Image(np.random.rand(10,10)) for a in range(3)})
        hmap2 = HoloMap({a: Image(np.random.rand(10,10)) for a in range(3)})
        plot = bokeh_renderer.get_plot(hmap1+hmap2)
        plot.update(1)
        title = plot.handles['title']
        self.assertIsInstance(title, Div)
        text = "<span style='font-size: 16pt'><b>Default: 1</b></font>"
        self.assertEqual(title.text, text)

    def test_layout_gridspaces(self):
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
        toolbar, column = plot.children
        self.assertIsInstance(toolbar, ToolbarBox)
        self.assertIsInstance(column, Column)
        self.assertEqual(len(column.children), 2)
        row1, row2 = column.children
        self.assertIsInstance(row1, Row)
        self.assertIsInstance(row2, Row)

        # Check the row of GridSpaces
        self.assertEqual(len(row1.children), 2)
        grid1, grid2 = row1.children
        self.assertIsInstance(grid1, Column)
        self.assertIsInstance(grid2, Column)
        self.assertEqual(len(grid1.children), 1)
        self.assertEqual(len(grid2.children), 1)
        grid1, grid2 = grid1.children[0], grid2.children[0]
        self.assertIsInstance(grid1, Column)
        self.assertIsInstance(grid2, Column)
        for grid in [grid1, grid2]:
            self.assertEqual(len(grid.children), 2)
            grow1, grow2 = grid.children
            self.assertIsInstance(grow1, Row)
            self.assertIsInstance(grow2, Row)
            self.assertEqual(len(grow1.children), 2)
            self.assertEqual(len(grow2.children), 2)
            ax_row, grid_row = grow1.children
            grow1, grow2 = grid_row.children[0].children
            gfig1, gfig2 = grow1.children
            gfig3, gfig4 = grow2.children
            self.assertIsInstance(gfig1, Figure)
            self.assertIsInstance(gfig2, Figure)
            self.assertIsInstance(gfig3, Figure)
            self.assertIsInstance(gfig4, Figure)

        # Check the row of Curve and a spacer
        self.assertEqual(len(row2.children), 2)
        fig, spacer = row2.children
        self.assertIsInstance(fig, Figure)
        self.assertIsInstance(spacer, Figure)

    def test_layout_instantiate_subplots(self):
        layout = (Curve(range(10)) + Curve(range(10)) + Image(np.random.rand(10,10)) +
                  Curve(range(10)) + Curve(range(10)))
        plot = bokeh_renderer.get_plot(layout)
        positions = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0)]
        self.assertEqual(sorted(plot.subplots.keys()), positions)

    def test_layout_instantiate_subplots_transposed(self):
        layout = (Curve(range(10)) + Curve(range(10)) + Image(np.random.rand(10,10)) +
                  Curve(range(10)) + Curve(range(10)))
        plot = bokeh_renderer.get_plot(layout(plot=dict(transpose=True)))
        positions = [(0, 0), (0, 1), (1, 0), (2, 0), (3, 0)]
        self.assertEqual(sorted(plot.subplots.keys()), positions)

    
    def test_empty_adjoint_plot(self):
        adjoint = Curve([0,1,1,2,3]) << Empty() << Curve([0,1,1,0,1])
        plot = bokeh_renderer.get_plot(adjoint)
        adjoint_plot = plot.subplots[(0, 0)]
        self.assertEqual(len(adjoint_plot.subplots), 3)
        column = plot.state.children[1]
        row1, row2 = column.children
        self.assertEqual(row1.children[0].plot_height, row1.children[1].plot_height)
        self.assertEqual(row1.children[1].plot_width, 0)
        self.assertEqual(row2.children[1].plot_width, 0)
        self.assertEqual(row2.children[0].plot_height, row2.children[1].plot_height)

    def test_layout_shared_source_synced_update(self):
        hmap = HoloMap({i: Dataset({chr(65+j): np.random.rand(i+2)
                                    for j in range(4)}, kdims=['A', 'B', 'C', 'D'])
                        for i in range(3)})

        # Create two holomaps of points sharing the same data source
        hmap1=  hmap.map(lambda x: Points(x.clone(kdims=['A', 'B'])), Dataset)
        hmap2 = hmap.map(lambda x: Points(x.clone(kdims=['D', 'C'])), Dataset)

        # Pop key (1,) for one of the HoloMaps and make Layout
        hmap2.pop((1,))
        layout = (hmap1 + hmap2).opts(plot=dict(shared_datasource=True))

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
        img = Image(np.random.rand(10,10)).opts(plot=dict(shared_axes=False))
        plot = bokeh_renderer.get_plot(curve+img)
        plot = plot.subplots[(0, 1)].subplots['main']
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual((x_range.start, x_range.end), (-.5, .5))
        self.assertEqual((y_range.start, y_range.end), (-.5, .5))

    def test_layout_empty_subplots(self):
        layout = Curve(range(10)) + NdOverlay() + HoloMap() + HoloMap({1: Image(np.random.rand(10,10))})
        plot = bokeh_renderer.get_plot(layout)
        self.assertEqual(len(plot.subplots.values()), 2)

    def test_layout_set_toolbar_location(self):
        layout = (Curve([]) + Points([])).options(toolbar='left')
        plot = bokeh_renderer.get_plot(layout)
        self.assertIsInstance(plot.state, Row)
        self.assertIsInstance(plot.state.children[0], ToolbarBox)

    def test_layout_disable_toolbar(self):
        layout = (Curve([]) + Points([])).options(toolbar=None)
        plot = bokeh_renderer.get_plot(layout)
        self.assertIsInstance(plot.state, Column)
        self.assertEqual(len(plot.state.children), 1)
