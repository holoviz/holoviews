import numpy as np
from bokeh.layouts import Column
from bokeh.models import Div, Toolbar

from holoviews.core import (
    Dataset,
    DynamicMap,
    GridMatrix,
    GridSpace,
    HoloMap,
    NdOverlay,
)
from holoviews.element import Curve, Image, Points
from holoviews.operation import gridmatrix
from holoviews.streams import Stream

from .test_plot import TestBokehPlot, bokeh_renderer


class TestGridPlot(TestBokehPlot):

    def test_grid_title(self):
        grid = GridSpace({(i, j): HoloMap({a: Image(np.random.rand(10,10))
                                           for a in range(3)}, kdims=['X'])
                          for i in range(2) for j in range(3)})
        plot = bokeh_renderer.get_plot(grid)
        title = plot.handles['title']
        self.assertIsInstance(title, Div)
        text = ('<span style="color:black;font-family:Arial;'
                'font-style:bold;font-weight:bold;font-size:16pt">X: 0</span>')
        self.assertEqual(title.text, text)

    def test_grid_title_update(self):
        grid = GridSpace({(i, j): HoloMap({a: Image(np.random.rand(10,10))
                                           for a in range(3)}, kdims=['X'])
                          for i in range(2) for j in range(3)})
        plot = bokeh_renderer.get_plot(grid)
        plot.update(1)
        title = plot.handles['title']
        self.assertIsInstance(title, Div)
        text = ('<span style="color:black;font-family:Arial;'
                'font-style:bold;font-weight:bold;font-size:16pt">X: 1</span>')
        self.assertEqual(title.text, text)

    def test_gridmatrix_overlaid_batched(self):
        ds = Dataset((['A']*5+['B']*5, np.random.rand(10), np.random.rand(10)),
                     kdims=['a', 'b', 'c'])
        gmatrix = gridmatrix(ds.groupby('a', container_type=NdOverlay))
        plot = bokeh_renderer.get_plot(gmatrix)

        sp1 = plot.subplots[('b', 'c')]
        self.assertEqual(sp1.state.xaxis[0].visible, False)
        self.assertEqual(sp1.state.yaxis[0].visible, True)
        sp2 = plot.subplots[('b', 'b')]
        self.assertEqual(sp2.state.xaxis[0].visible, True)
        self.assertEqual(sp2.state.yaxis[0].visible, True)
        sp3 = plot.subplots[('c', 'b')]
        self.assertEqual(sp3.state.xaxis[0].visible, True)
        self.assertEqual(sp3.state.yaxis[0].visible, False)
        sp4 = plot.subplots[('c', 'c')]
        self.assertEqual(sp4.state.xaxis[0].visible, False)
        self.assertEqual(sp4.state.yaxis[0].visible, False)

    def test_gridspace_sparse(self):
        grid = GridSpace({(i, j): Curve(range(i+j)) for i in range(1, 3)
                            for j in range(2,4) if not (i==1 and j == 2)})
        plot = bokeh_renderer.get_plot(grid)
        size = bokeh_renderer.get_size(plot.state)
        self.assertEqual(size, (320, 311))

    def test_grid_shared_source_synced_update(self):
        hmap = HoloMap({i: Dataset({chr(65+j): np.random.rand(i+2)
                                    for j in range(4)}, kdims=['A', 'B', 'C', 'D'])
                        for i in range(3)})

        # Create two holomaps of points sharing the same data source
        hmap1=  hmap.map(lambda x: Points(x.clone(kdims=['A', 'B'])), Dataset)
        hmap2 = hmap.map(lambda x: Points(x.clone(kdims=['D', 'C'])), Dataset)

        # Pop key (1,) for one of the HoloMaps and make GridSpace
        hmap2.pop(1)
        grid = GridSpace({0: hmap1, 2: hmap2}, kdims=['X']).opts(shared_datasource=True)

        # Get plot
        plot = bokeh_renderer.get_plot(grid)

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
        self.assertEqual(data['C'], np.full_like(hmap1[1].dimension_values(0), np.nan))
        self.assertEqual(data['D'], np.full_like(hmap1[1].dimension_values(0), np.nan))

    def test_grid_set_toolbar_location(self):
        grid = GridSpace({0: Curve([]), 1: Points([])}, 'X').opts(toolbar='left')
        plot = bokeh_renderer.get_plot(grid)
        self.assertIsInstance(plot.state, Column)
        self.assertIsInstance(plot.state.children[0].toolbar, Toolbar)


    def test_grid_disable_toolbar(self):
        grid = GridSpace({0: Curve([]), 1: Points([])}, 'X').opts(toolbar=None)
        plot = bokeh_renderer.get_plot(grid)
        self.assertIsInstance(plot.state, Column)
        self.assertEqual([p for p in plot.state.children if isinstance(p, Toolbar)], [])

    def test_grid_dimensioned_stream_title_update(self):
        stream = Stream.define('Test', test=0)()
        dmap = DynamicMap(lambda test: Curve([]), kdims=['test'], streams=[stream])
        grid = GridMatrix({0: dmap, 1: Curve([])}, 'X')
        plot = bokeh_renderer.get_plot(grid)
        self.assertIn('test: 0', plot.handles['title'].text)
        stream.event(test=1)
        self.assertIn('test: 1', plot.handles['title'].text)
        plot.cleanup()
        self.assertEqual(stream._subscribers, [])
