import numpy as np

from holoviews.core.spaces import DynamicMap
from holoviews.element import Spread
from holoviews.streams import Buffer

from .testplot import TestBokehPlot, bokeh_renderer



class TestSpreadPlot(TestBokehPlot):

    def test_spread_stream_data(self):
        buffer = Buffer({'y': np.array([]), 'yerror': np.array([]), 'x': np.array([])})
        dmap = DynamicMap(Spread, streams=[buffer])
        plot = bokeh_renderer.get_plot(dmap)
        buffer.send({'y': [1, 2, 1, 4], 'yerror': [.5, .2, .1, .5], 'x': [0,1,2,3]})
        cds = plot.handles['cds']
        self.assertEqual(cds.data['x'], np.array([0., 1., 2., 3., 3., 2., 1., 0.]))
        self.assertEqual(cds.data['y'], np.array([0.5, 1.8, 0.9, 3.5, 4.5, 1.1, 2.2, 1.5]))

    def test_spread_with_nans(self):
        spread = Spread([(0, 0, 0, 1), (1, 0, 0, 2), (2, 0, 0, 3), (3, np.nan, np.nan, np.nan),
                         (4, 0, 0, 5), (5, 0, 0, 6), (6, 0, 0, 7)], vdims=['y', 'neg', 'pos'])
        plot = bokeh_renderer.get_plot(spread)
        cds = plot.handles['cds']
        self.assertEqual(cds.data['x'], np.array([0., 1., 2., 2., 1., 0., np.nan,
                                                  4., 5., 6., 6., 5., 4.]))
        self.assertEqual(cds.data['y'], np.array([0., 0., 0., 3., 2., 1., np.nan,
                                                  0., 0., 0., 7., 6., 5.]))

    def test_spread_empty(self):
        spread = Spread([])
        plot = bokeh_renderer.get_plot(spread)
        cds = plot.handles['cds']
        self.assertEqual(cds.data['x'], [])
        self.assertEqual(cds.data['y'], [])
