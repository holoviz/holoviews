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

    def test_spread_padding_square(self):
        spread = Spread([(1, 1, 0.5), (2, 2, 0.5), (3, 3, 0.5)]).options(padding=0.1)
        plot = bokeh_renderer.get_plot(spread)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 0.8)
        self.assertEqual(x_range.end, 3.2)
        self.assertEqual(y_range.start, 0.19999999999999996)
        self.assertEqual(y_range.end, 3.8)

    def test_spread_padding_hard_range(self):
        spread = Spread([(1, 1, 0.5), (2, 2, 0.5), (3, 3, 0.5)]).redim.range(y=(0, 4)).options(padding=0.1)
        plot = bokeh_renderer.get_plot(spread)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 0.8)
        self.assertEqual(x_range.end, 3.2)
        self.assertEqual(y_range.start, 0)
        self.assertEqual(y_range.end, 4)

    def test_spread_padding_soft_range(self):
        spread = Spread([(1, 1, 0.5), (2, 2, 0.5), (3, 3, 0.5)]).redim.soft_range(y=(0, 3.5)).options(padding=0.1)
        plot = bokeh_renderer.get_plot(spread)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 0.8)
        self.assertEqual(x_range.end, 3.2)
        self.assertEqual(y_range.start, 0)
        self.assertEqual(y_range.end, 3.5)

    def test_spread_padding_nonsquare(self):
        spread = Spread([(1, 1, 0.5), (2, 2, 0.5), (3, 3, 0.5)]).options(padding=0.1, width=600)
        plot = bokeh_renderer.get_plot(spread)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 0.9)
        self.assertEqual(x_range.end, 3.1)
        self.assertEqual(y_range.start, 0.19999999999999996)
        self.assertEqual(y_range.end, 3.8)

    def test_spread_padding_logx(self):
        spread = Spread([(1, 1, 0.5), (2, 2, 0.5), (3,3, 0.5)]).options(padding=0.1, logx=True)
        plot = bokeh_renderer.get_plot(spread)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 0.89595845984076228)
        self.assertEqual(x_range.end, 3.3483695221017129)
        self.assertEqual(y_range.start, 0.19999999999999996)
        self.assertEqual(y_range.end, 3.8)
    
    def test_spread_padding_logy(self):
        spread = Spread([(1, 1, 0.5), (2, 2, 0.5), (3, 3, 0.5)]).options(padding=0.1, logy=True)
        plot = bokeh_renderer.get_plot(spread)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 0.8)
        self.assertEqual(x_range.end, 3.2)
        self.assertEqual(y_range.start, 0.41158562699652224)
        self.assertEqual(y_range.end, 4.2518491541367327)
