import numpy as np

from holoviews.core import DynamicMap
from holoviews.element import Curve 
from holoviews.streams import Buffer

from .testplot import TestBokehPlot, bokeh_renderer


class TestBufferStreamPlot(TestBokehPlot):

    def test_buffer_stream_following(self):
        stream = Buffer(data={'x': np.array([1]), 'y': np.array([1])}, following=True)
        dmap = DynamicMap(Curve, streams=[stream])
        
        plot = bokeh_renderer.get_plot(dmap)

        x_range = plot.handles['x_range']
        y_range = plot.handles['y_range']

        self.assertEqual(x_range.start, 0)
        self.assertEqual(x_range.end, 2)
        self.assertEqual(y_range.start, 0)
        self.assertEqual(y_range.end, 2)

        stream.send({'x': np.array([2]), 'y': np.array([-1])})

        self.assertEqual(x_range.start, 1)
        self.assertEqual(x_range.end, 2)
        self.assertEqual(y_range.start, -1)
        self.assertEqual(y_range.end, 1)

        stream.following = False

        stream.send({'x': np.array([3]), 'y': np.array([3])})

        self.assertEqual(x_range.start, 1)
        self.assertEqual(x_range.end, 2)
        self.assertEqual(y_range.start, -1)
        self.assertEqual(y_range.end, 1)
