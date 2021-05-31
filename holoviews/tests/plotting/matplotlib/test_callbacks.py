from collections import deque

import numpy as np

from holoviews.core import DynamicMap
from holoviews.element import Points, Curve
from holoviews.streams import PointerXY, PointerX


from .test_plot import TestMPLPlot, mpl_renderer


class TestCallbackPlot(TestMPLPlot):

    def test_dynamic_streams_refresh(self):
        stream = PointerXY(x=0, y=0)
        dmap = DynamicMap(lambda x, y: Points([(x, y)]),
                             kdims=[], streams=[stream])
        plot = mpl_renderer.get_plot(dmap)
        pre = mpl_renderer(plot, fmt='png')
        plot.state.set_dpi(72)
        stream.event(x=1, y=1)
        post = mpl_renderer(plot, fmt='png')
        self.assertNotEqual(pre, post)

    def test_stream_callback_single_call(self):
        def history_callback(x, history=deque(maxlen=10)):
            history.append(x)
            return Curve(list(history))
        stream = PointerX(x=0)
        dmap = DynamicMap(history_callback, kdims=[], streams=[stream])
        plot = mpl_renderer.get_plot(dmap)
        mpl_renderer(plot)
        for i in range(20):
            plot.state.set_dpi(72)
            stream.event(x=i)
        x, y = plot.handles['artist'].get_data()
        self.assertEqual(x, np.arange(10))
        self.assertEqual(y, np.arange(10, 20))
