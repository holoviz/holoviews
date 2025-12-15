from collections import deque

import numpy as np

from holoviews.core import DynamicMap
from holoviews.element import Curve, Points
from holoviews.streams import PointerX, PointerXY
from holoviews.testing import assert_data_equal

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
        assert pre != post

    def test_stream_callback_single_call(self):
        history = deque(maxlen=10)
        def history_callback(x):
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
        assert_data_equal(x, np.arange(10))
        assert_data_equal(y, np.arange(10, 20))
