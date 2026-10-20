from __future__ import annotations

import numpy as np
from bokeh.core.property.bases import validation_on

import holoviews as hv
from holoviews.streams import Buffer, Pipe

from .test_plot import TestBokehPlot, bokeh_renderer


class TestBufferStreamPlot(TestBokehPlot):
    def test_buffer_stream_following(self):
        stream = Buffer(data={"x": np.array([1]), "y": np.array([1])}, following=True)
        dmap = hv.DynamicMap(hv.Curve, streams=[stream])

        plot = bokeh_renderer.get_plot(dmap)

        x_range = plot.handles["x_range"]
        y_range = plot.handles["y_range"]

        assert x_range.start == 0
        assert x_range.end == 2
        assert y_range.start == 0
        assert y_range.end == 2

        stream.send({"x": np.array([2]), "y": np.array([-1])})

        assert x_range.start == 1
        assert x_range.end == 2
        assert y_range.start == -1
        assert y_range.end == 1

        stream.following = False

        stream.send({"x": np.array([3]), "y": np.array([3])})

        assert x_range.start == 1
        assert x_range.end == 2
        assert y_range.start == -1
        assert y_range.end == 1


def test_stream_update_keeps_bokeh_validation_on():
    # Bokeh's validation flag is process-wide, and since Bokeh 3.10 the server
    # runs callbacks of different sessions on concurrent threads, so toggling it
    # while updating a source would affect, and could leak into, other sessions.
    pipe = Pipe(data=[(0, 0), (1, 1)])
    plot = bokeh_renderer.get_plot(hv.DynamicMap(hv.Curve, streams=[pipe]))
    seen = []
    plot.handles["source"].on_change("data", lambda attr, old, new: seen.append(validation_on()))
    pipe.send([(0, 1), (1, 0)])
    assert seen == [True]
