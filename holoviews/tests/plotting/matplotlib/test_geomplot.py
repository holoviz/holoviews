from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.dates import date2num

import holoviews as hv

from .test_plot import TestMPLPlot, mpl_renderer


class TestSegmentPlot(TestMPLPlot):
    def test_segments_datetime_yaxis(self):
        t = pd.to_datetime(["2016-01-01 08:00", "2016-01-01 08:01", "2016-01-01 08:02"])
        segments = hv.Segments(
            pd.DataFrame({"x0": [0.0, 1.0], "y0": t[:2], "x1": [1.0, 2.0], "y1": t[1:]}),
            ["x0", "y0", "x1", "y1"],
        )
        plot = mpl_renderer.get_plot(segments)
        lines = plot.handles["artist"].get_segments()
        expected = date2num(t.to_numpy())
        np.testing.assert_array_equal(lines[0][:, 1], expected[:2])
        np.testing.assert_array_equal(lines[1][:, 1], expected[1:])
        bottom, top = plot.handles["axis"].get_ylim()
        assert bottom <= lines[0][0, 1] <= top
