import datetime as dt

import numpy as np

from holoviews.element import Dataset
from holoviews.operation import histogram

from .testplot import TestMPLPlot, mpl_renderer


class TestCurvePlot(TestMPLPlot):

    def test_histogram_datetime64_plot(self):
        dates = np.array([dt.datetime(2017, 1, i) for i in range(1, 5)])
        hist = histogram(Dataset(dates, 'Date'), num_bins=4)
        plot = mpl_renderer.get_plot(hist)
        artist = plot.handles['artist']
        ax = plot.handles['axis']
        self.assertEqual(ax.get_xlim(), (736330.0, 736333.0))
        self.assertEqual([p.get_x() for p in artist.patches], [736330.0, 736330.75, 736331.5, 736332.25])
