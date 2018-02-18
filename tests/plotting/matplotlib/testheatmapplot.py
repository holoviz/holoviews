import numpy as np

from holoviews.element import HeatMap, Image

from .testplot import TestMPLPlot, mpl_renderer


class TestLayoutPlot(TestMPLPlot):

    def test_heatmap_invert_axes(self):
        arr = np.array([[0, 1, 2], [3, 4, 5]])
        hm = HeatMap(Image(arr)).opts(plot=dict(invert_axes=True))
        plot = mpl_renderer.get_plot(hm)
        artist = plot.handles['artist']
        self.assertEqual(artist.get_array().data, arr.T[::-1, ::-1])
        self.assertEqual(artist.get_extent(), (0, 2, 0, 3))
