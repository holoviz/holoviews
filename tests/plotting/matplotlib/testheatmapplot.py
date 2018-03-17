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

    def test_heatmap_xmarks_int(self):
        hmap = HeatMap([('A',1, 1), ('B', 2, 2)]).options(xmarks=2)
        plot = mpl_renderer.get_plot(hmap)
        for marker, pos in zip(plot.handles['xmarks'], (0, 1)):
            self.assertEqual(marker.get_xdata(), [pos, pos])

    def test_heatmap_xmarks_tuple(self):
        hmap = HeatMap([('A',1, 1), ('B', 2, 2)]).options(xmarks=('A', 'B'))
        plot = mpl_renderer.get_plot(hmap)
        for marker, pos in zip(plot.handles['xmarks'], (0, 1)):
            self.assertEqual(marker.get_xdata(), [pos, pos])

    def test_heatmap_xmarks_list(self):
        hmap = HeatMap([('A',1, 1), ('B', 2, 2)]).options(xmarks=[0, 1])
        plot = mpl_renderer.get_plot(hmap)
        for marker, pos in zip(plot.handles['xmarks'], (0, 1)):
            self.assertEqual(marker.get_xdata(), [pos, pos])

    def test_heatmap_ymarks_int(self):
        hmap = HeatMap([('A',1, 1), ('B', 2, 2)]).options(ymarks=2)
        plot = mpl_renderer.get_plot(hmap)
        for marker, pos in zip(plot.handles['ymarks'], (0, 1)):
            self.assertEqual(marker.get_ydata(), [pos, pos])

    def test_heatmap_ymarks_tuple(self):
        hmap = HeatMap([('A',1, 1), ('B', 2, 2)]).options(ymarks=('A', 'B'))
        plot = mpl_renderer.get_plot(hmap)
        for marker, pos in zip(plot.handles['ymarks'], (0, 1)):
            self.assertEqual(marker.get_ydata(), [pos, pos])

    def test_heatmap_ymarks_list(self):
        hmap = HeatMap([('A',1, 1), ('B', 2, 2)]).options(ymarks=[0, 1])
        plot = mpl_renderer.get_plot(hmap)
        for marker, pos in zip(plot.handles['ymarks'], (0, 1)):
            self.assertEqual(marker.get_ydata(), [pos, pos])
