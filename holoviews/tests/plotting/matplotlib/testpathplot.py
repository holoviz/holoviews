import numpy as np

from holoviews.core import NdOverlay
from holoviews.element import Polygons, Contours

from .testplot import TestMPLPlot, mpl_renderer


class TestPolygonPlot(TestMPLPlot):

    def test_polygons_colored(self):
        polygons = NdOverlay({j: Polygons([[(i**j, i) for i in range(10)]], level=j)
                              for j in range(5)})
        plot = mpl_renderer.get_plot(polygons)
        for j, splot in enumerate(plot.subplots.values()):
            artist = splot.handles['artist']
            self.assertEqual(artist.get_array(), np.array([j]))
            self.assertEqual(artist.get_clim(), (0, 4))


class TestContoursPlot(TestMPLPlot):

    def test_contours_categorical_color(self):
        path = Contours([{('x', 'y'): np.random.rand(10, 2), 'z': cat}
                     for cat in ('B', 'A', 'B')],
                    vdims='z').opts(plot=dict(color_index='z'))
        plot = mpl_renderer.get_plot(path)
        artist = plot.handles['artist']
        self.assertEqual(artist.get_array(), np.array([1, 0, 1]))
