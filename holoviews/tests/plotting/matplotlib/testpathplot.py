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

    def test_polygon_with_hole_plot(self):
        xs = [1, 2, 3]
        ys = [2, 0, 7]
        holes = [[[(1.5, 2), (2, 3), (1.6, 1.6)], [(2.1, 4.5), (2.5, 5), (2.3, 3.5)]]]
        poly = Polygons([{'x': xs, 'y': ys, 'holes': holes}])
        plot = mpl_renderer.get_plot(poly)
        artist = plot.handles['artist']
        paths = artist.get_paths()
        self.assertEqual(len(paths), 1)
        path = paths[0]
        self.assertEqual(path.vertices, np.array([
            (1, 2), (2, 0), (3, 7), (1.5, 2), (2, 3), (1.6, 1.6),
            (2.1, 4.5), (2.5, 5), (2.3, 3.5)])
        )
        self.assertEqual(path.codes, np.array([1, 2, 2, 1, 2, 2, 1, 2, 2]))

    def test_multi_polygon_hole_plot(self):
        xs = [1, 2, 3, np.nan, 6, 7, 3]
        ys = [2, 0, 7, np.nan, 7, 5, 2]
        holes = [
            [[(1.5, 2), (2, 3), (1.6, 1.6)], [(2.1, 4.5), (2.5, 5), (2.3, 3.5)]],
            []
        ]
        poly = Polygons([{'x': xs, 'y': ys, 'holes': holes, 'value': 1}], vdims=['value'])
        plot = mpl_renderer.get_plot(poly)
        artist = plot.handles['artist']
        self.assertEqual(artist.get_array(), np.array([1, 1]))
        paths = artist.get_paths()
        self.assertEqual(len(paths), 2)
        path = paths[0]
        self.assertEqual(path.vertices, np.array([
            (1, 2), (2, 0), (3, 7), (1.5, 2), (2, 3), (1.6, 1.6),
            (2.1, 4.5), (2.5, 5), (2.3, 3.5)])
        )
        self.assertEqual(path.codes, np.array([1, 2, 2, 1, 2, 2, 1, 2, 2]))
        path2 = paths[1]
        self.assertEqual(path2.vertices, np.array([(6, 7), (7, 5), (3, 2)]))
        self.assertEqual(path2.codes, np.array([1, 2, 2]))


class TestContoursPlot(TestMPLPlot):

    def test_contours_categorical_color(self):
        path = Contours([{('x', 'y'): np.random.rand(10, 2), 'z': cat}
                     for cat in ('B', 'A', 'B')],
                    vdims='z').opts(plot=dict(color_index='z'))
        plot = mpl_renderer.get_plot(path)
        artist = plot.handles['artist']
        self.assertEqual(artist.get_array(), np.array([1, 0, 1]))
