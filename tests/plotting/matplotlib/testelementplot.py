import numpy as np

from holoviews.core.spaces import DynamicMap
from holoviews.element import Image, Curve
from holoviews.streams import Stream

from .testplot import TestMPLPlot, mpl_renderer


class TestElementPlot(TestMPLPlot):

    def test_stream_cleanup(self):
        stream = Stream.define(str('Test'), test=1)()
        dmap = DynamicMap(lambda test: Curve([]), streams=[stream])
        plot = mpl_renderer.get_plot(dmap)
        self.assertTrue(bool(stream._subscribers))
        plot.cleanup()
        self.assertFalse(bool(stream._subscribers))


class TestColorbarPlot(TestMPLPlot):

    def test_colormapper_unsigned_int(self):
        img = Image(np.array([[1, 1, 1, 2], [2, 2, 3, 4]]).astype('uint16'))
        plot = mpl_renderer.get_plot(img)
        artist = plot.handles['artist']
        self.assertEqual(artist.get_clim(), (1, 4))

    def test_colormapper_symmetric(self):
        img = Image(np.array([[0, 1], [2, 3]])).options(symmetric=True)
        plot = mpl_renderer.get_plot(img)
        artist = plot.handles['artist']
        self.assertEqual(artist.get_clim(), (-3, 3))

    def test_colormapper_clims(self):
        img = Image(np.array([[0, 1], [2, 3]])).options(clims=(0, 4))
        plot = mpl_renderer.get_plot(img)
        artist = plot.handles['artist']
        self.assertEqual(artist.get_clim(), (0, 4))
        
    def test_colormapper_color_levels(self):
        img = Image(np.array([[0, 1], [2, 3]])).options(color_levels=5)
        plot = mpl_renderer.get_plot(img)
        artist = plot.handles['artist']
        self.assertEqual(len(artist.cmap.colors), 5)

    def test_colormapper_transparent_nan(self):
        img = Image(np.array([[0, 1], [2, 3]])).options(clipping_colors={'NaN': 'transparent'})
        plot = mpl_renderer.get_plot(img)
        cmap = plot.handles['artist'].cmap
        self.assertEqual(cmap._rgba_bad, (1.0, 1.0, 1.0, 0))

    def test_colormapper_min_max_colors(self):
        img = Image(np.array([[0, 1], [2, 3]])).options(clipping_colors={'min': 'red', 'max': 'blue'})
        plot = mpl_renderer.get_plot(img)
        cmap = plot.handles['artist'].cmap
        print(dir(cmap))
        self.assertEqual(cmap._rgba_under, (1.0, 0, 0, 1))
        self.assertEqual(cmap._rgba_over, (0, 0, 1.0, 1))
