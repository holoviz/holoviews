from unittest import SkipTest

import numpy as np

from holoviews.element import Raster, Image, ImageStack
from holoviews.plotting.mpl.raster import RGBPlot

from .test_plot import TestMPLPlot, mpl_renderer

from matplotlib.colors import ListedColormap


class TestRasterPlot(TestMPLPlot):

    def test_raster_invert_axes(self):
        arr = np.array([[0, 1, 2], [3, 4, 5]])
        raster = Raster(arr).opts(invert_axes=True)
        plot = mpl_renderer.get_plot(raster)
        artist = plot.handles['artist']
        np.testing.assert_equal(artist.get_array().data, arr.T[::-1])
        assert artist.get_extent() == [0, 2, 0, 3]

    def test_raster_nodata(self):
        arr = np.array([[0, 1, 2], [3, 4, 5]])
        expected = np.array([[3, 4, 5],
                             [np.NaN, 1, 2]])

        raster = Raster(arr).opts(nodata=0)
        plot = mpl_renderer.get_plot(raster)
        artist = plot.handles['artist']
        np.testing.assert_equal(artist.get_array().data, expected)

    def test_raster_nodata_uint(self):
        arr = np.array([[0, 1, 2], [3, 4, 5]], dtype='uint32')
        expected = np.array([[3, 4, 5],
                             [np.NaN, 1, 2]])

        raster = Raster(arr).opts(nodata=0)
        plot = mpl_renderer.get_plot(raster)
        artist = plot.handles['artist']
        np.testing.assert_equal(artist.get_array().data, expected)


    def test_image_invert_axes(self):
        arr = np.array([[0, 1, 2], [3, 4, 5]])
        raster = Image(arr).opts(invert_axes=True)
        plot = mpl_renderer.get_plot(raster)
        artist = plot.handles['artist']
        np.testing.assert_equal(artist.get_array().data, arr.T[::-1, ::-1])
        assert artist.get_extent() == [-0.5, 0.5, -0.5, 0.5]

    def test_image_listed_cmap(self):
        colors = ['#ffffff','#000000']
        img = Image(np.array([[0, 1, 2], [3, 4, 5]])).opts(cmap=colors)
        plot = mpl_renderer.get_plot(img)
        artist = plot.handles['artist']
        cmap = artist.get_cmap()
        self.assertIsInstance(cmap, ListedColormap)
        assert cmap.colors == colors

    def test_image_cbar_extend_both(self):
        img = Image(np.array([[0, 1], [2, 3]])).redim(z=dict(range=(1,2)))
        plot = mpl_renderer.get_plot(img.opts(colorbar=True))
        assert plot.handles['cbar'].extend == 'both'

    def test_image_cbar_extend_min(self):
        img = Image(np.array([[0, 1], [2, 3]])).redim(z=dict(range=(1, None)))
        plot = mpl_renderer.get_plot(img.opts(colorbar=True))
        assert plot.handles['cbar'].extend == 'min'

    def test_image_cbar_extend_max(self):
        img = Image(np.array([[0, 1], [2, 3]])).redim(z=dict(range=(None, 2)))
        plot = mpl_renderer.get_plot(img.opts(colorbar=True))
        assert plot.handles['cbar'].extend == 'max'

    def test_image_cbar_extend_clim(self):
        img = Image(np.array([[0, 1], [2, 3]])).opts(
            clim=(np.nan, np.nan), colorbar=True)
        plot = mpl_renderer.get_plot(img)
        assert plot.handles['cbar'].extend == 'neither'

    def test_image_stack(self):
        try:
            import datashader  # noqa: F401
        except ImportError:
            raise SkipTest("Test requires datashader")
        x = np.arange(0, 3)
        y = np.arange(5, 8)
        a = np.array([[np.nan, np.nan, 1], [np.nan] * 3, [np.nan] * 3])
        b = np.array([[np.nan] * 3, [1, 1, np.nan], [np.nan] * 3])
        c = np.array([[np.nan] * 3, [np.nan] * 3, [1, 1, 1]])

        img_stack = ImageStack((x, y, a, b, c), kdims=["x", "y"], vdims=["a", "b", "c"])
        plot = mpl_renderer.get_plot(img_stack)
        artist = plot.handles['artist']
        array = artist.get_array().data
        assert array.shape == (3, 3, 4)
        assert artist.get_extent() == [-0.5, 2.5, 4.5, 7.5]
        assert isinstance(plot, RGBPlot)
