import numpy as np

from holoviews.element import Dataset, Image, QuadMesh

from .test_plot import MPL_GE_3_8_0, TestMPLPlot, mpl_renderer


class TestQuadMeshPlot(TestMPLPlot):

    def test_quadmesh_invert_axes(self):
        arr = np.array([[0, 1, 2], [3, 4, 5]])
        qmesh = QuadMesh(Image(arr)).opts(invert_axes=True)
        plot = mpl_renderer.get_plot(qmesh)
        artist = plot.handles['artist']
        if MPL_GE_3_8_0:
            np.testing.assert_equal(artist.get_array().data, arr[::-1].T)
        else:
            np.testing.assert_equal(artist.get_array().data, arr.T[::-1].flatten())

    def test_quadmesh_nodata(self):
        arr = np.array([[0, 1, 2], [3, 4, 5]])
        qmesh = QuadMesh(Image(arr)).opts(nodata=0)
        plot = mpl_renderer.get_plot(qmesh)
        artist = plot.handles['artist']
        if MPL_GE_3_8_0:
            expected = np.array([[3, 4, 5], [np.nan, 1, 2]])
        else:
            expected = np.array([3, 4, 5, np.nan, 1, 2])

        np.testing.assert_equal(artist.get_array().data, expected)

    def test_quadmesh_nodata_uint(self):
        arr = np.array([[0, 1, 2], [3, 4, 5]], dtype='uint32')
        qmesh = QuadMesh(Image(arr)).opts(nodata=0)
        plot = mpl_renderer.get_plot(qmesh)
        artist = plot.handles['artist']
        if MPL_GE_3_8_0:
            expected = np.array([[3, 4, 5], [np.nan, 1, 2]])
        else:
            expected = np.array([3, 4, 5, np.nan, 1, 2])
        np.testing.assert_equal(artist.get_array().data, expected)

    def test_quadmesh_update_cbar(self):
        xs = ys = np.linspace(0, 6, 10)
        zs = np.linspace(1, 2, 5)
        XS, _YS, ZS = np.meshgrid(xs, ys, zs)
        values = np.sin(XS) * ZS
        ds = Dataset((xs, ys, zs, values.T), ['x', 'y', 'z'], 'values')
        hmap = ds.to(QuadMesh).opts(colorbar=True, framewise=True)
        plot = mpl_renderer.get_plot(hmap)
        cbar = plot.handles['cbar']
        np.testing.assert_allclose([cbar.vmin, cbar.vmax], [-0.9989549170979283, 0.9719379013633128])
        plot.update(3)
        np.testing.assert_allclose([cbar.vmin, cbar.vmax], [-1.7481711049213744, 1.7008913273857975])
