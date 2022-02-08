import numpy as np

from holoviews.element import QuadMesh, Image, Dataset

from .test_plot import TestMPLPlot, mpl_renderer


class TestQuadMeshPlot(TestMPLPlot):

    def test_quadmesh_invert_axes(self):
        arr = np.array([[0, 1, 2], [3, 4,  5]])
        qmesh = QuadMesh(Image(arr)).opts(plot=dict(invert_axes=True))
        plot = mpl_renderer.get_plot(qmesh)
        artist = plot.handles['artist']
        self.assertEqual(artist.get_array().data, arr.T[:, ::-1].flatten())

    def test_quadmesh_nodata(self):
        arr = np.array([[0, 1, 2], [3, 4,  5]])
        qmesh = QuadMesh(Image(arr)).opts(nodata=0)
        plot = mpl_renderer.get_plot(qmesh)
        artist = plot.handles['artist']
        self.assertEqual(artist.get_array().data,
                         np.array([3, 4, 5, np.NaN, 1, 2]))

    def test_quadmesh_nodata_uint(self):
        arr = np.array([[0, 1, 2], [3, 4,  5]], dtype='uint32')
        qmesh = QuadMesh(Image(arr)).opts(nodata=0)
        plot = mpl_renderer.get_plot(qmesh)
        artist = plot.handles['artist']
        self.assertEqual(artist.get_array().data,
                         np.array([3, 4, 5, np.NaN, 1, 2]))

    def test_quadmesh_update_cbar(self):
        xs = ys = np.linspace(0, 6, 10)
        zs = np.linspace(1, 2, 5)
        XS, YS, ZS = np.meshgrid(xs, ys, zs)
        values = np.sin(XS) * ZS
        ds = Dataset((xs, ys, zs, values.T), ['x', 'y', 'z'], 'values')
        hmap = ds.to(QuadMesh).options(colorbar=True, framewise=True)
        plot = mpl_renderer.get_plot(hmap)
        cbar = plot.handles['cbar']
        self.assertEqual((cbar.vmin, cbar.vmax), (-0.998954917097928, 0.971937901363312))
        plot.update(3)
        self.assertEqual((cbar.vmin, cbar.vmax), (-1.748171104921374, 1.700891327385797))
