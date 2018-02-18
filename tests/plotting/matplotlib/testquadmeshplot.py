import numpy as np

from holoviews.element import QuadMesh, Image

from .testplot import TestMPLPlot, mpl_renderer


class TestQuadMeshPlot(TestMPLPlot):

    def test_quadmesh_invert_axes(self):
        arr = np.array([[0, 1, 2], [3, 4,  5]])
        qmesh = QuadMesh(Image(arr)).opts(plot=dict(invert_axes=True))
        plot = mpl_renderer.get_plot(qmesh)
        artist = plot.handles['artist']
        self.assertEqual(artist.get_array().data, arr.T[:, ::-1].flatten())
