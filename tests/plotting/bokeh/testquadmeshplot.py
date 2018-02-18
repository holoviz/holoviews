import numpy as np

from holoviews.element import QuadMesh, Image

from .testplot import TestBokehPlot, bokeh_renderer


class TestQuadMeshPlot(TestBokehPlot):

    def test_quadmesh_colormapping(self):
        n = 21
        xs = np.logspace(1, 3, n)
        ys = np.linspace(1, 10, n)
        qmesh = QuadMesh((xs, ys, np.random.rand(n-1, n-1)))
        self._test_colormapping(qmesh, 2)

    def test_quadmesh_invert_axes(self):
        arr = np.array([[0, 1, 2], [3, 4, 5]])
        qmesh = QuadMesh(Image(arr)).opts(plot=dict(invert_axes=True, tools=['hover']))
        plot = bokeh_renderer.get_plot(qmesh)
        source = plot.handles['source']
        self.assertEqual(source.data['z'], qmesh.dimension_values(2, flat=False).flatten())
        self.assertEqual(source.data['x'], qmesh.dimension_values(0))
        self.assertEqual(source.data['y'], qmesh.dimension_values(1))
