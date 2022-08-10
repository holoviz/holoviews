import numpy as np

from holoviews.element import QuadMesh, Image

from .test_plot import TestBokehPlot, bokeh_renderer

from bokeh.models import ColorBar


class TestQuadMeshPlot(TestBokehPlot):

    def test_quadmesh_colormapping(self):
        n = 21
        xs = np.logspace(1, 3, n)
        ys = np.linspace(1, 10, n)
        qmesh = QuadMesh((xs, ys, np.random.rand(n-1, n-1)))
        self._test_colormapping(qmesh, 2)

    def test_quadmesh_invert_axes(self):
        arr = np.array([[0, 1, 2], [3, 4, 5]])
        qmesh = QuadMesh(Image(arr)).opts(invert_axes=True, tools=['hover'])
        plot = bokeh_renderer.get_plot(qmesh)
        source = plot.handles['source']
        self.assertEqual(source.data['z'], qmesh.dimension_values(2, flat=False).flatten())
        self.assertEqual(source.data['x'], qmesh.dimension_values(0))
        self.assertEqual(source.data['y'], qmesh.dimension_values(1))

    def test_quadmesh_colorbar(self):
        n = 21
        xs = np.logspace(1, 3, n)
        ys = np.linspace(1, 10, n)
        qmesh = QuadMesh((xs, ys, np.random.rand(n-1, n-1))).opts(colorbar=True)
        plot = bokeh_renderer.get_plot(qmesh)
        self.assertIsInstance(plot.handles['colorbar'], ColorBar)
        self.assertIs(plot.handles['colorbar'].color_mapper, plot.handles['color_mapper'])

    def test_quadmesh_inverted_coords(self):
        xs = [0, 1, 2]
        ys = [2, 1, 0]
        qmesh = QuadMesh((xs, ys, np.random.rand(3, 3)))
        plot = bokeh_renderer.get_plot(qmesh)
        source = plot.handles['source']
        self.assertEqual(source.data['z'], qmesh.dimension_values(2, flat=False).T.flatten())
        self.assertEqual(source.data['left'], np.array([-0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5]))
        self.assertEqual(source.data['right'], np.array([0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 2.5, 2.5, 2.5]))
        self.assertEqual(source.data['top'], np.array([0.5, 1.5, 2.5, 0.5, 1.5, 2.5, 0.5, 1.5, 2.5]))
        self.assertEqual(source.data['bottom'], np.array([-0.5, 0.5, 1.5, -0.5, 0.5, 1.5, -0.5, 0.5, 1.5]))

    def test_quadmesh_nodata(self):
        xs = [0, 1, 2]
        ys = [2, 1, 0]
        data = np.array([[0,1,2], [3,4,5], [6,7,8]])
        flattened = np.array([6, 3, np.NaN, 7, 4, 1, 8, 5, 2])
        qmesh = QuadMesh((xs, ys, data)).opts(nodata=0)
        plot = bokeh_renderer.get_plot(qmesh)
        source = plot.handles['source']
        self.assertEqual(source.data['z'], flattened)

    def test_quadmesh_nodata_uint(self):
        xs = [0, 1, 2]
        ys = [2, 1, 0]
        data = np.array([[0,1,2], [3,4,5], [6,7,8]], dtype='uint32')
        flattened = np.array([6, 3, np.NaN, 7, 4, 1, 8, 5, 2])
        qmesh = QuadMesh((xs, ys, data)).opts(nodata=0)
        plot = bokeh_renderer.get_plot(qmesh)
        source = plot.handles['source']
        self.assertEqual(source.data['z'], flattened)
