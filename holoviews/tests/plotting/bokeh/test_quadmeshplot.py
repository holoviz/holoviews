import numpy as np
from bokeh.models import ColorBar

from holoviews.element import Image, QuadMesh

from .test_plot import TestBokehPlot, bokeh_renderer


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
        flattened = np.array([6, 3, np.nan, 7, 4, 1, 8, 5, 2])
        qmesh = QuadMesh((xs, ys, data)).opts(nodata=0)
        plot = bokeh_renderer.get_plot(qmesh)
        source = plot.handles['source']
        self.assertEqual(source.data['z'], flattened)

    def test_quadmesh_non_sanitized_name(self):
        # https://github.com/holoviz/holoviews/issues/6460
        xs = [0, 1, 2]
        ys = [2, 1, 0]
        data = np.array([[0,1,2], [3,4,5], [6,7,8]])
        qmesh = QuadMesh((xs, ys, data), vdims="z (x,y)")
        plot = bokeh_renderer.get_plot(qmesh)
        source = plot.handles['source']

        flattened = np.array([6, 3, 0, 7, 4, 1, 8, 5, 2])
        sanitized_name = 'z_left_parenthesis_x_comma_y_right_parenthesis'
        np.testing.assert_equal(source.data[sanitized_name], flattened)

    def test_quadmesh_non_sanitized_name_grid(self):
        # https://github.com/holoviz/holoviews/issues/6460
        xs = [0, 1, 2]
        ys = [2, 1, 0]
        data = np.array([[0,1,2], [3,4,5], [6,7,8]])
        XX, YY = np.meshgrid(xs, ys)
        qmesh = QuadMesh((XX, YY, data), vdims="z (x,y)")
        plot = bokeh_renderer.get_plot(qmesh)
        source = plot.handles['source']

        flattened = data.flatten()
        sanitized_name = 'z_left_parenthesis_x_comma_y_right_parenthesis'
        np.testing.assert_equal(source.data[sanitized_name], flattened)

    def test_quadmesh_nodata_uint(self):
        xs = [0, 1, 2]
        ys = [2, 1, 0]
        data = np.array([[0,1,2], [3,4,5], [6,7,8]], dtype='uint32')
        flattened = np.array([6, 3, np.nan, 7, 4, 1, 8, 5, 2])
        qmesh = QuadMesh((xs, ys, data)).opts(nodata=0)
        plot = bokeh_renderer.get_plot(qmesh)
        source = plot.handles['source']
        self.assertEqual(source.data['z'], flattened)

    def test_quadmesh_regular_centers(self):
        X = [0.5, 1.5]
        Y = [0.5, 1.5, 2.5]
        Z = np.array([[1., 2., 3.], [4., 5., np.nan]]).T
        LABELS = np.array([['0-0', '0-1', '0-2'], ['1-0', '1-1', '1-2']]).T
        qmesh = QuadMesh((X, Y, Z, LABELS), vdims=['Value', 'Label'])
        plot = bokeh_renderer.get_plot(qmesh.opts(tools=['hover']))
        source = plot.handles['source']
        expected = {
                'left': [0., 0., 0., 1., 1., 1.],
                'right': [1., 1., 1., 2., 2., 2.],
                'Value': [ 1.,  2.,  3.,  4.,  5., np.nan],
                'bottom': [0., 1., 2., 0., 1., 2.],
                'top': [1., 2., 3., 1., 2., 3.],
                'Label': ['0-0', '0-1', '0-2', '1-0', '1-1', '1-2'],
                'x': [0.5, 0.5, 0.5, 1.5, 1.5, 1.5],
                'y': [0.5, 1.5, 2.5, 0.5, 1.5, 2.5]}
        self.assertEqual(source.data.keys(), expected.keys())
        for key in expected:  # noqa: PLC0206
            self.assertEqual(list(source.data[key]), expected[key])

    def test_quadmesh_irregular_centers(self):
        X = [[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]]
        Y = [[0.5, 1.5, 2.5], [0.5, 1.5, 2.5]]
        Z = np.array([[1., 2., 3.], [4., 5., np.nan]])
        LABELS = np.array([['0-0', '0-1', '0-2'], ['1-0', '1-1', '1-2']])
        qmesh = QuadMesh((X, Y, Z, LABELS), vdims=['Value', 'Label'])
        plot = bokeh_renderer.get_plot(qmesh.opts(tools=['hover']))
        source = plot.handles['source']
        expected = {'xs': [[0., 1., 1., 0.], [0., 1., 1., 0.], [0., 1., 1., 0.],
                           [1., 2., 2., 1.], [1., 2., 2., 1.]],
                    'ys': [[0., 0., 1., 1.], [1., 1., 2., 2.], [2., 2., 3., 3.],
                           [0., 0., 1., 1.], [1., 1., 2., 2.]],
                    'Value': [1., 2., 3., 4., 5.],
                    'Label': ['0-0', '0-1', '0-2', '1-0', '1-1'],
                    'x': [0.5, 0.5, 0.5, 1.5, 1.5],
                    'y': [0.5, 1.5, 2.5, 0.5, 1.5]}
        self.assertEqual(source.data.keys(), expected.keys())
        for key in expected:  # noqa: PLC0206
            self.assertEqual(list(source.data[key]), expected[key])

    def test_quadmesh_irregular_edges(self):
        X = [[0., 0., 0., 0.], [1., 1., 1., 1.], [2., 2., 2., 2.]]
        Y = [[0., 1., 2., 3.], [0., 1., 2., 3.], [0., 1., 2., 3.]]
        Z = np.array([[1., 2., 3.], [4., 5., np.nan]])
        LABELS = np.array([['0-0', '0-1', '0-2'], ['1-0', '1-1', '1-2']])
        qmesh = QuadMesh((X, Y, Z, LABELS), vdims=['Value', 'Label'])
        plot = bokeh_renderer.get_plot(qmesh.opts(tools=['hover']))
        source = plot.handles['source']
        expected = {'xs': [[0., 1., 1., 0.], [0., 1., 1., 0.], [0., 1., 1., 0.],
                           [1., 2., 2., 1.], [1., 2., 2., 1.]],
                    'ys': [[0., 0., 1., 1.], [1., 1., 2., 2.], [2., 2., 3., 3.],
                           [0., 0., 1., 1.], [1., 1., 2., 2.]],
                    'Value': [1., 2., 3., 4., 5.],
                    'Label': ['0-0', '0-1', '0-2', '1-0', '1-1'],
                    'x': [0.5, 0.5, 0.5, 1.5, 1.5],
                    'y': [0.5, 1.5, 2.5, 0.5, 1.5]}
        self.assertEqual(source.data.keys(), expected.keys())
        for key in expected:  # noqa: PLC0206
            self.assertEqual(list(source.data[key]), expected[key])
