import numpy as np

from holoviews.element import QuadMesh

from .test_plot import TestPlotlyPlot


class TestQuadMeshPlot(TestPlotlyPlot):

    def test_quadmesh_state(self):
        img = QuadMesh(([1, 2, 4], [0, 1], np.array([[0, 1, 2], [2, 3, 4]])))
        state = self._get_plot_state(img)
        self.assertEqual(state['data'][0]['type'], 'heatmap')
        self.assertEqual(state['data'][0]['x'], np.array([0.5, 1.5, 3., 5.]))
        self.assertEqual(state['data'][0]['y'], np.array([-0.5, .5, 1.5]))
        self.assertEqual(state['data'][0]['z'], np.array([[0, 1, 2], [2, 3, 4]]))
        self.assertEqual(state['data'][0]['zmin'], 0)
        self.assertEqual(state['data'][0]['zmax'], 4)
        self.assertEqual(state['layout']['xaxis']['range'], [0.5, 5])
        self.assertEqual(state['layout']['yaxis']['range'], [-0.5, 1.5])

    def test_quadmesh_nodata(self):
        img = QuadMesh(([1, 2, 4], [0, 1],
                        np.array([[0, 1, 2], [2, 3, 4]]))).opts(nodata=0)
        state = self._get_plot_state(img)
        self.assertEqual(state['data'][0]['type'], 'heatmap')
        self.assertEqual(state['data'][0]['z'], np.array([[np.nan, 1, 2], [2, 3, 4]]))

    def test_quadmesh_nodata_uint(self):
        img = QuadMesh(([1, 2, 4], [0, 1],
                        np.array([[0, 1, 2], [2, 3, 4]], dtype='uint32'))).opts(nodata=0)
        state = self._get_plot_state(img)
        self.assertEqual(state['data'][0]['type'], 'heatmap')
        self.assertEqual(state['data'][0]['z'], np.array([[np.nan, 1, 2], [2, 3, 4]]))

    def test_quadmesh_state_inverted(self):
        img = QuadMesh(([1, 2, 4], [0, 1], np.array([[0, 1, 2], [2, 3, 4]]))).opts(
            invert_axes=True)
        state = self._get_plot_state(img)
        self.assertEqual(state['data'][0]['x'], np.array([-0.5, .5, 1.5]))
        self.assertEqual(state['data'][0]['y'], np.array([0.5, 1.5, 3., 5.]))
        self.assertEqual(state['data'][0]['z'], np.array([[0, 1, 2], [2, 3, 4]]).T)
        self.assertEqual(state['data'][0]['zmin'], 0)
        self.assertEqual(state['data'][0]['zmax'], 4)
        self.assertEqual(state['layout']['xaxis']['range'], [-0.5, 1.5])
        self.assertEqual(state['layout']['yaxis']['range'], [0.5, 5])

    def test_visible(self):
        element = QuadMesh(
            ([1, 2, 4], [0, 1], np.array([[0, 1, 2], [2, 3, 4]]))
        ).opts(visible=False)
        state = self._get_plot_state(element)
        self.assertEqual(state['data'][0]['visible'], False)
