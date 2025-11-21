import numpy as np

from holoviews.element import QuadMesh
from holoviews.testing import assert_data_equal

from .test_plot import TestPlotlyPlot


class TestQuadMeshPlot(TestPlotlyPlot):

    def test_quadmesh_state(self):
        img = QuadMesh(([1, 2, 4], [0, 1], np.array([[0, 1, 2], [2, 3, 4]])))
        state = self._get_plot_state(img)
        assert state['data'][0]['type'] == 'heatmap'
        assert_data_equal(state['data'][0]['x'], np.array([0.5, 1.5, 3., 5.]))
        assert_data_equal(state['data'][0]['y'], np.array([-0.5, .5, 1.5]))
        assert_data_equal(state['data'][0]['z'], np.array([[0, 1, 2], [2, 3, 4]]))
        assert state['data'][0]['zmin'] == 0
        assert state['data'][0]['zmax'] == 4
        assert state['layout']['xaxis']['range'] == [0.5, 5]
        assert state['layout']['yaxis']['range'] == [-0.5, 1.5]

    def test_quadmesh_nodata(self):
        img = QuadMesh(([1, 2, 4], [0, 1],
                        np.array([[0, 1, 2], [2, 3, 4]]))).opts(nodata=0)
        state = self._get_plot_state(img)
        assert state['data'][0]['type'] == 'heatmap'
        assert_data_equal(state['data'][0]['z'], np.array([[np.nan, 1, 2], [2, 3, 4]]))

    def test_quadmesh_nodata_uint(self):
        img = QuadMesh(([1, 2, 4], [0, 1],
                        np.array([[0, 1, 2], [2, 3, 4]], dtype='uint32'))).opts(nodata=0)
        state = self._get_plot_state(img)
        assert state['data'][0]['type'] == 'heatmap'
        assert_data_equal(state['data'][0]['z'], np.array([[np.nan, 1, 2], [2, 3, 4]]))

    def test_quadmesh_state_inverted(self):
        img = QuadMesh(([1, 2, 4], [0, 1], np.array([[0, 1, 2], [2, 3, 4]]))).opts(
            invert_axes=True)
        state = self._get_plot_state(img)
        assert_data_equal(state['data'][0]['x'], np.array([-0.5, .5, 1.5]))
        assert_data_equal(state['data'][0]['y'], np.array([0.5, 1.5, 3., 5.]))
        assert_data_equal(state['data'][0]['z'], np.array([[0, 1, 2], [2, 3, 4]]).T)
        assert state['data'][0]['zmin'] == 0
        assert state['data'][0]['zmax'] == 4
        assert state['layout']['xaxis']['range'] == [-0.5, 1.5]
        assert state['layout']['yaxis']['range'] == [0.5, 5]

    def test_visible(self):
        element = QuadMesh(
            ([1, 2, 4], [0, 1], np.array([[0, 1, 2], [2, 3, 4]]))
        ).opts(visible=False)
        state = self._get_plot_state(element)
        assert state['data'][0]['visible'] is False
