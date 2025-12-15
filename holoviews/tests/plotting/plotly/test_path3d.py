import numpy as np

from holoviews.element import Path3D
from holoviews.testing import assert_data_equal

from .test_plot import TestPlotlyPlot


class TestPath3DPlot(TestPlotlyPlot):

    def test_path3D_state(self):
        path3D = Path3D([(0, 1, 0), (1, 2, 1), (2, 3, 2)])
        state = self._get_plot_state(path3D)
        assert_data_equal(state['data'][0]['x'], np.array([0, 1, 2]))
        assert_data_equal(state['data'][0]['y'], np.array([1, 2, 3]))
        assert state['data'][0]['mode'] == 'lines'
        assert state['data'][0]['type'] == 'scatter3d'
        assert state['layout']['scene']['xaxis']['range'] == [0, 2]
        assert state['layout']['scene']['yaxis']['range'] == [1, 3]
        assert state['layout']['scene']['zaxis']['range'] == [0, 2]

    def test_path3D_multi(self):
        path3D = Path3D([[(0, 1, 0), (1, 2, 1), (2, 3, 2)], [(-1, 1, 3), (-2, 2, 4), (-3, 3, 5)]])
        state = self._get_plot_state(path3D)
        assert_data_equal(state['data'][0]['x'], np.array([0, 1, 2]))
        assert_data_equal(state['data'][0]['y'], np.array([1, 2, 3]))
        assert_data_equal(state['data'][0]['z'], np.array([0, 1, 2]))
        assert state['data'][0]['mode'] == 'lines'
        assert state['data'][0]['type'] == 'scatter3d'
        assert_data_equal(state['data'][1]['x'], np.array([-1, -2, -3]))
        assert_data_equal(state['data'][1]['y'], np.array([1, 2, 3]))
        assert_data_equal(state['data'][1]['z'], np.array([3, 4, 5]))
        assert state['data'][1]['mode'] == 'lines'
        assert state['data'][1]['type'] == 'scatter3d'
        assert state['layout']['scene']['xaxis']['range'] == [-3, 2]
        assert state['layout']['scene']['yaxis']['range'] == [1, 3]
        assert state['layout']['scene']['zaxis']['range'] == [0, 5]

    def test_path3D_multi_colors(self):
        path3D = Path3D([[(0, 1, 0, 'red'), (1, 2, 1, 'red'), (2, 3, 2, 'red')],
                         [(-1, 1, 3, 'blue'), (-2, 2, 4, 'blue'), (-3, 3, 5, 'blue')]],
                        vdims='color').opts(color='color')
        state = self._get_plot_state(path3D)
        assert state['data'][0]['line']['color'] == 'red'
        assert state['data'][1]['line']['color'] == 'blue'

    def test_visible(self):
        element = Path3D([(0, 1, 0), (1, 2, 1), (2, 3, 2)]).opts(visible=False)
        state = self._get_plot_state(element)
        assert state['data'][0]['visible'] is False
