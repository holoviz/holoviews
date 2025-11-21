import numpy as np

from holoviews.element import Scatter3D
from holoviews.testing import assert_data_equal

from .test_plot import TestPlotlyPlot


class TestScatter3DPlot(TestPlotlyPlot):

    def test_scatter3d_state(self):
        scatter = Scatter3D(([0,1], [2,3], [4,5]))
        state = self._get_plot_state(scatter)
        assert_data_equal(state['data'][0]['x'], np.array([0, 1]))
        assert_data_equal(state['data'][0]['y'], np.array([2, 3]))
        assert_data_equal(state['data'][0]['z'], np.array([4, 5]))
        assert state['layout']['scene']['xaxis']['range'] == [0, 1]
        assert state['layout']['scene']['yaxis']['range'] == [2, 3]
        assert state['layout']['scene']['zaxis']['range'] == [4, 5]

    def test_scatter3d_color_mapped(self):
        scatter = Scatter3D(([0,1], [2,3], [4,5])).opts(color='y')
        state = self._get_plot_state(scatter)
        assert_data_equal(state['data'][0]['marker']['color'], np.array([2, 3]))
        assert state['data'][0]['marker']['cmin'] == 2
        assert state['data'][0]['marker']['cmax'] == 3

    def test_scatter3d_size(self):
        scatter = Scatter3D(([0,1], [2,3], [4,5])).opts(size='y')
        state = self._get_plot_state(scatter)
        assert_data_equal(state['data'][0]['marker']['size'], np.array([2, 3]))

    def test_visible(self):
        element = Scatter3D(([0, 1], [2, 3], [4, 5])).opts(visible=False)
        state = self._get_plot_state(element)
        assert state['data'][0]['visible'] is False
