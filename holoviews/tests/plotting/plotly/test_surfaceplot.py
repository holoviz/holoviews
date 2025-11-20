import numpy as np

from holoviews.element import Surface
from holoviews.testing import assert_data_equal

from .test_plot import TestPlotlyPlot


class TestSurfacePlot(TestPlotlyPlot):

    def test_surface_state(self):
        img = Surface(([1, 2, 3], [0, 1], np.array([[0, 1, 2], [2, 3, 4]])))
        state = self._get_plot_state(img)
        assert state['data'][0]['type'] == 'surface'
        assert_data_equal(state['data'][0]['x'], np.array([1, 2, 3]))
        assert_data_equal(state['data'][0]['y'], np.array([0, 1]))
        assert_data_equal(state['data'][0]['z'], np.array([[0, 1, 2], [2, 3, 4]]))
        assert state['data'][0]['cmin'] == 0
        assert state['data'][0]['cmax'] == 4
        assert state['layout']['scene']['xaxis']['range'] == [0.5, 3.5]
        assert state['layout']['scene']['yaxis']['range'] == [-0.5, 1.5]
        assert state['layout']['scene']['zaxis']['range'] == [0, 4]

    def test_surface_colorbar(self):
        img = Surface(([1, 2, 3], [0, 1], np.array([[0, 1, 2], [2, 3, 4]])))
        img.opts(colorbar=True)
        state = self._get_plot_state(img)
        trace = state['data'][0]
        assert trace['showscale']

        img.opts(colorbar=False)
        state = self._get_plot_state(img)
        trace = state['data'][0]
        assert not trace['showscale']

    def test_visible(self):
        element = Surface(
            ([1, 2, 3], [0, 1], np.array([[0, 1, 2], [2, 3, 4]]))
        ).opts(visible=False)
        state = self._get_plot_state(element)
        assert state['data'][0]['visible'] is False
