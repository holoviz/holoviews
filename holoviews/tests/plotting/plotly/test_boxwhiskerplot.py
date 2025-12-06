import numpy as np

from holoviews.element import BoxWhisker
from holoviews.testing import assert_data_equal

from .test_plot import TestPlotlyPlot


class TestBoxWhiskerPlot(TestPlotlyPlot):

    def test_boxwhisker_single(self):
        box = BoxWhisker([1, 1, 2, 3, 3, 4, 5, 5])
        state = self._get_plot_state(box)
        assert len(state['data']) == 1
        assert state['data'][0]['type'] == 'box'
        assert state['data'][0]['name'] == ''
        assert_data_equal(state['data'][0]['y'], np.array([1, 1, 2, 3, 3, 4, 5, 5]))
        assert state['layout'].get('xaxis', {}) == {}
        assert state['layout']['yaxis']['range'] == [1, 5]
        assert state['layout']['yaxis']['title']['text'] == 'y'

    def test_boxwhisker_single_invert_axes(self):
        box = BoxWhisker([1, 1, 2, 3, 3, 4, 5, 5]).opts(invert_axes=True)
        state = self._get_plot_state(box)
        assert len(state['data']) == 1
        assert state['data'][0]['type'] == 'box'
        assert state['data'][0]['name'] == ''
        assert_data_equal(state['data'][0]['x'], np.array([1, 1, 2, 3, 3, 4, 5, 5]))
        assert state['layout'].get('yaxis', {}) == {}
        assert state['layout']['xaxis']['range'] == [1, 5]
        assert state['layout']['xaxis']['title']['text'] == 'y'

    def test_boxwhisker_multi(self):
        box = BoxWhisker((['A']*8+['B']*8, [1, 1, 2, 3, 3, 4, 5, 5]*2), 'x', 'y')
        state = self._get_plot_state(box)
        assert len(state['data']) == 2
        assert state['data'][0]['type'] == 'box'
        assert state['data'][0]['name'] == 'A'
        assert_data_equal(state['data'][0]['y'], np.array([1, 1, 2, 3, 3, 4, 5, 5]))
        assert state['data'][1]['type'] == 'box'
        assert state['data'][1]['name'] == 'B'
        assert_data_equal(state['data'][1]['y'], np.array([1, 1, 2, 3, 3, 4, 5, 5]))
        assert state['layout']['xaxis']['title']['text'] == 'x'
        assert state['layout']['yaxis']['range'] == [1, 5]
        assert state['layout']['yaxis']['title']['text'] == 'y'

    def test_boxwhisker_multi_invert_axes(self):
        box = BoxWhisker((['A']*8+['B']*8, [1, 1, 2, 3, 3, 4, 5, 5]*2), 'x', 'y').opts(
            invert_axes=True)
        state = self._get_plot_state(box)
        assert len(state['data']) == 2
        assert state['data'][0]['type'] == 'box'
        assert state['data'][0]['name'] == 'A'
        assert_data_equal(state['data'][0]['x'], np.array([1, 1, 2, 3, 3, 4, 5, 5]))
        assert state['data'][1]['type'] == 'box'
        assert state['data'][1]['name'] == 'B'
        assert_data_equal(state['data'][1]['x'], np.array([1, 1, 2, 3, 3, 4, 5, 5]))
        assert state['layout']['yaxis']['title']['text'] == 'x'
        assert state['layout']['xaxis']['range'] == [1, 5]
        assert state['layout']['xaxis']['title']['text'] == 'y'

    def test_visible(self):
        element = BoxWhisker(([3, 2, 1], [0, 1, 2])).opts(visible=False)
        state = self._get_plot_state(element)
        assert state['data'][0]['visible'] is False
