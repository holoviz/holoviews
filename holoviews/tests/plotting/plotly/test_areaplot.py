import numpy as np

from holoviews.element import Area

from .test_plot import TestPlotlyPlot


class TestAreaPlot(TestPlotlyPlot):

    def test_area_to_zero_y(self):
        curve = Area([1, 2, 3])
        state = self._get_plot_state(curve)
        self.assertEqual(state['data'][0]['x'], np.array([0, 1, 2]))
        self.assertEqual(state['data'][0]['y'], np.array([1, 2, 3]))
        self.assertEqual(state['data'][0]['mode'], 'lines')
        self.assertEqual(state['data'][0]['fill'], 'tozeroy')
        self.assertEqual(state['layout']['yaxis']['range'], [0, 3])

    def test_area_to_zero_x(self):
        curve = Area([1, 2, 3]).opts(invert_axes=True)
        state = self._get_plot_state(curve)
        self.assertEqual(state['data'][0]['x'], np.array([1, 2, 3]))
        self.assertEqual(state['data'][0]['y'], np.array([0, 1, 2]))
        self.assertEqual(state['data'][0]['mode'], 'lines')
        self.assertEqual(state['data'][0]['fill'], 'tozerox')
        self.assertEqual(state['layout']['xaxis']['range'], [0, 3])
        self.assertEqual(state['layout']['yaxis']['range'], [0, 2])
        self.assertEqual(state['layout']['xaxis']['title']['text'], 'y')
        self.assertEqual(state['layout']['yaxis']['title']['text'], 'x')

    def test_area_fill_between_ys(self):
        area = Area([(0, 1, 0.5), (1, 2, 1), (2, 3, 2.25)], vdims=['y', 'y2'])
        state = self._get_plot_state(area)
        self.assertEqual(state['data'][0]['y'], np.array([0.5, 1, 2.25]))
        self.assertEqual(state['data'][0]['mode'], 'lines')
        self.assertEqual(state['data'][0].get('fill', None), None)
        self.assertEqual(state['data'][1]['y'], np.array([1, 2, 3]))
        self.assertEqual(state['data'][1]['mode'], 'lines')
        self.assertEqual(state['data'][1]['fill'], 'tonexty')
        self.assertEqual(state['layout']['yaxis']['range'], [0.5, 3])

    def test_area_fill_between_xs(self):
        area = Area([(0, 1, 0.5), (1, 2, 1), (2, 3, 2.25)], vdims=['y', 'y2']).opts(invert_axes=True)
        state = self._get_plot_state(area)
        self.assertEqual(state['data'][0]['x'], np.array([0.5, 1, 2.25]))
        self.assertEqual(state['data'][0]['mode'], 'lines')
        self.assertEqual(state['data'][0].get('fill', None), None)
        self.assertEqual(state['data'][1]['x'], np.array([1, 2, 3]))
        self.assertEqual(state['data'][1]['mode'], 'lines')
        self.assertEqual(state['data'][1]['fill'], 'tonextx')
        self.assertEqual(state['layout']['xaxis']['range'], [0.5, 3])
        self.assertEqual(state['layout']['yaxis']['range'], [0, 2])

    def test_area_visible(self):
        curve = Area([1, 2, 3]).opts(visible=False)
        state = self._get_plot_state(curve)
        self.assertEqual(state['data'][0]['visible'], False)
