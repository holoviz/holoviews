import numpy as np

from holoviews.element import Curve

from .testplot import TestPlotlyPlot


class TestCurvePlot(TestPlotlyPlot):

    def test_curve_state(self):
        curve = Curve([1, 2, 3])
        state = self._get_plot_state(curve)
        self.assertEqual(state['data'][0]['x'], np.array([0, 1, 2]))
        self.assertEqual(state['data'][0]['y'], np.array([1, 2, 3]))
        self.assertEqual(state['data'][0]['mode'], 'lines')
        self.assertEqual(state['layout']['xaxis']['range'], [0, 2])
        self.assertEqual(state['layout']['yaxis']['range'], [1, 3])

    def test_curve_inverted(self):
        curve = Curve([1, 2, 3]).options(invert_axes=True)
        state = self._get_plot_state(curve)
        self.assertEqual(state['data'][0]['x'], np.array([1, 2, 3]))
        self.assertEqual(state['data'][0]['y'], np.array([0, 1, 2]))
        self.assertEqual(state['data'][0]['mode'], 'lines')
        self.assertEqual(state['layout']['xaxis']['range'], [1, 3])
        self.assertEqual(state['layout']['yaxis']['range'], [0, 2])
        self.assertEqual(state['layout']['xaxis']['title']['text'], 'y')
        self.assertEqual(state['layout']['yaxis']['title']['text'], 'x')

    def test_curve_interpolation(self):
        curve = Curve([1, 2, 3]).options(interpolation='steps-mid')
        state = self._get_plot_state(curve)
        self.assertEqual(state['data'][0]['x'], np.array([0., 0.5, 0.5, 1.5, 1.5, 2.]))
        self.assertEqual(state['data'][0]['y'], np.array([1, 1, 2, 2, 3, 3]))

    def test_curve_color(self):
        curve = Curve([1, 2, 3]).options(color='red')
        state = self._get_plot_state(curve)
        self.assertEqual(state['data'][0]['line']['color'], 'red')

    def test_curve_color_mapping_error(self):
        curve = Curve([1, 2, 3]).options(color='x')
        with self.assertRaises(ValueError):
            self._get_plot_state(curve)

    def test_curve_dash(self):
        curve = Curve([1, 2, 3]).options(dash='dash')
        state = self._get_plot_state(curve)
        self.assertEqual(state['data'][0]['line']['dash'], 'dash')

    def test_curve_line_width(self):
        curve = Curve([1, 2, 3]).options(line_width=5)
        state = self._get_plot_state(curve)
        self.assertEqual(state['data'][0]['line']['width'], 5)

    def test_visible(self):
        element = Curve([1, 2, 3]).options(visible=False)
        state = self._get_plot_state(element)
        self.assertEqual(state['data'][0]['visible'], False)
