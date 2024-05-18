import numpy as np

from holoviews.element import Histogram

from .test_plot import TestPlotlyPlot


class TestHistogramPlot(TestPlotlyPlot):

    def setUp(self):
        super().setUp()
        self.frequencies = [1, 3, 5, 4, 2, 0]
        self.edges = [-3, -2, -1, 0, 1, 2]

    def test_histogram_plot(self):
        hist = Histogram((self.edges, self.frequencies))
        state = self._get_plot_state(hist)
        np.testing.assert_equal(state['data'][0]['x'], self.edges)
        np.testing.assert_equal(state['data'][0]['y'], self.frequencies)
        self.assertEqual(state['data'][0]['type'], 'bar')
        self.assertEqual(state['data'][0]['orientation'], 'v')
        self.assertEqual(state['data'][0]['width'], 1)
        self.assertEqual(state['layout']['xaxis']['range'], [-3.5, 2.5])
        self.assertEqual(state['layout']['xaxis']['title']['text'], 'x')
        self.assertEqual(state['layout']['yaxis']['range'], [0, 5])
        self.assertEqual(state['layout']['yaxis']['title']['text'], 'Frequency')

    def test_histogram_plot_inverted(self):
        hist = Histogram(
            (self.edges, self.frequencies)
        ).opts(invert_axes=True)

        state = self._get_plot_state(hist)
        np.testing.assert_equal(state['data'][0]['y'], self.edges)
        np.testing.assert_equal(state['data'][0]['x'], self.frequencies)
        self.assertEqual(state['data'][0]['type'], 'bar')
        self.assertEqual(state['data'][0]['orientation'], 'h')
        self.assertEqual(state['data'][0]['width'], 1)
        self.assertEqual(state['layout']['yaxis']['range'], [-3.5, 2.5])
        self.assertEqual(state['layout']['yaxis']['title']['text'], 'x')
        self.assertEqual(state['layout']['xaxis']['range'], [0, 5])
        self.assertEqual(state['layout']['xaxis']['title']['text'], 'Frequency')

    def test_histogram_plot_styling(self):
        props = {
            'color': 'orange',
            'line_width': 7,
            'line_color': 'green',
        }
        hist = Histogram((self.edges, self.frequencies)).opts(**props)
        state = self._get_plot_state(hist)
        marker = state['data'][0]['marker']
        self.assert_property_values(marker, props)

    def test_visible(self):
        element = Histogram((self.edges, self.frequencies)).opts(visible=False)
        state = self._get_plot_state(element)
        self.assertEqual(state['data'][0]['visible'], False)
