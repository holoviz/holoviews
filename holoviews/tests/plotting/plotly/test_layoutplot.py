import numpy as np

from holoviews.element import Curve, Image

from .testplot import TestPlotlyPlot, plotly_renderer


class TestLayoutPlot(TestPlotlyPlot):

    def test_layout_instantiate_subplots(self):
        layout = (Curve(range(10)) + Curve(range(10)) + Image(np.random.rand(10,10)) +
                  Curve(range(10)) + Curve(range(10)))
        plot = plotly_renderer.get_plot(layout)
        positions = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
        self.assertEqual(sorted(plot.subplots.keys()), positions)

    def test_layout_instantiate_subplots_transposed(self):
        layout = (Curve(range(10)) + Curve(range(10)) + Image(np.random.rand(10,10)) +
                  Curve(range(10)) + Curve(range(10)))
        plot = plotly_renderer.get_plot(layout.options(transpose=True))
        positions = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]
        self.assertEqual(sorted(plot.subplots.keys()), positions)
    
    def test_layout_state(self):
        layout = Curve([1, 2, 3]) + Curve([2, 4, 6])
        state = self._get_plot_state(layout)
        self.assertEqual(state['data'][0]['y'], np.array([1, 2, 3]))
        self.assertEqual(state['data'][0]['yaxis'], 'y')
        self.assertEqual(state['data'][1]['y'], np.array([2, 4, 6]))
        self.assertEqual(state['data'][1]['yaxis'], 'y2')
