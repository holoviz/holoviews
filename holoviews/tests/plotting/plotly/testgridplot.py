import numpy as np

from holoviews.core.spaces import GridSpace
from holoviews.element import Scatter, Curve

from .testplot import TestPlotlyPlot


class TestGridPlot(TestPlotlyPlot):

    def test_layout_with_grid(self):
        # Create GridSpace
        grid = GridSpace({(i, j): Curve([i, j]) for i in [0, 1]
                          for j in [0, 1]})
        grid = grid.options(vspacing=0, hspacing=0)

        # Create Scatter
        scatter = Scatter([-10, 0])

        # Create Horizontal Layout
        layout = (scatter + grid).options(vspacing=0, hspacing=0)

        state = self._get_plot_state(layout)

        # Check the scatter plot on the left
        self.assertEqual(state['data'][0]['y'], np.array([-10, 0]))
        self.assertEqual(state['data'][0]['mode'], 'markers')
        self.assertEqual(state['data'][0]['xaxis'], 'x')
        self.assertEqual(state['data'][0]['yaxis'], 'y')
        self.assertEqual(state['layout']['xaxis']['range'], [0, 1])
        self.assertEqual(state['layout']['xaxis']['domain'], [0, 0.5])
        self.assertEqual(state['layout']['yaxis']['range'], [-10, 0])
        self.assertEqual(state['layout']['yaxis']['domain'], [0, 1])

        # Check the grid plot on the right

        # (0, 0) - bottom-left
        self.assertEqual(state['data'][1]['y'], np.array([0, 0]))
        self.assertEqual(state['data'][1]['mode'], 'lines')
        self.assertEqual(state['data'][1]['xaxis'], 'x2')
        self.assertEqual(state['data'][1]['yaxis'], 'y2')

        # (1, 0) - bottom-right
        self.assertEqual(state['data'][2]['y'], np.array([1, 0]))
        self.assertEqual(state['data'][2]['mode'], 'lines')
        self.assertEqual(state['data'][2]['xaxis'], 'x3')
        self.assertEqual(state['data'][2]['yaxis'], 'y2')

        # (0, 1) - top-left
        self.assertEqual(state['data'][3]['y'], np.array([0, 1]))
        self.assertEqual(state['data'][3]['mode'], 'lines')
        self.assertEqual(state['data'][3]['xaxis'], 'x2')
        self.assertEqual(state['data'][3]['yaxis'], 'y3')

        # (1, 1) - top-right
        self.assertEqual(state['data'][4]['y'], np.array([1, 1]))
        self.assertEqual(state['data'][4]['mode'], 'lines')
        self.assertEqual(state['data'][4]['xaxis'], 'x3')
        self.assertEqual(state['data'][4]['yaxis'], 'y3')

        # Axes
        self.assertEqual(state['layout']['xaxis2']['domain'], [0.5, 0.75])
        self.assertEqual(state['layout']['xaxis3']['domain'], [0.75, 1.0])
        self.assertEqual(state['layout']['yaxis2']['domain'], [0, 0.5])
        self.assertEqual(state['layout']['yaxis3']['domain'], [0.5, 1.0])

    
    def test_grid_state(self):
        grid = GridSpace({(i, j): Curve([i, j]) for i in [0, 1]
                          for j in [0, 1]})
        state = self._get_plot_state(grid)
        self.assertEqual(state['data'][0]['y'], np.array([0, 0]))
        self.assertEqual(state['data'][0]['xaxis'], 'x')
        self.assertEqual(state['data'][0]['yaxis'], 'y')
        self.assertEqual(state['data'][1]['y'], np.array([1, 0]))
        self.assertEqual(state['data'][1]['xaxis'], 'x2')
        self.assertEqual(state['data'][1]['yaxis'], 'y')
        self.assertEqual(state['data'][2]['y'], np.array([0, 1]))
        self.assertEqual(state['data'][2]['xaxis'], 'x')
        self.assertEqual(state['data'][2]['yaxis'], 'y2')
        self.assertEqual(state['data'][3]['y'], np.array([1, 1]))
        self.assertEqual(state['data'][3]['xaxis'], 'x2')
        self.assertEqual(state['data'][3]['yaxis'], 'y2')
