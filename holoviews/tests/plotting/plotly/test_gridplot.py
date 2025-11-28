import numpy as np

from holoviews.core.spaces import GridSpace
from holoviews.element import Curve, Scatter
from holoviews.testing import assert_data_equal

from .test_plot import TestPlotlyPlot


class TestGridPlot(TestPlotlyPlot):

    def test_layout_with_grid(self):
        # Create GridSpace
        grid = GridSpace({(i, j): Curve([i, j]) for i in [0, 1]
                          for j in [0, 1]})
        grid = grid.opts(vspacing=0, hspacing=0)

        # Create Scatter
        scatter = Scatter([-10, 0])

        # Create Horizontal Layout
        layout = (scatter + grid).opts(vspacing=0, hspacing=0)

        state = self._get_plot_state(layout)

        # Compute expected x domain break
        start_fig_width = 400
        grid_fig_width = 400 * 1.1
        x_domain_break1 = start_fig_width / (start_fig_width + grid_fig_width)

        # Compute expect y domain break for left scatter plot
        start_fig_height = 400
        grid_fig_height = start_fig_height * 1.1
        y_domain_break1 = start_fig_height / grid_fig_height

        # Check the scatter plot on the left
        assert_data_equal(state['data'][0]['y'], np.array([-10, 0]))
        assert state['data'][0]['mode'] == 'markers'
        assert state['data'][0]['xaxis'] == 'x'
        assert state['data'][0]['yaxis'] == 'y'
        assert state['layout']['xaxis']['range'] == [0, 1]
        assert state['layout']['xaxis']['domain'] == [0, x_domain_break1]
        assert state['layout']['yaxis']['range'] == [-10, 1]
        assert state['layout']['yaxis']['domain'] == [0, y_domain_break1]

        # Check the grid plot on the right

        # (0, 0) - bottom-left
        assert_data_equal(state['data'][1]['y'], np.array([0, 0]))
        assert state['data'][1]['mode'] == 'lines'
        assert state['data'][1]['xaxis'] == 'x2'
        assert state['data'][1]['yaxis'] == 'y2'

        # (1, 0) - bottom-right
        assert_data_equal(state['data'][2]['y'], np.array([1, 0]))
        assert state['data'][2]['mode'] == 'lines'
        assert state['data'][2]['xaxis'] == 'x3'
        assert state['data'][2]['yaxis'] == 'y2'

        # (0, 1) - top-left
        assert_data_equal(state['data'][3]['y'], np.array([0, 1]))
        assert state['data'][3]['mode'] == 'lines'
        assert state['data'][3]['xaxis'] == 'x2'
        assert state['data'][3]['yaxis'] == 'y3'

        # (1, 1) - top-right
        assert_data_equal(state['data'][4]['y'], np.array([1, 1]))
        assert state['data'][4]['mode'] == 'lines'
        assert state['data'][4]['xaxis'] == 'x3'
        assert state['data'][4]['yaxis'] == 'y3'

        # Axes
        x_dimain_break2 = x_domain_break1 + (1 - x_domain_break1) / 2
        assert state['layout']['xaxis2']['domain'] == [x_domain_break1, x_dimain_break2]
        assert state['layout']['xaxis3']['domain'] == [x_dimain_break2, 1.0]
        assert state['layout']['yaxis2']['domain'] == [0, 0.5]
        assert state['layout']['yaxis3']['domain'] == [0.5, 1.0]


    def test_grid_state(self):
        grid = GridSpace({(i, j): Curve([i, j]) for i in [0, 1]
                          for j in [0, 1]})
        state = self._get_plot_state(grid)
        assert_data_equal(state['data'][0]['y'], np.array([0, 0]))
        assert state['data'][0]['xaxis'] == 'x'
        assert state['data'][0]['yaxis'] == 'y'
        assert_data_equal(state['data'][1]['y'], np.array([1, 0]))
        assert state['data'][1]['xaxis'] == 'x2'
        assert state['data'][1]['yaxis'] == 'y'
        assert_data_equal(state['data'][2]['y'], np.array([0, 1]))
        assert state['data'][2]['xaxis'] == 'x'
        assert state['data'][2]['yaxis'] == 'y2'
        assert_data_equal(state['data'][3]['y'], np.array([1, 1]))
        assert state['data'][3]['xaxis'] == 'x2'
        assert state['data'][3]['yaxis'] == 'y2'
