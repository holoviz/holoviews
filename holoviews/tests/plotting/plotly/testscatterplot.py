import numpy as np

from holoviews.element import Scatter

from .testplot import TestPlotlyPlot


class TestScatterPlot(TestPlotlyPlot):

    def test_scatter_state(self):
        scatter = Scatter([3, 2, 1])
        state = self._get_plot_state(scatter)
        self.assertEqual(state['data'][0]['y'], np.array([3, 2, 1]))
        self.assertEqual(state['data'][0]['mode'], 'markers')
        self.assertEqual(state['layout']['yaxis']['range'], [1, 3])

    def test_scatter_inverted(self):
        scatter = Scatter([1, 2, 3]).options(invert_axes=True)
        state = self._get_plot_state(scatter)
        self.assertEqual(state['data'][0]['x'], np.array([1, 2, 3]))
        self.assertEqual(state['data'][0]['y'], np.array([0, 1, 2]))
        self.assertEqual(state['data'][0]['mode'], 'markers')
        self.assertEqual(state['layout']['xaxis']['range'], [1, 3])
        self.assertEqual(state['layout']['yaxis']['range'], [0, 2])
        self.assertEqual(state['layout']['xaxis']['title']['text'], 'y')
        self.assertEqual(state['layout']['yaxis']['title']['text'], 'x')

    def test_scatter_color_mapped(self):
        scatter = Scatter([3, 2, 1]).options(color='x')
        state = self._get_plot_state(scatter)
        self.assertEqual(state['data'][0]['marker']['color'], np.array([0, 1, 2]))
        self.assertEqual(state['data'][0]['marker']['cmin'], 0)
        self.assertEqual(state['data'][0]['marker']['cmax'], 2)

    def test_scatter_size(self):
        scatter = Scatter([3, 2, 1]).options(size='y')
        state = self._get_plot_state(scatter)
        self.assertEqual(state['data'][0]['marker']['size'], np.array([3, 2, 1]))

    def test_scatter_colors(self):
        scatter = Scatter([
            (0, 1, 'red'), (1, 2, 'green'), (2, 3, 'blue')
        ], vdims=['y', 'color']).options(color='color')
        state = self._get_plot_state(scatter)
        self.assertEqual(state['data'][0]['marker']['color'],
                         np.array(['red', 'green', 'blue']))

    def test_scatter_markers(self):
        scatter = Scatter([
            (0, 1, 'square'), (1, 2, 'circle'), (2, 3, 'triangle-up')
        ], vdims=['y', 'marker']).options(marker='marker')
        state = self._get_plot_state(scatter)
        self.assertEqual(state['data'][0]['marker']['symbol'],
                         np.array(['square', 'circle', 'triangle-up']))

    def test_scatter_selectedpoints(self):
        scatter = Scatter([
            (0, 1,), (1, 2), (2, 3)
        ]).options(selectedpoints=[1, 2])
        state = self._get_plot_state(scatter)
        self.assertEqual(state['data'][0]['selectedpoints'], [1, 2])

    def test_visible(self):
        element = Scatter([3, 2, 1]).options(visible=False)
        state = self._get_plot_state(element)
        self.assertEqual(state['data'][0]['visible'], False)
