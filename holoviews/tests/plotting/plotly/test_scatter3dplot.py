import numpy as np

from holoviews.element import Scatter3D

from .test_plot import TestPlotlyPlot


class TestScatter3DPlot(TestPlotlyPlot):

    def test_scatter3d_state(self):
        scatter = Scatter3D(([0,1], [2,3], [4,5]))
        state = self._get_plot_state(scatter)
        self.assertEqual(state['data'][0]['x'], np.array([0, 1]))
        self.assertEqual(state['data'][0]['y'], np.array([2, 3]))
        self.assertEqual(state['data'][0]['z'], np.array([4, 5]))
        self.assertEqual(state['layout']['scene']['xaxis']['range'], [0, 1])
        self.assertEqual(state['layout']['scene']['yaxis']['range'], [2, 3])
        self.assertEqual(state['layout']['scene']['zaxis']['range'], [4, 5])

    def test_scatter3d_color_mapped(self):
        scatter = Scatter3D(([0,1], [2,3], [4,5])).opts(color='y')
        state = self._get_plot_state(scatter)
        self.assertEqual(state['data'][0]['marker']['color'], np.array([2, 3]))
        self.assertEqual(state['data'][0]['marker']['cmin'], 2)
        self.assertEqual(state['data'][0]['marker']['cmax'], 3)

    def test_scatter3d_size(self):
        scatter = Scatter3D(([0,1], [2,3], [4,5])).opts(size='y')
        state = self._get_plot_state(scatter)
        self.assertEqual(state['data'][0]['marker']['size'], np.array([2, 3]))

    def test_visible(self):
        element = Scatter3D(([0, 1], [2, 3], [4, 5])).opts(visible=False)
        state = self._get_plot_state(element)
        self.assertEqual(state['data'][0]['visible'], False)
