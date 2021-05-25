import numpy as np

from holoviews.element import Surface

from .testplot import TestPlotlyPlot


class TestSurfacePlot(TestPlotlyPlot):

    def test_surface_state(self):
        img = Surface(([1, 2, 3], [0, 1], np.array([[0, 1, 2], [2, 3, 4]])))
        state = self._get_plot_state(img)
        self.assertEqual(state['data'][0]['type'], 'surface')
        self.assertEqual(state['data'][0]['x'], np.array([1, 2, 3]))
        self.assertEqual(state['data'][0]['y'], np.array([0, 1]))
        self.assertEqual(state['data'][0]['z'], np.array([[0, 1, 2], [2, 3, 4]]))
        self.assertEqual(state['data'][0]['cmin'], 0)
        self.assertEqual(state['data'][0]['cmax'], 4)
        self.assertEqual(state['layout']['scene']['xaxis']['range'], [0.5, 3.5])
        self.assertEqual(state['layout']['scene']['yaxis']['range'], [-0.5, 1.5])
        self.assertEqual(state['layout']['scene']['zaxis']['range'], [0, 4])

    def test_surface_colorbar(self):
        img = Surface(([1, 2, 3], [0, 1], np.array([[0, 1, 2], [2, 3, 4]])))
        img.opts(colorbar=True)
        state = self._get_plot_state(img)
        trace = state['data'][0]
        self.assertTrue(trace['showscale'])

        img.opts(colorbar=False)
        state = self._get_plot_state(img)
        trace = state['data'][0]
        self.assertFalse(trace['showscale'])

    def test_visible(self):
        element = Surface(
            ([1, 2, 3], [0, 1], np.array([[0, 1, 2], [2, 3, 4]]))
        ).options(visible=False)
        state = self._get_plot_state(element)
        self.assertEqual(state['data'][0]['visible'], False)
