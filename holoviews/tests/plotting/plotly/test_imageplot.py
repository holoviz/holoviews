import numpy as np

from holoviews.element import Image

from .testplot import TestPlotlyPlot


class TestImagePlot(TestPlotlyPlot):

    def test_image_state(self):
        img = Image(([1, 2, 3], [0, 1], np.array([[0, 1, 2], [2, 3, 4]])))
        state = self._get_plot_state(img)
        self.assertEqual(state['data'][0]['type'], 'heatmap')
        self.assertEqual(state['data'][0]['x0'], 1)
        self.assertEqual(state['data'][0]['dx'], 1)
        self.assertEqual(state['data'][0]['y0'], 0)
        self.assertEqual(state['data'][0]['dy'], 1)
        self.assertEqual(state['data'][0]['z'], np.array([[0, 1, 2], [2, 3, 4]]))
        self.assertEqual(state['data'][0]['zmin'], 0)
        self.assertEqual(state['data'][0]['zmax'], 4)
        self.assertEqual(state['layout']['xaxis']['range'], [0.5, 3.5])
        self.assertEqual(state['layout']['yaxis']['range'], [-0.5, 1.5])

    def test_image_nodata(self):
        img = Image(([1, 2, 3], [0, 1], np.array([[0, 1, 2], [2, 3, 4]]))).opts(nodata=0)
        state = self._get_plot_state(img)
        self.assertEqual(state['data'][0]['type'], 'heatmap')
        self.assertEqual(state['data'][0]['z'], np.array([[np.NaN, 1, 2], [2, 3, 4]]))

    def test_image_nodata_unint(self):
        img = Image(([1, 2, 3], [0, 1], np.array([[0, 1, 2], [2, 3, 4]],
                                                 dtype='uint32'))).opts(nodata=0)
        state = self._get_plot_state(img)
        self.assertEqual(state['data'][0]['type'], 'heatmap')
        self.assertEqual(state['data'][0]['z'], np.array([[np.NaN, 1, 2], [2, 3, 4]]))

    def test_image_state_inverted(self):
        img = Image(([1, 2, 3], [0, 1], np.array([[0, 1, 2], [2, 3, 4]]))).options(
            invert_axes=True)
        state = self._get_plot_state(img)
        self.assertEqual(state['data'][0]['y0'], 1)
        self.assertEqual(state['data'][0]['dy'], 1)
        self.assertEqual(state['data'][0]['x0'], 0)
        self.assertEqual(state['data'][0]['dx'], 1)
        self.assertEqual(state['data'][0]['z'], np.array([[0, 1, 2], [2, 3, 4]]).T)
        self.assertEqual(state['data'][0]['zmin'], 0)
        self.assertEqual(state['data'][0]['zmax'], 4)
        self.assertEqual(state['layout']['yaxis']['range'], [0.5, 3.5])
        self.assertEqual(state['layout']['xaxis']['range'], [-0.5, 1.5])

    def test_visible(self):
        element = Image(
            ([1, 2, 3], [0, 1], np.array([[0, 1, 2], [2, 3, 4]]))
        ).options(visible=False)
        state = self._get_plot_state(element)
        self.assertEqual(state['data'][0]['visible'], False)
