import numpy as np
from holoviews.element import ImageStack
from holoviews.plotting.plotly import RGBPlot

from .test_plot import TestPlotlyPlot, plotly_renderer


class TestImageStackPlot(TestPlotlyPlot):

    def test_image_stack(self):
        x = np.arange(0, 3)
        y = np.arange(5, 8)
        a = np.array([[np.nan, np.nan, 1], [np.nan] * 3, [np.nan] * 3])
        b = np.array([[np.nan] * 3, [1, 1, np.nan], [np.nan] * 3])
        c = np.array([[np.nan] * 3, [np.nan] * 3, [1, 1, 1]])
        image_stack = ImageStack((x, y, a, b, c), kdims=["x", "y"], vdims=["a", "b", "c"])
        assert isinstance(plotly_renderer.get_plot(image_stack), RGBPlot)
        fig_dict = plotly_renderer.get_plot_state(image_stack)
        x_range = fig_dict['layout']['xaxis']['range']
        self.assertEqual(x_range[0], -0.5)
        self.assertEqual(x_range[1], 2.5)

        y_range = fig_dict['layout']['yaxis']['range']
        self.assertEqual(y_range[0], 4.5)
        self.assertEqual(y_range[1], 7.5)

        # Check layout.image object
        images = fig_dict['layout']['images']
        self.assertEqual(len(images), 1)
        image = images[0]

        # Check location properties
        self.assert_property_values(image, {
            'xref': 'x',
            'yref': 'y',
            'x': -0.5,
            'y': 7.5,
            'sizex': 3,
            'sizey': 3,
            'sizing': 'stretch',
            'layer': 'above',
        })
