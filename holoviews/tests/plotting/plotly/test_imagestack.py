from unittest import SkipTest
try:
    import datashader  # noqa: F401
except ImportError:
   raise SkipTest("Test requires datashader")

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
        image_stack = ImageStack(
            (x, y, a, b, c), kdims=["x", "y"], vdims=["a", "b", "c"]
        )
        assert isinstance(plotly_renderer.get_plot(image_stack), RGBPlot)
        fig_dict = plotly_renderer.get_plot_state(image_stack)
        x_range = fig_dict["layout"]["xaxis"]["range"]
        assert x_range[0] == -0.5
        assert x_range[1] == 2.5

        y_range = fig_dict["layout"]["yaxis"]["range"]
        assert y_range[0] == 4.5
        assert y_range[1] == 7.5

        # Check layout.image object
        images = fig_dict["layout"]["images"]
        assert len(images) == 1
        image = images[0]

        # Check location properties
        assert image["xref"] == "x"
        assert image["yref"] == "y"
        assert image["x"] == -0.5
        assert image["y"] == 7.5
        assert image["sizex"] == 3
        assert image["sizey"] == 3
        assert image["sizing"] == "stretch"
        assert image["layer"] == "above"
