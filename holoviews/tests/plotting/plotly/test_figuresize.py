from holoviews.element import Points
from .test_plot import TestPlotlyPlot


class TestImagePlot(TestPlotlyPlot):

    def test_image_state(self):
        img = Points([(0, 0)]).opts(width=345, height=456)
        state = self._get_plot_state(img)

        self.assertEqual(state["layout"]["width"], 345)
        self.assertEqual(state["layout"]["height"], 456)
