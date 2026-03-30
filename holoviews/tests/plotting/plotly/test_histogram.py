import numpy as np

import holoviews as hv

from .test_plot import TestPlotlyPlot


class TestHistogramPlot(TestPlotlyPlot):
    def setup_method(self):
        super().setup_method()
        self.frequencies = [1, 3, 5, 4, 2, 0]
        self.edges = [-3, -2, -1, 0, 1, 2]

    def test_histogram_plot(self):
        hist = hv.Histogram((self.edges, self.frequencies))
        state = self._get_plot_state(hist)
        np.testing.assert_equal(state["data"][0]["x"], self.edges)
        np.testing.assert_equal(state["data"][0]["y"], self.frequencies)
        assert state["data"][0]["type"] == "bar"
        assert state["data"][0]["orientation"] == "v"
        assert state["data"][0]["width"] == 1
        assert state["layout"]["xaxis"]["range"] == [-3.5, 2.5]
        assert state["layout"]["xaxis"]["title"]["text"] == "x"
        assert state["layout"]["yaxis"]["range"] == [0, 5]
        assert state["layout"]["yaxis"]["title"]["text"] == "Frequency"

    def test_histogram_plot_inverted(self):
        hist = hv.Histogram((self.edges, self.frequencies)).opts(invert_axes=True)

        state = self._get_plot_state(hist)
        np.testing.assert_equal(state["data"][0]["y"], self.edges)
        np.testing.assert_equal(state["data"][0]["x"], self.frequencies)
        assert state["data"][0]["type"] == "bar"
        assert state["data"][0]["orientation"] == "h"
        assert state["data"][0]["width"] == 1
        assert state["layout"]["yaxis"]["range"] == [-3.5, 2.5]
        assert state["layout"]["yaxis"]["title"]["text"] == "x"
        assert state["layout"]["xaxis"]["range"] == [0, 5]
        assert state["layout"]["xaxis"]["title"]["text"] == "Frequency"

    def test_histogram_plot_styling(self):
        props = {
            "color": "orange",
            "line_width": 7,
            "line_color": "green",
        }
        hist = hv.Histogram((self.edges, self.frequencies)).opts(**props)
        state = self._get_plot_state(hist)
        marker = state["data"][0]["marker"]
        self.assert_property_values(marker, props)

    def test_visible(self):
        element = hv.Histogram((self.edges, self.frequencies)).opts(visible=False)
        state = self._get_plot_state(element)
        assert state["data"][0]["visible"] is False

    def test_histogram_stack_ndoverlay(self):
        edges = np.array([0, 1, 2, 3])
        h1 = hv.Histogram((edges, np.array([1, 2, 3])))
        h2 = hv.Histogram((edges, np.array([6, 4, 2])))
        h3 = hv.Histogram((edges, np.array([8, 1, 2])))
        overlay = hv.NdOverlay({0: h1, 1: h2, 2: h3})
        stacked = hv.Histogram.stack(overlay)
        state = self._get_plot_state(stacked)
        # Trace 0: h1 values, base=0
        np.testing.assert_array_equal(state["data"][0]["y"], [1, 2, 3])
        np.testing.assert_array_equal(state["data"][0]["base"], [0, 0, 0])
        # Trace 1: h2 values (delta), base=h1 top
        np.testing.assert_array_equal(state["data"][1]["y"], [6, 4, 2])
        np.testing.assert_array_equal(state["data"][1]["base"], [1, 2, 3])
        # Trace 2: h3 values (delta), base=h1+h2 top
        np.testing.assert_array_equal(state["data"][2]["y"], [8, 1, 2])
        np.testing.assert_array_equal(state["data"][2]["base"], [7, 6, 5])

    def test_histogram_stack_overlay(self):
        edges = np.array([0, 1, 2, 3])
        h1 = hv.Histogram((edges, np.array([1, 2, 3])), label="A")
        h2 = hv.Histogram((edges, np.array([6, 4, 2])), label="B")
        stacked = hv.Histogram.stack(h1 * h2)
        state = self._get_plot_state(stacked)
        # First trace: h1 values, base=0
        np.testing.assert_array_equal(state["data"][0]["y"], [1, 2, 3])
        np.testing.assert_array_equal(state["data"][0]["base"], [0, 0, 0])
        # Second trace: h2 values (delta), base=h1 top
        np.testing.assert_array_equal(state["data"][1]["y"], [6, 4, 2])
        np.testing.assert_array_equal(state["data"][1]["base"], [1, 2, 3])
