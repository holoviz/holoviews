import numpy as np
import pandas as pd
import pytest

import holoviews as hv

from .test_plot import TestBokehPlot, bokeh_renderer


class TestDonutPlot(TestBokehPlot):
    def test_donut_simple(self):
        d = hv.Donut([("A", 30), ("B", 70)])
        plot = bokeh_renderer.get_plot(d)
        source = plot.handles["source"]
        assert len(source.data["start_angle"]) == 2
        assert len(source.data["end_angle"]) == 2

    def test_donut_angles_sum_to_2pi(self):
        d = hv.Donut([("A", 25), ("B", 50), ("C", 25)])
        plot = bokeh_renderer.get_plot(d)
        source = plot.handles["source"]
        ends = source.data["end_angle"]
        np.testing.assert_almost_equal(ends[-1], 2 * np.pi)

    @pytest.mark.parametrize(
        ("values", "expected_fracs"),
        [
            ([("A", 50), ("B", 50)], [0.5, 0.5]),
            ([("A", 25), ("B", 50), ("C", 25)], [0.25, 0.5, 0.25]),
            ([("Only", 100)], [1.0]),
        ],
    )
    def test_donut_angle_proportions(self, values, expected_fracs):
        d = hv.Donut(values)
        plot = bokeh_renderer.get_plot(d)
        source = plot.handles["source"]
        starts = source.data["start_angle"]
        ends = source.data["end_angle"]
        widths = np.array(ends) - np.array(starts)
        actual_fracs = widths / (2 * np.pi)
        np.testing.assert_almost_equal(actual_fracs, expected_fracs)

    def test_donut_dataframe_input(self):
        df = pd.DataFrame({"Category": ["A", "B", "C"], "Value": [10, 20, 30]})
        d = hv.Donut(df, kdims="Category", vdims="Value")
        plot = bokeh_renderer.get_plot(d)
        source = plot.handles["source"]
        assert len(source.data["start_angle"]) == 3

    def test_donut_inner_radius(self):
        d = hv.Donut([("A", 30), ("B", 70)]).opts(inner_radius=0.6)
        plot = bokeh_renderer.get_plot(d)
        glyph = plot.handles["glyph"]
        assert glyph.inner_radius == 0.6

    def test_donut_outer_radius(self):
        d = hv.Donut([("A", 30), ("B", 70)]).opts(outer_radius=2.0)
        plot = bokeh_renderer.get_plot(d)
        glyph = plot.handles["glyph"]
        assert glyph.outer_radius == 2.0

    def test_donut_inner_radius_zero(self):
        d = hv.Donut([("A", 30), ("B", 70)]).opts(inner_radius=0)
        plot = bokeh_renderer.get_plot(d)
        glyph = plot.handles["glyph"]
        assert glyph.inner_radius == 0

    def test_donut_hover_data(self):
        d = hv.Donut([("A", 30), ("B", 70)]).opts(tools=["hover"])
        plot = bokeh_renderer.get_plot(d)
        source = plot.handles["source"]
        assert "percentage" in source.data
        np.testing.assert_almost_equal(source.data["percentage"], [30.0, 70.0])

    def test_donut_legend_present(self):
        """Legend should be auto-generated via color=dim('x') default."""
        d = hv.Donut([("A", 30), ("B", 70)])
        plot = bokeh_renderer.get_plot(d)
        bokeh_plot = plot.handles["plot"]
        legends = bokeh_plot.legend
        assert len(legends) > 0
        all_labels = []
        for legend in legends:
            for item in legend.items:
                if hasattr(item.label, "get"):
                    all_labels.append(item.label.get("value", item.label.get("field", "")))
        assert len(all_labels) > 0

    def test_donut_empty(self):
        d = hv.Donut([], kdims=["x"], vdims=["y"])
        plot = bokeh_renderer.get_plot(d)
        source = plot.handles["source"]
        assert len(source.data["start_angle"]) == 0

    def test_donut_single_slice(self):
        d = hv.Donut([("Only", 42)])
        plot = bokeh_renderer.get_plot(d)
        source = plot.handles["source"]
        assert len(source.data["start_angle"]) == 1
        np.testing.assert_almost_equal(source.data["start_angle"][0], 0)
        np.testing.assert_almost_equal(source.data["end_angle"][0], 2 * np.pi)

    @pytest.mark.parametrize(
        ("values", "n_slices"),
        [
            ([("A", 30), ("B", 70)], 2),
            ([("A", 10), ("B", 20), ("C", 30), ("D", 40)], 4),
        ],
    )
    def test_donut_source_lengths(self, values, n_slices):
        d = hv.Donut(values)
        plot = bokeh_renderer.get_plot(d)
        source = plot.handles["source"]
        for col in ("start_angle", "end_angle"):
            assert len(source.data[col]) == n_slices

    def test_donut_all_zero_values(self):
        d = hv.Donut([("A", 0), ("B", 0)])
        plot = bokeh_renderer.get_plot(d)
        source = plot.handles["source"]
        assert len(source.data["start_angle"]) == 2

    def test_donut_show_labels(self):
        d = hv.Donut([("A", 30), ("B", 70)]).opts(show_labels=True)
        plot = bokeh_renderer.get_plot(d)
        assert "label_source" in plot.handles
        label_data = plot.handles["label_source"].data
        assert list(label_data["text"]) == ["A", "B"]

    def test_donut_total_label_total(self):
        d = hv.Donut([("A", 300), ("B", 700)]).opts(total_label="total")
        plot = bokeh_renderer.get_plot(d)
        assert "total_label" in plot.handles
        assert plot.handles["total_label"].text == "$1.0K"

    def test_donut_total_label_custom(self):
        d = hv.Donut([("A", 30), ("B", 70)]).opts(total_label="Budget")
        plot = bokeh_renderer.get_plot(d)
        assert "total_label" in plot.handles
        assert plot.handles["total_label"].text == "Budget"

    def test_donut_total_label_disabled_for_pie(self):
        d = hv.Donut([("A", 30), ("B", 70)]).opts(total_label="total", inner_radius=0)
        plot = bokeh_renderer.get_plot(d)
        assert "total_label" not in plot.handles
