from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from bokeh.models import Range1d

import holoviews as hv
from holoviews.plotting.bokeh.chart import DonutPlot

from ..._deps import ibis, ibis_skip, pl, pl_skip
from .test_plot import TestBokehPlot, bokeh_renderer


class TestDonutPlot(TestBokehPlot):
    def _get_plot_source(self, donut):
        plot = bokeh_renderer.get_plot(donut)
        return plot, plot.handles["annular_wedge_1_source"]

    def _get_width_fractions(self, source):
        starts = source.data["start_angle"]
        ends = source.data["end_angle"]
        return (np.array(ends) - np.array(starts)) / (2 * np.pi)

    def test_donut_angles_sum_to_2pi(self):
        d = hv.Donut([("A", 25), ("B", 50), ("C", 25)])
        _, source = self._get_plot_source(d)
        ends = source.data["end_angle"]
        np.testing.assert_almost_equal(ends[-1], 2 * np.pi)

    @pytest.mark.parametrize(
        ("values", "expected_fracs", "expected_labels"),
        [
            ([("A", 50), ("B", 50)], [0.5, 0.5], ["A", "B"]),
            ([("A", 25), ("B", 50), ("C", 25)], [0.25, 0.5, 0.25], ["A", "B", "C"]),
            ([("Only", 100)], [1.0], ["Only"]),
            ([("A", 0), ("B", 0)], [0.0, 0.0], ["A", "B"]),
            ([("A", 3), ("B", 0), ("C", 1)], [0.75, 0.0, 0.25], ["A", "B", "C"]),
            ([("A", 1), ("B", pd.NA), ("C", 3)], [0.25, 0.75], ["A", "C"]),
            (
                [("Rent", pd.NA), (None, 1200), ("Food", 400), ("Transport", 200)],
                [2 / 3, 1 / 3],
                ["Food", "Transport"],
            ),
        ],
    )
    def test_donut_slice_geometry(self, values, expected_fracs, expected_labels):
        _, source = self._get_plot_source(hv.Donut(values))
        assert len(source.data["start_angle"]) == len(expected_labels)
        assert len(source.data["end_angle"]) == len(expected_labels)
        assert list(source.data["x"]) == expected_labels
        np.testing.assert_almost_equal(self._get_width_fractions(source), expected_fracs)

    def test_donut_dataframe_input(self):
        df = pd.DataFrame({"Category": ["A", "B", "C"], "Value": [10, 20, 30]})
        d = hv.Donut(df, kdims="Category", vdims="Value")
        _, source = self._get_plot_source(d)
        assert len(source.data["start_angle"]) == 3

    def test_donut_inner_radius(self):
        d = hv.Donut([("A", 30), ("B", 70)]).opts(inner_radius=0.6)
        plot = bokeh_renderer.get_plot(d)
        glyph = plot.handles["annular_wedge_1_glyph"]
        assert glyph.inner_radius == 0.6

    def test_donut_outer_radius(self):
        d = hv.Donut([("A", 30), ("B", 70)]).opts(outer_radius=2.0)
        plot = bokeh_renderer.get_plot(d)
        glyph = plot.handles["annular_wedge_1_glyph"]
        assert glyph.outer_radius == 2.0

    def test_donut_inner_radius_zero(self):
        d = hv.Donut([("A", 30), ("B", 70)]).opts(inner_radius=0)
        plot = bokeh_renderer.get_plot(d)
        glyph = plot.handles["annular_wedge_1_glyph"]
        assert glyph.inner_radius == 0

    def test_donut_hover_data(self):
        d = hv.Donut([("A", 30), ("B", 70)]).opts(tools=["hover"])
        _, source = self._get_plot_source(d)
        assert "percentage" in source.data
        np.testing.assert_almost_equal(source.data["percentage"], [30.0, 70.0])

    def test_donut_legend_present(self):
        """Legend should be auto-generated via color=dim('x') default."""
        d = hv.Donut([("A", 30), ("B", 70)])
        plot = bokeh_renderer.get_plot(d)
        bokeh_plot = plot.handles["plot"]
        legends = bokeh_plot.legend
        assert len(legends) > 0
        all_labels = [item.label.field for legend in legends for item in legend.items]
        assert len(all_labels) > 0

    def test_donut_empty(self):
        d = hv.Donut([], kdims=["x"], vdims=["y"])
        _, source = self._get_plot_source(d)
        assert len(source.data["start_angle"]) == 0

    def test_donut_show_labels(self):
        d = hv.Donut([("A", 30), ("B", 70)]).opts(show_labels=True)
        plot = bokeh_renderer.get_plot(d)
        assert "text_1_source" in plot.handles
        label_data = plot.handles["text_1_source"].data
        assert list(label_data["text"]) == ["A", "B"]

    @pytest.mark.parametrize(
        ("values", "opts", "expected_text", "present"),
        [
            ([("A", 300), ("B", 700)], {"center_label": "total"}, "1000.0", True),
            ([("A", 300), ("B", pd.NA), ("C", 700)], {"center_label": "total"}, "1000.0", True),
            ([("A", 300), (None, 700)], {"center_label": "total"}, "300.0", True),
            ([("A", 30), ("B", 70)], {"center_label": "Budget"}, "Budget", True),
            ([("A", 30), ("B", 70)], {"center_label": "total", "inner_radius": 0}, "100.0", True),
        ],
    )
    def test_donut_center_label(self, values, opts, expected_text, present):
        plot = bokeh_renderer.get_plot(hv.Donut(values).opts(**opts))
        assert ("center_label" in plot.handles) is present
        if present:
            assert plot.handles["center_label"].text == expected_text

    @pytest.fixture
    def datetime_donut(self):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
                "amt": [1200, 400, 200],
            }
        )
        return hv.Donut(df, kdims="date", vdims="amt")

    def test_donut_datetime_labels_are_formatted(self, datetime_donut):
        plot = bokeh_renderer.get_plot(datetime_donut.opts(show_labels="{date}\n{fraction:.2%}"))
        source = plot.handles["annular_wedge_1_source"]
        assert list(source.data["date"]) == [
            "2024-01-01 00:00:00",
            "2024-01-02 00:00:00",
            "2024-01-03 00:00:00",
        ]
        assert list(plot.handles["text_1_source"].data["text"]) == [
            "2024-01-01 00:00:00\n66.67%",
            "2024-01-02 00:00:00\n22.22%",
            "2024-01-03 00:00:00\n11.11%",
        ]

    @pytest.mark.parametrize(
        ("template", "expected"),
        [
            (
                "{date}\n{fraction:.2%}",
                [
                    "2024-01-01 00:00:00\n66.67%",
                    "2024-01-02 00:00:00\n22.22%",
                    "2024-01-03 00:00:00\n11.11%",
                ],
            ),
            (
                "{date:%Y-%m-%d}\n{fraction:.2%}",
                [
                    "2024-01-01\n66.67%",
                    "2024-01-02\n22.22%",
                    "2024-01-03\n11.11%",
                ],
            ),
        ],
    )
    def test_donut_datetime_label_templates(self, datetime_donut, template, expected):
        plot = bokeh_renderer.get_plot(datetime_donut.opts(show_labels=template))
        assert list(plot.handles["text_1_source"].data["text"]) == expected

    @pytest.mark.parametrize(
        ("cmap", "n_colors"),
        [
            ("Category20", 3),
            ({"A": "red", "B": "green", "C": "blue"}, 3),
            (["#ff0000", "#00ff00", "#0000ff"], 3),
        ],
    )
    def test_donut_cmap(self, cmap, n_colors):
        d = hv.Donut([("A", 10), ("B", 20), ("C", 30)]).opts(cmap=cmap)
        _, source = self._get_plot_source(d)
        assert len(source.data["start_angle"]) == n_colors

    def test_donut_color_raises(self):
        d = hv.Donut([("A", 30), ("B", 70)]).opts(color="red")
        with pytest.raises(ValueError, match="cmap"):
            bokeh_renderer.get_plot(d)

    def test_donut_extra_vdims_in_hover(self):
        df = pd.DataFrame(
            {
                "Category": ["A", "B", "C"],
                "Amount": [10, 20, 30],
                "Cumulative": [10, 30, 60],
            }
        )
        d = hv.Donut(df, kdims=["Category"], vdims=["Amount", "Cumulative"]).opts(tools=["hover"])
        _, source = self._get_plot_source(d)
        assert "Cumulative" in source.data
        np.testing.assert_array_equal(source.data["Cumulative"], [10, 30, 60])

    def test_donut_extra_vdims_filtered(self):
        """Extra vdims should be filtered when rows are dropped."""
        d = hv.Donut(
            [("A", 10, 100), (None, 20, 200), ("C", 30, 300)],
            kdims=["x"],
            vdims=["y", "extra"],
        )
        _, source = self._get_plot_source(d)
        np.testing.assert_array_equal(source.data["extra"], [100, 300])

    def test_donut_axes_not_categorical(self):
        plot = bokeh_renderer.get_plot(hv.Donut([("A", 30), ("B", 70)]))
        assert isinstance(plot.state.x_range, Range1d)
        assert isinstance(plot.state.y_range, Range1d)
        assert (plot.state.x_range.start, plot.state.x_range.end) == (-1.1, 1.1)

    @pytest.mark.parametrize(
        "sibling", [hv.Curve([(-5, 1), (5, 2)]), hv.Bars([("A", 1), ("B", 2)])]
    )
    def test_donut_ranges_not_shared_with_layout_siblings(self, sibling):
        plot = bokeh_renderer.get_plot(hv.Donut([("A", 30), ("B", 70)]) + sibling)
        donut_plot = plot.traverse(specs=[DonutPlot])[0]
        x_range = donut_plot.state.x_range
        assert isinstance(x_range, Range1d)
        assert (x_range.start, x_range.end) == (-1.1, 1.1)

    def test_donut_overlay_axes_not_categorical(self):
        plot = bokeh_renderer.get_plot(hv.Donut([("A", 30), ("B", 70)]) * hv.Text(0, 0, "total"))
        assert isinstance(plot.state.x_range, Range1d)
        assert (plot.state.x_range.start, plot.state.x_range.end) == (-1, 1)

    def test_donut_negative_values_raise(self):
        d = hv.Donut([("A", 30), ("B", -10)], vdims=["Amount"])
        with pytest.raises(ValueError, match=r"non-negative.*'Amount'"):
            bokeh_renderer.get_plot(d)

    def test_donut_layout_colors_consistent(self):
        layout = hv.Donut([("A", 1), ("B", 2)]) + hv.Donut([("A", 1), ("C", 2)])
        plot = bokeh_renderer.get_plot(layout)
        cmappers = [sp.handles["color_mapper"] for sp in plot.traverse(specs=[DonutPlot])]
        colors = [dict(zip(cm.factors, cm.palette, strict=True)) for cm in cmappers]
        assert colors[0]["A"] == colors[1]["A"]
        assert colors[0]["B"] != colors[1]["C"]

    def test_donut_holomap_colors_consistent(self):
        hmap = hv.HoloMap({0: hv.Donut([("A", 1), ("B", 2)]), 1: hv.Donut([("C", 2)])})
        plot = bokeh_renderer.get_plot(hmap)
        cmapper = plot.handles["color_mapper"]
        assert set(cmapper.factors) == {"A", "B", "C"}
        plot.update((1,))
        assert set(cmapper.factors) == {"A", "B", "C"}

    def test_donut_duplicated_category_one_factor(self):
        plot = bokeh_renderer.get_plot(hv.Donut([("A", 1), ("B", 2), ("A", 3)]))
        assert plot.handles["color_mapper"].factors == ["A", "B"]

    def test_donut_missing_category_takes_no_color(self):
        plot = bokeh_renderer.get_plot(hv.Donut([(None, 1), ("A", 2), ("B", 3)]))
        assert plot.handles["color_mapper"].factors == ["A", "B"]

    @pl_skip
    @pytest.mark.parametrize("lazy", [False, True])
    def test_donut_polars(self, lazy):
        df = pl.DataFrame({"x": ["A", None, "C"], "y": [10, 20, 30], "extra": [1, 2, 3]})
        d = hv.Donut(df.lazy() if lazy else df, kdims="x", vdims=["y", "extra"])
        _, source = self._get_plot_source(d)
        assert list(source.data["x"]) == ["A", "C"]
        np.testing.assert_array_equal(source.data["extra"], [1, 3])

    @ibis_skip
    # Raised by ibis' default backend with pandas 3
    @pytest.mark.filterwarnings(r"ignore:The 'future\.infer_string' option is deprecated")
    def test_donut_ibis(self):
        df = pd.DataFrame({"x": ["A", None, "C"], "y": [10, 20, 30]})
        _, source = self._get_plot_source(hv.Donut(ibis.memtable(df), kdims="x", vdims="y"))
        assert list(source.data["x"]) == ["A", "C"]

    def test_donut_layout_of_holomaps_colors_by_category(self):
        layout = hv.NdLayout(
            {
                0: hv.Donut([("Rent", 1), ("Food", 2)]),
                1: hv.Donut([("Travel", 1), ("Misc", 2), ("Rent", 3)]),
                2: hv.Donut([("Misc", 1), ("Gym", 2)]),
            }
        )
        plot = bokeh_renderer.get_plot(layout)
        cmappers = [sp.handles["color_mapper"] for sp in plot.traverse(specs=[DonutPlot])]
        assert len({tuple(cm.factors) for cm in cmappers}) == 1
        assert len({tuple(cm.palette) for cm in cmappers}) == 1
        assert len(set(cmappers[0].factors)) == 5

    def test_donut_holomap_colors_by_category(self):
        hmap = hv.HoloMap(
            {
                0: hv.Donut([("Rent", 1), ("Food", 2)]),
                1: hv.Donut([("Travel", 1), ("Misc", 2), ("Rent", 3)]),
            }
        )
        plot = bokeh_renderer.get_plot(hmap)
        cmapper = plot.handles["color_mapper"]
        rent = cmapper.palette[cmapper.factors.index("Rent")]
        plot.update((1,))
        assert cmapper.palette[cmapper.factors.index("Rent")] == rent
        assert {"Rent", "Food", "Travel", "Misc"} <= set(cmapper.factors)
