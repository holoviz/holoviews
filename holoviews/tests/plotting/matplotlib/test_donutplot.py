from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import holoviews as hv

from ..._deps import pl, pl_skip
from .test_plot import TestMPLPlot, mpl_renderer


class TestDonutPlot(TestMPLPlot):
    def _get_patches(self, donut):
        plot = mpl_renderer.get_plot(donut)
        return plot, plot.handles["axis"].patches

    def _get_width_fractions(self, patches):
        return [(patch.theta2 - patch.theta1) / 360 for patch in patches]

    @pytest.mark.parametrize(
        ("values", "expected_widths"),
        [
            ([("A", 50), ("B", 50)], [0.5, 0.5]),
            ([("A", 10), ("B", 20), ("C", 30)], [1 / 6, 1 / 3, 1 / 2]),
            ([("Only", 42)], [1.0]),
            ([("A", 0), ("B", 0)], [0.0, 0.0]),
            ([("A", 3), ("B", 0), ("C", 1)], [0.75, 0.0, 0.25]),
            ([("A", 1), ("B", pd.NA), ("C", 3)], [0.25, 0.75]),
            ([("Rent", pd.NA), (None, 1200), ("Food", 400), ("Transport", 200)], [2 / 3, 1 / 3]),
        ],
    )
    def test_donut_slice_geometry(self, values, expected_widths):
        _, patches = self._get_patches(hv.Donut(values))
        assert len(patches) == len(expected_widths)
        np.testing.assert_almost_equal(self._get_width_fractions(patches), expected_widths)

    def test_donut_dataframe_input(self):
        df = pd.DataFrame({"Category": ["A", "B", "C"], "Value": [10, 20, 30]})
        d = hv.Donut(df, kdims="Category", vdims="Value")
        _, patches = self._get_patches(d)
        assert len(patches) == 3

    def test_donut_inner_radius_zero(self):
        """inner_radius=0 should produce full pie wedges."""
        d = hv.Donut([("A", 30), ("B", 70)]).opts(inner_radius=0)
        _, patches = self._get_patches(d)
        assert len(patches) == 2
        np.testing.assert_almost_equal(patches[0].width, 1.0)

    def test_donut_wedge_angles(self):
        """Wedge angles should span the full 360 degrees."""
        _, patches = self._get_patches(hv.Donut([("A", 25), ("B", 75)]))
        total_angle = sum(p.theta2 - p.theta1 for p in patches)
        np.testing.assert_almost_equal(total_angle, 360.0)

    def test_donut_legend_present(self):
        d = hv.Donut([("A", 30), ("B", 70)]).opts(show_legend=True)
        plot = mpl_renderer.get_plot(d)
        legend = plot.handles["axis"].get_legend()
        assert legend is not None
        texts = [t.get_text() for t in legend.get_texts()]
        assert "A" in texts
        assert "B" in texts

    def test_donut_legend_disabled(self):
        d = hv.Donut([("A", 30), ("B", 70)]).opts(show_legend=False)
        plot = mpl_renderer.get_plot(d)
        legend = plot.handles["axis"].get_legend()
        assert legend is None

    def test_donut_empty(self):
        d = hv.Donut([], kdims=["x"], vdims=["y"])
        _, patches = self._get_patches(d)
        assert len(patches) == 0

    def test_donut_equal_aspect(self):
        d = hv.Donut([("A", 30), ("B", 70)])
        plot = mpl_renderer.get_plot(d)
        np.testing.assert_almost_equal(plot.handles["axis"].get_aspect(), 1.0)

    def test_donut_show_labels(self):
        d = hv.Donut([("A", 30), ("B", 70)]).opts(show_labels=True)
        plot = mpl_renderer.get_plot(d)
        texts = [t.get_text() for t in plot.handles["axis"].texts]
        assert "A" in texts
        assert "B" in texts

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
        plot = mpl_renderer.get_plot(hv.Donut(values).opts(**opts))
        texts = [text.get_text() for text in plot.handles["axis"].texts]
        if present:
            assert expected_text in texts
        else:
            assert expected_text is None

    @pytest.fixture
    def datetime_donut(self):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
                "amt": [1200, 400, 200],
            }
        )
        return hv.Donut(df, kdims="date", vdims="amt")

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
        plot = mpl_renderer.get_plot(datetime_donut.opts(show_labels=template))
        texts = [t.get_text() for t in plot.handles["axis"].texts]
        assert texts == expected

    @pytest.mark.parametrize(
        "cmap",
        [
            "Set2",
            {"A": "red", "B": "green", "C": "blue"},
            ["#ff0000", "#00ff00", "#0000ff"],
        ],
    )
    def test_donut_cmap(self, cmap):
        d = hv.Donut([("A", 10), ("B", 20), ("C", 30)]).opts(cmap=cmap)
        _, patches = self._get_patches(d)
        # All wedges should be rendered; colors should not all be identical.
        assert len(patches) == 3
        colors = [p.get_facecolor() for p in patches]
        assert len(set(map(tuple, colors))) > 1

    def test_donut_negative_values_raise(self):
        d = hv.Donut([("A", 30), ("B", -10)], vdims=["Amount"])
        with pytest.raises(ValueError, match=r"non-negative.*'Amount'"):
            mpl_renderer.get_plot(d)

    def test_donut_duplicated_category_same_color(self):
        plot, patches = self._get_patches(hv.Donut([("A", 1), ("B", 2), ("A", 3)]))
        colors = [p.get_facecolor() for p in patches]
        assert colors[0] == colors[2] != colors[1]
        texts = [t.get_text() for t in plot.handles["axis"].get_legend().get_texts()]
        assert texts == ["A", "B"]

    def test_donut_missing_category_takes_no_color(self):
        _, patches = self._get_patches(hv.Donut([(None, 1), ("A", 2), ("B", 3)]))
        _, expected = self._get_patches(hv.Donut([("A", 2), ("B", 3)]))
        assert [p.get_facecolor() for p in patches] == [p.get_facecolor() for p in expected]

    @pytest.mark.parametrize("cmap", [None, "Set2"])
    def test_donut_layout_colors_consistent(self, cmap):
        d1 = hv.Donut([("A", 1), ("B", 2)])
        d2 = hv.Donut([("A", 1), ("C", 2)])
        layout = (d1 + d2).opts(hv.opts.Donut(cmap=cmap)) if cmap else d1 + d2
        plot = mpl_renderer.get_plot(layout)
        p1, p2 = (sp.handles["artist"] for sp in plot.traverse(specs=[hv.plotting.mpl.DonutPlot]))
        assert p1[0].get_facecolor() == p2[0].get_facecolor()
        assert p1[1].get_facecolor() != p2[1].get_facecolor()

    def test_donut_dict_cmap_duplicated_category(self):
        d = hv.Donut([("A", 1), ("B", 2), ("A", 3), ("C", 4)]).opts(cmap={"B": "red"})
        _, patches = self._get_patches(d)
        colors = [p.get_facecolor() for p in patches]
        assert colors[0] == colors[2]
        assert colors[1] == (1.0, 0.0, 0.0, 1.0)

    def test_donut_layout_colors_by_category(self):
        layout = hv.NdLayout(
            {
                0: hv.Donut([("Rent", 1), ("Food", 2)]),
                1: hv.Donut([("Travel", 1), ("Misc", 2), ("Rent", 3)]),
            }
        )
        plot = mpl_renderer.get_plot(layout)
        p1, p2 = (sp.handles["artist"] for sp in plot.traverse(specs=[hv.plotting.mpl.DonutPlot]))
        # Rent is the first wedge of p1 and the last one of p2
        assert p1[0].get_facecolor() == p2[2].get_facecolor()
        assert p1[1].get_facecolor() not in [p.get_facecolor() for p in p2]

    def test_donut_holomap_colors_by_category(self):
        hmap = hv.HoloMap(
            {
                0: hv.Donut([("Rent", 1), ("Food", 2)]),
                1: hv.Donut([("Travel", 1), ("Misc", 2), ("Rent", 3)]),
            }
        )
        plot = mpl_renderer.get_plot(hmap)
        rent = plot.handles["artist"][0].get_facecolor()
        plot.update((1,))
        assert plot.handles["artist"][2].get_facecolor() == rent

    @pl_skip
    def test_donut_polars(self):
        df = pl.DataFrame({"x": ["A", None, "C"], "y": [10, 20, 30]})
        _, patches = self._get_patches(hv.Donut(df, kdims="x", vdims="y"))
        assert len(patches) == 2
