import numpy as np
import pandas as pd
import panel as pn
import pytest
from bokeh.models import FactorRange, FixedTicker, HoverTool, Range1d, Span

import holoviews as hv
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.streams import Pipe, Stream, Tap
from holoviews.testing import assert_data_equal
from holoviews.util import Dynamic

from ...utils import LoggingComparison
from .test_plot import TestBokehPlot, bokeh_renderer


class TestOverlayPlot(LoggingComparison, TestBokehPlot):
    def test_overlay_apply_ranges_disabled(self):
        overlay = (hv.Curve(range(10)) * hv.Curve(range(10))).opts("Curve", apply_ranges=False)
        plot = bokeh_renderer.get_plot(overlay)
        assert all(np.isnan(e) for e in plot.get_extents(overlay, {}))

    def test_overlay_update_sources(self):
        hmap = hv.HoloMap(
            {
                i: (hv.Curve(np.arange(i), label="A") * hv.Curve(np.arange(i) * 2, label="B"))
                for i in range(10, 13)
            }
        )
        plot = bokeh_renderer.get_plot(hmap)
        plot.update((12,))
        subplot1, subplot2 = plot.subplots.values()
        assert_data_equal(subplot1.handles["source"].data["y"], np.arange(12))
        assert_data_equal(subplot2.handles["source"].data["y"], np.arange(12) * 2)

    def test_overlay_framewise_norm(self):
        a = {"X": [0, 1, 2], "Y": [0, 1, 2], "Z": [0, 50, 100]}
        b = {"X": [3, 4, 5], "Y": [0, 10, 20], "Z": [50, 50, 150]}
        sa = hv.Scatter(a, "X", ["Y", "Z"]).opts(color="Z", framewise=True)
        sb = hv.Scatter(b, "X", ["Y", "Z"]).opts(color="Z", framewise=True)
        plot = bokeh_renderer.get_plot(sa * sb)
        sa_plot, sb_plot = plot.subplots.values()
        sa_cmapper = sa_plot.handles["color_color_mapper"]
        sb_cmapper = sb_plot.handles["color_color_mapper"]
        assert sa_cmapper.low == 0
        assert sb_cmapper.low == 0
        assert sa_cmapper.high == 150
        assert sb_cmapper.high == 150

    def test_overlay_update_visible(self):
        hmap = hv.HoloMap({i: hv.Curve(np.arange(i), label="A") for i in range(1, 3)})
        hmap2 = hv.HoloMap({i: hv.Curve(np.arange(i), label="B") for i in range(3, 5)})
        plot = bokeh_renderer.get_plot(hmap * hmap2)
        subplot1, subplot2 = plot.subplots.values()
        assert subplot1.handles["glyph_renderer"].visible
        assert not subplot2.handles["glyph_renderer"].visible
        plot.update((4,))
        assert not subplot1.handles["glyph_renderer"].visible
        assert subplot2.handles["glyph_renderer"].visible

    def test_hover_tool_instance_renderer_association(self):
        tooltips = [("index", "$index")]
        hover = HoverTool(tooltips=tooltips)
        overlay = hv.Curve(np.random.rand(10, 2)).opts(tools=[hover]) * hv.Points(
            np.random.rand(10, 2)
        )
        plot = bokeh_renderer.get_plot(overlay)
        curve_plot = plot.subplots[("Curve", "I")]
        assert len(curve_plot.handles["hover"].renderers) == 1
        assert curve_plot.handles["glyph_renderer"] in curve_plot.handles["hover"].renderers
        assert plot.handles["hover"].tooltips == tooltips

    # def test_hover_tool_overlay_renderers(self):
    #     overlay = Curve(range(2)).opts(tools=['hover']) * ErrorBars([]).opts(tools=['hover'])
    #     plot = bokeh_renderer.get_plot(overlay)
    #     assert len(plot.handles['hover'].renderers) == 1
    #     assert plot.handles['hover'].tooltips == [('x', '@{x}'), ('y', '@{y}')]

    def test_hover_tool_nested_overlay_renderers(self):
        overlay1 = hv.NdOverlay({0: hv.Curve(range(2)), 1: hv.Curve(range(3))}, kdims=["Test"])
        overlay2 = hv.NdOverlay({0: hv.Curve(range(4)), 1: hv.Curve(range(5))}, kdims=["Test"])
        nested_overlay = (overlay1 * overlay2).opts("Curve", tools=["hover"])
        plot = bokeh_renderer.get_plot(nested_overlay)
        assert len(plot.handles["hover"].renderers) == 4
        assert plot.handles["hover"].tooltips == [
            ("Test", "@{Test}"),
            ("x", "@{x}"),
            ("y", "@{y}"),
        ]

    def test_overlay_empty_layers(self):
        overlay = hv.Curve(range(10)) * hv.NdOverlay()
        plot = bokeh_renderer.get_plot(overlay)
        assert len(plot.subplots) == 1
        self.log_handler.assert_contains("WARNING", "is empty and will be skipped during plotting")

    def test_overlay_show_frame_disabled(self):
        overlay = (hv.Curve(range(10)) * hv.Curve(range(10))).opts(show_frame=False)
        plot = bokeh_renderer.get_plot(overlay).state
        assert plot.outline_line_alpha == 0

    def test_overlay_no_xaxis(self):
        overlay = (hv.Curve(range(10)) * hv.Curve(range(10))).opts(xaxis=None)
        plot = bokeh_renderer.get_plot(overlay).state
        assert not plot.xaxis[0].visible

    def test_overlay_no_yaxis(self):
        overlay = (hv.Curve(range(10)) * hv.Curve(range(10))).opts(yaxis=None)
        plot = bokeh_renderer.get_plot(overlay).state
        assert not plot.yaxis[0].visible

    def test_overlay_xlabel_override(self):
        overlay = (hv.Curve(range(10)) * hv.Curve(range(10))).opts(xlabel="custom x-label")
        plot = bokeh_renderer.get_plot(overlay).state
        assert plot.xaxis[0].axis_label == "custom x-label"

    def test_overlay_ylabel_override(self):
        overlay = (hv.Curve(range(10)) * hv.Curve(range(10))).opts(ylabel="custom y-label")
        plot = bokeh_renderer.get_plot(overlay).state
        assert plot.yaxis[0].axis_label == "custom y-label"

    def test_overlay_xlabel_override_propagated(self):
        overlay = hv.Curve(range(10)).opts(xlabel="custom x-label") * hv.Curve(range(10))
        plot = bokeh_renderer.get_plot(overlay).state
        assert plot.xaxis[0].axis_label == "custom x-label"

    def test_overlay_ylabel_override_propagated(self):
        overlay = hv.Curve(range(10)).opts(ylabel="custom y-label") * hv.Curve(range(10))
        plot = bokeh_renderer.get_plot(overlay).state
        assert plot.yaxis[0].axis_label == "custom y-label"

    def test_overlay_xrotation(self):
        overlay = (hv.Curve(range(10)) * hv.Curve(range(10))).opts(xrotation=90)
        plot = bokeh_renderer.get_plot(overlay).state
        assert plot.xaxis[0].major_label_orientation == np.pi / 2

    def test_overlay_yrotation(self):
        overlay = (hv.Curve(range(10)) * hv.Curve(range(10))).opts(yrotation=90)
        plot = bokeh_renderer.get_plot(overlay).state
        assert plot.yaxis[0].major_label_orientation == np.pi / 2

    def test_overlay_xticks_list(self):
        overlay = (hv.Curve(range(10)) * hv.Curve(range(10))).opts(xticks=[0, 5, 10])
        plot = bokeh_renderer.get_plot(overlay).state
        assert isinstance(plot.xaxis[0].ticker, FixedTicker)
        assert plot.xaxis[0].ticker.ticks == [0, 5, 10]

    def test_overlay_yticks_list(self):
        overlay = (hv.Curve(range(10)) * hv.Curve(range(10))).opts(yticks=[0, 5, 10])
        plot = bokeh_renderer.get_plot(overlay).state
        assert isinstance(plot.yaxis[0].ticker, FixedTicker)
        assert plot.yaxis[0].ticker.ticks == [0, 5, 10]

    def test_overlay_update_plot_opts(self):
        hmap = hv.HoloMap(
            {
                0: (hv.Curve([]) * hv.Curve([])).opts(title="A"),
                1: (hv.Curve([]) * hv.Curve([])).opts(title="B"),
            }
        )
        plot = bokeh_renderer.get_plot(hmap)
        assert plot.state.title.text == "A"
        plot.update((1,))
        assert plot.state.title.text == "B"

    def test_overlay_update_plot_opts_inherited(self):
        hmap = hv.HoloMap(
            {
                0: (hv.Curve([]).opts(title="A") * hv.Curve([])),
                1: (hv.Curve([]).opts(title="B") * hv.Curve([])),
            }
        )
        plot = bokeh_renderer.get_plot(hmap)
        assert plot.state.title.text == "A"
        plot.update((1,))
        assert plot.state.title.text == "B"

    def test_points_errorbars_text_ndoverlay_categorical_xaxis(self):
        overlay = hv.NdOverlay(
            {i: hv.Points(([chr(65 + i)] * 10, np.random.randn(10))) for i in range(5)}
        )
        error = hv.ErrorBars([(el["x"][0], np.mean(el["y"]), np.std(el["y"])) for el in overlay])
        text = hv.Text("C", 0, "Test")
        plot = bokeh_renderer.get_plot(overlay * error * text)
        x_range = plot.handles["x_range"]
        y_range = plot.handles["y_range"]
        assert isinstance(x_range, FactorRange)
        factors = ["A", "B", "C", "D", "E"]
        assert x_range.factors == ["A", "B", "C", "D", "E"]
        assert isinstance(y_range, Range1d)
        error_plot = plot.subplots[("ErrorBars", "I")]
        for xs, factor in zip(error_plot.handles["source"].data["base"], factors, strict=True):
            assert factor == xs

    def test_overlay_categorical_two_level(self):
        bars = hv.Bars(
            [("A", "a", 1), ("B", "b", 2), ("A", "b", 3), ("B", "a", 4)], kdims=["Upper", "Lower"]
        )

        plot = bokeh_renderer.get_plot(bars * hv.HLine(2))
        x_range = plot.handles["x_range"]
        assert isinstance(x_range, FactorRange)
        assert x_range.factors == [("A", "a"), ("A", "b"), ("B", "a"), ("B", "b")]
        assert isinstance(plot.state.renderers[-1], Span)

    def test_points_errorbars_text_ndoverlay_categorical_xaxis_invert_axes(self):
        overlay = hv.NdOverlay(
            {i: hv.Points(([chr(65 + i)] * 10, np.random.randn(10))) for i in range(5)}
        )
        error = hv.ErrorBars(
            [(el["x"][0], np.mean(el["y"]), np.std(el["y"])) for el in overlay]
        ).opts(invert_axes=True)
        text = hv.Text("C", 0, "Test")
        plot = bokeh_renderer.get_plot(overlay * error * text)
        x_range = plot.handles["x_range"]
        y_range = plot.handles["y_range"]
        assert isinstance(x_range, Range1d)
        assert isinstance(y_range, FactorRange)
        assert y_range.factors == ["A", "B", "C", "D", "E"]

    def test_overlay_empty_element_extent(self):
        overlay = hv.Curve([]).redim.range(x=(-10, 10)) * hv.Points([]).redim.range(y=(-20, 20))
        plot = bokeh_renderer.get_plot(overlay)
        extents = plot.get_extents(overlay, {})
        assert extents == (-10, -20, 10, 20)

    def test_dynamic_subplot_creation(self):
        def cb(X):
            return hv.NdOverlay({i: hv.Curve(np.arange(10) + i) for i in range(X)})

        dmap = hv.DynamicMap(cb, kdims=["X"]).redim.range(X=(1, 10))
        plot = bokeh_renderer.get_plot(dmap)
        assert len(plot.subplots) == 1
        plot.update((3,))
        assert len(plot.subplots) == 3
        for i, subplot in enumerate(plot.subplots.values()):
            assert subplot.cyclic_index == i

    def test_complex_range_example(self):
        errors = [
            (0.1 * i, np.sin(0.1 * i), (i + 1) / 3.0, (i + 1) / 5.0)
            for i in np.linspace(0, 100, 11)
        ]
        errors = hv.ErrorBars(errors, vdims=["y", "yerrneg", "yerrpos"]).redim.range(y=(0, None))
        overlay = hv.Curve(errors) * errors * hv.VLine(4)
        plot = bokeh_renderer.get_plot(overlay)
        x_range = plot.handles["x_range"]
        y_range = plot.handles["y_range"]
        assert x_range.start == 0
        assert x_range.end == 10.0
        assert y_range.start == 0
        assert y_range.end == 19.655978889110628

    def test_overlay_muted_renderer(self):
        overlay = hv.Curve((np.arange(5)), label="increase") * hv.Curve(
            (np.arange(5) * -1 + 5), label="decrease"
        ).opts(muted=True)
        plot = bokeh_renderer.get_plot(overlay)
        unmuted, muted = plot.subplots.values()
        assert not unmuted.handles["glyph_renderer"].muted
        assert muted.handles["glyph_renderer"].muted

    def test_overlay_params_bind_linked_stream(self):
        tap = Tap()

        def test(x):
            return hv.Curve([1, 2, 3]) * hv.VLine(x or 0)

        dmap = hv.DynamicMap(pn.bind(test, x=tap.param.x))
        plot = bokeh_renderer.get_plot(dmap)

        tap.event(x=1)
        _, vline_plot = plot.subplots.values()
        assert vline_plot.handles["glyph"].location == 1

    def test_overlay_params_dict_linked_stream(self):
        tap = Tap()

        def test(x):
            return hv.Curve([1, 2, 3]) * hv.VLine(x or 0)

        dmap = hv.DynamicMap(test, streams={"x": tap.param.x})
        plot = bokeh_renderer.get_plot(dmap)

        tap.event(x=1)
        _, vline_plot = plot.subplots.values()
        assert vline_plot.handles["glyph"].location == 1

    def test_ndoverlay_subcoordinate_y_no_batching(self):
        overlay = hv.NdOverlay(
            {i: hv.Curve(np.arange(10) * i).opts(subcoordinate_y=True) for i in range(10)}
        ).opts(legend_limit=1)
        plot = bokeh_renderer.get_plot(overlay)

        assert plot.batched == False
        assert len(plot.subplots) == 10

    def test_ndoverlay_subcoordinate_y_ranges(self):
        data = {
            "x": np.arange(10),
            "A": np.arange(10),
            "B": np.arange(10) * 2,
            "C": np.arange(10) * 3,
        }
        overlay = hv.NdOverlay(
            {
                "A": hv.Curve(data, "x", ("A", "y")).opts(subcoordinate_y=True),
                "B": hv.Curve(data, "x", ("B", "y")).opts(subcoordinate_y=True),
                "C": hv.Curve(data, "x", ("C", "y")).opts(subcoordinate_y=True),
            }
        )
        plot = bokeh_renderer.get_plot(overlay)

        assert plot.state.y_range.start == -0.5
        assert plot.state.y_range.end == 2.5
        for sp in plot.subplots.values():
            assert sp.handles["y_range"].start == 0
            assert sp.handles["y_range"].end == 27

    def test_ndoverlay_subcoordinate_y_ranges_update(self):
        def lines(data):
            return hv.NdOverlay(
                {
                    "A": hv.Curve(data, "x", ("A", "y1")).opts(
                        subcoordinate_y=True, framewise=True
                    ),
                    "B": hv.Curve(data, "x", ("B", "y2")).opts(
                        subcoordinate_y=True, framewise=True
                    ),
                    "C": hv.Curve(data, "x", ("C", "y3")).opts(
                        subcoordinate_y=True, framewise=True
                    ),
                }
            )

        data = {
            "x": np.arange(10),
            "A": np.arange(10),
            "B": np.arange(10) * 2,
            "C": np.arange(10) * 3,
        }
        stream = Pipe(data=data)
        dmap = hv.DynamicMap(lines, streams=[stream])
        plot = bokeh_renderer.get_plot(dmap)

        assert plot.state.y_range.start == -0.5
        assert plot.state.y_range.end == 2.5
        for sp in plot.subplots.values():
            y_range = sp.handles["y_range"]
            assert y_range.start == 0
            assert y_range.end == data[y_range.name].max()

        new_data = {"x": np.arange(10), "A": data["A"] + 1, "B": data["B"] + 2, "C": data["C"] + 3}
        stream.event(data=new_data)
        for sp in plot.subplots.values():
            y_range = sp.handles["y_range"]
            ydata = new_data[y_range.name]
            assert y_range.start == ydata.min()
            assert y_range.end == ydata.max()

    def test_overlay_subcoordinate_y_ranges(self):
        data = {
            "x": np.arange(10),
            "A": np.arange(10),
            "B": np.arange(10) * 2,
            "C": np.arange(10) * 3,
        }
        overlay = hv.Overlay(
            [
                hv.Curve(data, "x", ("A", "y"), label="A").opts(subcoordinate_y=True),
                hv.Curve(data, "x", ("B", "y"), label="B").opts(subcoordinate_y=True),
                hv.Curve(data, "x", ("C", "y"), label="C").opts(subcoordinate_y=True),
            ]
        )
        plot = bokeh_renderer.get_plot(overlay)

        assert plot.state.y_range.start == -0.5
        assert plot.state.y_range.end == 2.5
        for sp in plot.subplots.values():
            assert sp.handles["y_range"].start == 0
            assert sp.handles["y_range"].end == 27

    def test_overlay_subcoordinate_y_ranges_update(self):
        def lines(data):
            return hv.Overlay(
                [
                    hv.Curve(data, "x", ("A", "y1"), label="A").opts(
                        subcoordinate_y=True, framewise=True
                    ),
                    hv.Curve(data, "x", ("B", "y2"), label="B").opts(
                        subcoordinate_y=True, framewise=True
                    ),
                    hv.Curve(data, "x", ("C", "y3"), label="C").opts(
                        subcoordinate_y=True, framewise=True
                    ),
                ]
            )

        data = {
            "x": np.arange(10),
            "A": np.arange(10),
            "B": np.arange(10) * 2,
            "C": np.arange(10) * 3,
        }
        stream = Pipe(data=data)
        dmap = hv.DynamicMap(lines, streams=[stream])
        plot = bokeh_renderer.get_plot(dmap)

        assert plot.state.y_range.start == -0.5
        assert plot.state.y_range.end == 2.5
        for sp in plot.subplots.values():
            y_range = sp.handles["y_range"]
            assert y_range.start == 0
            assert y_range.end == data[y_range.name].max()

        new_data = {"x": np.arange(10), "A": data["A"] + 1, "B": data["B"] + 2, "C": data["C"] + 3}
        stream.event(data=new_data)
        for sp in plot.subplots.values():
            y_range = sp.handles["y_range"]
            ydata = new_data[y_range.name]
            assert y_range.start == ydata.min()
            assert y_range.end == ydata.max()


@pytest.mark.parametrize("order", [("str", "date"), ("date", "str")])
def test_ndoverlay_categorical_y_ranges(order):
    df = pd.DataFrame(
        {
            "str": ["apple", "banana", "cherry", "date", "elderberry"],
            "date": pd.to_datetime(
                ["2023-01-01", "2023-02-14", "2023-03-21", "2023-04-30", "2023-05-15"]
            ),
        }
    )
    overlay = hv.NdOverlay({col: hv.Scatter(df, kdims="index", vdims=col) for col in order})
    plot = bokeh_renderer.get_plot(overlay)
    output = plot.handles["y_range"].factors
    expected = sorted(map(str, df.values.ravel()))
    assert output == expected


class TestLegends(TestBokehPlot):
    def test_overlay_legend(self):
        overlay = hv.Curve(range(10), label="A") * hv.Curve(range(10), label="B")
        plot = bokeh_renderer.get_plot(overlay)
        legend_labels = [l.label["value"] for l in plot.state.legend[0].items]
        assert legend_labels == ["A", "B"]

    def test_overlay_legend_with_labels(self):
        overlay = (hv.Curve(range(10), label="A") * hv.Curve(range(10), label="B")).opts(
            legend_labels={"A": "A Curve", "B": "B Curve"}
        )
        plot = bokeh_renderer.get_plot(overlay)
        legend_labels = [l.label["value"] for l in plot.state.legend[0].items]
        assert legend_labels == ["A Curve", "B Curve"]

    def test_holomap_legend_updates(self):
        hmap = hv.HoloMap(
            {
                i: hv.Curve([1, 2, 3], label=chr(65 + i + 2)) * hv.Curve([1, 2, 3], label="B")
                for i in range(3)
            }
        )
        plot = bokeh_renderer.get_plot(hmap)
        legend_labels = [property_to_dict(item.label) for item in plot.state.legend[0].items]
        assert legend_labels == [{"value": "C"}, {"value": "B"}]
        plot.update((1,))
        legend_labels = [property_to_dict(item.label) for item in plot.state.legend[0].items]
        assert legend_labels == [{"value": "B"}, {"value": "D"}]
        plot.update((2,))
        legend_labels = [property_to_dict(item.label) for item in plot.state.legend[0].items]
        assert legend_labels == [{"value": "B"}, {"value": "E"}]

    def test_holomap_legend_updates_varying_lengths(self):
        hmap = hv.HoloMap(
            {
                i: hv.Overlay([hv.Curve([1, 2, j], label=chr(65 + j)) for j in range(i)])
                for i in range(1, 4)
            }
        )
        plot = bokeh_renderer.get_plot(hmap)
        legend_labels = [property_to_dict(item.label) for item in plot.state.legend[0].items]
        assert legend_labels == [{"value": "A"}]
        plot.update((2,))
        legend_labels = [property_to_dict(item.label) for item in plot.state.legend[0].items]
        assert legend_labels == [{"value": "A"}, {"value": "B"}]
        plot.update((3,))
        legend_labels = [property_to_dict(item.label) for item in plot.state.legend[0].items]
        assert legend_labels == [{"value": "A"}, {"value": "B"}, {"value": "C"}]

    def test_dynamicmap_legend_updates(self):
        hmap = hv.HoloMap(
            {
                i: hv.Curve([1, 2, 3], label=chr(65 + i + 2)) * hv.Curve([1, 2, 3], label="B")
                for i in range(3)
            }
        )
        dmap = Dynamic(hmap)
        plot = bokeh_renderer.get_plot(dmap)
        legend_labels = [property_to_dict(item.label) for item in plot.state.legend[0].items]
        assert legend_labels == [{"value": "C"}, {"value": "B"}]
        plot.update((1,))
        legend_labels = [property_to_dict(item.label) for item in plot.state.legend[0].items]
        assert legend_labels == [{"value": "B"}, {"value": "D"}]
        plot.update((2,))
        legend_labels = [property_to_dict(item.label) for item in plot.state.legend[0].items]
        assert legend_labels == [{"value": "B"}, {"value": "E"}]

    def test_dynamicmap_legend_updates_add_dynamic_plots(self):
        hmap = hv.HoloMap(
            {
                i: hv.Overlay([hv.Curve([1, 2, j], label=chr(65 + j)) for j in range(i)])
                for i in range(1, 4)
            }
        )
        dmap = Dynamic(hmap)
        plot = bokeh_renderer.get_plot(dmap)
        legend_labels = [property_to_dict(item.label) for item in plot.state.legend[0].items]
        assert legend_labels == [{"value": "A"}]
        plot.update((2,))
        legend_labels = [property_to_dict(item.label) for item in plot.state.legend[0].items]
        assert legend_labels == [{"value": "A"}, {"value": "B"}]
        plot.update((3,))
        legend_labels = [property_to_dict(item.label) for item in plot.state.legend[0].items]
        assert legend_labels == [{"value": "A"}, {"value": "B"}, {"value": "C"}]

    def test_dynamicmap_ndoverlay_shrink_number_of_items(self):
        selected = Stream.define("selected", items=3)()

        def callback(items):
            return hv.NdOverlay({j: hv.Overlay([hv.Curve([1, 2, j])]) for j in range(items)})

        dmap = hv.DynamicMap(callback, streams=[selected])
        plot = bokeh_renderer.get_plot(dmap)
        selected.event(items=2)
        assert len([r for r in plot.state.renderers if r.visible]) == 2

    def test_dynamicmap_variable_length_overlay(self):
        selected = Stream.define("selected", items=[1])()

        def callback(items):
            return hv.Overlay([hv.Box(0, 0, radius * 2) for radius in items])

        dmap = hv.DynamicMap(callback, streams=[selected])
        plot = bokeh_renderer.get_plot(dmap)
        assert len(plot.subplots) == 1
        selected.event(items=[1, 2, 4])
        assert len(plot.subplots) == 3
        selected.event(items=[1, 4])
        sp1, sp2, sp3 = plot.subplots.values()
        assert sp1.handles["cds"].data["xs"][0].min() == -1
        assert sp1.handles["glyph_renderer"].visible
        assert sp2.handles["cds"].data["xs"][0].min() == -4
        assert sp2.handles["glyph_renderer"].visible
        assert sp3.handles["cds"].data["xs"][0].min() == -4
        assert not sp3.handles["glyph_renderer"].visible
