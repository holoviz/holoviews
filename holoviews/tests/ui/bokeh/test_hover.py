import numpy as np
import pandas as pd
import panel as pn
import pytest

import holoviews as hv
from holoviews.plotting.bokeh.util import BOKEH_GE_3_7_0, BOKEH_GE_3_8_0

from .. import expect

pytestmark = pytest.mark.ui

bokeh_3_7_0 = pytest.mark.skipif(not BOKEH_GE_3_7_0, reason="Added in Bokeh 3.7")
bokeh_3_8_0 = pytest.mark.skipif(not BOKEH_GE_3_8_0, reason="Added in Bokeh 3.8")

@pytest.mark.usefixtures("bokeh_backend")
def test_hover_tooltips_list(serve_hv):
    hv_image = hv.Image(
        np.random.rand(10, 10), bounds=(0, 0, 1, 1), kdims=["xc", "yc"]
    ).opts(hover_tooltips=["$x", "xc", "@yc", "@z"])

    page = serve_hv(hv_image)
    hv_plot = page.locator(".bk-events")
    expect(hv_plot).to_have_count(1)
    bbox = hv_plot.bounding_box()

    # Hover over the plot
    page.mouse.move(bbox["x"] + bbox["width"] / 2, bbox["y"] + bbox["height"] / 2)
    page.mouse.up()

    expect(page.locator(".bk-Tooltip")).to_have_count(1)

    expect(page.locator(".bk-Tooltip")).to_contain_text("x:")
    expect(page.locator(".bk-Tooltip")).to_contain_text("xc:")
    expect(page.locator(".bk-Tooltip")).to_contain_text("yc:")
    expect(page.locator(".bk-Tooltip")).to_contain_text("z:")
    expect(page.locator(".bk-Tooltip")).not_to_contain_text("?")


@pytest.mark.usefixtures("bokeh_backend")
def test_hover_tooltips_unit_format(serve_hv):
    dim = hv.Dimension("Test", unit="Unit")
    hv_image = hv.Image(
        np.zeros((10, 10)), bounds=(0, 0, 1, 1), kdims=["xc", "yc"], vdims=[dim]
    ).opts(hover_tooltips=[("Test", "@Test{%0.2f}")])

    page = serve_hv(hv_image)
    hv_plot = page.locator(".bk-events")
    expect(hv_plot).to_have_count(1)
    bbox = hv_plot.bounding_box()

    # Hover over the plot
    page.mouse.move(bbox["x"] + bbox["width"] / 2, bbox["y"] + bbox["height"] / 2)
    page.mouse.up()

    expect(page.locator(".bk-Tooltip")).to_have_count(1)

    expect(page.locator(".bk-Tooltip")).to_contain_text("Test: 0.00%")


@pytest.mark.usefixtures("bokeh_backend")
def test_hover_tooltips_list_mix_tuple_string(serve_hv):
    hv_image = hv.Image(
        np.random.rand(10, 10), bounds=(0, 0, 1, 1), kdims=["xc", "yc"]
    ).opts(hover_tooltips=[("xs", "($x, @xc)"), "yc", "z"])

    page = serve_hv(hv_image)
    hv_plot = page.locator(".bk-events")
    expect(hv_plot).to_have_count(1)
    bbox = hv_plot.bounding_box()

    # Hover over the plot
    page.mouse.move(bbox["x"] + bbox["width"] / 2, bbox["y"] + bbox["height"] / 2)
    page.mouse.up()

    expect(page.locator(".bk-Tooltip")).to_have_count(1)

    expect(page.locator(".bk-Tooltip")).to_contain_text("xs:")
    expect(page.locator(".bk-Tooltip")).to_contain_text("yc:")
    expect(page.locator(".bk-Tooltip")).to_contain_text("z:")
    expect(page.locator(".bk-Tooltip")).not_to_contain_text("?")


@pytest.mark.usefixtures("bokeh_backend")
def test_hover_tooltips_label_group(serve_hv):
    hv_image = hv.Image(
        np.random.rand(10, 10),
        bounds=(0, 0, 1, 1),
        kdims=["xc", "yc"],
        label="Image Label",
        group="Image Group",
    ).opts(
        hover_tooltips=[
            "$label",
            "$group",
            ("Plot Label", "$label"),
            ("Plot Group", "$group"),
        ]
    )

    page = serve_hv(hv_image)
    hv_plot = page.locator(".bk-events")
    expect(hv_plot).to_have_count(1)
    bbox = hv_plot.bounding_box()

    # Hover over the plot
    page.mouse.move(bbox["x"] + bbox["width"] / 2, bbox["y"] + bbox["height"] / 2)
    page.mouse.up()

    expect(page.locator(".bk-Tooltip")).to_have_count(1)

    expect(page.locator(".bk-Tooltip")).to_contain_text("label:")
    expect(page.locator(".bk-Tooltip")).to_contain_text("group:")
    expect(page.locator(".bk-Tooltip")).to_contain_text("Plot Label:")
    expect(page.locator(".bk-Tooltip")).to_contain_text("Plot Group:")
    expect(page.locator(".bk-Tooltip")).not_to_contain_text("?")


@pytest.mark.usefixtures("bokeh_backend")
def test_hover_tooltips_missing(serve_hv):
    hv_image = hv.Image(
        np.random.rand(10, 10), bounds=(0, 0, 1, 1), kdims=["xc", "yc"]
    ).opts(hover_tooltips=["abc"])

    page = serve_hv(hv_image)
    hv_plot = page.locator(".bk-events")
    expect(hv_plot).to_have_count(1)
    bbox = hv_plot.bounding_box()

    # Hover over the plot
    page.mouse.move(bbox["x"] + bbox["width"] / 2, bbox["y"] + bbox["height"] / 2)
    page.mouse.up()

    expect(page.locator(".bk-Tooltip")).to_have_count(1)

    expect(page.locator(".bk-Tooltip")).to_contain_text("?")


@pytest.mark.usefixtures("bokeh_backend")
def test_hover_tooltips_html_string(serve_hv):
    hv_image = hv.Image(
        np.random.rand(10, 10), bounds=(0, 0, 1, 1), kdims=["xc", "yc"]
    ).opts(hover_tooltips="<b>x</b>: $x<br>y: @yc")

    page = serve_hv(hv_image)
    hv_plot = page.locator(".bk-events")
    expect(hv_plot).to_have_count(1)
    bbox = hv_plot.bounding_box()

    # Hover over the plot
    page.mouse.move(bbox["x"] + bbox["width"] / 2, bbox["y"] + bbox["height"] / 2)
    page.mouse.up()

    expect(page.locator(".bk-Tooltip")).to_have_count(1)

    expect(page.locator(".bk-Tooltip")).to_contain_text("x:")
    expect(page.locator(".bk-Tooltip")).to_contain_text("y:")
    expect(page.locator(".bk-Tooltip")).not_to_contain_text("?")


@pytest.mark.usefixtures("bokeh_backend")
def test_hover_tooltips_formatters(serve_hv):
    hv_image = hv.Image(
        np.random.rand(10, 10), bounds=(0, 0, 1, 1), kdims=["xc", "yc"]
    ).opts(
        hover_tooltips=[("X", "($x, @xc{%0.3f})")], hover_formatters={"@xc": "printf"}
    )

    page = serve_hv(hv_image)
    hv_plot = page.locator(".bk-events")
    expect(hv_plot).to_have_count(1)
    bbox = hv_plot.bounding_box()

    # Hover over the plot
    page.mouse.move(bbox["x"] + bbox["width"] / 2, bbox["y"] + bbox["height"] / 2)
    page.mouse.up()

    expect(page.locator(".bk-Tooltip")).to_have_count(1)

    expect(page.locator(".bk-Tooltip")).to_contain_text("X:")
    expect(page.locator(".bk-Tooltip")).to_contain_text("%")
    expect(page.locator(".bk-Tooltip")).not_to_contain_text("?")


@pytest.mark.usefixtures("bokeh_backend")
@pytest.mark.parametrize("hover_mode", ["hline", "vline"])
def test_hover_mode(serve_hv, hover_mode):
    hv_curve = hv.Curve([0, 10, 2]).opts(tools=["hover"], hover_mode=hover_mode)

    page = serve_hv(hv_curve)
    hv_plot = page.locator(".bk-events")
    expect(hv_plot).to_have_count(1)
    bbox = hv_plot.bounding_box()

    # Hover over the plot
    page.mouse.move(bbox["x"] + bbox["width"] / 2, bbox["y"] + bbox["height"] / 2)
    page.mouse.up()

    expect(page.locator(".bk-Tooltip")).to_have_count(1)

    expect(page.locator(".bk-Tooltip")).to_contain_text("x:")
    expect(page.locator(".bk-Tooltip")).to_contain_text("y:")
    expect(page.locator(".bk-Tooltip")).not_to_contain_text("?")


@pytest.mark.usefixtures("bokeh_backend")
@pytest.mark.parametrize(
    "hover_tooltip",
    ["Amplitude", "@Amplitude", ("Amplitude", "@Amplitude")],
)
def test_hover_tooltips_dimension_unit(serve_hv, hover_tooltip):
    amplitude_dim = hv.Dimension("Amplitude", unit="µV")
    hv_curve = hv.Curve([0, 10, 2], vdims=[amplitude_dim]).opts(
        hover_tooltips=[hover_tooltip], hover_mode="vline"
    )

    page = serve_hv(hv_curve)
    hv_plot = page.locator(".bk-events")
    expect(hv_plot).to_have_count(1)
    bbox = hv_plot.bounding_box()

    # Hover over the plot
    page.mouse.move(bbox["x"] + bbox["width"] / 2, bbox["y"] + bbox["height"] / 2)
    page.mouse.up()

    expect(page.locator(".bk-Tooltip")).to_have_count(1)

    expect(page.locator(".bk-Tooltip")).to_contain_text("Amplitude (µV): 10")


@bokeh_3_7_0
@pytest.mark.usefixtures("bokeh_backend")
def test_hover_tooltips_rasterize_server_hover(serve_hv, rng):
    import datashader as ds

    from holoviews.operation.datashader import rasterize

    df = pd.DataFrame({
        "x": rng.normal(45, 1, 100),
        "y": rng.normal(85, 1, 100),
        "s": 1,
        "val": 10,
        "cat": "cat1",
    })
    img = rasterize(hv.Points(df), selector=ds.first("val"), width=10, height=10, dynamic=False).opts(tools=["hover"])

    page = serve_hv(img)
    hv_plot = page.locator(".bk-events")
    expect(hv_plot).to_have_count(1)
    bbox = hv_plot.bounding_box()

    # Hover over the plot, first time the hovertool only have null
    # we then timeout and hover again to get hovertool with actual values
    page.mouse.move(bbox["x"] + bbox["width"] / 2, bbox["y"] + bbox["height"] / 2)
    page.mouse.up()

    expect(page.locator(".bk-Tooltip")).to_have_count(1)
    page.wait_for_timeout(100)

    page.mouse.move(bbox["x"] + bbox["width"] / 2, bbox["y"] + bbox["height"] / 2)
    page.mouse.up()

    expect(page.locator(".bk-Tooltip")).to_have_count(1)
    expect(page.locator(".bk-Tooltip")).to_contain_text("x:4")
    expect(page.locator(".bk-Tooltip")).to_contain_text("y:8")
    expect(page.locator(".bk-Tooltip")).to_contain_text("val:10")
    expect(page.locator(".bk-Tooltip")).to_contain_text("cat:cat1")


@bokeh_3_7_0
@pytest.mark.usefixtures("bokeh_backend")
@pytest.mark.parametrize(
    "hover_tooltips",
    [None, ["x", "y"], ["x", "y", "s"], ["s"]],
    ids=["default", "only_coords", "reduced_vars", "only_vars"],
)
@pytest.mark.parametrize("selector_in_hovertool", [True, False])
def test_hover_tooltips_rasterize_server_hover_selector_ux(serve_hv, rng, hover_tooltips, selector_in_hovertool):
    import datashader as ds

    from holoviews.operation.datashader import rasterize

    df = pd.DataFrame({
        "x": rng.normal(45, 1, 100),
        "y": rng.normal(85, 1, 100),
        "s": 2,
        "val": 10,
        "cat": "cat1",
    })
    img = rasterize(hv.Points(df), selector=ds.first("val"), width=10, height=10, dynamic=False)
    img.opts(tools=["hover"], hover_tooltips=hover_tooltips, selector_in_hovertool=selector_in_hovertool)

    page = serve_hv(img)
    hv_plot = page.locator(".bk-events")
    expect(hv_plot).to_have_count(1)
    bbox = hv_plot.bounding_box()

    # Hover over the plot, first time the hovertool only have null
    # we then timeout and hover again to get hovertool with actual values
    page.mouse.move(bbox["x"] + bbox["width"] / 2, bbox["y"] + bbox["height"] / 2)
    page.mouse.up()

    expect(page.locator(".bk-Tooltip")).to_have_count(1)
    page.wait_for_timeout(100)

    page.mouse.move(bbox["x"] + bbox["width"] / 2, bbox["y"] + bbox["height"] / 2)
    page.mouse.up()


    # Selector test
    if hover_tooltips == ["x", "y"] or not selector_in_hovertool:
        expect(page.locator(".bk-Tooltip")).not_to_contain_text("Selector:first('val')")
    else:
        expect(page.locator(".bk-Tooltip")).to_contain_text("Selector:first('val')")

    # The dividing line
    line_expect = expect(page.locator("div[style*='height: 1px'][style*='grid-column: span 2']"))
    if hover_tooltips in (["x", "y"], ["s"]):
        line_expect.to_have_count(0)
    else:
        line_expect.to_have_count(1)


@bokeh_3_7_0
@pytest.mark.usefixtures("bokeh_backend")
@pytest.mark.parametrize("convert_x", [True, False])
@pytest.mark.parametrize("convert_y", [True, False])
def test_hover_tooltips_rasterize_server_datetime_axis(serve_hv, rng, convert_x, convert_y):
    if not convert_x and not convert_y:
        pytest.skip("Skipping case where both convert_x and convert_y are False")
    import datashader as ds

    from holoviews.operation.datashader import rasterize

    df = pd.DataFrame({
        "x": rng.normal(0, 0.1, 500),
        "y": rng.normal(0, 0.1, 500),
        "s": 1,
        "val": 10,
        "cat": "cat1",
    })
    if convert_x:
        df['x'] = pd.Timestamp(2020, 1, 1, 12) + (df['x'] * 5e8).apply(pd.Timedelta)
    if convert_y:
        df['y'] = pd.Timestamp(2020, 1, 1, 12) + (df['y'] * 5e8).apply(pd.Timedelta)
    img = rasterize(hv.Points(df), selector=ds.first("val"), width=10, height=10, dynamic=False).opts(tools=["hover"])

    page = serve_hv(img)
    hv_plot = page.locator(".bk-events")
    expect(hv_plot).to_have_count(1)
    bbox = hv_plot.bounding_box()

    # Hover over the plot, first time the hovertool only have null
    # we then timeout and hover again to get hovertool with actual values
    page.mouse.move(bbox["x"] + bbox["width"] / 2, bbox["y"] + bbox["height"] / 2)
    page.mouse.up()

    expect(page.locator(".bk-Tooltip")).to_have_count(1)
    page.wait_for_timeout(100)

    page.mouse.move(bbox["x"] + bbox["width"] / 2, bbox["y"] + bbox["height"] / 2)
    page.mouse.up()

    expect(page.locator(".bk-Tooltip")).to_have_count(1)
    if convert_x:
        expect(page.locator(".bk-Tooltip")).to_contain_text('x:2020-01-01')
    if convert_y:
        expect(page.locator(".bk-Tooltip")).to_contain_text('y:2020-01-01')


@bokeh_3_7_0
@pytest.mark.usefixtures("bokeh_backend")
def test_hover_tooltips_selector_update_plot(serve_panel):
    import datashader as ds

    from holoviews.operation.datashader import rasterize

    N_OBS = 1000
    x_data = np.random.random((N_OBS, N_OBS))

    def get_plot(color_by):
        if color_by == 'option1':
            color_data = np.random.choice(['A', 'B', 'C', 'D'], size=N_OBS)
        else:
            color_data = np.random.choice(['a', 'b', 'c', 'd'], size=N_OBS)

        dataset = hv.Dataset(
            (x_data[:, 0], x_data[:, 1], color_data),
            ['X', 'Y'],
            color_by,
        )
        plot = dataset.to(hv.Points)
        plot = rasterize(
            plot,
            aggregator=ds.count_cat(color_by),
            selector=ds.first('X'),
        )
        plot = plot.opts(tools=["hover"], title=color_by)
        return plot

    scb = pn.widgets.Select(name="Color By", options=['option1', 'option2'])
    layout = pn.Row(scb, pn.bind(get_plot, scb))

    page = serve_panel(layout)
    page.wait_for_timeout(500)

    # Locate the plot and move mouse over it
    hv_plot = page.locator(".bk-events")
    expect(hv_plot).to_have_count(1)
    bbox = hv_plot.bounding_box()
    page.mouse.move(bbox["x"] + bbox["width"] / 2, bbox["y"] + bbox["height"] / 2)
    page.wait_for_timeout(200)

    tooltip = page.locator(".bk-Tooltip")
    expect(tooltip).to_have_count(1)
    expect(tooltip).to_contain_text('A:')
    expect(tooltip).to_contain_text('B:')
    expect(tooltip).to_contain_text('C:')
    expect(tooltip).to_contain_text('D:')

    # Change the selector to 'option2'
    scb.value = "option2"
    page.wait_for_timeout(1000)

    # Move the mouse again to trigger updated tooltip
    page.mouse.move(bbox["x"] + bbox["width"] / 4, bbox["y"] + bbox["height"] / 4)
    page.wait_for_timeout(200)
    tooltip = page.locator(".bk-Tooltip")
    expect(tooltip).to_have_count(1)
    expect(tooltip).to_contain_text('a:')
    expect(tooltip).to_contain_text('b:')
    expect(tooltip).to_contain_text('c:')
    expect(tooltip).to_contain_text('d:')


@bokeh_3_8_0
@pytest.mark.usefixtures("bokeh_backend")
def test_hover_tooltips_rasterize_server_hover_filter(serve_hv, rng):
    import datashader as ds

    from holoviews.operation.datashader import rasterize

    df = pd.DataFrame({
        "x": rng.normal(45, 1, 100),
        "y": rng.normal(85, 1, 100),
        "s": 1,
        "val": 10,
        "cat": "cat1",
    })

    hover_models = []
    def watch_hook(plot, element):
        hover_models[:] = [plot.handles["hover"].filters[""].args["hover_model"]]

    img = rasterize(
        hv.Points(df),
        selector=ds.first("val"),
        width=10,
        height=10,
        dynamic=False
    ).opts(tools=["hover"], hooks=[watch_hook])

    page = serve_hv(img)
    hv_plot = page.locator(".bk-events")
    expect(hv_plot).to_have_count(1)
    bbox = hv_plot.bounding_box()

    # Hover over the plot, first time the hovertool only have null
    # we then timeout and hover again to get hovertool with actual values
    page.mouse.move(bbox["x"] + bbox["width"] / 2, bbox["y"] + bbox["height"] / 2)
    page.mouse.up()

    expect(page.locator(".bk-Tooltip")).to_have_count(1)
    page.wait_for_timeout(100)

    assert len(hover_models) == 1
    assert hover_models[0].data["__index__"] != -1

    # Move to no data part of the plot
    page.mouse.move(bbox["x"] + bbox["width"] / 2, bbox["y"] + bbox["height"] * 3 / 4)
    page.mouse.move(bbox["x"] + bbox["width"] / 2, bbox["y"] + bbox["height"] * 3 / 4)
    page.mouse.up()

    # Should not show anything
    expect(page.locator(".bk-Tooltip")).to_have_count(0)
    page.wait_for_timeout(100)

    assert len(hover_models) == 1
    assert hover_models[0].data["__index__"] == -1


@pytest.mark.usefixtures("bokeh_backend")
@pytest.mark.parametrize("x_axis_type", [int, str, lambda x: x+0.1], ids=["int", "str", "float"])
@pytest.mark.parametrize("y_axis_type", [int, str, lambda x: x+0.1], ids=["int", "str", "float"])
def test_hover_heatmap_image(serve_hv, x_axis_type, y_axis_type):
    x = list(map(x_axis_type, range(0, 24, 2)))
    y = list(map(y_axis_type, range(10)))
    z = np.arange(10 * 12).reshape(10, 12)

    heatmap = hv.HeatMap((x, y, z)).opts(tools=["hover"])

    page = serve_hv(heatmap)
    hv_plot = page.locator(".bk-events")
    expect(hv_plot).to_have_count(1)
    bbox = hv_plot.bounding_box()

    # Hover over the center of the plot
    page.mouse.move(bbox["x"] + bbox["width"] / 2, bbox["y"] + bbox["height"] / 2)
    page.mouse.up()

    expect(page.locator(".bk-Tooltip")).to_have_count(1)
    tooltip = page.locator(".bk-Tooltip")

    is_lambda = lambda x: x.__name__ == "<lambda>"
    expect(tooltip).to_contain_text("x: 10.100" if is_lambda(x_axis_type) else "x: 10")
    expect(tooltip).to_contain_text("y: 4.100" if is_lambda(y_axis_type) else "y: 4")
    expect(tooltip).to_contain_text("z: 53")
