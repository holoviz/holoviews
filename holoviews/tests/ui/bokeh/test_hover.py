import numpy as np
import pandas as pd
import pytest

import holoviews as hv

from .. import expect

pytestmark = pytest.mark.ui

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
    img = rasterize(hv.Points(df), selector=ds.first("val")).opts(tools=["hover"])

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

    page.mouse.move(bbox["x"] + bbox["width"] / 4, bbox["y"] + bbox["height"] / 4)
    page.mouse.up()

    expect(page.locator(".bk-Tooltip")).to_have_count(1)
    expect(page.locator(".bk-Tooltip")).to_contain_text('x:4')
    expect(page.locator(".bk-Tooltip")).to_contain_text('y:8')
    expect(page.locator(".bk-Tooltip")).to_contain_text("val:NaN")
    expect(page.locator(".bk-Tooltip")).to_contain_text('cat:"-"')
