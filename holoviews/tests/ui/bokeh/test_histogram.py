import pytest
from bokeh.models import CustomJS

from holoviews.element import Histogram

from .. import expect, wait_until


def watch_hook(dim, pos):
    # Setting up monitoring, as I cannot get the data to sync back to Python
    def create_hook(plot, element):
        range_ = getattr(plot.handles["plot"], f"{dim}_range")
        cjs = CustomJS(
            args=dict(source=plot.handles["source"], range_=range_),
            code=f"""
                window.bokeh_cds_data = source.data['bottom']
                window.bokeh_range = range_.{pos}
            """,
        )
        range_.js_on_change("start", cjs)

    return create_hook

@pytest.mark.usefixtures("bokeh_backend")
def test_histogram_logy(serve_hv):
    hist = Histogram(([1, 3, 6, 9], [1, 10, 0, 100])).opts(logy=True, hooks=[watch_hook("y", "start")])

    page = serve_hv(hist)
    hv_plot = page.locator(".bk-events")
    expect(hv_plot).to_have_count(1)
    bbox = hv_plot.bounding_box()

    center_x = bbox["x"] + bbox["width"] / 2
    center_y = bbox["y"] + bbox["height"] / 2
    page.mouse.move(center_x, center_y)

    bottom_y = bbox["y"] - bbox["height"]
    page.mouse.down()
    page.mouse.move(center_x, bottom_y)
    page.mouse.up()


    def f():
        cds_value = page.evaluate("() => window.bokeh_cds_data")[0]
        range_value = page.evaluate("() => window.bokeh_range")
        assert cds_value != 0.01  # default value
        assert cds_value == range_value

    wait_until(f, page=page)


@pytest.mark.usefixtures("bokeh_backend")
def test_histogram_logx(serve_hv):
    hist = Histogram(([1, 3, 6, 9], [1, 10, 0, 100])).opts(logx=True, invert_axes=True, hooks=[watch_hook("x", "start")])

    page = serve_hv(hist)
    hv_plot = page.locator(".bk-events")
    expect(hv_plot).to_have_count(1)
    bbox = hv_plot.bounding_box()

    center_x = bbox["x"] + bbox["width"] / 2
    center_y = bbox["y"] + bbox["height"] / 2
    page.mouse.move(center_x, center_y)

    bottom_x = bbox["x"] + bbox["width"]
    page.mouse.down()
    page.mouse.move(bottom_x, center_y)
    page.mouse.up()


    def f():
        cds_value = page.evaluate("() => window.bokeh_cds_data")[0]
        range_value = page.evaluate("() => window.bokeh_range")
        assert cds_value != 0.01  # default value
        assert cds_value == range_value

    wait_until(f, page=page)

@pytest.mark.usefixtures("bokeh_backend")
def test_histogram_logy_invert(serve_hv):
    hist = Histogram(([1, 3, 6, 9], [1, 10, 0, 100])).opts(logy=True, invert_yaxis=True, hooks=[watch_hook("y", "end")])

    page = serve_hv(hist)
    hv_plot = page.locator(".bk-events")
    expect(hv_plot).to_have_count(1)
    bbox = hv_plot.bounding_box()

    center_x = bbox["x"] + bbox["width"] / 2
    center_y = bbox["y"] + bbox["height"] / 2
    page.mouse.move(center_x, center_y)

    bottom_y = bbox["y"] + bbox["height"]
    page.mouse.down()
    page.mouse.move(center_x, bottom_y)
    page.mouse.up()

    def f():
        cds_value = page.evaluate("() => window.bokeh_cds_data")[0]
        range_value = page.evaluate("() => window.bokeh_range")
        assert cds_value != 0.01  # default value
        assert cds_value == range_value

    wait_until(f, page=page)


@pytest.mark.usefixtures("bokeh_backend")
def test_histogram_logx_invert(serve_hv):
    hist = Histogram(([1, 3, 6, 9], [1, 10, 0, 100])).opts(logx=True, invert_axes=True, invert_xaxis=True, hooks=[watch_hook("x", "end")])

    page = serve_hv(hist)
    hv_plot = page.locator(".bk-events")
    expect(hv_plot).to_have_count(1)
    bbox = hv_plot.bounding_box()

    center_x = bbox["x"] + bbox["width"] / 2
    center_y = bbox["y"] + bbox["height"] / 2
    page.mouse.move(center_x, center_y)

    bottom_x = bbox["x"] - bbox["width"]
    page.mouse.down()
    page.mouse.move(bottom_x, center_y)
    page.mouse.up()


    def f():
        cds_value = page.evaluate("() => window.bokeh_cds_data")[0]
        range_value = page.evaluate("() => window.bokeh_range")
        assert cds_value != 0.01  # default value
        assert cds_value == range_value

    wait_until(f, page=page)
