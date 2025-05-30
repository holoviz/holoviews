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
@pytest.mark.parametrize(
    ["opts_kwargs", "hook", "drag_direction"],
    [
        ({"logy": True}, watch_hook("y", "start"), "down"),
        ({"logx": True, "invert_axes": True}, watch_hook("x", "start"), "right"),
        ({"logy": True, "invert_yaxis": True}, watch_hook("y", "end"), "up"),
        ({"logx": True, "invert_axes": True, "invert_xaxis": True}, watch_hook("x", "end"), "left"),
    ],
    ids=["logy", "logx", "logy-invert", "logx-invert"],
)
def test_histogram_log_scaling(serve_hv, opts_kwargs, hook, drag_direction):
    hist = Histogram(([1, 3, 6, 9], [1, 10, 0, 100])).opts(hooks=[hook], **opts_kwargs)

    page = serve_hv(hist)
    hv_plot = page.locator(".bk-events")
    expect(hv_plot).to_have_count(1)
    bbox = hv_plot.bounding_box()

    center_x = bbox["x"] + bbox["width"] / 2
    center_y = bbox["y"] + bbox["height"] / 2
    page.mouse.move(center_x, center_y)

    match drag_direction:
        case "down":
            target_x, target_y = center_x, bbox["y"] - bbox["height"]
        case "up":
            target_x, target_y = center_x, bbox["y"] + bbox["height"]
        case "right":
            target_x, target_y = bbox["x"] + bbox["width"], center_y
        case "left":
            target_x, target_y = bbox["x"] - bbox["width"], center_y
        case _:
            raise ValueError(f"Unknown drag direction: {drag_direction}")

    page.mouse.down()
    page.mouse.move(target_x, target_y)
    page.mouse.up()

    def f():
        cds_value = page.evaluate("() => window.bokeh_cds_data")[0]
        range_value = page.evaluate("() => window.bokeh_range")
        assert cds_value != 0.01  # default value
        assert cds_value == range_value

    wait_until(f, page=page)
