from __future__ import annotations

import pytest

import holoviews as hv
from holoviews.plotting.bokeh.renderer import BokehRenderer

from .. import expect, wait_until

pytestmark = pytest.mark.ui


@pytest.mark.usefixtures("bokeh_backend")
def test_link_selections_programmatic_clear_removes_region(serve_hv):
    points = hv.Points([1, 2, 3, 4, 5, 6, 7, 8]).opts(width=400, height=300)

    # Helper to inspect bokeh model renderers for region-like glyphs
    def count_highlighted_region_renderers(bokeh_plot):
        cnt = 0
        for r in bokeh_plot.renderers:
            data = r.data_source.data
            if len(data.get("left", [])) > 0:
                cnt += 1
                continue
        return cnt

    ls = hv.link_selections.instance()
    linked = ls(points).opts(active_tools=["box_select"])

    page = serve_hv(linked)
    hv_plot = page.locator(".bk-events")
    expect(hv_plot).to_have_count(1)
    bbox = hv_plot.bounding_box()

    get_count = lambda: count_highlighted_region_renderers(
        BokehRenderer.get_plot(linked[()]).state
    )
    wait_until(lambda: get_count() == 0, page)

    # Box-drag selection
    start_x = bbox["x"] + bbox["width"] * 0.25
    start_y = bbox["y"] + bbox["height"] * 0.25
    end_x = bbox["x"] + bbox["width"] * 0.75
    end_y = bbox["y"] + bbox["height"] * 0.75

    hv_plot.click()
    page.mouse.move(start_x, start_y)
    page.mouse.down(button="left")
    page.mouse.move(end_x, end_y, steps=10)
    page.mouse.up(button="left")

    wait_until(lambda: ls.selection_expr is not None, page)
    # Ensure at least one new region renderer was created by the interaction
    wait_until(lambda: get_count() == 1, page)

    ls.selection_expr = None
    # Final region count should be back to initial
    wait_until(lambda: get_count() == 0, page)
