import time

import pytest

try:
    from playwright.sync_api import expect
except ImportError:
    pytestmark = pytest.mark.skip('playwright not available')

pytestmark = pytest.mark.ui

from holoviews import Scatter
from holoviews.streams import BoundsXY
from holoviews.plotting.bokeh import BokehRenderer
from panel.pane.holoviews import HoloViews
from panel.io.server import serve
from panel.tests.util import wait_until


def test_box_select(page, port):
    hv_scatter = Scatter([1, 2, 3]).opts(
        backend='bokeh', tools=['box_select'], active_tools=['box_select']
    )

    bounds = BoundsXY(source=hv_scatter)

    pn_scatter = HoloViews(hv_scatter, renderer=BokehRenderer)

    serve(pn_scatter, port=port, threaded=True, show=False)

    time.sleep(0.5)

    page.goto(f"http://localhost:{port}")

    hv_plot = page.locator('.bk-events')

    expect(hv_plot).to_have_count(1)

    bbox = hv_plot.bounding_box()
    hv_plot.click()

    page.mouse.move(bbox['x']+100, bbox['y']+100)
    page.mouse.down()
    page.mouse.move(bbox['x']+150, bbox['y']+150, steps=5)
    page.mouse.up()

    EXPECTED_BOUNDS = (0.32844036697247725, 1.8285714285714285, 0.8788990825688077, 2.3183673469387758)
    wait_until(lambda: bounds.bounds == EXPECTED_BOUNDS, page)
