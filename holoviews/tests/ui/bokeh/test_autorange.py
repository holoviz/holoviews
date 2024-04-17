import numpy as np
import pytest

from holoviews.element import Curve
from holoviews.plotting.bokeh.renderer import BokehRenderer

from .. import expect, wait_until

pytestmark = pytest.mark.ui


@pytest.mark.usefixtures("bokeh_backend")
def test_autorange_single(serve_hv):
    curve = Curve(np.arange(1000)).opts(autorange='y', active_tools=['box_zoom'])

    plot = BokehRenderer.get_plot(curve)

    page = serve_hv(plot)

    hv_plot = page.locator('.bk-events')

    expect(hv_plot).to_have_count(1)

    bbox = hv_plot.bounding_box()
    hv_plot.click()

    page.mouse.move(bbox['x']+100, bbox['y']+100)
    page.mouse.down()
    page.mouse.move(bbox['x']+150, bbox['y']+150, steps=5)
    page.mouse.up()

    y_range = plot.handles['y_range']
    wait_until(lambda: y_range.start == 163.2 and y_range.end == 448.8, page)


@pytest.mark.usefixtures("bokeh_backend")
def test_autorange_single_in_overlay(serve_hv):
    c1 = Curve(np.arange(1000))
    c2 = Curve(-np.arange(1000)).opts(autorange='y')

    overlay = (c1*c2).opts(active_tools=['box_zoom'])

    plot = BokehRenderer.get_plot(overlay)

    page = serve_hv(plot)

    hv_plot = page.locator('.bk-events')

    expect(hv_plot).to_have_count(1)

    bbox = hv_plot.bounding_box()
    hv_plot.click()

    page.mouse.move(bbox['x']+100, bbox['y']+100)
    page.mouse.down()
    page.mouse.move(bbox['x']+150, bbox['y']+150, steps=5)
    page.mouse.up()

    y_range = plot.handles['y_range']
    wait_until(lambda: y_range.start == -486 and y_range.end == 486, page)

@pytest.mark.usefixtures("bokeh_backend")
def test_autorange_overlay(serve_hv):
    c1 = Curve(np.arange(1000))
    c2 = Curve(-np.arange(1000))

    overlay = (c1*c2).opts(active_tools=['box_zoom'], autorange='y')

    plot = BokehRenderer.get_plot(overlay)

    page = serve_hv(plot)

    hv_plot = page.locator('.bk-events')

    expect(hv_plot).to_have_count(1)

    bbox = hv_plot.bounding_box()
    hv_plot.click()

    page.mouse.move(bbox['x']+100, bbox['y']+100)
    page.mouse.down()
    page.mouse.move(bbox['x']+150, bbox['y']+150, steps=5)
    page.mouse.up()

    y_range = plot.handles['y_range']
    wait_until(lambda: y_range.start == -486 and y_range.end == 486, page)
