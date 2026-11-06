from __future__ import annotations

from itertools import product

import numpy as np
import pytest

import holoviews as hv
from holoviews.operation import gridmatrix

from .. import expect, wait_until

pytestmark = pytest.mark.ui

_MEASURE_JS = """() => {
    function fixedTicks(axes) {
        const axis = (axes || []).find((a) => a.ticker && a.ticker.type === "FixedTicker")
        return axis && axis.ticker.ticks.length ? axis.ticker.ticks.slice() : null
    }
    const out = []
    for (const view of Bokeh.index.all_views()) {
        const model = view.model
        if (model == null || model.type !== "Figure") continue
        const inner = view.frame_view ? view.frame_view.bbox : null
        if (inner == null) continue
        const outerRect = view.el.getBoundingClientRect()
        out.push({
            start_x: outerRect.x + inner.x0,
            start_y: outerRect.y + inner.y0,
            width: inner.x1 - inner.x0,
            height: inner.y1 - inner.y0,
            ticks_x: fixedTicks([...(model.above || []), ...(model.below || [])]),
            ticks_y: fixedTicks([...(model.left || []), ...(model.right || [])]),
        })
    }
    return out
}"""


def _measure_axes(page):
    figs = page.evaluate(_MEASURE_JS)
    axes = [f for f in figs if f["ticks_x"] or f["ticks_y"]]
    return axes, [f for f in figs if f not in axes]


def assert_axes(page, x=False, y=False) -> None:
    """Check that the GridSpace's x/y axis spans its subplots' frames, with a tick at each column/row center."""
    wait_until(lambda: len(_measure_axes(page)[0]) == x + y, page)
    axes, cells = _measure_axes(page)

    for dim, start, size, present in [("x", "start_x", "width", x), ("y", "start_y", "height", y)]:
        if not present:
            continue
        entry = next(a for a in axes if a[f"ticks_{dim}"])
        lo = min(c[start] for c in cells)
        hi = max(c[start] + c[size] for c in cells)
        assert entry[start] == pytest.approx(lo, abs=2)
        assert entry[start] + entry[size] == pytest.approx(hi, abs=2)
        # one tick per column/row: unique frame centers, merging subpixel differences
        centers = np.unique(np.round([c[start] + c[size] / 2 for c in cells]))
        centers = centers[np.insert(np.diff(centers) > 2, 0, True)]
        ticks = np.array(entry[f"ticks_{dim}"])
        if dim == "y":  # bokeh y ticks count from the bottom, page coordinates from the top
            ticks = 1 - ticks
        np.testing.assert_allclose(np.sort(entry[start] + ticks * entry[size]), centers, atol=2)


@pytest.mark.usefixtures("bokeh_backend")
def test_gridspace(serve_hv):
    plot = hv.Curve([0, 1])
    plots = dict.fromkeys(product(range(2), range(2, 4)), plot)
    gridspace = hv.GridSpace(plots, kdims=["x", "y"])

    page = serve_hv(gridspace)
    assert_axes(page, x=True, y=True)


@pytest.mark.usefixtures("bokeh_backend")
def test_gridspace_legend_alignment(serve_hv):
    plot = hv.NdOverlay({0: hv.Curve([0, 1]), 1: hv.Curve([1, 0])})
    plots = dict.fromkeys(product(range(3), range(3, 6)), plot)
    gridspace = hv.GridSpace(plots, kdims=["x", "y"]).opts(show_legend=True)

    page = serve_hv(gridspace)
    assert_axes(page, x=True, y=True)


@pytest.mark.usefixtures("bokeh_backend")
def test_gridspace_colorbar_alignment(serve_hv):
    x, y = np.arange(4), np.arange(3)
    z = np.arange(12).reshape((3, 4))
    img = hv.Image((x, y, z), kdims=["x", "y"], vdims=["z"]).opts(colorbar=True)
    gridspace = hv.GridSpace(dict.fromkeys(range(5), img), kdims=["t"])

    page = serve_hv(gridspace)
    assert_axes(page, x=True)


@pytest.mark.usefixtures("bokeh_backend")
@pytest.mark.parametrize("xaxis", [None, "bottom", "top"])
@pytest.mark.parametrize("yaxis", [None, "right", "left"])
@pytest.mark.parametrize("shared_xaxis", [True, False])
@pytest.mark.parametrize("shared_yaxis", [True, False])
def test_gridspace_axis_alignment(serve_hv, xaxis, yaxis, shared_xaxis, shared_yaxis):
    plot = hv.Curve([0, 1]).opts(xaxis=xaxis, yaxis=yaxis)
    plots = dict.fromkeys(product(range(2), range(2, 5)), plot)
    gridspace = hv.GridSpace(plots, kdims=["x", "y"]).opts(
        shared_xaxis=shared_xaxis, shared_yaxis=shared_yaxis
    )

    page = serve_hv(gridspace)
    assert_axes(page, x=True, y=True)


@pytest.mark.usefixtures("bokeh_backend")
@pytest.mark.parametrize("id", [".bk-Canvas", ".bk-CartesianFrame"])
def test_gridmatrix_alignment(serve_hv, id):
    ds = hv.Dataset({"a": [0, 1, 2], "b": [0, 50000, 100000]}, kdims=["a", "b"])
    grid = gridmatrix(ds, chart_type=hv.Points)

    page = serve_hv(grid)
    frames = page.locator(id)
    expect(frames).to_have_count(4)
    xs = sorted(frames.nth(i).bounding_box()["x"] for i in range(4))

    assert xs[0] == xs[1]
    assert xs[2] == xs[3]
