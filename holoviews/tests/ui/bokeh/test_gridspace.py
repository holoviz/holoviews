from __future__ import annotations

from itertools import product

import numpy as np
import pytest

import holoviews as hv

from .. import wait_until

pytestmark = pytest.mark.ui

SIZE_TOL = 2  # pixels

_MEASURE_JS = """() => {
    function collectViews() {
        const map = {}
        function walk(view) {
            if (view.model != null) map[view.model.id] = view
            const cv = view.child_views
            if (cv != null) for (const child of cv.values()) walk(child)
        }
        for (const rootId of Object.keys(Bokeh.index)) walk(Bokeh.index[rootId])
        return map
    }
    function fixedTicks(axes) {
        const axis = (axes || []).find((a) => a.ticker && a.ticker.type === "FixedTicker")
        return axis && axis.ticker.ticks.length ? axis.ticker.ticks.slice() : null
    }
    const views = collectViews()
    const out = []
    for (const id of Object.keys(views)) {
        const model = views[id].model
        if (model == null || model.type !== "Figure") continue
        const inner = views[id].frame_view ? views[id].frame_view.bbox : null
        if (inner == null) continue
        const xTicks = fixedTicks([...(model.above || []), ...(model.below || [])])
        const yTicks = fixedTicks([...(model.left || []), ...(model.right || [])])
        if (xTicks || yTicks) {
            const outerRect = views[id].el.getBoundingClientRect()
            out.push({
                start_x: outerRect.x + inner.x0,
                start_y: outerRect.y + inner.y0,
                width: inner.x1 - inner.x0,
                height: inner.y1 - inner.y0,
                ticks_x: xTicks,
                ticks_y: yTicks,
            })
        }
    }
    return out
}"""


def assert_axes(page, x=None, y=None) -> None:
    wait_until(lambda: bool(page.evaluate(_MEASURE_JS)), page)
    axes = page.evaluate(_MEASURE_JS)

    if x is not None:
        start, *ticks, end = x
        entry = next(a for a in axes if a["ticks_x"])
        assert entry["start_x"] == pytest.approx(start, abs=SIZE_TOL)
        assert entry["start_x"] + entry["width"] == pytest.approx(end, abs=SIZE_TOL)
        tick_px = entry["start_x"] + np.array(entry["ticks_x"]) * entry["width"]
        np.testing.assert_allclose(tick_px, ticks, atol=SIZE_TOL)

    if y is not None:
        start, *ticks, end = y
        entry = next(a for a in axes if a["ticks_y"])
        assert entry["start_y"] == pytest.approx(start, abs=SIZE_TOL)
        assert entry["start_y"] + entry["height"] == pytest.approx(end, abs=SIZE_TOL)
        tick_px = entry["start_y"] + np.array(entry["ticks_y"]) * entry["height"]
        np.testing.assert_allclose(tick_px, ticks, atol=SIZE_TOL)


@pytest.mark.usefixtures("bokeh_backend")
def test_gridspace(serve_hv):
    curves = {(x, y): hv.Curve([0, 1]) for x, y in product(range(2), range(2, 4))}
    gridspace = hv.GridSpace(curves, kdims=["x", "y"])
    page = serve_hv(gridspace)

    assert_axes(page, x=(71, 131, 257, 317), y=(33, 93, 219, 279))


@pytest.mark.usefixtures("bokeh_backend")
def test_gridspace_legend_alignment(serve_hv):
    plot = hv.NdOverlay({0: hv.Curve([0, 1]), 1: hv.Curve([1, 0])})
    gridspace = hv.GridSpace(kdims=["x", "y"]).opts(show_legend=True)
    for power, amplitude in product(range(3), range(3, 6)):
        gridspace[amplitude, power] = plot
    page = serve_hv(gridspace)

    assert_axes(page, x=(71, 131, 257, 383, 443), y=(33, 93, 219, 345, 405))


@pytest.mark.usefixtures("bokeh_backend")
@pytest.mark.parametrize("colorbar", [True, False])
def test_gridspace_colorbar_alignment(serve_hv, colorbar):
    x, y = np.arange(4), np.arange(3)
    z = np.arange(12).reshape((3, 4))
    img = hv.Image((x, y, z), kdims=["x", "y"], vdims=["z"]).opts(colorbar=colorbar)
    gridspace = hv.GridSpace(dict.fromkeys(range(5), img), kdims=["t"])
    page = serve_hv(gridspace)

    if colorbar:
        x = (3, 63, 301, 539, 777, 1015, 1075)
    else:  # Without colorbar it has toolbar to right
        x = (3, 63, 216, 369, 522, 675, 735)

    assert_axes(page, x=x)


@pytest.mark.usefixtures("bokeh_backend")
@pytest.mark.parametrize("xaxis", [None, "bottom", "top"])
@pytest.mark.parametrize("yaxis", [None, "right", "left"])
@pytest.mark.parametrize("shared_xaxis", [True, False])
@pytest.mark.parametrize("shared_yaxis", [True, False])
def test_gridspace_axis_alignment(serve_hv, xaxis, yaxis, shared_xaxis, shared_yaxis):
    curves = {(x, y): hv.Curve([0, 1]) for x, y in product(range(2), range(2, 5))}
    gridspace = hv.GridSpace(curves, kdims=["x", "y"]).opts(
        hv.opts.Curve(xaxis=xaxis, yaxis=yaxis),
        hv.opts.GridSpace(shared_xaxis=shared_xaxis, shared_yaxis=shared_yaxis),
    )
    page = serve_hv(gridspace)

    match shared_yaxis, yaxis:
        case False, None:
            x = (71, 131, 257, 317)
        case False, "left":
            x = (120, 180, 355, 415)
        case False, "right":
            x = (71, 131, 306, 366)
        case True, None:
            x = (3, 63, 189, 249)
        case True, "left":
            x = (52, 112, 287, 347)
        case True, "right":
            x = (3, 63, 238, 298)

    match shared_xaxis, xaxis:
        case False, None:
            y = (33, 93, 219, 345, 405)
        case False, "bottom":
            y = (33, 93, 261, 429, 489)
        case False, "top":
            y = (75, 135, 303, 471, 531)
        case True, None:
            y = (92, 152, 278, 404, 464)
        case True, "bottom":
            y = (92, 152, 320, 488, 548)
        case True, "top":
            y = (134, 194, 362, 530, 590)

    assert_axes(page, x=x, y=y)
