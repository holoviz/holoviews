from __future__ import annotations

from itertools import product

import numpy as np
import pytest

import holoviews as hv
from holoviews.plotting.bokeh.renderer import BokehRenderer

from .. import wait_until

pytestmark = pytest.mark.ui

TICK_TOL = 0.01
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
        start, end, *ticks = x
        entry = next(a for a in axes if a["ticks_x"])
        assert entry["start_x"] == pytest.approx(start, abs=SIZE_TOL)
        assert entry["start_x"] + entry["width"] == pytest.approx(end, abs=SIZE_TOL)
        np.testing.assert_allclose(entry["ticks_x"], ticks, atol=TICK_TOL)

    if y is not None:
        start, end, *ticks = y
        entry = next(a for a in axes if a["ticks_y"])
        assert entry["start_y"] == pytest.approx(start, abs=SIZE_TOL)
        assert entry["start_y"] + entry["height"] == pytest.approx(end, abs=SIZE_TOL)
        np.testing.assert_allclose(entry["ticks_y"], ticks, atol=TICK_TOL)


@pytest.mark.usefixtures("bokeh_backend")
def test_gridplot(serve_hv):
    curves = {(x, y): hv.Curve([0, 1]) for x, y in product(range(2), range(2, 4))}
    gridspace = hv.GridSpace(curves, kdims=["x", "y"])

    page = serve_hv(gridspace)

    ticks = (0.243902, 0.756098)
    assert_axes(page, x=(71, 317, *ticks), y=(33, 279, *ticks))


@pytest.mark.usefixtures("bokeh_backend")
def test_gridplot_legend_alignment(serve_hv):
    plot = hv.NdOverlay({0: hv.Curve([0, 1]), 1: hv.Curve([1, 0])})
    gridspace = hv.GridSpace(kdims=["x", "y"]).opts(show_legend=True)
    for power, amplitude in product(range(3), range(3, 6)):
        gridspace[amplitude, power] = plot

    page = serve_hv(gridspace)

    ticks = (0.161290, 0.5, 0.838710)
    assert_axes(page, x=(71, 443, *ticks), y=(33, 405, *ticks))


@pytest.mark.usefixtures("bokeh_backend")
@pytest.mark.parametrize("colorbar", [True, False])
def test_gridplot_colorbar_alignment(serve_hv, colorbar):
    x, y = np.arange(4), np.arange(3)
    z = np.arange(12).reshape((3, 4))
    img = hv.Image((x, y, z), kdims=["x", "y"], vdims=["z"]).opts(colorbar=colorbar)
    gridspace = hv.GridSpace(dict.fromkeys(range(5), img), kdims=["t"])

    page = serve_hv(gridspace)

    if colorbar:
        x = (3, 1075, 0.055762, 0.277881, 0.5, 0.722119, 0.944238)
    else:  # Without colorbar it has toolbar to right
        x = (3, 735, 0.081967, 0.290984, 0.5, 0.709016, 0.918033)

    assert_axes(page, x=x)


@pytest.mark.usefixtures("bokeh_backend")
@pytest.mark.parametrize("xaxis", [None, "bottom"])
@pytest.mark.parametrize("yaxis", [None, "right"])
def test_gridplot_per_cell_axis_alignment(serve_hv, xaxis, yaxis):
    # A real (visible) per-cell x/y-axis on every cell adds the same
    # "leading gutter" to every cell in a row/column, unlike a legend
    # or colorbar which only widens the one cell that carries it -- a
    # different code path in bind_dynamic_axis_sizing's leading-edge
    # tracking. Sparse on purpose, to also exercise empty cells.
    years = [2023, 2024]
    months = [5, 6, 7]
    points = {(2023, 5): [(1, 4)], (2024, 6): [(1, 8)], (2024, 7): [(1, 8)]}
    items = {
        (year, month): hv.Scatter(points.get((year, month), []))
        for year in years
        for month in months
    }
    gridspace = hv.GridSpace(items, kdims=["year", "month"]).opts(
        hv.opts.Scatter(xaxis=xaxis, yaxis=yaxis, show_legend=False),
        hv.opts.GridSpace(shared_xaxis=True, shared_yaxis=True),
    )

    plot = BokehRenderer.get_plot(gridspace)
    page = serve_hv(plot)

    # A real per-cell y-axis grows every cell's horizontal space, which
    # the fake x-axis has to track; a real per-cell x-axis grows every
    # cell's vertical space, tracked by the fake y-axis -- crossed, not
    # matched by name.
    x_start_end, x_ticks = {
        None: ((3, 249), [0.243902, 0.756098]),
        "right": ((3, 289), [0.209790, 0.790210]),
    }[yaxis]
    y_start_end, y_ticks = {
        None: ((92, 464), [0.161290, 0.5, 0.838710]),
        "bottom": ((92, 548), [0.131579, 0.5, 0.868421]),
    }[xaxis]

    assert_axes(
        page,
        x=(*x_start_end, *x_ticks),
        y=(*y_start_end, *y_ticks),
    )
