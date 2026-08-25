from __future__ import annotations

import numpy as np
import pytest

import holoviews as hv
from holoviews.plotting.bokeh.renderer import BokehRenderer

from .. import wait_until

pytestmark = pytest.mark.ui

TICK_TOL = 0.01
SIZE_TOL = 2  # pixels

# Reads the real rendered geometry (Bokeh.index walk + view.frame_view.bbox,
# same technique GridPlot's own dynamic axis sizing uses) of GridPlot's
# shared fake x/y axis figures -- identified by carrying a FixedTicker,
# which only make_axis's axes use.
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
    const views = collectViews()
    const out = []
    for (const id of Object.keys(views)) {
        const model = views[id].model
        if (model == null || model.type !== "Figure") continue
        const inner = views[id].frame_view ? views[id].frame_view.bbox : null
        if (inner == null) continue
        // model.xaxis/model.yaxis are Python-only convenience properties,
        // not real BokehJS getters -- find axes via the actual above/
        // below/left/right layout lists instead.
        function fixedTicks(axes) {
            const axis = (axes || []).find((a) => a.ticker && a.ticker.type === "FixedTicker")
            return axis && axis.ticker.ticks.length ? axis.ticker.ticks.slice() : null
        }
        const xTicks = fixedTicks([...(model.above || []), ...(model.below || [])])
        const yTicks = fixedTicks([...(model.left || []), ...(model.right || [])])
        if (xTicks || yTicks) {
            const outerRect = views[id].el.getBoundingClientRect()
            out.push({
                frameStartX: outerRect.x + inner.x0,
                frameStartY: outerRect.y + inner.y0,
                frameWidth: inner.x1 - inner.x0,
                frameHeight: inner.y1 - inner.y0,
                xTicks, yTicks,
            })
        }
    }
    return out
}"""


def _measure_axes(page):
    return page.evaluate(_MEASURE_JS)


def _assert_axes(page, x=None, y=None):
    """Asserts GridPlot's shared fake x/y axes render at fixed, previously
    verified pixel start/end positions and tick fractions -- golden values,
    not values re-measured from the same page, so a Bokeh-side layout
    change actually surfaces as a failure instead of silently moving both
    sides together.

    `x`/`y` are (start, end, ticks) triples; pass None to skip an axis.
    """
    wait_until(lambda: bool(_measure_axes(page)), page)
    axes = _measure_axes(page)

    if x is not None:
        start, end, ticks = x
        entry = next(a for a in axes if a["xTicks"])
        assert entry["frameStartX"] == pytest.approx(start, abs=SIZE_TOL)
        assert entry["frameStartX"] + entry["frameWidth"] == pytest.approx(end, abs=SIZE_TOL)
        np.testing.assert_allclose(entry["xTicks"], ticks, atol=TICK_TOL)

    if y is not None:
        start, end, ticks = y
        entry = next(a for a in axes if a["yTicks"])
        assert entry["frameStartY"] == pytest.approx(start, abs=SIZE_TOL)
        assert entry["frameStartY"] + entry["frameHeight"] == pytest.approx(end, abs=SIZE_TOL)
        np.testing.assert_allclose(entry["yTicks"], ticks, atol=TICK_TOL)


@pytest.mark.usefixtures("bokeh_backend")
@pytest.mark.parametrize("shared_xaxis", [True, False])
@pytest.mark.parametrize("shared_yaxis", [True, False])
def test_gridplot_axis_alignment_combinations(serve_hv, shared_xaxis, shared_yaxis):
    curves = {(x, y): hv.Curve(np.arange(10) * (x + y + 1)) for x in range(2) for y in range(2)}
    gridspace = hv.GridSpace(curves, kdims=["x", "y"]).opts(
        shared_xaxis=shared_xaxis, shared_yaxis=shared_yaxis
    )

    plot = BokehRenderer.get_plot(gridspace)
    page = serve_hv(plot)

    ticks = [0.243902, 0.756098]
    # Flipping shared_xaxis/shared_yaxis moves the fake axis to the far
    # side of the grid, so start/end differ per combination even though
    # size and tick fractions don't.
    x_start_end, y_start_end = {
        (True, True): ((49, 295), (92, 338)),
        (True, False): ((71, 317), (92, 338)),
        (False, True): ((49, 295), (33, 279)),
        (False, False): ((71, 317), (33, 279)),
    }[shared_xaxis, shared_yaxis]

    _assert_axes(
        page,
        x=(*x_start_end, ticks),
        y=(*y_start_end, ticks),
    )


@pytest.mark.usefixtures("bokeh_backend")
def test_gridplot_legend_alignment(serve_hv):
    # A legend on one cell grows that cell's rendered canvas past
    # frame_width -- the fake axis has to track the real width, not the
    # nominal frame_width, or its ticks drift out from under later columns.
    powers = [1, 2, 3]
    amplitudes = [0.5, 0.75, 1.0]
    gridspace = hv.GridSpace(kdims=["Amplitude", "Power"])
    for power in powers:
        for amplitude in amplitudes:
            lines = hv.NdOverlay({0: hv.Curve([0, 1]), 1: hv.Curve([1, 0])}, kdims="Phase")
            gridspace[amplitude, power] = lines
    gridspace = gridspace.opts(show_legend=True)

    plot = BokehRenderer.get_plot(gridspace)
    page = serve_hv(plot)

    ticks = [0.161290, 0.5, 0.838710]
    _assert_axes(page, x=(71, 443, ticks), y=(33, 405, ticks))


@pytest.mark.usefixtures("bokeh_backend")
@pytest.mark.parametrize(
    ("colorbar", "end"),
    [(True, 1132), (False, 788)],
)
def test_gridplot_colorbar_alignment(serve_hv, colorbar, end):
    # A per-cell colorbar grows that cell's canvas the same way a legend
    # does; exercised separately since colorbar width tracks its own
    # formatter/label width rather than a legend's entries.
    np.random.seed(0)
    data = np.random.randn(5, 3, 4)
    lons = np.linspace(1, 100, 4)
    lats = np.linspace(1, 100, 3)
    images = {
        t: hv.Image((lons, lats, data[i]), kdims=["lon", "lat"], vdims=["value"])
        for i, t in enumerate([1, 2, 3, 4, 5])
    }
    gridspace = hv.GridSpace(images, kdims=["time"]).opts(
        hv.opts.Image(colorbar=colorbar),
        hv.opts.GridSpace(shared_xaxis=True, shared_yaxis=True),
    )

    plot = BokehRenderer.get_plot(gridspace)
    page = serve_hv(plot)

    ticks = {
        True: [0.055762, 0.277881, 0.5, 0.722119, 0.944238],
        False: [0.081967, 0.290984, 0.5, 0.709016, 0.918033],
    }[colorbar]
    _assert_axes(page, x=(56, end, ticks))


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

    _assert_axes(
        page,
        x=(*x_start_end, x_ticks),
        y=(*y_start_end, y_ticks),
    )
