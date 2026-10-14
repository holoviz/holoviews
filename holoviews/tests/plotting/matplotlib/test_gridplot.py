from __future__ import annotations

import numpy as np
import pytest

import holoviews as hv

from .test_plot import TestMPLPlot, mpl_renderer


class TestGridPlot(TestMPLPlot):
    @pytest.mark.parametrize("invert_xaxis", [False, True])
    @pytest.mark.parametrize("invert_yaxis", [False, True])
    def test_grid_invert_axis(self, invert_xaxis, invert_yaxis):
        grid = hv.GridSpace(
            {(x, y): hv.Curve([x, y], label=f"{x}-{y}") for x in range(3) for y in range(2)}
        ).opts(invert_xaxis=invert_xaxis, invert_yaxis=invert_yaxis)
        plot = mpl_renderer.get_plot(grid)
        xs = [2, 1, 0] if invert_xaxis else [0, 1, 2]
        ys = [1, 0] if invert_yaxis else [0, 1]
        for r, y in enumerate(ys):
            for c, x in enumerate(xs):
                assert plot.subplots[(r, c)].hmap.last.label == f"{x}-{y}"
                bbox = plot.subaxes[(r, c)].get_position()
                assert bbox.x0 == pytest.approx(plot.subaxes[(0, c)].get_position().x0)
                assert bbox.y0 == pytest.approx(plot.subaxes[(r, 0)].get_position().y0)
                if r:
                    assert bbox.y0 > plot.subaxes[(r - 1, c)].get_position().y0
                if c:
                    assert bbox.x0 > plot.subaxes[(r, c - 1)].get_position().x0
        axis = plot.handles["axis"]
        assert [t.get_text() for t in axis.get_xticklabels()] == list(map(str, xs))
        assert [t.get_text() for t in axis.get_yticklabels()] == list(map(str, ys))

    @pytest.mark.parametrize("invert_xaxis", [False, True])
    @pytest.mark.parametrize("invert_yaxis", [False, True])
    def test_raster_grid_invert_axis(self, invert_xaxis, invert_yaxis):
        grid = hv.GridSpace(
            {(x, y): hv.Image(np.full((2, 2), x + y)) for x in range(3) for y in range(2)}
        ).opts(invert_xaxis=invert_xaxis, invert_yaxis=invert_yaxis)
        plot = mpl_renderer.get_plot(grid)
        xs = [2, 1, 0] if invert_xaxis else [0, 1, 2]
        ys = [1, 0] if invert_yaxis else [0, 1]
        axis = plot.handles["axis"]

        def center(key):
            l, r, b, t = plot.handles["projs"][key].get_extent()
            return axis.transData.transform(((l + r) / 2, (b + t) / 2))

        for r, y in enumerate(ys):
            for c, x in enumerate(xs):
                cx, cy = center((x, y))
                if r:
                    assert cy > center((x, ys[r - 1]))[1]
                if c:
                    assert cx > center((xs[c - 1], y))[0]
        assert [t.get_text() for t in axis.get_xticklabels()] == list(map(str, xs))
        assert [t.get_text() for t in axis.get_yticklabels()] == list(map(str, ys))
