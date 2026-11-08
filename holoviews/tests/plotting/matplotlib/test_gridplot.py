from __future__ import annotations

import holoviews as hv
from holoviews.operation import gridmatrix

from .test_plot import TestMPLPlot, mpl_renderer


class TestGridPlot(TestMPLPlot):
    def test_gridmatrix_diagonal_left_edge_labelled_with_row_dimension(self, rng):
        ds = hv.Dataset({k: rng.random(10) for k in "abc"}, kdims=list("abc"))
        plot = mpl_renderer.get_plot(gridmatrix(ds))

        axis = plot.subplots[(0, 0)].handles["axis"]
        assert axis.get_ylabel() == "a"
        assert len(axis.get_yticks()) == 0
        assert plot.subplots[(1, 0)].handles["axis"].get_ylabel() == "b"
