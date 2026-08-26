from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

import holoviews as hv
from holoviews.core.options import AbbreviatedException
from holoviews.testing import assert_data_equal

from .test_plot import TestMPLPlot, mpl_renderer


class TestPointPlot(TestMPLPlot):
    def test_points_rcparams_do_not_persist(self):
        opts = dict(fig_rcparams={"text.usetex": True})
        points = hv.Points(([0, 1], [0, 3])).opts(**opts)
        mpl_renderer.get_plot(points)
        assert not plt.rcParams["text.usetex"]

    def test_points_rcparams_used(self):
        opts = dict(fig_rcparams={"grid.color": "red"})
        points = hv.Points(([0, 1], [0, 3])).opts(**opts)
        plot = mpl_renderer.get_plot(points)
        ax = plot.state.axes[0]
        lines = ax.get_xgridlines()
        assert lines[0].get_color() == "red"

    def test_points_padding_square(self):
        points = hv.Points([1, 2, 3]).opts(padding=0.1)
        plot = mpl_renderer.get_plot(points)
        x_range, y_range = plot.handles["axis"].get_xlim(), plot.handles["axis"].get_ylim()
        assert x_range[0] == -0.2
        assert x_range[1] == 2.2
        assert y_range[0] == 0.8
        assert y_range[1] == 3.2

    def test_curve_padding_square_per_axis(self):
        curve = hv.Points([1, 2, 3]).opts(padding=((0, 0.1), (0.1, 0.2)))
        plot = mpl_renderer.get_plot(curve)
        x_range, y_range = plot.handles["axis"].get_xlim(), plot.handles["axis"].get_ylim()
        assert x_range[0] == 0
        assert x_range[1] == 2.2
        assert y_range[0] == 0.8
        assert y_range[1] == 3.4

    def test_points_padding_hard_xrange(self):
        points = hv.Points([1, 2, 3]).redim.range(x=(0, 3)).opts(padding=0.1)
        plot = mpl_renderer.get_plot(points)
        x_range, y_range = plot.handles["axis"].get_xlim(), plot.handles["axis"].get_ylim()
        assert x_range[0] == 0
        assert x_range[1] == 3
        assert y_range[0] == 0.8
        assert y_range[1] == 3.2

    def test_points_padding_soft_xrange(self):
        points = hv.Points([1, 2, 3]).redim.soft_range(x=(0, 3)).opts(padding=0.1)
        plot = mpl_renderer.get_plot(points)
        x_range, y_range = plot.handles["axis"].get_xlim(), plot.handles["axis"].get_ylim()
        assert x_range[0] == 0
        assert x_range[1] == 3
        assert y_range[0] == 0.8
        assert y_range[1] == 3.2

    def test_points_padding_unequal(self):
        points = hv.Points([1, 2, 3]).opts(padding=(0.05, 0.1))
        plot = mpl_renderer.get_plot(points)
        x_range, y_range = plot.handles["axis"].get_xlim(), plot.handles["axis"].get_ylim()
        assert x_range[0] == -0.1
        assert x_range[1] == 2.1
        assert y_range[0] == 0.8
        assert y_range[1] == 3.2

    def test_points_padding_nonsquare(self):
        points = hv.Points([1, 2, 3]).opts(padding=0.1, aspect=2)
        plot = mpl_renderer.get_plot(points)
        x_range, y_range = plot.handles["axis"].get_xlim(), plot.handles["axis"].get_ylim()
        assert x_range[0] == -0.1
        assert x_range[1] == 2.1
        assert y_range[0] == 0.8
        assert y_range[1] == 3.2

    def test_points_padding_logx(self):
        points = hv.Points([(1, 1), (2, 2), (3, 3)]).opts(padding=0.1, logx=True)
        plot = mpl_renderer.get_plot(points)
        x_range, y_range = plot.handles["axis"].get_xlim(), plot.handles["axis"].get_ylim()
        assert x_range[0] == 0.89595845984076228
        assert x_range[1] == 3.3483695221017129
        assert y_range[0] == 0.8
        assert y_range[1] == 3.2

    def test_points_padding_logy(self):
        points = hv.Points([1, 2, 3]).opts(padding=0.1, logy=True)
        plot = mpl_renderer.get_plot(points)
        x_range, y_range = plot.handles["axis"].get_xlim(), plot.handles["axis"].get_ylim()
        assert x_range[0] == -0.2
        assert x_range[1] == 2.2
        assert y_range[0] == 0.89595845984076228
        assert y_range[1] == 3.3483695221017129

    def test_points_padding_datetime_square(self):
        points = hv.Points([(np.datetime64(f"2016-04-0{i}"), i) for i in range(1, 4)]).opts(
            padding=0.1
        )
        plot = mpl_renderer.get_plot(points)
        x_range, y_range = plot.handles["axis"].get_xlim(), plot.handles["axis"].get_ylim()
        assert x_range[0] == 16891.8
        assert x_range[1] == 16894.2
        assert y_range[0] == 0.8
        assert y_range[1] == 3.2

    def test_points_padding_datetime_nonsquare(self):
        points = hv.Points([(np.datetime64(f"2016-04-0{i}"), i) for i in range(1, 4)]).opts(
            padding=0.1, aspect=2
        )
        plot = mpl_renderer.get_plot(points)
        x_range, y_range = plot.handles["axis"].get_xlim(), plot.handles["axis"].get_ylim()
        assert x_range[0] == 16891.9
        assert x_range[1] == 16894.1
        assert y_range[0] == 0.8
        assert y_range[1] == 3.2

    def test_points_sizes_scalar_update(self):
        hmap = hv.HoloMap({i: hv.Points([1, 2, 3]).opts(s=i * 10) for i in range(1, 3)})
        plot = mpl_renderer.get_plot(hmap)
        artist = plot.handles["artist"]
        plot.update((1,))
        assert_data_equal(artist.get_sizes(), np.array([10]))
        plot.update((2,))
        assert_data_equal(artist.get_sizes(), np.array([20]))

    ###########################
    #    Styling mapping      #
    ###########################

    def test_point_color_op(self):
        points = hv.Points(
            [(0, 0, "#000000"), (0, 1, "#FF0000"), (0, 2, "#00FF00")], vdims="color"
        ).opts(color="color")
        plot = mpl_renderer.get_plot(points)
        artist = plot.handles["artist"]
        assert_data_equal(
            artist.get_facecolors(), np.array([[0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1]])
        )

    def test_point_color_op_update(self):
        points = hv.HoloMap(
            {
                0: hv.Points(
                    [(0, 0, "#000000"), (0, 1, "#FF0000"), (0, 2, "#00FF00")], vdims="color"
                ),
                1: hv.Points(
                    [(0, 0, "#0000FF"), (0, 1, "#00FF00"), (0, 2, "#FF0000")], vdims="color"
                ),
            }
        ).opts(color="color")
        plot = mpl_renderer.get_plot(points)
        artist = plot.handles["artist"]
        plot.update((1,))
        assert_data_equal(
            artist.get_facecolors(), np.array([[0, 0, 1, 1], [0, 1, 0, 1], [1, 0, 0, 1]])
        )

    def test_point_line_color_op(self):
        points = hv.Points(
            [(0, 0, "#000000"), (0, 1, "#FF0000"), (0, 2, "#00FF00")], vdims="color"
        ).opts(edgecolors="color")
        plot = mpl_renderer.get_plot(points)
        artist = plot.handles["artist"]
        assert_data_equal(
            artist.get_edgecolors(), np.array([[0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1]])
        )

    def test_point_line_color_op_update(self):
        points = hv.HoloMap(
            {
                0: hv.Points(
                    [(0, 0, "#000000"), (0, 1, "#FF0000"), (0, 2, "#00FF00")], vdims="color"
                ),
                1: hv.Points(
                    [(0, 0, "#0000FF"), (0, 1, "#00FF00"), (0, 2, "#FF0000")], vdims="color"
                ),
            }
        ).opts(edgecolors="color")
        plot = mpl_renderer.get_plot(points)
        artist = plot.handles["artist"]
        plot.update((1,))
        assert_data_equal(
            artist.get_edgecolors(), np.array([[0, 0, 1, 1], [0, 1, 0, 1], [1, 0, 0, 1]])
        )

    def test_point_fill_color_op(self):
        points = hv.Points(
            [(0, 0, "#000000"), (0, 1, "#FF0000"), (0, 2, "#00FF00")], vdims="color"
        ).opts(facecolors="color")
        plot = mpl_renderer.get_plot(points)
        artist = plot.handles["artist"]
        assert_data_equal(
            artist.get_facecolors(), np.array([[0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1]])
        )

    def test_point_linear_color_op(self):
        points = hv.Points([(0, 0, 0), (0, 1, 1), (0, 2, 2)], vdims="color").opts(color="color")
        plot = mpl_renderer.get_plot(points)
        artist = plot.handles["artist"]
        assert_data_equal(np.asarray(artist.get_array()), np.array([0, 1, 2]))
        assert artist.get_clim() == (0, 2)

    def test_point_linear_color_op_update(self):
        points = hv.HoloMap(
            {
                0: hv.Points([(0, 0, 0), (0, 1, 1), (0, 2, 2)], vdims="color"),
                1: hv.Points([(0, 0, 2.5), (0, 1, 3), (0, 2, 1.2)], vdims="color"),
            }
        ).opts(color="color", framewise=True)
        plot = mpl_renderer.get_plot(points)
        artist = plot.handles["artist"]
        assert artist.get_clim() == (0, 2)
        plot.update((1,))
        assert_data_equal(np.asarray(artist.get_array()), np.array([2.5, 3, 1.2]))
        assert artist.get_clim() == (1.2, 3)

    def test_point_categorical_color_op(self):
        points = hv.Points([(0, 0, "A"), (0, 1, "B"), (0, 2, "A")], vdims="color").opts(
            color="color"
        )
        plot = mpl_renderer.get_plot(points)
        artist = plot.handles["artist"]
        assert_data_equal(np.asarray(artist.get_array()), np.array([0, 1, 0]))
        assert artist.get_clim() == (0, 1)

    def test_point_categorical_color_op_legend(self):
        points = hv.Points([(0, 0, "A"), (0, 1, "B"), (0, 2, "A")], vdims="color").opts(
            color="color", show_legend=True
        )
        plot = mpl_renderer.get_plot(points)
        leg = plot.handles["axis"].get_legend()
        legend_labels = [l.get_text() for l in leg.texts]
        assert legend_labels == ["A", "B"]

    def test_point_categorical_color_op_legend_with_labels(self):
        points = hv.Points([(0, 0, "A"), (0, 1, "B"), (0, 2, "A")], vdims="color").opts(
            color="color", show_legend=True, legend_labels={"A": "A point", "B": "B point"}
        )
        plot = mpl_renderer.get_plot(points)
        leg = plot.handles["axis"].get_legend()
        legend_labels = [l.get_text() for l in leg.texts]
        assert legend_labels == ["A point", "B point"]

    def test_point_size_op(self):
        points = hv.Points([(0, 0, 1), (0, 1, 4), (0, 2, 8)], vdims="size").opts(s="size")
        plot = mpl_renderer.get_plot(points)
        artist = plot.handles["artist"]
        assert_data_equal(artist.get_sizes(), np.array([1, 4, 8]))

    def test_point_size_op_update(self):
        points = hv.HoloMap(
            {
                0: hv.Points([(0, 0, 3), (0, 1, 1), (0, 2, 2)], vdims="size"),
                1: hv.Points([(0, 0, 2.5), (0, 1, 3), (0, 2, 1.2)], vdims="size"),
            }
        ).opts(s="size")
        plot = mpl_renderer.get_plot(points)
        artist = plot.handles["artist"]
        assert_data_equal(artist.get_sizes(), np.array([3, 1, 2]))
        plot.update((1,))
        assert_data_equal(artist.get_sizes(), np.array([2.5, 3, 1.2]))

    def test_point_line_width_op(self):
        points = hv.Points([(0, 0, 1), (0, 1, 4), (0, 2, 8)], vdims="line_width").opts(
            linewidth="line_width"
        )
        plot = mpl_renderer.get_plot(points)
        artist = plot.handles["artist"]
        assert artist.get_linewidths() == [1, 4, 8]

    def test_point_line_width_op_update(self):
        points = hv.HoloMap(
            {
                0: hv.Points([(0, 0, 3), (0, 1, 1), (0, 2, 2)], vdims="line_width"),
                1: hv.Points([(0, 0, 2.5), (0, 1, 3), (0, 2, 1.2)], vdims="line_width"),
            }
        ).opts(linewidth="line_width")
        plot = mpl_renderer.get_plot(points)
        artist = plot.handles["artist"]
        assert artist.get_linewidths() == [3, 1, 2]
        plot.update((1,))
        assert artist.get_linewidths() == [2.5, 3, 1.2]

    def test_point_marker_op(self):
        points = hv.Points(
            [(0, 0, "circle"), (0, 1, "triangle"), (0, 2, "square")], vdims="marker"
        ).opts(marker="marker")
        msg = 'ValueError: Mapping a dimension to the "marker" style'
        with pytest.raises(AbbreviatedException, match=msg):
            mpl_renderer.get_plot(points)

    def test_point_alpha_op(self):
        points = hv.Points([(0, 0, 0), (0, 1, 0.2), (0, 2, 0.7)], vdims="alpha").opts(
            alpha="alpha"
        )
        msg = 'ValueError: Mapping a dimension to the "alpha" style'
        with pytest.raises(AbbreviatedException, match=msg):
            mpl_renderer.get_plot(points)

    def test_op_ndoverlay_value(self):
        markers = ["d", "s"]
        overlay = hv.NdOverlay(
            {marker: hv.Points(np.arange(i)) for i, marker in enumerate(markers)}, "Marker"
        ).opts("Points", marker="Marker")
        plot = mpl_renderer.get_plot(overlay)
        for subplot, marker in zip(plot.subplots.values(), markers, strict=True):
            style = dict(subplot.style[subplot.cyclic_index])
            style = subplot._apply_transforms(subplot.current_frame, {}, style)
            assert style["marker"] == marker
