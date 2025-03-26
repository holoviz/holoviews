import numpy as np

import holoviews as hv
from holoviews.element import HLines, HSpans, VLines, VSpans
from holoviews.plotting.mpl.util import MPL_GE_3_9_0

from .test_plot import TestMPLPlot, mpl_renderer


class TestHVLinesPlot(TestMPLPlot):
    def test_hlines_plot(self):
        hlines = HLines(
            {"y": [0, 1, 2, 5.5], "extra": [-1, -2, -3, -44]}, vdims=["extra"]
        )
        plot = mpl_renderer.get_plot(hlines)
        assert plot.handles["fig"].axes[0].get_xlabel() == "x"
        assert plot.handles["fig"].axes[0].get_ylabel() == "y"

        xlim = plot.handles["fig"].axes[0].get_xlim()
        ylim = plot.handles["fig"].axes[0].get_ylim()
        assert np.allclose(xlim, (-0.055, 0.055))
        assert np.allclose(ylim, (0, 5.5))

        sources = plot.handles["annotations"]
        assert len(sources) == 4
        for source, val in zip(sources, hlines.data["y"], strict=None):
            assert source.get_data() == ([0, 1], [val, val])

    def test_hlines_array(self):
        hlines = HLines(np.array([0, 1, 2, 5.5]))
        plot = mpl_renderer.get_plot(hlines)
        assert plot.handles["fig"].axes[0].get_xlabel() == "x"
        assert plot.handles["fig"].axes[0].get_ylabel() == "y"

        xlim = plot.handles["fig"].axes[0].get_xlim()
        ylim = plot.handles["fig"].axes[0].get_ylim()
        assert np.allclose(xlim, (-0.055, 0.055))
        assert np.allclose(ylim, (0, 5.5))

        sources = plot.handles["annotations"]
        assert len(sources) == 4
        for source, val in zip(sources, hlines.data, strict=None):
            assert source.get_data() == ([0, 1], [val, val])

    def test_hlines_plot_invert_axes(self):
        hlines = HLines(
            {"y": [0, 1, 2, 5.5], "extra": [-1, -2, -3, -44]}, vdims=["extra"]
        ).opts(invert_axes=True)
        plot = mpl_renderer.get_plot(hlines)
        assert plot.handles["fig"].axes[0].get_xlabel() == "y"
        assert plot.handles["fig"].axes[0].get_ylabel() == "x"

        xlim = plot.handles["fig"].axes[0].get_xlim()
        ylim = plot.handles["fig"].axes[0].get_ylim()
        assert np.allclose(xlim, (0, 5.5))
        assert np.allclose(ylim, (-0.055, 0.055))

        sources = plot.handles["annotations"]
        assert len(sources) == 4
        for source, val in zip(sources, hlines.data["y"], strict=None):
            assert source.get_data() == ([val, val], [0, 1])

    def test_hlines_nondefault_kdim(self):
        hlines = HLines({"other": [0, 1, 2, 5.5]}, kdims=["other"])
        plot = mpl_renderer.get_plot(hlines)
        assert plot.handles["fig"].axes[0].get_xlabel() == "x"
        assert plot.handles["fig"].axes[0].get_ylabel() == "y"

        xlim = plot.handles["fig"].axes[0].get_xlim()
        ylim = plot.handles["fig"].axes[0].get_ylim()
        assert np.allclose(xlim, (-0.055, 0.055))
        assert np.allclose(ylim, (0, 5.5))

        sources = plot.handles["annotations"]
        assert len(sources) == 4
        for source, val in zip(sources, hlines.data["other"], strict=None):
            assert source.get_data() == ([0, 1], [val, val])

    def test_vlines_plot(self):
        vlines = VLines(
            {"x": [0, 1, 2, 5.5], "extra": [-1, -2, -3, -44]}, vdims=["extra"]
        )
        plot = mpl_renderer.get_plot(vlines)
        assert plot.handles["fig"].axes[0].get_xlabel() == "x"
        assert plot.handles["fig"].axes[0].get_ylabel() == "y"

        xlim = plot.handles["fig"].axes[0].get_xlim()
        ylim = plot.handles["fig"].axes[0].get_ylim()
        assert np.allclose(xlim, (0, 5.5))
        assert np.allclose(ylim, (-0.055, 0.055))

        sources = plot.handles["annotations"]
        assert len(sources) == 4
        for source, val in zip(sources, vlines.data["x"], strict=None):
            assert source.get_data() == ([val, val], [0, 1])

    def test_vlines_plot_invert_axes(self):
        vlines = VLines(
            {"x": [0, 1, 2, 5.5], "extra": [-1, -2, -3, -44]}, vdims=["extra"]
        ).opts(invert_axes=True)
        plot = mpl_renderer.get_plot(vlines)
        assert plot.handles["fig"].axes[0].get_xlabel() == "y"
        assert plot.handles["fig"].axes[0].get_ylabel() == "x"

        xlim = plot.handles["fig"].axes[0].get_xlim()
        ylim = plot.handles["fig"].axes[0].get_ylim()
        assert np.allclose(xlim, (-0.055, 0.055))
        assert np.allclose(ylim, (0, 5.5))

        sources = plot.handles["annotations"]
        assert len(sources) == 4
        for source, val in zip(sources, vlines.data["x"], strict=None):
            assert source.get_data() == ([0, 1], [val, val])

    def test_vlines_nondefault_kdim(self):
        vlines = VLines({"other": [0, 1, 2, 5.5]}, kdims=["other"])
        plot = mpl_renderer.get_plot(vlines)
        assert plot.handles["fig"].axes[0].get_xlabel() == "x"
        assert plot.handles["fig"].axes[0].get_ylabel() == "y"

        xlim = plot.handles["fig"].axes[0].get_xlim()
        ylim = plot.handles["fig"].axes[0].get_ylim()
        assert np.allclose(xlim, (0, 5.5))
        assert np.allclose(ylim, (-0.055, 0.055))

        sources = plot.handles["annotations"]
        assert len(sources) == 4
        for source, val in zip(sources, vlines.data["other"], strict=None):
            assert source.get_data() == ([val, val], [0, 1])

    def test_vlines_hlines_overlay(self):
        hlines = HLines(
            {"y": [0, 1, 2, 5.5], "extra": [-1, -2, -3, -44]}, vdims=["extra"]
        )
        vlines = VLines(
            {"x": [0, 1, 2, 5.5], "extra": [-1, -2, -3, -44]}, vdims=["extra"]
        )

        plot = mpl_renderer.get_plot(hlines * vlines)
        assert plot.handles["fig"].axes[0].get_xlabel() == "x"
        assert plot.handles["fig"].axes[0].get_ylabel() == "y"

        xlim = plot.handles["fig"].axes[0].get_xlim()
        ylim = plot.handles["fig"].axes[0].get_ylim()
        assert np.allclose(xlim, (0, 5.5))
        assert np.allclose(ylim, (0, 5.5))

        sources = plot.handles["fig"].axes[0].get_children()
        for source, val in zip(sources[:4], hlines.data["y"], strict=None):
            assert source.get_data() == ([0, 1], [val, val])

        for source, val in zip(sources[4:], vlines.data["x"], strict=None):
            assert source.get_data() == ([val, val], [0, 1])


class TestHVSpansPlot(TestMPLPlot):

    def _hspans_check(self, source, v0, v1):
        # Matplotlib 3.9+ uses a rectangle instead of polygon
        if MPL_GE_3_9_0:
            rect = [source.get_x(), source.get_y(), source.get_width(), source.get_height()]
            assert np.allclose(rect, [0, v0, 1, v1 - v0])
        else:
            assert np.allclose(source.xy[:, 0], [0, 0, 1, 1, 0])
            assert np.allclose(source.xy[:, 1], [v0, v1, v1, v0, v0])

    def _vspans_check(self, source, v0, v1):
        # Matplotlib 3.9+ uses a rectangle instead of polygon
        if MPL_GE_3_9_0:
            rect = [source.get_x(), source.get_y(), source.get_width(), source.get_height()]
            assert np.allclose(rect, [v0, 0, v1 - v0, 1])
        else:
            assert np.allclose(source.xy[:, 1], [0, 1, 1, 0, 0])
            assert np.allclose(source.xy[:, 0], [v0, v0, v1, v1, v0])

    def test_hspans_plot(self):
        hspans = HSpans(
            {"y0": [0, 3, 5.5], "y1": [1, 4, 6.5], "extra": [-1, -2, -3]},
            vdims=["extra"],
        )
        plot = mpl_renderer.get_plot(hspans)
        assert plot.handles["fig"].axes[0].get_xlabel() == "x"
        assert plot.handles["fig"].axes[0].get_ylabel() == "y"

        xlim = plot.handles["fig"].axes[0].get_xlim()
        ylim = plot.handles["fig"].axes[0].get_ylim()
        assert np.allclose(xlim, (-0.055, 0.055))
        assert np.allclose(ylim, (0, 6.5))

        sources = plot.handles["annotations"]
        assert len(sources) == 3
        for source, v0, v1 in zip(sources, hspans.data["y0"], hspans.data["y1"], strict=None):
            self._hspans_check(source, v0, v1)

    def test_hspans_inverse_plot(self):
        hspans = HSpans(
            {"y0": [0, 3, 5.5], "y1": [1, 4, 6.5], "extra": [-1, -2, -3]},
            vdims=["extra"],
        ).opts(invert_axes=True)
        plot = mpl_renderer.get_plot(hspans)
        assert plot.handles["fig"].axes[0].get_xlabel() == "y"
        assert plot.handles["fig"].axes[0].get_ylabel() == "x"

        xlim = plot.handles["fig"].axes[0].get_xlim()
        ylim = plot.handles["fig"].axes[0].get_ylim()
        assert np.allclose(xlim, (0, 6.5))
        assert np.allclose(ylim, (-0.055, 0.055))

        sources = plot.handles["annotations"]
        assert len(sources) == 3
        for source, v0, v1 in zip(sources, hspans.data["y0"], hspans.data["y1"], strict=None):
            self._vspans_check(source, v0, v1)

    def test_dynamicmap_overlay_hspans(self):
        el = HSpans(data=[[1, 3], [2, 4]])
        dmap = hv.DynamicMap(lambda: hv.Overlay([el]))

        plot_el = mpl_renderer.get_plot(el)
        plot_dmap = mpl_renderer.get_plot(dmap)

        xlim_el = plot_el.handles["fig"].axes[0].get_xlim()
        ylim_el = plot_el.handles["fig"].axes[0].get_ylim()
        xlim_dmap = plot_dmap.handles["fig"].axes[0].get_xlim()
        ylim_dmap = plot_dmap.handles["fig"].axes[0].get_ylim()

        assert np.allclose(xlim_el, xlim_dmap)
        assert np.allclose(ylim_el, ylim_dmap)

    def test_hspans_nondefault_kdim(self):
        hspans = HSpans(
            {"other0": [0, 3, 5.5], "other1": [1, 4, 6.5]}, kdims=["other0", "other1"]
        )
        plot = mpl_renderer.get_plot(hspans)
        assert plot.handles["fig"].axes[0].get_xlabel() == "x"
        assert plot.handles["fig"].axes[0].get_ylabel() == "y"

        xlim = plot.handles["fig"].axes[0].get_xlim()
        ylim = plot.handles["fig"].axes[0].get_ylim()
        assert np.allclose(xlim, (-0.055, 0.055))
        assert np.allclose(ylim, (0, 6.5))

        sources = plot.handles["annotations"]
        assert len(sources) == 3
        for source, v0, v1 in zip(
            sources, hspans.data["other0"], hspans.data["other1"], strict=None
        ):
            self._hspans_check(source, v0, v1)

    def test_vspans_plot(self):
        vspans = VSpans(
            {"x0": [0, 3, 5.5], "x1": [1, 4, 6.5], "extra": [-1, -2, -3]},
            vdims=["extra"],
        )
        plot = mpl_renderer.get_plot(vspans)
        assert plot.handles["fig"].axes[0].get_xlabel() == "x"
        assert plot.handles["fig"].axes[0].get_ylabel() == "y"

        xlim = plot.handles["fig"].axes[0].get_xlim()
        ylim = plot.handles["fig"].axes[0].get_ylim()
        assert np.allclose(xlim, (0, 6.5))
        assert np.allclose(ylim, (-0.055, 0.055))

        sources = plot.handles["annotations"]
        assert len(sources) == 3
        for source, v0, v1 in zip(sources, vspans.data["x0"], vspans.data["x1"], strict=None):
            self._vspans_check(source, v0, v1)

    def test_vspans_inverse_plot(self):
        vspans = VSpans(
            {"x0": [0, 3, 5.5], "x1": [1, 4, 6.5], "extra": [-1, -2, -3]},
            vdims=["extra"],
        ).opts(invert_axes=True)
        plot = mpl_renderer.get_plot(vspans)
        assert plot.handles["fig"].axes[0].get_xlabel() == "y"
        assert plot.handles["fig"].axes[0].get_ylabel() == "x"

        xlim = plot.handles["fig"].axes[0].get_xlim()
        ylim = plot.handles["fig"].axes[0].get_ylim()
        assert np.allclose(xlim, (-0.055, 0.055))
        assert np.allclose(ylim, (0, 6.5))

        sources = plot.handles["annotations"]
        assert len(sources) == 3
        for source, v0, v1 in zip(sources, vspans.data["x0"], vspans.data["x1"], strict=None):
            self._hspans_check(source, v0, v1)

    def test_vspans_nondefault_kdims(self):
        vspans = VSpans(
            {"other0": [0, 3, 5.5], "other1": [1, 4, 6.5]}, kdims=["other0", "other1"]
        )
        plot = mpl_renderer.get_plot(vspans)
        assert plot.handles["fig"].axes[0].get_xlabel() == "x"
        assert plot.handles["fig"].axes[0].get_ylabel() == "y"

        xlim = plot.handles["fig"].axes[0].get_xlim()
        ylim = plot.handles["fig"].axes[0].get_ylim()
        assert np.allclose(xlim, (0, 6.5))
        assert np.allclose(ylim, (-0.055, 0.055))

        sources = plot.handles["annotations"]
        assert len(sources) == 3
        for source, v0, v1 in zip(
            sources, vspans.data["other0"], vspans.data["other1"], strict=None
        ):
            self._vspans_check(source, v0, v1)

    def test_vspans_hspans_overlay(self):
        hspans = HSpans(
            {"y0": [0, 3, 5.5], "y1": [1, 4, 6.5], "extra": [-1, -2, -3]},
            vdims=["extra"],
        )
        vspans = VSpans(
            {"x0": [0, 3, 5.5], "x1": [1, 4, 6.5], "extra": [-1, -2, -3]},
            vdims=["extra"],
        )
        plot = mpl_renderer.get_plot(hspans * vspans)
        assert plot.handles["fig"].axes[0].get_xlabel() == "x"
        assert plot.handles["fig"].axes[0].get_ylabel() == "y"

        xlim = plot.handles["fig"].axes[0].get_xlim()
        ylim = plot.handles["fig"].axes[0].get_ylim()
        assert np.allclose(xlim, (0, 6.5))
        assert np.allclose(ylim, (0, 6.5))

        sources = plot.handles["fig"].axes[0].get_children()
        for source, v0, v1 in zip(sources[:3], hspans.data["y0"], hspans.data["y1"], strict=None):
            self._hspans_check(source, v0, v1)

        for source, v0, v1 in zip(sources[3:6], vspans.data["x0"], vspans.data["x1"], strict=None):
            self._vspans_check(source, v0, v1)

    def test_dynamicmap_overlay_vspans(self):
        el = VSpans(data=[[1, 3], [2, 4]])
        dmap = hv.DynamicMap(lambda: hv.Overlay([el]))

        plot_el = mpl_renderer.get_plot(el)
        plot_dmap = mpl_renderer.get_plot(dmap)

        xlim_el = plot_el.handles["fig"].axes[0].get_xlim()
        ylim_el = plot_el.handles["fig"].axes[0].get_ylim()
        xlim_dmap = plot_dmap.handles["fig"].axes[0].get_xlim()
        ylim_dmap = plot_dmap.handles["fig"].axes[0].get_ylim()

        assert np.allclose(xlim_el, xlim_dmap)
        assert np.allclose(ylim_el, ylim_dmap)
