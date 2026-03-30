"""
Test cases for rendering exporters
"""

import base64
import re
import shutil
import sys
from io import BytesIO

import numpy as np
import panel as pn
import param
import pytest
from matplotlib import style
from panel.widgets import DiscreteSlider, FloatSlider, Player
from PIL import Image
from pyviz_comms import CommManager

import holoviews as hv
from holoviews.plotting.mpl import CurvePlot, MPLRenderer
from holoviews.plotting.renderer import Renderer
from holoviews.streams import Stream


class MPLRendererTest:
    """
    Note if not possible to compare the hashes of SVG and WebM formats
    as the hashes are not stable across exports.
    """

    def setup_method(self):
        self.basename = "no-file"
        self.image1 = hv.Image(np.array([[0, 1], [2, 3]]), label="Image1")
        self.image2 = hv.Image(np.array([[1, 0], [4, -2]]), label="Image2")
        self.map1 = hv.HoloMap({1: self.image1, 2: self.image2}, label="TestMap")

        self.unicode_table = hv.ItemTable(
            [("β", "Δ1"), ("°C", "3×4")], label="Poincaré", group="α Festkörperphysik"
        )

        self.renderer = MPLRenderer.instance()
        self.nbcontext = Renderer.notebook_context
        self.comm_manager = Renderer.comm_manager
        with param.logging_level("ERROR"):
            Renderer.notebook_context = False
            Renderer.comm_manager = CommManager

    def teardown_method(self):
        with param.logging_level("ERROR"):
            Renderer.notebook_context = self.nbcontext
            Renderer.comm_manager = self.comm_manager

    def test_get_size_single_plot(self):
        plot = self.renderer.get_plot(self.image1)
        w, h = self.renderer.get_size(plot)
        assert (w, h) == (288, 288)

    def test_get_size_row_plot(self):
        with style.context("default"):
            plot = self.renderer.get_plot(self.image1 + self.image2)
        w, h = self.renderer.get_size(plot)
        # Depending on the backend the height may be slightly different
        assert (w, h) == (576, 257) or (w, h) == (576, 259)

    def test_get_size_column_plot(self):
        with style.context("default"):
            plot = self.renderer.get_plot((self.image1 + self.image2).cols(1))
        w, h = self.renderer.get_size(plot)
        # Depending on the backend the height may be slightly different
        assert (w, h) == (288, 509) or (w, h) == (288, 511)

    def test_get_size_grid_plot(self):
        grid = hv.GridSpace({(i, j): self.image1 for i in range(3) for j in range(3)})
        plot = self.renderer.get_plot(grid)
        w, h = self.renderer.get_size(plot)
        assert (w, h) == (345, 345)

    def test_get_size_table(self):
        table = hv.Table(range(10), kdims=["x"])
        plot = self.renderer.get_plot(table)
        w, h = self.renderer.get_size(plot)
        assert (w, h) == (288, 288)

    def test_render_gif(self):
        data, _metadata = self.renderer.components(self.map1, "gif")
        assert "<img src='data:image/gif" in data["text/html"]

    @pytest.mark.skipif(sys.platform == "win32", reason="Skip on Windows")
    @pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not available")
    def test_render_mp4(self):
        data, _metadata = self.renderer.components(self.map1, "mp4")
        assert "<source src='data:video/mp4" in data["text/html"]

    def test_render_static(self):
        curve = hv.Curve([])
        obj, _ = self.renderer._validate(curve, None)
        assert isinstance(obj, CurvePlot)

    def test_render_holomap_individual(self):
        hmap = hv.HoloMap({i: hv.Curve([1, 2, i]) for i in range(5)})
        obj, _ = self.renderer._validate(hmap, None)
        assert isinstance(obj, pn.pane.HoloViews)
        assert obj.center is True
        assert obj.widget_location == "right"
        assert obj.widget_type == "individual"
        widgets = obj.layout.select(DiscreteSlider)
        assert len(widgets) == 1
        slider = widgets[0]
        assert slider.options == dict([(str(i), i) for i in range(5)])

    def test_render_holomap_embedded(self):
        hmap = hv.HoloMap({i: hv.Curve([1, 2, i]) for i in range(5)})
        data, _ = self.renderer.components(hmap)
        assert 'State"' in data["text/html"]

    def test_render_holomap_not_embedded(self):
        hmap = hv.HoloMap({i: hv.Curve([1, 2, i]) for i in range(5)})
        data, _ = self.renderer.instance(widget_mode="live").components(hmap)
        assert 'State"' not in data["text/html"]

    def test_render_holomap_scrubber(self):
        hmap = hv.HoloMap({i: hv.Curve([1, 2, i]) for i in range(5)})
        obj, _ = self.renderer._validate(hmap, "scrubber")
        assert isinstance(obj, pn.pane.HoloViews)
        assert obj.center is True
        assert obj.widget_location == "bottom"
        assert obj.widget_type == "scrubber"
        widgets = obj.layout.select(Player)
        assert len(widgets) == 1
        player = widgets[0]
        assert player.start == 0
        assert player.end == 4

    def test_render_holomap_scrubber_fps(self):
        hmap = hv.HoloMap({i: hv.Curve([1, 2, i]) for i in range(5)})
        obj, _ = self.renderer.instance(fps=2)._validate(hmap, "scrubber")
        assert isinstance(obj, pn.pane.HoloViews)
        widgets = obj.layout.select(Player)
        assert len(widgets) == 1
        player = widgets[0]
        assert player.interval == 500

    def test_render_holomap_individual_widget_position(self):
        hmap = hv.HoloMap({i: hv.Curve([1, 2, i]) for i in range(5)})
        obj, _ = self.renderer.instance(widget_location="top")._validate(hmap, None)
        assert isinstance(obj, pn.pane.HoloViews)
        assert obj.center is True
        assert obj.widget_location == "top"
        assert obj.widget_type == "individual"

    @pytest.mark.filterwarnings("ignore:Attempted to send message over Jupyter Comm:UserWarning")
    def test_render_dynamicmap_with_dims(self):
        dmap = hv.DynamicMap(lambda y: hv.Curve([1, 2, y]), kdims=["y"]).redim.range(y=(0.1, 5))
        obj, _ = self.renderer._validate(dmap, None)
        self.renderer.components(obj)
        [(plot, _pane)] = obj._plots.values()
        artist = plot.handles["artist"]

        (_, y) = artist.get_data()
        assert y[2] == 0.1
        slider = obj.layout.select(FloatSlider)[0]
        slider.value = 3.1
        (_, y) = artist.get_data()
        assert y[2] == 3.1

    @pytest.mark.filterwarnings("ignore:Attempted to send message over Jupyter Comm:UserWarning")
    def test_render_dynamicmap_with_stream(self):
        stream = Stream.define("Custom", y=2)()
        dmap = hv.DynamicMap(lambda y: hv.Curve([1, 2, y]), kdims=["y"], streams=[stream])
        obj, _ = self.renderer._validate(dmap, None)
        self.renderer.components(obj)
        [(plot, _pane)] = obj._plots.values()
        artist = plot.handles["artist"]

        (_, y) = artist.get_data()
        assert y[2] == 2
        stream.event(y=3)
        (_, y) = artist.get_data()
        assert y[2] == 3

    @pytest.mark.filterwarnings("ignore:Attempted to send message over Jupyter Comm:UserWarning")
    def test_render_dynamicmap_with_stream_dims(self):
        stream = Stream.define("Custom", y=2)()
        dmap = hv.DynamicMap(
            lambda x, y: hv.Curve([x, 1, y]), kdims=["x", "y"], streams=[stream]
        ).redim.values(x=[1, 2, 3])
        obj, _ = self.renderer._validate(dmap, None)
        self.renderer.components(obj)
        [(plot, _pane)] = obj._plots.values()
        artist = plot.handles["artist"]

        (_, y) = artist.get_data()
        assert y[2] == 2
        stream.event(y=3)
        (_, y) = artist.get_data()
        assert y[2] == 3

        (_, y) = artist.get_data()
        assert y[0] == 1
        slider = obj.layout.select(DiscreteSlider)[0]
        slider.value = 3
        (_, y) = artist.get_data()
        assert y[0] == 3


class TestAnimationBbox:
    """Test that matplotlib animations are not clipped"""

    def setup_method(self):
        self.renderer = MPLRenderer.instance()

    def _gif_frame_size(self, obj):
        """Render obj as GIF and return the (width, height) of the first frame."""
        data, _ = self.renderer.components(obj, "gif")
        # Extract the base64 data from the img tag
        match = re.search(r"base64,([^'\"]+)", data["text/html"])
        raw = base64.b64decode(match.group(1))
        img = Image.open(BytesIO(raw))
        return img.size

    def _png_size(self, obj):
        """Render obj as PNG and return the (width, height)."""
        data = self.renderer(self.renderer.get_plot(obj), "png")[0]

        img = Image.open(BytesIO(data))
        return img.size

    def _all_labels_visible(self, obj):
        """Check that the GIF frame is at least as large as the PNG.

        The PNG uses bbox_inches='tight' which tightly crops to content.
        If the GIF is smaller in either dimension, content is clipped.
        """
        gif_w, gif_h = self._gif_frame_size(obj)
        png_w, png_h = self._png_size(obj)
        return gif_w >= png_w and gif_h >= png_h

    def test_gif_single_plot_not_clipped(self):
        hmap = hv.HoloMap({i: hv.Curve(np.random.rand(10)) for i in range(3)})
        assert self._all_labels_visible(hmap)

    def test_gif_layout_not_clipped(self):
        hmap = hv.HoloMap({i: hv.Curve(np.random.rand(10)) for i in range(3)})
        layout = hmap + hmap
        assert self._all_labels_visible(layout)

    def test_gif_with_long_labels_not_clipped(self):
        hmap = hv.HoloMap(
            {
                i: hv.Curve(np.random.rand(10), "long x-axis label", "long y-axis label")
                for i in range(3)
            },
            kdims=["parameter with a very long name"],
        )
        assert self._all_labels_visible(hmap)

    @pytest.mark.skipif(sys.platform == "win32", reason="Skip on Windows")
    @pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not available")
    def test_video_has_even_dimensions(self):
        hmap = hv.HoloMap({i: hv.Curve(np.random.rand(10)) for i in range(3)})
        plot = self.renderer.get_plot(hmap)
        self.renderer._adjust_figure_for_anim(plot, "mp4")
        fig = plot.state
        dpi = self.renderer.dpi or fig.dpi
        w_px = int(fig.get_size_inches()[0] * dpi)
        h_px = int(fig.get_size_inches()[1] * dpi)
        assert w_px % 2 == 0
        assert h_px % 2 == 0
