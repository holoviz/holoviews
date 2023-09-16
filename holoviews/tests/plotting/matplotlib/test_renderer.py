"""
Test cases for rendering exporters
"""
import subprocess

from unittest import SkipTest

import numpy as np
import param
import panel as pn
from matplotlib import style

from holoviews import (DynamicMap, HoloMap, Image, ItemTable,
                       GridSpace, Table, Curve)
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import Stream
from holoviews.plotting.mpl import MPLRenderer, CurvePlot
from holoviews.plotting.renderer import Renderer
from panel.widgets import DiscreteSlider, Player, FloatSlider
from pyviz_comms import CommManager


class MPLRendererTest(ComparisonTestCase):
    """
    Note if not possible to compare the hashes of SVG and WebM formats
    as the hashes are not stable across exports.
    """

    def setUp(self):
        self.basename = 'no-file'
        self.image1 = Image(np.array([[0,1],[2,3]]), label='Image1')
        self.image2 = Image(np.array([[1,0],[4,-2]]), label='Image2')
        self.map1 = HoloMap({1:self.image1, 2:self.image2}, label='TestMap')

        self.unicode_table = ItemTable([('β','Δ1'), ('°C', '3×4')],
                                       label='Poincaré', group='α Festkörperphysik')

        self.renderer = MPLRenderer.instance()
        self.nbcontext = Renderer.notebook_context
        self.comm_manager = Renderer.comm_manager
        with param.logging_level('ERROR'):
            Renderer.notebook_context = False
            Renderer.comm_manager = CommManager

    def tearDown(self):
        with param.logging_level('ERROR'):
            Renderer.notebook_context = self.nbcontext
            Renderer.comm_manager = self.comm_manager

    def test_get_size_single_plot(self):
        plot = self.renderer.get_plot(self.image1)
        w, h = self.renderer.get_size(plot)
        self.assertEqual((w, h), (288, 288))

    def test_get_size_row_plot(self):
        with style.context("default"):
            plot = self.renderer.get_plot(self.image1 + self.image2)
        w, h = self.renderer.get_size(plot)
        self.assertEqual((w, h), (576, 257))

    def test_get_size_column_plot(self):
        with style.context("default"):
            plot = self.renderer.get_plot((self.image1 + self.image2).cols(1))
        w, h = self.renderer.get_size(plot)
        self.assertEqual((w, h), (288, 509))

    def test_get_size_grid_plot(self):
        grid = GridSpace({(i, j): self.image1 for i in range(3) for j in range(3)})
        plot = self.renderer.get_plot(grid)
        w, h = self.renderer.get_size(plot)
        self.assertEqual((w, h), (345, 345))

    def test_get_size_table(self):
        table = Table(range(10), kdims=['x'])
        plot = self.renderer.get_plot(table)
        w, h = self.renderer.get_size(plot)
        self.assertEqual((w, h), (288, 288))

    def test_render_gif(self):
        data, metadata = self.renderer.components(self.map1, 'gif')
        self.assertIn("<img src='data:image/gif", data['text/html'])

    def test_render_mp4(self):
        devnull = subprocess.DEVNULL
        try:
            subprocess.call(['ffmpeg', '-h'], stdout=devnull, stderr=devnull)
        except Exception:
            raise SkipTest('ffmpeg not available, skipping mp4 export test')
        data, metadata = self.renderer.components(self.map1, 'mp4')
        self.assertIn("<source src='data:video/mp4", data['text/html'])

    def test_render_static(self):
        curve = Curve([])
        obj, _ = self.renderer._validate(curve, None)
        self.assertIsInstance(obj, CurvePlot)

    def test_render_holomap_individual(self):
        hmap = HoloMap({i: Curve([1, 2, i]) for i in range(5)})
        obj, _ = self.renderer._validate(hmap, None)
        self.assertIsInstance(obj, pn.pane.HoloViews)
        self.assertEqual(obj.center, True)
        self.assertEqual(obj.widget_location, 'right')
        self.assertEqual(obj.widget_type, 'individual')
        widgets = obj.layout.select(DiscreteSlider)
        self.assertEqual(len(widgets), 1)
        slider = widgets[0]
        self.assertEqual(slider.options, dict([(str(i), i) for i in range(5)]))

    def test_render_holomap_embedded(self):
        hmap = HoloMap({i: Curve([1, 2, i]) for i in range(5)})
        data, _ = self.renderer.components(hmap)
        self.assertIn('State"', data['text/html'])

    # def test_render_holomap_not_embedded(self):
    #     hmap = HoloMap({i: Curve([1, 2, i]) for i in range(5)})
    #     data, _ = self.renderer.instance(widget_mode='live').components(hmap)
    #     self.assertNotIn('State"', data['text/html'])

    def test_render_holomap_scrubber(self):
        hmap = HoloMap({i: Curve([1, 2, i]) for i in range(5)})
        obj, _ = self.renderer._validate(hmap, 'scrubber')
        self.assertIsInstance(obj, pn.pane.HoloViews)
        self.assertEqual(obj.center, True)
        self.assertEqual(obj.widget_location, 'bottom')
        self.assertEqual(obj.widget_type, 'scrubber')
        widgets = obj.layout.select(Player)
        self.assertEqual(len(widgets), 1)
        player = widgets[0]
        self.assertEqual(player.start, 0)
        self.assertEqual(player.end, 4)

    def test_render_holomap_scrubber_fps(self):
        hmap = HoloMap({i: Curve([1, 2, i]) for i in range(5)})
        obj, _ = self.renderer.instance(fps=2)._validate(hmap, 'scrubber')
        self.assertIsInstance(obj, pn.pane.HoloViews)
        widgets = obj.layout.select(Player)
        self.assertEqual(len(widgets), 1)
        player = widgets[0]
        self.assertEqual(player.interval, 500)

    def test_render_holomap_individual_widget_position(self):
        hmap = HoloMap({i: Curve([1, 2, i]) for i in range(5)})
        obj, _ = self.renderer.instance(widget_location='top')._validate(hmap, None)
        self.assertIsInstance(obj, pn.pane.HoloViews)
        self.assertEqual(obj.center, True)
        self.assertEqual(obj.widget_location, 'top')
        self.assertEqual(obj.widget_type, 'individual')

    def test_render_dynamicmap_with_dims(self):
        dmap = DynamicMap(lambda y: Curve([1, 2, y]), kdims=['y']).redim.range(y=(0.1, 5))
        obj, _ = self.renderer._validate(dmap, None)
        self.renderer.components(obj)
        [(plot, pane)] = obj._plots.values()
        artist = plot.handles['artist']

        (_, y) = artist.get_data()
        self.assertEqual(y[2], 0.1)
        slider = obj.layout.select(FloatSlider)[0]
        slider.value = 3.1
        (_, y) = artist.get_data()
        self.assertEqual(y[2], 3.1)

    def test_render_dynamicmap_with_stream(self):
        stream = Stream.define('Custom', y=2)()
        dmap = DynamicMap(lambda y: Curve([1, 2, y]), kdims=['y'], streams=[stream])
        obj, _ = self.renderer._validate(dmap, None)
        self.renderer.components(obj)
        [(plot, pane)] = obj._plots.values()
        artist = plot.handles['artist']

        (_, y) = artist.get_data()
        self.assertEqual(y[2], 2)
        stream.event(y=3)
        (_, y) = artist.get_data()
        self.assertEqual(y[2], 3)

    def test_render_dynamicmap_with_stream_dims(self):
        stream = Stream.define('Custom', y=2)()
        dmap = DynamicMap(lambda x, y: Curve([x, 1, y]), kdims=['x', 'y'],
                          streams=[stream]).redim.values(x=[1, 2, 3])
        obj, _ = self.renderer._validate(dmap, None)
        self.renderer.components(obj)
        [(plot, pane)] = obj._plots.values()
        artist = plot.handles['artist']

        (_, y) = artist.get_data()
        self.assertEqual(y[2], 2)
        stream.event(y=3)
        (_, y) = artist.get_data()
        self.assertEqual(y[2], 3)

        (_, y) = artist.get_data()
        self.assertEqual(y[0], 1)
        slider = obj.layout.select(DiscreteSlider)[0]
        slider.value = 3
        (_, y) = artist.get_data()
        self.assertEqual(y[0], 3)
