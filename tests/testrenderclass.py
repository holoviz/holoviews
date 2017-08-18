# -*- coding: utf-8 -*-
"""
Test cases for rendering exporters
"""
from __future__ import unicode_literals

from io import BytesIO
from unittest import SkipTest
from nose.plugins.attrib import attr
import numpy as np

from holoviews import HoloMap, Image, ItemTable, Store, GridSpace, Table, Curve
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting import Renderer

try:
    # Standardize backend due to random inconsistencies
    from matplotlib import pyplot
    from holoviews.plotting.mpl import MPLRenderer
    pyplot.switch_backend('agg')
except:
    pass

try:
    from bokeh.io import curdoc

    from holoviews.plotting.bokeh import BokehRenderer
    from holoviews.plotting.bokeh.util import bokeh_version
except:
    pass


class TestRenderer(ComparisonTestCase):
    """
    Test the basic serializer and deserializer (i.e. using pickle),
    including metadata access.
    """

    def test_renderer_encode_unicode_types(self):
        mime_types = ['image/svg+xml', 'text/html', 'text/json']
        for mime in mime_types:
            info = {'mime_type': mime}
            encoded = Renderer.encode(('Testing «ταБЬℓσ»: 1<2 & 4+1>3', info))
            self.assertTrue(isinstance(encoded, bytes))


@attr(optional=1)
class MPLRendererTest(ComparisonTestCase):
    """
    Note if not possible to compare the hashes of SVG and WebM formats
    as the hashes are not stable across exports.
    """

    def setUp(self):
        if 'matplotlib' not in Store.renderers:
            raise SkipTest("Matplotlib required to test widgets")

        self.basename = 'no-file'
        self.image1 = Image(np.array([[0,1],[2,3]]), label='Image1')
        self.image2 = Image(np.array([[1,0],[4,-2]]), label='Image2')
        self.map1 = HoloMap({1:self.image1, 2:self.image2}, label='TestMap')

        self.unicode_table = ItemTable([('β','Δ1'), ('°C', '3×4')],
                                       label='Poincaré', group='α Festkörperphysik')

        self.renderer = MPLRenderer.instance()

    def test_get_size_single_plot(self):
        plot = self.renderer.get_plot(self.image1)
        w, h = self.renderer.get_size(plot)
        self.assertEqual((w, h), (288, 288))

    def test_get_size_row_plot(self):
        plot = self.renderer.get_plot(self.image1+self.image2)
        w, h = self.renderer.get_size(plot)
        self.assertEqual((w, h), (576, 258))

    def test_get_size_column_plot(self):
        plot = self.renderer.get_plot((self.image1+self.image2).cols(1))
        w, h = self.renderer.get_size(plot)
        self.assertEqual((w, h), (288, 510))

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

    def test_get_size_tables_in_layout(self):
        table = Table(range(10), kdims=['x'])
        plot = self.renderer.get_plot(table+table)
        w, h = self.renderer.get_size(plot)
        self.assertEqual((w, h), (576, 231))

@attr(optional=1)
class BokehRendererTest(ComparisonTestCase):

    def setUp(self):
        if 'bokeh' not in Store.renderers:
            raise SkipTest("Bokeh required to test widgets")
        self.image1 = Image(np.array([[0,1],[2,3]]), label='Image1')
        self.image2 = Image(np.array([[1,0],[4,-2]]), label='Image2')
        self.map1 = HoloMap({1:self.image1, 2:self.image2}, label='TestMap')
        self.renderer = BokehRenderer.instance()

    def test_save_html(self):
        bytesio = BytesIO()
        self.renderer.save(self.image1, bytesio)

    def test_export_widgets(self):
        bytesio = BytesIO()
        self.renderer.export_widgets(self.map1, bytesio, fmt='widgets')

    def test_render_get_plot_server_doc(self):
        renderer = self.renderer.instance(mode='server')
        plot = renderer.get_plot(self.image1)
        self.assertIs(plot.document, curdoc())

    def test_get_size_single_plot(self):
        plot = self.renderer.get_plot(self.image1)
        w, h = self.renderer.get_size(plot)
        self.assertEqual((w, h), (300, 300))

    def test_get_size_row_plot(self):
        plot = self.renderer.get_plot(self.image1+self.image2)
        w, h = self.renderer.get_size(plot)
        self.assertEqual((w, h), (600, 300))

    def test_get_size_column_plot(self):
        plot = self.renderer.get_plot((self.image1+self.image2).cols(1))
        w, h = self.renderer.get_size(plot)
        self.assertEqual((w, h), (300, 600))

    def test_get_size_grid_plot(self):
        grid = GridSpace({(i, j): self.image1 for i in range(3) for j in range(3)})
        plot = self.renderer.get_plot(grid)
        w, h = self.renderer.get_size(plot)
        self.assertEqual((w, h), (419, 413))

    def test_get_size_table(self):
        table = Table(range(10), kdims=['x'])
        plot = self.renderer.get_plot(table)
        w, h = self.renderer.get_size(plot)
        self.assertEqual((w, h), (400, 300))

    def test_get_size_tables_in_layout(self):
        table = Table(range(10), kdims=['x'])
        plot = self.renderer.get_plot(table+table)
        w, h = self.renderer.get_size(plot)
        self.assertEqual((w, h), (680, 300))

    def test_render_to_png(self):
        if bokeh_version < str('0.12.6'):
            raise SkipTest('Bokeh static png rendering requires bokeh>=0.12.6')
        curve = Curve([])
        renderer = BokehRenderer.instance(fig='png')
        png, info = renderer(curve)
        self.assertIsInstance(png, bytes)
        self.assertEqual(info['file-ext'], 'png')
