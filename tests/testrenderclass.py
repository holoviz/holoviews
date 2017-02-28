# -*- coding: utf-8 -*-
"""
Test cases for rendering exporters
"""
from __future__ import unicode_literals

from io import BytesIO
from hashlib import sha256
from unittest import SkipTest
import numpy as np

from holoviews import HoloMap, Image, ItemTable, Store, GridSpace, Table
from holoviews.core.util import unicode
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting import Renderer

from nose.plugins.attrib import attr

from .testwidgets import normalize

try:
    # Standardize backend due to random inconsistencies
    from matplotlib import pyplot
    from holoviews.plotting.mpl import MPLRenderer
    pyplot.switch_backend('agg')
except:
    pass

try:
    from holoviews.plotting.bokeh import BokehRenderer
except:
    pass

def digest_data(data):
    hashfn = sha256()
    hashfn.update(data)
    return hashfn.hexdigest()


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

    def test_simple_export_gif(self):
        data = self.renderer(self.map1, fmt='gif')[0]
        self.assertEqual(digest_data(data),
                         '95258c17d10620f20604c9cbd17e6b65e886a6163c96d6574f3eb812e0f149c2')

    def test_simple_export_gif_double_size(self):
        data = self.renderer.instance(size=200)(self.map1, fmt='gif')[0]
        self.assertEqual(digest_data(data),
                         'fbe6d753df1471315cbd83d370379591af0fdea114601c5ce1a615777749ca91')

    def test_simple_export_gif_half_fps(self):
        data = self.renderer.instance(fps=5)(self.map1, fmt='gif', )[0]
        self.assertEqual(digest_data(data),
                         'add756aa3caeb4c5f2396cdd5bd0122128c6a1275de9d3a44a0c21a734c4d5f4')

    def test_simple_export_png1(self):
        data = self.renderer(self.image1, fmt='png')[0]
        self.assertEqual(digest_data(data),
                         '08b7d97e79f715f9d8593416b1f6561ebe3e75bb038172ffd6048286ab09e671')

    def test_simple_export_png1_double_size(self):
        data = self.renderer.instance(size=200)(self.image1, fmt='png')[0]
        self.assertEqual(digest_data(data),
                         '0b48a3e366b300fcfea3cba47d6a07631a8fcc02a96860f1291233ef2c976764')

    def test_simple_export_png2(self):
        data = self.renderer(self.image2, fmt='png')[0]
        self.assertEqual(digest_data(data),
                         '2b5f1639584cd4c18c01cfb3f26d26dfa582fff39b869223269c0d941f17cc8b')

    def test_simple_export_png2_double_size(self):
        data = self.renderer.instance(size=200)(self.image2, fmt='png')[0]
        self.assertEqual(digest_data(data),
                         'ef8e2df9c3d3a27e738ae1ca9f5b0704b6467cb44265f7933a3c137ce8a8a519')

    def test_simple_export_unicode_table_png(self):
        "Test that unicode support and rendering is working"
        data = self.renderer.instance(size=200)(self.unicode_table, fmt='png')[0]
        self.assertEqual(digest_data(data),
                         'a3dd68a888de14064cb621c14be5b175d96781cdbc932a3f778def34beaee1ff')

    def test_static_html_scrubber(self):
        data = normalize(self.renderer.static_html(self.map1, fmt='scrubber'))
        self.assertEqual(digest_data(data),
                         '631f32e5be35211e49e1dcd13a7ea117331deddafb97fc4815000ca1ed80397f')

    def test_static_html_widgets(self):
        data = normalize(self.renderer.static_html(self.map1, fmt='widgets'))
        self.assertEqual(digest_data(data),
                         '9c4ac8fc5e5689c4f671b8483b06a7d6042559539b224adf82a3ed4946c8eae6')

    def test_static_html_gif(self):
        data = self.renderer.static_html(self.map1, fmt='gif')
        self.assertEqual(digest_data(normalize(data)),
                         '9d43822e0f368f3c673b19aaf66d22252849947b7dc4a157306c610c42d319b5')

    def test_export_widgets(self):
        bytesio = BytesIO()
        self.renderer.export_widgets(self.map1, bytesio, fmt='widgets')
        data = normalize(bytesio.read())
        self.assertEqual(digest_data(data),
                         '91bbc7b4efebd07b1ee595b902d9899b27f2c7e353dfc87c57c2dfd5d0404301')

    def test_get_size_single_plot(self):
        plot = self.renderer.get_plot(self.image1)
        w, h = self.renderer.get_size(plot)
        self.assertEqual((w, h), (400, 400))

    def test_get_size_row_plot(self):
        plot = self.renderer.get_plot(self.image1+self.image2)
        w, h = self.renderer.get_size(plot)
        self.assertEqual((w, h), (800, 355))

    def test_get_size_column_plot(self):
        plot = self.renderer.get_plot((self.image1+self.image2).cols(1))
        w, h = self.renderer.get_size(plot)
        self.assertEqual((w, h), (400, 702))

    def test_get_size_grid_plot(self):
        grid = GridSpace({(i, j): self.image1 for i in range(3) for j in range(3)})
        plot = self.renderer.get_plot(grid)
        w, h = self.renderer.get_size(plot)
        self.assertEqual((w, h), (480, 480))

    def test_get_size_table(self):
        table = Table(range(10), kdims=['x'])
        plot = self.renderer.get_plot(table)
        w, h = self.renderer.get_size(plot)
        self.assertEqual((w, h), (400, 400))

    def test_get_size_tables_in_layout(self):
        table = Table(range(10), kdims=['x'])
        plot = self.renderer.get_plot(table+table)
        w, h = self.renderer.get_size(plot)
        self.assertEqual((w, h), (800, 320))

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
        self.assertEqual((w, h), (422, 418))

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
