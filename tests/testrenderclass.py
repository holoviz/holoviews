# -*- coding: utf-8 -*-
"""
Test cases for rendering exporters
"""
from hashlib import sha256
from unittest import SkipTest
import numpy as np

from holoviews import plotting  # pyflakes:ignore (Sets Store.renderer)
from holoviews import HoloMap, Store, Image, ItemTable
from holoviews.element.comparison import ComparisonTestCase

from nose.plugins.attrib import attr

try:
    # Standardize backend due to random inconsistencies
    from matplotlib import pyplot
    pyplot.switch_backend('agg')
except:
    pyplot = None

def digest_data(data):
    hashfn = sha256()
    hashfn.update(data)
    return hashfn.hexdigest()


@attr(optional=1)
class MPLPlotRendererTest(ComparisonTestCase):
    """
    Note if not possible to compare the hashes of SVG and WebM formats
    as the hashes are not stable across exports.
    """

    def setUp(self):
        if pyplot is None:
            raise SkipTest("Matplotlib required to test widgets")

        self.basename = 'no-file'
        self.image1 = Image(np.array([[0,1],[2,3]]), label='Image1')
        self.image2 = Image(np.array([[1,0],[4,-2]]), label='Image2')
        self.map1 = HoloMap({1:self.image1, 2:self.image2}, label='TestMap')

        self.unicode_table = ItemTable([('β','Δ1'), ('°C', '3×4')],
                                       label='Poincaré', group='α Festkörperphysik')

        self.renderer = Store.renderer.instance()

    def test_simple_export_gif(self):
        data = self.renderer(self.map1, fmt='gif')[0]
        self.assertEqual(digest_data(data),
                         '1b62bad83de5cd9b480beec0d6aba8a07e4e0ae1a5ede837bbdb1ddd70790884')

    def test_simple_export_gif_double_size(self):
        data = self.renderer.instance(size=200)(self.map1, fmt='gif')[0]
        self.assertEqual(digest_data(data),
                         'da5a1813d5cdb1d499bb2768b079c3b9622a65b63396265f67558e72194b4b94')

    def test_simple_export_gif_half_fps(self):
        data = self.renderer.instance(fps=10)(self.map1, fmt='gif', )[0]
        self.assertEqual(digest_data(data),
                         '1b62bad83de5cd9b480beec0d6aba8a07e4e0ae1a5ede837bbdb1ddd70790884')

    def test_simple_export_png1(self):
        data = self.renderer(self.image1, fmt='png')[0]
        self.assertEqual(digest_data(data),
                         'bcc2596acf53f9cc29b49088455a54d4d3104eda3563752f9529f8922c51536f')

    def test_simple_export_png1_double_size(self):
        data = self.renderer.instance(size=200)(self.image1, fmt='png')[0]
        self.assertEqual(digest_data(data),
                         'd7e4a4fe43163f57b134b5f29c71c1cf47b21f93db79aa1f809fd3a120747c9e')

    def test_simple_export_png2(self):
        data = self.renderer(self.image2, fmt='png')[0]
        self.assertEqual(digest_data(data),
                         'e83defd1f41d2edbab17faadb50d9ca5a8dbfbd9501a012c254a7b26549b54ef')

    def test_simple_export_png2_double_size(self):
        data = self.renderer.instance(size=200)(self.image2, fmt='png')[0]
        self.assertEqual(digest_data(data),
                         '972281f53afc9083787fa2b9d3039b37e96f3bd6111cc2372bacb94cd3cf61f7')

    def test_simple_export_unicode_table_png(self):
        "Test that unicode support and rendering is working"
        data = self.renderer.instance(size=200)(self.unicode_table, fmt='png')[0]
        self.assertEqual(digest_data(data),
                         '75ae13a85383f50e9e4b5f5ae7b4ac63d90edd5e553166730e024817a647965c')
