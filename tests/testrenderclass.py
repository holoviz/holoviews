# -*- coding: utf-8 -*-
"""
Test cases for rendering exporters
"""
from hashlib import sha256
from unittest import SkipTest
import numpy as np

from holoviews.plotting.mpl.renderer import MPLRenderer
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
class MPLRendererTest(ComparisonTestCase):
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

        self.renderer = MPLRenderer.instance()

    def test_simple_export_gif(self):
        data = self.renderer(self.map1, fmt='gif')[0]
        self.assertEqual(digest_data(data),
                         '3ae99baf8765381db448421a5343aea318f04b9a6a3fadc3c97d3093339e6789')

    def test_simple_export_gif_double_size(self):
        data = self.renderer.instance(size=200)(self.map1, fmt='gif')[0]
        self.assertEqual(digest_data(data),
                         '99134eda29e76ceba217c00878116af9cdb5cf733430235137f2c53486607837')

    def test_simple_export_gif_half_fps(self):
        data = self.renderer.instance(fps=10)(self.map1, fmt='gif', )[0]
        self.assertEqual(digest_data(data),
                         '3ae99baf8765381db448421a5343aea318f04b9a6a3fadc3c97d3093339e6789')

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
