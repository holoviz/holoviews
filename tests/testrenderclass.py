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

        self.renderer = Store.renderer.instance()

    def test_simple_export_gif(self):
        data = self.renderer(self.map1, fmt='gif')[0]
        self.assertEqual(digest_data(data),
                         'f71b5aa9e001e22d502e6c4ad0e9fc4aea5c04cb2d1a68f2f8a4296e563107e1')

    def test_simple_export_gif_double_size(self):
        data = self.renderer.instance(size=200)(self.map1, fmt='gif')[0]
        self.assertEqual(digest_data(data),
                         '11baa26abaf19572dc1a44e80e0a11c81db716124ad99ea84a272d44ca99526d')

    def test_simple_export_gif_half_fps(self):
        data = self.renderer.instance(fps=10)(self.map1, fmt='gif', )[0]
        self.assertEqual(digest_data(data),
                         'f71b5aa9e001e22d502e6c4ad0e9fc4aea5c04cb2d1a68f2f8a4296e563107e1')

    def test_simple_export_png1(self):
        data = self.renderer(self.image1, fmt='png')[0]
        self.assertEqual(digest_data(data),
                         'ed1930b4d25870b0630e030e607bb6e487952e576ebd0a1a7232ca72ef7cf50c')

    def test_simple_export_png1_double_size(self):
        data = self.renderer.instance(size=200)(self.image1, fmt='png')[0]
        self.assertEqual(digest_data(data),
                         '10ad6d8969aa12d51e17c30244044264bab566c4bb5a2e900ace937fa48c2105')

    def test_simple_export_png2(self):
        data = self.renderer(self.image2, fmt='png')[0]
        self.assertEqual(digest_data(data),
                         '16bb9117d5cc0959167264a9183179641318f5f8ae269ccdab66df9b2b5b40d7')

    def test_simple_export_png2_double_size(self):
        data = self.renderer.instance(size=200)(self.image2, fmt='png')[0]
        self.assertEqual(digest_data(data),
                         '1fa233a601bc7942031e434c20253a8551639bd8cf574440ea9b4485a185c2a1')

    def test_simple_export_unicode_table_png(self):
        "Test that unicode support and rendering is working"
        data = self.renderer.instance(size=200)(self.unicode_table, fmt='png')[0]
        self.assertEqual(digest_data(data),
                         'a3dd68a888de14064cb621c14be5b175d96781cdbc932a3f778def34beaee1ff')
