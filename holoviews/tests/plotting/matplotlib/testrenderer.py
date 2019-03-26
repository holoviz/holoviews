# -*- coding: utf-8 -*-
"""
Test cases for rendering exporters
"""
from __future__ import unicode_literals

import os
import sys
import subprocess

from unittest import SkipTest

import numpy as np

from holoviews import HoloMap, Image, ItemTable, Store, GridSpace, Table
from holoviews.element.comparison import ComparisonTestCase

try:
    from holoviews.plotting.mpl import MPLRenderer
except:
    pass


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
        self.assertEqual((w, h), (576, 255))

    def test_get_size_column_plot(self):
        plot = self.renderer.get_plot((self.image1+self.image2).cols(1))
        w, h = self.renderer.get_size(plot)
        self.assertEqual((w, h), (288, 505))

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
        if sys.version_info.major > 2:
            devnull = subprocess.DEVNULL
        else:
            devnull = open(os.devnull, 'w')
        try:
            subprocess.call(['ffmpeg', '-h'], stdout=devnull, stderr=devnull)
        except:
            raise SkipTest('ffmpeg not available, skipping mp4 export test')
        data, metadata = self.renderer.components(self.map1, 'mp4')
        self.assertIn("<source src='data:video/mp4", data['text/html'])
