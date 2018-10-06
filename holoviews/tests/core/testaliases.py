# -*- coding: utf-8 -*-
"""
Unit tests of the Aliases helper function and aliases in general
"""
import holoviews as hv
import numpy as np

from holoviews.element.comparison import ComparisonTestCase


class TestAliases(ComparisonTestCase):
    """
    Tests of allowable and hasprefix method.
    """

    def setUp(self):
        self.data1 = np.random.rand(10,10)
        self.data2 = np.random.rand(10,10)
        super(TestAliases, self).setUp()


    def test_aliased_layout(self):
        im1 = hv.Image(self.data1, group=('Spectrum', 'Frequency spectrum'),
                       label=('Glucose', '$C_6H_{12}O_6$'))
        self.assertEqual(im1.label, '$C_6H_{12}O_6$')
        im2 = hv.Image(self.data2,
                       group=('Spectrum', 'Frequency spectrum'),
                       label=('Water', '$H_2O$'))
        self.assertEqual(im2.label, '$H_2O$')
        layout = im1 + im2
        self.assertEqual(layout.Spectrum.Glucose, im1)
        self.assertEqual(layout.Spectrum.Water, im2)


    def test_aliased_layout_helper(self):
        al = hv.util.Aliases(Spectrum='Frequency spectrum',
                             Water='$H_2O$',
                             Glucose='$C_6H_{12}O_6$')

        im1 = hv.Image(self.data1, group=al.Spectrum, label=al.Glucose)
        self.assertEqual(im1.label, '$C_6H_{12}O_6$')
        im2 = hv.Image(self.data2, group=al.Spectrum, label=al.Water)
        self.assertEqual(im2.label, '$H_2O$')
        layout = im1 + im2
        self.assertEqual(layout.Spectrum.Glucose, im1)
        self.assertEqual(layout.Spectrum.Water, im2)


    def test_dimension_aliases(self):
        im = hv.Image(self.data1,
                     kdims=[('Lambda', '$\Lambda$'),
                            ('Joules', 'Energy ($J$)')])
        self.assertEqual(im.kdims[0].label, '$\Lambda$')
        self.assertEqual(im.kdims[1].label, 'Energy ($J$)')
        sliced = im.select(Lambda=(-0.2, 0.2), Joules=(-0.3, 0.3))
        self.assertEqual(sliced.shape, (24,3))
