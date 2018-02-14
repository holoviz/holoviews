from __future__ import absolute_import

import numpy as np

from holoviews.element import Violin

from .testplot import TestMPLPlot, mpl_renderer


class TestMPLViolinPlot(TestMPLPlot):

    def test_violin_simple(self):
        values = np.random.rand(100)
        violin = Violin(values)
        plot = mpl_renderer.get_plot(violin)
        data, style, axis_opts = plot.get_data(violin, {}, {})
        self.assertEqual(data[0][0], values)
        self.assertEqual(style['positions'], [0])
        self.assertEqual(style['labels'], [''])

    def test_violin_multi(self):
        violin = Violin((np.random.randint(0, 2, 100), np.random.rand(100)), kdims=['A']).sort()
        r1, r2 = violin.range(1)
        plot = mpl_renderer.get_plot(violin)
        data, style, axis_opts = plot.get_data(violin, {}, {})
        self.assertEqual(data[0][0], violin.select(A=0).dimension_values(1))
        self.assertEqual(data[0][1], violin.select(A=1).dimension_values(1))
        self.assertEqual(style['positions'], [0, 1])
        self.assertEqual(style['labels'], ['0', '1'])
