import numpy as np

from holoviews.element import Violin
from holoviews.plotting.mpl.util import MPL_GE_3_9_0

from .test_plot import TestMPLPlot, mpl_renderer


class TestMPLViolinPlot(TestMPLPlot):

    def test_violin_simple(self):
        values = np.random.rand(100)
        violin = Violin(values)
        plot = mpl_renderer.get_plot(violin)
        data, style, axis_opts = plot.get_data(violin, {}, {})
        self.assertEqual(data[0][0], values)
        self.assertEqual(style['positions'], [0])
        if MPL_GE_3_9_0:
            self.assertEqual(style['tick_labels'], [''])
        else:
            self.assertEqual(style['labels'], [''])

    def test_violin_simple_overlay(self):
        values = np.random.rand(100)
        violin = Violin(values) * Violin(values)
        plot = mpl_renderer.get_plot(violin)
        p1, p2 = plot.subplots.values()
        self.assertEqual(p1.handles['boxes'][0].get_path().vertices,
                         p2.handles['boxes'][0].get_path().vertices)
        for b1, b2 in zip(p1.handles['bodies'][0].get_paths(), p2.handles['bodies'][0].get_paths(), strict=None):
            self.assertEqual(b1.vertices, b2.vertices)

    def test_violin_multi(self):
        violin = Violin((np.random.randint(0, 2, 100), np.random.rand(100)), kdims=['A']).sort()
        r1, r2 = violin.range(1)
        plot = mpl_renderer.get_plot(violin)
        data, style, axis_opts = plot.get_data(violin, {}, {})
        self.assertEqual(data[0][0], violin.select(A=0).dimension_values(1))
        self.assertEqual(data[0][1], violin.select(A=1).dimension_values(1))
        self.assertEqual(style['positions'], [0, 1])
        if MPL_GE_3_9_0:
            self.assertEqual(style['tick_labels'], ['0', '1'])
        else:
            self.assertEqual(style['labels'], ['0', '1'])
