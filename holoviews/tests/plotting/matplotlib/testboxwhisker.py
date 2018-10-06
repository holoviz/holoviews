from __future__ import absolute_import

import numpy as np

from holoviews.element import BoxWhisker

from .testplot import TestMPLPlot, mpl_renderer


class TestMPLBoxWhiskerPlot(TestMPLPlot):

    def test_boxwhisker_simple(self):
        values = np.random.rand(100)
        boxwhisker = BoxWhisker(values)
        plot = mpl_renderer.get_plot(boxwhisker)
        data, style, axis_opts = plot.get_data(boxwhisker, {}, {})
        self.assertEqual(data[0][0], values)
        self.assertEqual(style['labels'], [''])

    def test_boxwhisker_simple_overlay(self):
        values = np.random.rand(100)
        boxwhisker = BoxWhisker(values) * BoxWhisker(values)
        plot = mpl_renderer.get_plot(boxwhisker)
        p1, p2 = plot.subplots.values()
        self.assertEqual(p1.handles['boxes'][0].get_path().vertices,
                         p2.handles['boxes'][0].get_path().vertices)
