import numpy as np

from holoviews.element import BoxWhisker
from holoviews.plotting.mpl.util import MPL_GE_3_9_0
from holoviews.testing import assert_data_equal

from .test_plot import TestMPLPlot, mpl_renderer


class TestMPLBoxWhiskerPlot(TestMPLPlot):

    def test_boxwhisker_simple(self):
        values = np.random.rand(100)
        boxwhisker = BoxWhisker(values)
        plot = mpl_renderer.get_plot(boxwhisker)
        data, style, _axis_opts = plot.get_data(boxwhisker, {}, {})
        assert_data_equal(data[0][0], values)
        if MPL_GE_3_9_0:
            assert style['tick_labels'] == ['']
        else:
            assert style['labels'] == ['']

    def test_boxwhisker_simple_overlay(self):
        values = np.random.rand(100)
        boxwhisker = BoxWhisker(values) * BoxWhisker(values)
        plot = mpl_renderer.get_plot(boxwhisker)
        p1, p2 = plot.subplots.values()
        assert_data_equal(p1.handles['boxes'][0].get_path().vertices,
                         p2.handles['boxes'][0].get_path().vertices)

    def test_box_whisker_padding_square(self):
        curve = BoxWhisker([1, 2, 3]).opts(padding=0.1)
        plot = mpl_renderer.get_plot(curve)
        y_range = plot.handles['axis'].get_ylim()
        assert y_range[0] == 0.8
        assert y_range[1] == 3.2
