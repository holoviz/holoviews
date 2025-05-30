import numpy as np
import pandas as pd

from holoviews.element import Area, Overlay

from ...utils import LoggingComparisonTestCase
from .test_plot import TestMPLPlot, mpl_renderer


class TestAreaPlot(LoggingComparisonTestCase, TestMPLPlot):

    def test_area_padding_square(self):
        area = Area([(1, 1), (2, 2), (3, 3)]).opts(padding=0.1)
        plot = mpl_renderer.get_plot(area)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], 0.8)
        self.assertEqual(x_range[1], 3.2)
        self.assertEqual(y_range[0], 0)
        self.assertEqual(y_range[1], 3.2)

    def test_area_padding_square_per_axis(self):
        area = Area([(1, 1), (2, 2), (3, 3)]).opts(padding=((0, 0.1), (0.1, 0.2)))
        plot = mpl_renderer.get_plot(area)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], 1)
        self.assertEqual(x_range[1], 3.2)
        self.assertEqual(y_range[0], 0)
        self.assertEqual(y_range[1], 3.4)

    def test_area_with_lower_vdim(self):
        area = Area([(1, 0.5, 1), (2, 1.5, 2), (3, 2.5, 3)], vdims=['y', 'y2']).opts(padding=0.1)
        plot = mpl_renderer.get_plot(area)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], 0.8)
        self.assertEqual(x_range[1], 3.2)
        self.assertEqual(y_range[0], 0.25)
        self.assertEqual(y_range[1], 3.25)

    def test_area_padding_negative(self):
        area = Area([(1, -1), (2, -2), (3, -3)]).opts(padding=0.1)
        plot = mpl_renderer.get_plot(area)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], 0.8)
        self.assertEqual(x_range[1], 3.2)
        self.assertEqual(y_range[0], -3.2)
        self.assertEqual(y_range[1], 0)

    def test_area_padding_mixed(self):
        area = Area([(1, 1), (2, -2), (3, 3)]).opts(padding=0.1)
        plot = mpl_renderer.get_plot(area)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], 0.8)
        self.assertEqual(x_range[1], 3.2)
        self.assertEqual(y_range[0], -2.5)
        self.assertEqual(y_range[1], 3.5)

    def test_area_padding_hard_range(self):
        area = Area([(1, 1), (2, 2), (3, 3)]).redim.range(y=(0, 4)).opts(padding=0.1)
        plot = mpl_renderer.get_plot(area)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], 0.8)
        self.assertEqual(x_range[1], 3.2)
        self.assertEqual(y_range[0], 0)
        self.assertEqual(y_range[1], 4)

    def test_area_padding_soft_range(self):
        area = Area([(1, 1), (2, 2), (3, 3)]).redim.soft_range(y=(0, 3.5)).opts(padding=0.1)
        plot = mpl_renderer.get_plot(area)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], 0.8)
        self.assertEqual(x_range[1], 3.2)
        self.assertEqual(y_range[0], 0)
        self.assertEqual(y_range[1], 3.5)

    def test_area_padding_nonsquare(self):
        area = Area([(1, 1), (2, 2), (3, 3)]).opts(padding=0.1, aspect=2)
        plot = mpl_renderer.get_plot(area)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], 0.9)
        self.assertEqual(x_range[1], 3.1)
        self.assertEqual(y_range[0], 0)
        self.assertEqual(y_range[1], 3.2)

    def test_area_padding_logx(self):
        area = Area([(1, 1), (2, 2), (3,3)]).opts(padding=0.1, logx=True)
        plot = mpl_renderer.get_plot(area)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], 0.89595845984076228)
        self.assertEqual(x_range[1], 3.3483695221017129)
        self.assertEqual(y_range[0], 0)
        self.assertEqual(y_range[1], 3.2)

    def test_area_padding_logy(self):
        area = Area([(1, 1), (2, 2), (3, 3)]).opts(padding=0.1, logy=True)
        plot = mpl_renderer.get_plot(area)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], 0.8)
        self.assertEqual(x_range[1], 3.2)
        self.assertEqual(y_range[0], 0.01)
        self.assertEqual(y_range[1], 3.3483695221017129)
        self.log_handler.assertContains('WARNING', 'Logarithmic axis range encountered value less than')

    def test_area_stack_vdims(self):
        df = pd.DataFrame({'x': [1, 2, 3], 'y_1': [1, 2, 3], 'y_2': [6, 4, 2], 'y_3': [8, 1, 2]})
        overlay = Overlay([Area(df, kdims='x', vdims=col, label=col) for col in ['y_1', 'y_2', 'y_3']])
        plot = Area.stack(overlay)
        baselines = [np.array([0, 0, 0]), np.array([1., 2., 3.]), np.array([7., 6., 5.])]
        for n, baseline in zip(plot.data, baselines, strict=None):
            self.assertEqual(plot.data[n].data.Baseline.to_numpy(), baseline)
