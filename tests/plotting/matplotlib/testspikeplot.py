import numpy as np

from holoviews.element import Spikes

from .testplot import TestMPLPlot, mpl_renderer


class TestSpikesPlot(TestMPLPlot):

    def test_spikes_padding_square(self):
        spikes = Spikes([1, 2, 3]).options(padding=0.2)
        plot = mpl_renderer.get_plot(spikes)
        x_range = plot.handles['axis'].get_xlim()
        self.assertEqual(x_range[0], 0.8)
        self.assertEqual(x_range[1], 3.2)

    def test_spikes_padding_square_heights(self):
        spikes = Spikes([(1, 1), (2, 2), (3, 3)], vdims=['Height']).options(padding=0.2)
        plot = mpl_renderer.get_plot(spikes)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], 0.8)
        self.assertEqual(x_range[1], 3.2)
        self.assertEqual(y_range[0], 0)
        self.assertEqual(y_range[1], 3.2)

    def test_spikes_padding_hard_xrange(self):
        spikes = Spikes([1, 2, 3]).redim.range(x=(0, 3)).options(padding=0.2)
        plot = mpl_renderer.get_plot(spikes)
        x_range = plot.handles['axis'].get_xlim()
        self.assertEqual(x_range[0], 0)
        self.assertEqual(x_range[1], 3)

    def test_spikes_padding_soft_xrange(self):
        spikes = Spikes([1, 2, 3]).redim.soft_range(x=(0, 3)).options(padding=0.2)
        plot = mpl_renderer.get_plot(spikes)
        x_range = plot.handles['axis'].get_xlim()
        self.assertEqual(x_range[0], 0)
        self.assertEqual(x_range[1], 3.2)

    def test_spikes_padding_unequal(self):
        spikes = Spikes([1, 2, 3]).options(padding=(0.1, 0.2))
        plot = mpl_renderer.get_plot(spikes)
        x_range = plot.handles['axis'].get_xlim()
        self.assertEqual(x_range[0], 0.9)
        self.assertEqual(x_range[1], 3.1)

    def test_spikes_padding_nonsquare(self):
        spikes = Spikes([1, 2, 3]).options(padding=0.2, aspect=2)
        plot = mpl_renderer.get_plot(spikes)
        x_range = plot.handles['axis'].get_xlim()
        self.assertEqual(x_range[0], 0.9)
        self.assertEqual(x_range[1], 3.1)

    def test_spikes_padding_logx(self):
        spikes = Spikes([(1, 1), (2, 2), (3,3)]).options(padding=0.2, logx=True)
        plot = mpl_renderer.get_plot(spikes)
        x_range = plot.handles['axis'].get_xlim()
        self.assertEqual(x_range[0], 0.89595845984076228)
        self.assertEqual(x_range[1], 3.3483695221017129)

    def test_spikes_padding_datetime_square(self):
        spikes = Spikes([np.datetime64('2016-04-0%d' % i) for i in range(1, 4)]).options(
            padding=0.2
        )
        plot = mpl_renderer.get_plot(spikes)
        x_range = plot.handles['axis'].get_xlim()
        self.assertEqual(x_range[0], 736054.80000000005)
        self.assertEqual(x_range[1], 736057.19999999995)

    def test_spikes_padding_datetime_square_heights(self):
        spikes = Spikes([(np.datetime64('2016-04-0%d' % i), i) for i in range(1, 4)], vdims=['Height']).options(
            padding=0.2
        )
        plot = mpl_renderer.get_plot(spikes)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], 736054.80000000005)
        self.assertEqual(x_range[1], 736057.19999999995)
        self.assertEqual(y_range[0], 0)
        self.assertEqual(y_range[1], 3.2)

    def test_spikes_padding_datetime_nonsquare(self):
        spikes = Spikes([np.datetime64('2016-04-0%d' % i) for i in range(1, 4)]).options(
            padding=0.2, aspect=2
        )
        plot = mpl_renderer.get_plot(spikes)
        x_range = plot.handles['axis'].get_xlim()
        self.assertEqual(x_range[0], 736054.90000000002)
        self.assertEqual(x_range[1], 736057.09999999998)
