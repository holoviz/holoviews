import numpy as np

from holoviews.core import NdOverlay
from holoviews.element import Spikes

from .testplot import TestBokehPlot, bokeh_renderer


class TestSpikesPlot(TestBokehPlot):

    def test_spikes_colormapping(self):
        spikes = Spikes(np.random.rand(20, 2), vdims=['Intensity'])
        color_spikes = spikes.opts(plot=dict(color_index=1))
        self._test_colormapping(color_spikes, 1)

    def test_empty_spikes_plot(self):
        spikes = Spikes([], vdims=['Intensity'])
        plot = bokeh_renderer.get_plot(spikes)
        source = plot.handles['source']
        self.assertEqual(len(source.data['x']), 0)
        self.assertEqual(len(source.data['y0']), 0)
        self.assertEqual(len(source.data['y1']), 0)

    def test_batched_spike_plot(self):
        overlay = NdOverlay({i: Spikes([i], kdims=['Time']).opts(
            plot=dict(position=0.1*i, spike_length=0.1, show_legend=False))
                             for i in range(10)})
        plot = bokeh_renderer.get_plot(overlay)
        extents = plot.get_extents(overlay, {})
        self.assertEqual(extents, (0, 0, 9, 1))

    def test_spikes_padding_square(self):
        spikes = Spikes([1, 2, 3]).options(padding=0.2)
        plot = bokeh_renderer.get_plot(spikes)
        x_range = plot.handles['x_range']
        self.assertEqual(x_range.start, 0.8)
        self.assertEqual(x_range.end, 3.2)

    def test_spikes_padding_square_heights(self):
        spikes = Spikes([(1, 1), (2, 2), (3, 3)], vdims=['Height']).options(padding=0.2)
        plot = bokeh_renderer.get_plot(spikes)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 0.8)
        self.assertEqual(x_range.end, 3.2)
        self.assertEqual(y_range.start, 0)
        self.assertEqual(y_range.end, 3.2)

    def test_spikes_padding_hard_xrange(self):
        spikes = Spikes([1, 2, 3]).redim.range(x=(0, 3)).options(padding=0.2)
        plot = bokeh_renderer.get_plot(spikes)
        x_range = plot.handles['x_range']
        self.assertEqual(x_range.start, 0)
        self.assertEqual(x_range.end, 3)

    def test_spikes_padding_soft_xrange(self):
        spikes = Spikes([1, 2, 3]).redim.soft_range(x=(0, 3)).options(padding=0.2)
        plot = bokeh_renderer.get_plot(spikes)
        x_range = plot.handles['x_range']
        self.assertEqual(x_range.start, 0)
        self.assertEqual(x_range.end, 3.2)

    def test_spikes_padding_unequal(self):
        spikes = Spikes([1, 2, 3]).options(padding=(0.1, 0.2))
        plot = bokeh_renderer.get_plot(spikes)
        x_range = plot.handles['x_range']
        self.assertEqual(x_range.start, 0.9)
        self.assertEqual(x_range.end, 3.1)

    def test_spikes_padding_nonsquare(self):
        spikes = Spikes([1, 2, 3]).options(padding=0.2, width=600)
        plot = bokeh_renderer.get_plot(spikes)
        x_range = plot.handles['x_range']
        self.assertEqual(x_range.start, 0.9)
        self.assertEqual(x_range.end, 3.1)

    def test_spikes_padding_logx(self):
        spikes = Spikes([(1, 1), (2, 2), (3,3)]).options(padding=0.2, logx=True)
        plot = bokeh_renderer.get_plot(spikes)
        x_range = plot.handles['x_range']
        self.assertEqual(x_range.start, 0.89595845984076228)
        self.assertEqual(x_range.end, 3.3483695221017129)

    def test_spikes_padding_datetime_square(self):
        spikes = Spikes([np.datetime64('2016-04-0%d' % i) for i in range(1, 4)]).options(
            padding=0.2
        )
        plot = bokeh_renderer.get_plot(spikes)
        x_range = plot.handles['x_range']
        self.assertEqual(x_range.start, np.datetime64('2016-03-31T19:12:00.000000000'))
        self.assertEqual(x_range.end, np.datetime64('2016-04-03T04:48:00.000000000'))

    def test_spikes_padding_datetime_square_heights(self):
        spikes = Spikes([(np.datetime64('2016-04-0%d' % i), i) for i in range(1, 4)], vdims=['Height']).options(
            padding=0.2
        )
        plot = bokeh_renderer.get_plot(spikes)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, np.datetime64('2016-03-31T19:12:00.000000000'))
        self.assertEqual(x_range.end, np.datetime64('2016-04-03T04:48:00.000000000'))
        self.assertEqual(y_range.start, 0)
        self.assertEqual(y_range.end, 3.2)

    def test_spikes_padding_datetime_nonsquare(self):
        spikes = Spikes([np.datetime64('2016-04-0%d' % i) for i in range(1, 4)]).options(
            padding=0.2, width=600
        )
        plot = bokeh_renderer.get_plot(spikes)
        x_range = plot.handles['x_range']
        self.assertEqual(x_range.start, np.datetime64('2016-03-31T21:36:00.000000000'))
        self.assertEqual(x_range.end, np.datetime64('2016-04-03T02:24:00.000000000'))
