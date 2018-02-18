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
