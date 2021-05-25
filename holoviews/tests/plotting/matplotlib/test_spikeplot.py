import numpy as np

from holoviews.core.overlay import NdOverlay
from holoviews.core.spaces import HoloMap
from holoviews.element import Spikes

from ..utils import ParamLogStream
from .testplot import TestMPLPlot, mpl_renderer


class TestSpikesPlot(TestMPLPlot):

    def test_spikes_padding_square(self):
        spikes = Spikes([1, 2, 3]).options(padding=0.1)
        plot = mpl_renderer.get_plot(spikes)
        x_range = plot.handles['axis'].get_xlim()
        self.assertEqual(x_range[0], 0.8)
        self.assertEqual(x_range[1], 3.2)

    def test_spikes_padding_square_heights(self):
        spikes = Spikes([(1, 1), (2, 2), (3, 3)], vdims=['Height']).options(padding=0.1)
        plot = mpl_renderer.get_plot(spikes)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], 0.8)
        self.assertEqual(x_range[1], 3.2)
        self.assertEqual(y_range[0], 0)
        self.assertEqual(y_range[1], 3.2)

    def test_spikes_padding_hard_xrange(self):
        spikes = Spikes([1, 2, 3]).redim.range(x=(0, 3)).options(padding=0.1)
        plot = mpl_renderer.get_plot(spikes)
        x_range = plot.handles['axis'].get_xlim()
        self.assertEqual(x_range[0], 0)
        self.assertEqual(x_range[1], 3)

    def test_spikes_padding_soft_xrange(self):
        spikes = Spikes([1, 2, 3]).redim.soft_range(x=(0, 3)).options(padding=0.1)
        plot = mpl_renderer.get_plot(spikes)
        x_range = plot.handles['axis'].get_xlim()
        self.assertEqual(x_range[0], 0)
        self.assertEqual(x_range[1], 3)

    def test_spikes_padding_unequal(self):
        spikes = Spikes([1, 2, 3]).options(padding=(0.05, 0.1))
        plot = mpl_renderer.get_plot(spikes)
        x_range = plot.handles['axis'].get_xlim()
        self.assertEqual(x_range[0], 0.9)
        self.assertEqual(x_range[1], 3.1)

    def test_spikes_padding_nonsquare(self):
        spikes = Spikes([1, 2, 3]).options(padding=0.1, aspect=2)
        plot = mpl_renderer.get_plot(spikes)
        x_range = plot.handles['axis'].get_xlim()
        self.assertEqual(x_range[0], 0.9)
        self.assertEqual(x_range[1], 3.1)

    def test_spikes_padding_logx(self):
        spikes = Spikes([(1, 1), (2, 2), (3,3)]).options(padding=0.1, logx=True)
        plot = mpl_renderer.get_plot(spikes)
        x_range = plot.handles['axis'].get_xlim()
        self.assertEqual(x_range[0], 0.89595845984076228)
        self.assertEqual(x_range[1], 3.3483695221017129)

    def test_spikes_padding_datetime_square(self):
        spikes = Spikes([np.datetime64('2016-04-0%d' % i) for i in range(1, 4)]).options(
            padding=0.1
        )
        plot = mpl_renderer.get_plot(spikes)
        x_range = plot.handles['axis'].get_xlim()
        self.assertEqual(x_range[0], 16891.8)
        self.assertEqual(x_range[1], 16894.2)

    def test_spikes_padding_datetime_square_heights(self):
        spikes = Spikes([(np.datetime64('2016-04-0%d' % i), i) for i in range(1, 4)], vdims=['Height']).options(
            padding=0.1
        )
        plot = mpl_renderer.get_plot(spikes)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], 16891.8)
        self.assertEqual(x_range[1], 16894.2)
        self.assertEqual(y_range[0], 0)
        self.assertEqual(y_range[1], 3.2)

    def test_spikes_padding_datetime_nonsquare(self):
        spikes = Spikes([np.datetime64('2016-04-0%d' % i) for i in range(1, 4)]).options(
            padding=0.1, aspect=2
        )
        plot = mpl_renderer.get_plot(spikes)
        x_range = plot.handles['axis'].get_xlim()
        self.assertEqual(x_range[0], 16891.9)
        self.assertEqual(x_range[1], 16894.1)

    ###########################
    #    Styling mapping      #
    ###########################

    def test_spikes_color_op(self):
        spikes = Spikes([(0, 0, '#000000'), (0, 1, '#FF0000'), (0, 2, '#00FF00')],
                              vdims=['y', 'color']).options(color='color')
        plot = mpl_renderer.get_plot(spikes)
        artist = plot.handles['artist']
        self.assertEqual(artist.get_edgecolors(), np.array([
            [0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1]]
        ))

    def test_spikes_color_op_update(self):
        spikes = HoloMap({
            0: Spikes([(0, 0, '#000000'), (0, 1, '#FF0000'), (0, 2, '#00FF00')],
                      vdims=['y', 'color']),
            1: Spikes([(0, 0, '#FF0000'), (0, 1, '#00FF00'), (0, 2, '#0000FF')],
                      vdims=['y', 'color'])}).options(color='color')
        plot = mpl_renderer.get_plot(spikes)
        artist = plot.handles['artist']
        self.assertEqual(artist.get_edgecolors(), np.array([
            [0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1]]
        ))
        plot.update((1,))
        self.assertEqual(artist.get_edgecolors(), np.array([
            [1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
        ))
        
    def test_spikes_linear_color_op(self):
        spikes = Spikes([(0, 0, 0), (0, 1, 1), (0, 2, 2)],
                        vdims=['y', 'color']).options(color='color')
        plot = mpl_renderer.get_plot(spikes)
        artist = plot.handles['artist']
        self.assertEqual(artist.get_array(), np.array([0, 1, 2]))
        self.assertEqual(artist.get_clim(), (0, 2))

    def test_spikes_linear_color_op_update(self):
        spikes = HoloMap({
            0: Spikes([(0, 0, 0.5), (0, 1, 3.2), (0, 2, 1.8)],
                      vdims=['y', 'color']),
            1: Spikes([(0, 0, 0.1), (0, 1, 0.8), (0, 2, 0.3)],
                      vdims=['y', 'color'])}).options(color='color', framewise=True)
        plot = mpl_renderer.get_plot(spikes)
        artist = plot.handles['artist']
        self.assertEqual(artist.get_array(), np.array([0.5, 3.2, 1.8]))
        self.assertEqual(artist.get_clim(), (0.5, 3.2))
        plot.update((1,))
        self.assertEqual(artist.get_array(), np.array([0.1, 0.8, 0.3]))
        self.assertEqual(artist.get_clim(), (0.1, 0.8))

    def test_spikes_categorical_color_op(self):
        spikes = Spikes([(0, 0, 'A'), (0, 1, 'B'), (0, 2, 'A')],
                        vdims=['y', 'color']).options(color='color')
        plot = mpl_renderer.get_plot(spikes)
        artist = plot.handles['artist']
        self.assertEqual(artist.get_array(), np.array([0, 1, 0]))
        self.assertEqual(artist.get_clim(), (0, 1))

    def test_spikes_alpha_op(self):
        spikes = Spikes([(0, 0, 0), (0, 1, 0.2), (0, 2, 0.7)],
                              vdims=['y', 'alpha']).options(alpha='alpha')
        with self.assertRaises(Exception):
            mpl_renderer.get_plot(spikes)

    def test_spikes_line_width_op(self):
        spikes = Spikes([(0, 0, 1), (0, 1, 4), (0, 2, 8)],
                              vdims=['y', 'line_width']).options(linewidth='line_width')
        plot = mpl_renderer.get_plot(spikes)
        artist = plot.handles['artist']
        self.assertEqual(artist.get_linewidths(), [1, 4, 8])


    def test_spikes_line_width_op_update(self):
        spikes = HoloMap({
            0: Spikes([(0, 0, 0.5), (0, 1, 3.2), (0, 2, 1.8)],
                      vdims=['y', 'line_width']),
            1: Spikes([(0, 0, 0.1), (0, 1, 0.8), (0, 2, 0.3)],
                      vdims=['y', 'line_width'])}).options(linewidth='line_width')
        plot = mpl_renderer.get_plot(spikes)
        artist = plot.handles['artist']
        self.assertEqual(artist.get_linewidths(), [0.5, 3.2, 1.8])
        plot.update((1,))
        self.assertEqual(artist.get_linewidths(), [0.1, 0.8, 0.3])

    def test_op_ndoverlay_value(self):
        colors = ['blue', 'red']
        overlay = NdOverlay({color: Spikes(np.arange(i+2))
                             for i, color in enumerate(colors)}, 'Color').options(
                                     'Spikes', color='Color'
                             )
        plot = mpl_renderer.get_plot(overlay)
        colors = [(0, 0, 1, 1), (1, 0, 0, 1)]
        for subplot, color in zip(plot.subplots.values(),  colors):
            children = subplot.handles['artist'].get_children()
            for c in children:
                self.assertEqual(c.get_facecolor(), color)

    def test_spikes_color_index_color_clash(self):
        spikes = Spikes([(0, 0, 0), (0, 1, 1), (0, 2, 2)],
                        vdims=['y', 'color']).options(color='color', color_index='color')
        with ParamLogStream() as log:
            mpl_renderer.get_plot(spikes)
        log_msg = log.stream.read()
        warning = ("Cannot declare style mapping for 'color' option "
                   "and declare a color_index; ignoring the color_index.\n")
        self.assertEqual(log_msg, warning)

