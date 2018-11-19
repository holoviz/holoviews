import numpy as np

from holoviews.core.spaces import HoloMap
from holoviews.element import ErrorBars

from .testplot import TestMPLPlot, mpl_renderer


class TestErrorBarPlot(TestMPLPlot):


    ###########################
    #    Styling mapping      #
    ###########################

    def test_errorbars_color_op(self):
        errorbars = ErrorBars([(0, 0, 0.1, 0.2, '#000000'), (0, 1, 0.2, 0.4, '#FF0000'), (0, 2, 0.6, 1.2, '#00FF00')],
                              vdims=['y', 'perr', 'nerr', 'color']).options(color='color')
        plot = mpl_renderer.get_plot(errorbars)
        artist = plot.handles['artist']
        self.assertEqual(artist.get_edgecolors(),
                         np.array([[0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1]]))

    def test_errorbars_color_op_update(self):
        errorbars = HoloMap({
            0: ErrorBars([(0, 0, 0.1, 0.2, '#000000'), (0, 1, 0.2, 0.4, '#FF0000'),
                          (0, 2, 0.6, 1.2, '#00FF00')], vdims=['y', 'perr', 'nerr', 'color']),
            1: ErrorBars([(0, 0, 0.1, 0.2, '#FF0000'), (0, 1, 0.2, 0.4, '#00FF00'),
                          (0, 2, 0.6, 1.2, '#0000FF')], vdims=['y', 'perr', 'nerr', 'color'])
        }).options(color='color')
        plot = mpl_renderer.get_plot(errorbars)
        artist = plot.handles['artist']
        self.assertEqual(artist.get_edgecolors(),
                         np.array([[0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1]]))
        plot.update((1,))
        self.assertEqual(artist.get_edgecolors(),
                         np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]))

    def test_errorbars_linear_color_op(self):
        errorbars = ErrorBars([(0, 0, 0.1, 0.2, 0), (0, 1, 0.2, 0.4, 1), (0, 2, 0.6, 1.2, 2)],
                              vdims=['y', 'perr', 'nerr', 'color']).options(color='color')
        with self.assertRaises(Exception):
            mpl_renderer.get_plot(errorbars)

    def test_errorbars_categorical_color_op(self):
        errorbars = ErrorBars([(0, 0, 0.1, 0.2, 'A'), (0, 1, 0.2, 0.4, 'B'), (0, 2, 0.6, 1.2, 'C')],
                              vdims=['y', 'perr', 'nerr', 'color']).options(color='color')
        with self.assertRaises(Exception):
            mpl_renderer.get_plot(errorbars)

    def test_errorbars_line_color_op(self):
        errorbars = ErrorBars([(0, 0, 0.1, 0.2, '#000000'), (0, 1, 0.2, 0.4, '#FF0000'), (0, 2, 0.6, 1.2, '#00FF00')],
                              vdims=['y', 'perr', 'nerr', 'color']).options(edgecolor='color')
        plot = mpl_renderer.get_plot(errorbars)
        artist = plot.handles['artist']
        self.assertEqual(artist.get_edgecolors(), np.array([
            [0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1]]
        ))

    def test_errorbars_alpha_op(self):
        errorbars = ErrorBars([(0, 0, 0), (0, 1, 0.2), (0, 2, 0.7)],
                              vdims=['y', 'alpha']).options(alpha='alpha')
        with self.assertRaises(Exception):
            mpl_renderer.get_plot(errorbars)

    def test_errorbars_line_width_op(self):
        errorbars = ErrorBars([(0, 0, 0.1, 0.2, 1), (0, 1, 0.2, 0.4, 4), (0, 2, 0.6, 1.2, 8)],
                              vdims=['y', 'perr', 'nerr', 'line_width']).options(linewidth='line_width')
        plot = mpl_renderer.get_plot(errorbars)
        artist = plot.handles['artist']
        self.assertEqual(artist.get_linewidths(), [1, 4, 8])

    def test_errorbars_line_width_op_update(self):
        errorbars = HoloMap({
            0: ErrorBars([(0, 0, 0.1, 0.2, 1), (0, 1, 0.2, 0.4, 4),
                          (0, 2, 0.6, 1.2, 8)], vdims=['y', 'perr', 'nerr', 'line_width']),
            1: ErrorBars([(0, 0, 0.1, 0.2, 2), (0, 1, 0.2, 0.4, 6),
                          (0, 2, 0.6, 1.2, 4)], vdims=['y', 'perr', 'nerr', 'line_width'])
        }).options(linewidth='line_width')
        plot = mpl_renderer.get_plot(errorbars)
        artist = plot.handles['artist']
        self.assertEqual(artist.get_linewidths(), [1, 4, 8])
        plot.update((1,))
        self.assertEqual(artist.get_linewidths(), [2, 6, 4])
