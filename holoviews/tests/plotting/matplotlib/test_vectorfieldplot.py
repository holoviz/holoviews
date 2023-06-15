import numpy as np

from holoviews.core.spaces import HoloMap
from holoviews.element import VectorField

from .test_plot import TestMPLPlot, mpl_renderer
from ..utils import ParamLogStream


class TestVectorFieldPlot(TestMPLPlot):

    ###########################
    #    Styling mapping      #
    ###########################

    def test_vectorfield_color_op(self):
        vectorfield = VectorField([(0, 0, 0, 1, '#000000'), (0, 1, 0, 1,'#FF0000'), (0, 2, 0, 1,'#00FF00')],
                                  vdims=['A', 'M', 'color']).opts(color='color')
        plot = mpl_renderer.get_plot(vectorfield)
        artist = plot.handles['artist']
        self.assertEqual(artist.get_facecolors(), np.array([
            [0, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 0, 1]
        ]))

    def test_vectorfield_color_op_update(self):
        vectorfield = HoloMap({
            0: VectorField([(0, 0, 0, 1, '#000000'), (0, 1, 0, 1, '#FF0000'), (0, 2, 0, 1, '#00FF00')],
                           vdims=['A', 'M', 'color']),
            1: VectorField([(0, 0, 0, 1, '#0000FF'), (0, 1, 0, 1, '#00FF00'), (0, 2, 0, 1, '#FF0000')],
                           vdims=['A', 'M', 'color'])}).opts(color='color')
        plot = mpl_renderer.get_plot(vectorfield)
        artist = plot.handles['artist']
        self.assertEqual(artist.get_facecolors(), np.array([
            [0, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 0, 1]
        ]))
        plot.update((1,))
        self.assertEqual(artist.get_facecolors(), np.array([
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 0, 0, 1]
        ]))

    def test_vectorfield_linear_color_op_update(self):
        vectorfield = HoloMap({
            0: VectorField([(0, 0, 0, 1, 0), (0, 1, 0, 1, 1), (0, 2, 0, 1, 2)],
                           vdims=['A', 'M', 'color']),
            1: VectorField([(0, 0, 0, 1, 3.2), (0, 1, 0, 1, 2), (0, 2, 0, 1, 4)],
                           vdims=['A', 'M', 'color'])}).opts(color='color', framewise=True)
        plot = mpl_renderer.get_plot(vectorfield)
        artist = plot.handles['artist']
        self.assertEqual(np.asarray(artist.get_array()), np.array([0, 1, 2]))
        self.assertEqual(artist.get_clim(), (0, 2))
        plot.update((1,))
        self.assertEqual(np.asarray(artist.get_array()), np.array([3.2, 2, 4]))
        self.assertEqual(artist.get_clim(), (2, 4))

    def test_vectorfield_categorical_color_op(self):
        vectorfield = VectorField([(0, 0, 0, 1, 'A'), (0, 1, 0, 1, 'B'), (0, 2, 0, 1, 'C')],
                                  vdims=['A', 'M', 'color']).opts(color='color')
        plot = mpl_renderer.get_plot(vectorfield)
        artist = plot.handles['artist']
        self.assertEqual(np.asarray(artist.get_array()), np.array([0, 1, 2]))
        self.assertEqual(artist.get_clim(), (0, 2))

    def test_vectorfield_alpha_op(self):
        vectorfield = VectorField([(0, 0, 0, 1, 0), (0, 1, 0, 1, 0.2), (0, 2, 0, 1, 0.7)],
                                  vdims=['A', 'M', 'alpha']).opts(alpha='alpha')
        with self.assertRaises(Exception):
            mpl_renderer.get_plot(vectorfield)

    def test_vectorfield_line_width_op(self):
        vectorfield = VectorField([(0, 0, 0, 1, 1), (0, 1, 0, 1, 4), (0, 2, 0, 1, 8)],
                                  vdims=['A', 'M', 'line_width']).opts(linewidth='line_width')
        plot = mpl_renderer.get_plot(vectorfield)
        artist = plot.handles['artist']
        self.assertEqual(artist.get_linewidths(), [1, 4, 8])

    def test_vectorfield_line_width_op_update(self):
        vectorfield = HoloMap({
            0: VectorField([(0, 0, 0, 1, 1), (0, 1, 0, 1, 4), (0, 2, 0, 1, 8)],
                           vdims=['A', 'M', 'line_width']),
            1: VectorField([(0, 0, 0, 1, 3), (0, 1, 0, 1, 2), (0, 2, 0, 1, 5)],
                           vdims=['A', 'M', 'line_width'])}).opts(linewidth='line_width')
        plot = mpl_renderer.get_plot(vectorfield)
        artist = plot.handles['artist']
        self.assertEqual(artist.get_linewidths(), [1, 4, 8])
        plot.update((1,))
        self.assertEqual(artist.get_linewidths(), [3, 2, 5])

    def test_vectorfield_color_index_color_clash(self):
        vectorfield = VectorField([(0, 0, 0, 1, 0), (0, 1, 0, 1, 1), (0, 2, 0, 1, 2)],
                                  vdims=['A', 'M', 'color']).opts(color='color', color_index='A')
        with ParamLogStream() as log:
            mpl_renderer.get_plot(vectorfield)
        log_msg = log.stream.read()
        warning = (
            "The `color_index` parameter is deprecated in favor of color style mapping, e.g. "
            "`color=dim('color')` or `line_color=dim('color')`\nCannot declare style mapping "
            "for 'color' option and declare a color_index; ignoring the color_index.\n"
        )
        self.assertEqual(log_msg, warning)
