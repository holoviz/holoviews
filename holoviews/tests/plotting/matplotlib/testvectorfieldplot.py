import numpy as np

from holoviews.element import VectorField

from .testplot import TestMPLPlot, mpl_renderer
from ..utils import ParamLogStream


class TestVectorFieldPlot(TestMPLPlot):

    ###########################
    #    Styling mapping      #
    ###########################

    def test_vectorfield_color_op(self):
        vectorfield = VectorField([(0, 0, 0, 1, '#000000'), (0, 1, 0, 1,'#FF0000'), (0, 2, 0, 1,'#00FF00')],
                                  vdims=['A', 'M', 'color']).options(color='color')
        plot = mpl_renderer.get_plot(vectorfield)
        artist = plot.handles['artist']
        self.assertEqual(artist.get_facecolors(), np.array([
            [0, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 0, 1]
        ]))

    def test_vectorfield_linear_color_op(self):
        vectorfield = VectorField([(0, 0, 0, 1, 0), (0, 1, 0, 1, 1), (0, 2, 0, 1, 2)],
                                  vdims=['A', 'M', 'color']).options(color='color')
        plot = mpl_renderer.get_plot(vectorfield)
        artist = plot.handles['artist']
        self.assertEqual(artist.get_array(), np.array([0, 1, 2]))
        self.assertEqual(artist.get_clim(), (0, 2))

    def test_vectorfield_categorical_color_op(self):
        vectorfield = VectorField([(0, 0, 0, 1, 'A'), (0, 1, 0, 1, 'B'), (0, 2, 0, 1, 'C')],
                                  vdims=['A', 'M', 'color']).options(color='color')
        plot = mpl_renderer.get_plot(vectorfield)
        artist = plot.handles['artist']
        self.assertEqual(artist.get_array(), np.array([0, 1, 2]))
        self.assertEqual(artist.get_clim(), (0, 2))

    def test_vectorfield_alpha_op(self):
        vectorfield = VectorField([(0, 0, 0, 1, 0), (0, 1, 0, 1, 0.2), (0, 2, 0, 1, 0.7)],
                                  vdims=['A', 'M', 'alpha']).options(alpha='alpha')
        with self.assertRaises(Exception):
            mpl_renderer.get_plot(vectorfield)

    def test_vectorfield_line_width_op(self):
        vectorfield = VectorField([(0, 0, 0, 1, 1), (0, 1, 0, 1, 4), (0, 2, 0, 1, 8)],
                                  vdims=['A', 'M', 'line_width']).options(linewidth='line_width')
        plot = mpl_renderer.get_plot(vectorfield)
        artist = plot.handles['artist']
        self.assertEqual(artist.get_linewidths(), [1, 4, 8])

    def test_vectorfield_color_index_color_clash(self):
        vectorfield = VectorField([(0, 0, 0, 1, 0), (0, 1, 0, 1, 1), (0, 2, 0, 1, 2)],
                                  vdims=['A', 'M', 'color']).options(color='color', color_index='A')       
        with ParamLogStream() as log:
            plot = mpl_renderer.get_plot(vectorfield)
        log_msg = log.stream.read()
        warning = ("%s: Cannot declare style mapping for 'color' option "
                   "and declare a color_index, ignoring the color_index.\n"
                   % plot.name)
        self.assertEqual(log_msg, warning)
