import numpy as np

from holoviews.element import VectorField
from holoviews.plotting.bokeh.util import property_to_dict

from ..utils import ParamLogStream
from .test_plot import TestBokehPlot, bokeh_renderer

try:
    from bokeh.models import CategoricalColorMapper, LinearColorMapper
except ImportError:
    pass


class TestVectorFieldPlot(TestBokehPlot):

    ###########################
    #    Styling mapping      #
    ###########################

    def test_vectorfield_color_op(self):
        vectorfield = VectorField([(0, 0, 0, 1, '#000'), (0, 1, 0, 1,'#F00'), (0, 2, 0, 1,'#0F0')],
                                  vdims=['A', 'M', 'color']).opts(color='color')
        plot = bokeh_renderer.get_plot(vectorfield)
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        self.assertEqual(cds.data['color'], np.array(['#000', '#F00', '#0F0', '#000',
                                                      '#F00', '#0F0', '#000', '#F00', '#0F0']))
        self.assertEqual(property_to_dict(glyph.line_color), {'field': 'color'})

    def test_vectorfield_linear_color_op(self):
        vectorfield = VectorField([(0, 0, 0, 1, 0), (0, 1, 0, 1, 1), (0, 2, 0, 1, 2)],
                                  vdims=['A', 'M', 'color']).opts(color='color')
        plot = bokeh_renderer.get_plot(vectorfield)
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        cmapper = plot.handles['color_color_mapper']
        self.assertTrue(cmapper, LinearColorMapper)
        self.assertEqual(cmapper.low, 0)
        self.assertEqual(cmapper.high, 2)
        self.assertEqual(cds.data['color'], np.array([0, 1, 2, 0, 1, 2, 0, 1, 2]))
        self.assertEqual(property_to_dict(glyph.line_color), {'field': 'color', 'transform': cmapper})

    def test_vectorfield_categorical_color_op(self):
        vectorfield = VectorField([(0, 0, 0, 1, 'A'), (0, 1, 0, 1, 'B'), (0, 2, 0, 1, 'C')],
                                  vdims=['A', 'M', 'color']).opts(color='color')
        plot = bokeh_renderer.get_plot(vectorfield)
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        cmapper = plot.handles['color_color_mapper']
        self.assertTrue(cmapper, CategoricalColorMapper)
        self.assertEqual(cmapper.factors, ['A', 'B', 'C'])
        self.assertEqual(cds.data['color'], np.array(['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C']))
        self.assertEqual(property_to_dict(glyph.line_color), {'field': 'color', 'transform': cmapper})

    def test_vectorfield_alpha_op(self):
        vectorfield = VectorField([(0, 0, 0, 1, 0), (0, 1, 0, 1, 0.2), (0, 2, 0, 1, 0.7)],
                                  vdims=['A', 'M', 'alpha']).opts(alpha='alpha')
        plot = bokeh_renderer.get_plot(vectorfield)
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        self.assertEqual(cds.data['alpha'], np.array([0, 0.2, 0.7, 0, 0.2, 0.7, 0, 0.2, 0.7]))
        self.assertEqual(property_to_dict(glyph.line_alpha), {'field': 'alpha'})

    def test_vectorfield_line_width_op(self):
        vectorfield = VectorField([(0, 0, 0, 1, 1), (0, 1, 0, 1, 4), (0, 2, 0, 1, 8)],
                                  vdims=['A', 'M', 'line_width']).opts(line_width='line_width')
        plot = bokeh_renderer.get_plot(vectorfield)
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        self.assertEqual(cds.data['line_width'], np.array([1, 4, 8, 1, 4, 8, 1, 4, 8]))
        self.assertEqual(property_to_dict(glyph.line_width), {'field': 'line_width'})

    def test_vectorfield_color_index_color_clash(self):
        vectorfield = VectorField([(0, 0, 0), (0, 1, 1), (0, 2, 2)],
                        vdims='color').opts(line_color='color', color_index='color')
        with ParamLogStream() as log:
            bokeh_renderer.get_plot(vectorfield)
        log_msg = log.stream.read()
        warning = (
            "The `color_index` parameter is deprecated in favor of color style mapping, e.g. "
            "`color=dim('color')` or `line_color=dim('color')`\nCannot declare style mapping "
            "for 'line_color' option and declare a color_index; ignoring the color_index.\n"
        )
        self.assertEqual(log_msg, warning)

    def test_vectorfield_no_hover_columns(self):
        vectorfield = VectorField([(0, 0, 0), (0, 1, 1), (0, 2, 2)],
                        vdims='color').opts(tools=[])
        plot = bokeh_renderer.get_plot(vectorfield)
        keys = sorted(plot.handles["cds"].data)
        assert keys == ["x0", "x1", "y0", "y1"]

    def test_vectorfield_hover_columns(self):
        vectorfield = VectorField([(0, 0, 0), (0, 1, 1), (0, 2, 2)],
                        vdims='color').opts(tools=["hover"])
        plot = bokeh_renderer.get_plot(vectorfield)
        keys = sorted(plot.handles["cds"].data)
        assert keys == ["Angle", "Magnitude", "x", "x0", "x1", "y", "y0", "y1"]
