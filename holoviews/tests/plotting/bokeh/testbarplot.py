import numpy as np

from holoviews.core.overlay import NdOverlay
from holoviews.element import Bars

from bokeh.models import CategoricalColorMapper, LinearColorMapper

from ..utils import ParamLogStream
from .testplot import TestBokehPlot, bokeh_renderer


class TestBarPlot(TestBokehPlot):

    def test_bars_hover_ensure_kdims_sanitized(self):
        obj = Bars(np.random.rand(10,2), kdims=['Dim with spaces'])
        obj = obj.opts(tools=['hover'])
        self._test_hover_info(obj, [('Dim with spaces', '@{Dim_with_spaces}'), ('y', '@{y}')])

    def test_bars_hover_ensure_vdims_sanitized(self):
        obj = Bars(np.random.rand(10,2), vdims=['Dim with spaces'])
        obj = obj.opts(tools=['hover'])
        self._test_hover_info(obj, [('x', '@{x}'), ('Dim with spaces', '@{Dim_with_spaces}')])

    def test_bars_suppress_legend(self):
        bars = Bars([('A', 1), ('B', 2)]).opts(plot=dict(show_legend=False))
        plot = bokeh_renderer.get_plot(bars)
        plot.initialize_plot()
        fig = plot.state
        self.assertEqual(len(fig.legend), 0)

    def test_empty_bars(self):
        bars = Bars([], kdims=['x', 'y'], vdims=['z']).opts(plot=dict(group_index=1))
        plot = bokeh_renderer.get_plot(bars)
        plot.initialize_plot()
        source = plot.handles['source']
        for v in source.data.values():
            self.assertEqual(len(v), 0)

    def test_bars_grouped_categories(self):
        bars = Bars([('A', 0, 1), ('A', 1, -1), ('B', 0, 2)],
                    kdims=['Index', 'Category'], vdims=['Value'])
        plot = bokeh_renderer.get_plot(bars)
        source = plot.handles['source']
        self.assertEqual([tuple(x) for x in source.data['xoffsets']],
                         [('A', '0'), ('B', '0'), ('A', '1')])
        self.assertEqual(list(source.data['Category']), ['0', '0', '1'])
        self.assertEqual(source.data['Value'], np.array([1, 2, -1]))
        x_range = plot.handles['x_range']
        self.assertEqual(x_range.factors, [('A', '0'), ('A', '1'), ('B', '0'), ('B', '1')])

    def test_bars_multi_level_sorted(self):
        box= Bars((['A', 'B']*15, [3, 10, 1]*10, np.random.randn(30)),
                  ['Group', 'Category'], 'Value').aggregate(function=np.mean)
        plot = bokeh_renderer.get_plot(box)
        x_range = plot.handles['x_range']
        self.assertEqual(x_range.factors, [
            ('A', '1'), ('A', '3'), ('A', '10'), ('B', '1'), ('B', '3'), ('B', '10')])

    def test_box_whisker_multi_level_sorted_alphanumerically(self):
        box= Bars(([3, 10, 1]*10, ['A', 'B']*15, np.random.randn(30)),
                  ['Group', 'Category'], 'Value').aggregate(function=np.mean)
        plot = bokeh_renderer.get_plot(box)
        x_range = plot.handles['x_range']
        self.assertEqual(x_range.factors, [
            ('1', 'A'), ('1', 'B'), ('3', 'A'), ('3', 'B'), ('10', 'A'), ('10', 'B')])

    def test_bars_positive_negative_mixed(self):
        bars = Bars([('A', 0, 1), ('A', 1, -1), ('B', 0, 2)],
                    kdims=['Index', 'Category'], vdims=['Value'])
        plot = bokeh_renderer.get_plot(bars.opts(stacked=True))
        source = plot.handles['source']
        self.assertEqual(list(source.data['Category']), ['1', '0', '0'])
        self.assertEqual(list(source.data['Index']), ['A', 'A', 'B'])
        self.assertEqual(source.data['top'], np.array([0, 1, 2]))
        self.assertEqual(source.data['bottom'], np.array([-1, 0, 0]))

    def test_bars_logy(self):
        bars = Bars([('A', 1), ('B', 2), ('C', 3)],
                    kdims=['Index'], vdims=['Value'])
        plot = bokeh_renderer.get_plot(bars.opts(plot=dict(logy=True)))
        source = plot.handles['source']
        glyph = plot.handles['glyph']
        y_range = plot.handles['y_range']
        self.assertEqual(list(source.data['Index']), ['A', 'B', 'C'])
        self.assertEqual(source.data['Value'], np.array([1, 2, 3]))
        self.assertEqual(glyph.bottom, 10**(np.log10(3)-2))
        self.assertEqual(y_range.start, 0.03348369522101712)
        self.assertEqual(y_range.end, 3.348369522101713)

    def test_bars_logy_explicit_range(self):
        bars = Bars([('A', 1), ('B', 2), ('C', 3)],
                    kdims=['Index'], vdims=['Value']).redim.range(Value=(0.001, 3))
        plot = bokeh_renderer.get_plot(bars.opts(plot=dict(logy=True)))
        source = plot.handles['source']
        glyph = plot.handles['glyph']
        y_range = plot.handles['y_range']
        self.assertEqual(list(source.data['Index']), ['A', 'B', 'C'])
        self.assertEqual(source.data['Value'], np.array([1, 2, 3]))
        self.assertEqual(glyph.bottom, 0.001)
        self.assertEqual(y_range.start, 0.001)
        self.assertEqual(y_range.end, 3)

    def test_bars_ylim(self):
        bars = Bars([1, 2, 3]).opts(ylim=(0, 200))
        plot = bokeh_renderer.get_plot(bars)
        y_range = plot.handles['y_range']
        self.assertEqual(y_range.start, 0)
        self.assertEqual(y_range.end, 200)
        
    def test_bars_padding_square(self):
        points = Bars([(1, 2), (2, -1), (3, 3)]).options(padding=0.1)
        plot = bokeh_renderer.get_plot(points)
        y_range = plot.handles['y_range']
        self.assertEqual(y_range.start, -1.4)
        self.assertEqual(y_range.end, 3.4)

    def test_bars_padding_square_positive(self):
        points = Bars([(1, 2), (2, 1), (3, 3)]).options(padding=0.1)
        plot = bokeh_renderer.get_plot(points)
        y_range = plot.handles['y_range']
        self.assertEqual(y_range.start, 0)
        self.assertEqual(y_range.end, 3.2)

    def test_bars_padding_square_negative(self):
        points = Bars([(1, -2), (2, -1), (3, -3)]).options(padding=0.1)
        plot = bokeh_renderer.get_plot(points)
        y_range = plot.handles['y_range']
        self.assertEqual(y_range.start, -3.2)
        self.assertEqual(y_range.end, 0)

    def test_bars_padding_nonsquare(self):
        bars = Bars([(1, 2), (2, 1), (3, 3)]).options(padding=0.1, width=600)
        plot = bokeh_renderer.get_plot(bars)
        y_range = plot.handles['y_range']
        self.assertEqual(y_range.start, 0)
        self.assertEqual(y_range.end, 3.2)

    def test_bars_padding_logx(self):
        bars = Bars([(1, 1), (2, 2), (3,3)]).options(padding=0.1, logx=True)
        plot = bokeh_renderer.get_plot(bars)
        y_range = plot.handles['y_range']
        self.assertEqual(y_range.start, 0)
        self.assertEqual(y_range.end, 3.2)

    def test_bars_padding_logy(self):
        bars = Bars([(1, 2), (2, 1), (3, 3)]).options(padding=0.1, logy=True)
        plot = bokeh_renderer.get_plot(bars)
        y_range = plot.handles['y_range']
        self.assertEqual(y_range.start, 0.033483695221017122)
        self.assertEqual(y_range.end, 3.3483695221017129)

    ###########################
    #    Styling mapping      #
    ###########################

    def test_bars_color_op(self):
        bars = Bars([(0, 0, '#000'), (0, 1, '#F00'), (0, 2, '#0F0')],
                              vdims=['y', 'color']).options(color='color')
        plot = bokeh_renderer.get_plot(bars)
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        self.assertEqual(cds.data['color'], np.array(['#000', '#F00', '#0F0']))
        self.assertEqual(glyph.fill_color, {'field': 'color'})
        self.assertEqual(glyph.line_color, 'black')

    def test_bars_linear_color_op(self):
        bars = Bars([(0, 0, 0), (0, 1, 1), (0, 2, 2)],
                              vdims=['y', 'color']).options(color='color')
        plot = bokeh_renderer.get_plot(bars)
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        cmapper = plot.handles['color_color_mapper']
        self.assertTrue(cmapper, LinearColorMapper)
        self.assertEqual(cmapper.low, 0)
        self.assertEqual(cmapper.high, 2)
        self.assertEqual(cds.data['color'], np.array([0, 1, 2]))
        self.assertEqual(glyph.fill_color, {'field': 'color', 'transform': cmapper})
        self.assertEqual(glyph.line_color, 'black')

    def test_bars_categorical_color_op(self):
        bars = Bars([(0, 0, 'A'), (0, 1, 'B'), (0, 2, 'C')],
                              vdims=['y', 'color']).options(color='color')
        plot = bokeh_renderer.get_plot(bars)
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        cmapper = plot.handles['color_color_mapper']
        self.assertTrue(cmapper, CategoricalColorMapper)
        self.assertEqual(cmapper.factors, ['A', 'B', 'C'])
        self.assertEqual(cds.data['color'], np.array(['A', 'B', 'C']))
        self.assertEqual(glyph.fill_color, {'field': 'color', 'transform': cmapper})
        self.assertEqual(glyph.line_color, 'black')

    def test_bars_line_color_op(self):
        bars = Bars([(0, 0, '#000'), (0, 1, '#F00'), (0, 2, '#0F0')],
                              vdims=['y', 'color']).options(line_color='color')
        plot = bokeh_renderer.get_plot(bars)
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        self.assertEqual(cds.data['line_color'], np.array(['#000', '#F00', '#0F0']))
        self.assertNotEqual(glyph.fill_color, {'field': 'line_color'})
        self.assertEqual(glyph.line_color, {'field': 'line_color'})

    def test_bars_fill_color_op(self):
        bars = Bars([(0, 0, '#000'), (0, 1, '#F00'), (0, 2, '#0F0')],
                              vdims=['y', 'color']).options(fill_color='color')
        plot = bokeh_renderer.get_plot(bars)
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        self.assertEqual(cds.data['fill_color'], np.array(['#000', '#F00', '#0F0']))
        self.assertEqual(glyph.fill_color, {'field': 'fill_color'})
        self.assertNotEqual(glyph.line_color, {'field': 'fill_color'})

    def test_bars_alpha_op(self):
        bars = Bars([(0, 0, 0), (0, 1, 0.2), (0, 2, 0.7)],
                              vdims=['y', 'alpha']).options(alpha='alpha')
        plot = bokeh_renderer.get_plot(bars)
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        self.assertEqual(cds.data['alpha'], np.array([0, 0.2, 0.7]))
        self.assertEqual(glyph.fill_alpha, {'field': 'alpha'})

    def test_bars_line_alpha_op(self):
        bars = Bars([(0, 0, 0), (0, 1, 0.2), (0, 2, 0.7)],
                              vdims=['y', 'alpha']).options(line_alpha='alpha')
        plot = bokeh_renderer.get_plot(bars)
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        self.assertEqual(cds.data['line_alpha'], np.array([0, 0.2, 0.7]))
        self.assertEqual(glyph.line_alpha, {'field': 'line_alpha'})
        self.assertNotEqual(glyph.fill_alpha, {'field': 'line_alpha'})

    def test_bars_fill_alpha_op(self):
        bars = Bars([(0, 0, 0), (0, 1, 0.2), (0, 2, 0.7)],
                              vdims=['y', 'alpha']).options(fill_alpha='alpha')
        plot = bokeh_renderer.get_plot(bars)
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        self.assertEqual(cds.data['fill_alpha'], np.array([0, 0.2, 0.7]))
        self.assertNotEqual(glyph.line_alpha, {'field': 'fill_alpha'})
        self.assertEqual(glyph.fill_alpha, {'field': 'fill_alpha'})

    def test_bars_line_width_op(self):
        bars = Bars([(0, 0, 1), (0, 1, 4), (0, 2, 8)],
                              vdims=['y', 'line_width']).options(line_width='line_width')
        plot = bokeh_renderer.get_plot(bars)
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        self.assertEqual(cds.data['line_width'], np.array([1, 4, 8]))
        self.assertEqual(glyph.line_width, {'field': 'line_width'})

    def test_op_ndoverlay_value(self):
        colors = ['blue', 'red']
        overlay = NdOverlay({color: Bars(np.arange(i+2)) for i, color in enumerate(colors)}, 'Color').options('Bars', fill_color='Color')
        plot = bokeh_renderer.get_plot(overlay)
        for subplot, color in zip(plot.subplots.values(),  colors):
            self.assertEqual(subplot.handles['glyph'].fill_color, color)

    def test_bars_color_index_color_clash(self):
        bars = Bars([(0, 0, 0), (0, 1, 1), (0, 2, 2)],
                    vdims=['y', 'color']).options(color='color', color_index='color')
        with ParamLogStream() as log:
            bokeh_renderer.get_plot(bars)
        log_msg = log.stream.read()
        warning = ("Cannot declare style mapping for 'color' option "
                   "and declare a color_index; ignoring the color_index.\n")
        self.assertEqual(log_msg, warning)
