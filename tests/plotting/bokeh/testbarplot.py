import numpy as np

from holoviews.element import Bars

from .testplot import TestBokehPlot, bokeh_renderer


class TestBarPlot(TestBokehPlot):

    def test_bars_hover_ensure_kdims_sanitized(self):
        obj = Bars(np.random.rand(10,2), kdims=['Dim with spaces'])
        obj = obj(plot={'tools': ['hover']})
        self._test_hover_info(obj, [('Dim with spaces', '@{Dim_with_spaces}'), ('y', '@{y}')])

    def test_bars_hover_ensure_vdims_sanitized(self):
        obj = Bars(np.random.rand(10,2), vdims=['Dim with spaces'])
        obj = obj(plot={'tools': ['hover']})
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

    def test_bars_positive_negative_mixed(self):
        bars = Bars([('A', 0, 1), ('A', 1, -1), ('B', 0, 2)],
                    kdims=['Index', 'Category'], vdims=['Value'])
        plot = bokeh_renderer.get_plot(bars.opts(plot=dict(stack_index=1)))
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
        self.assertEqual(y_range.start, 10**(np.log10(3)-2))
        self.assertEqual(y_range.end, 3.)

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
        self.assertEqual(y_range.end, 3.)

    def test_bars_padding_square(self):
        points = Bars([(1, 2), (2, -1), (3, 3)]).options(padding=0.2)
        plot = bokeh_renderer.get_plot(points)
        y_range = plot.handles['y_range']
        self.assertEqual(y_range.start, -1.4)
        self.assertEqual(y_range.end, 3.4)

    def test_bars_padding_square_positive(self):
        points = Bars([(1, 2), (2, 1), (3, 3)]).options(padding=0.2)
        plot = bokeh_renderer.get_plot(points)
        y_range = plot.handles['y_range']
        self.assertEqual(y_range.start, 0)
        self.assertEqual(y_range.end, 3.2)

    def test_bars_padding_square_positive(self):
        points = Bars([(1, -2), (2, -1), (3, -3)]).options(padding=0.2)
        plot = bokeh_renderer.get_plot(points)
        y_range = plot.handles['y_range']
        self.assertEqual(y_range.start, -3.2)
        self.assertEqual(y_range.end, 0)

    def test_bars_padding_nonsquare(self):
        bars = Bars([(1, 2), (2, 1), (3, 3)]).options(padding=0.2, width=600)
        plot = bokeh_renderer.get_plot(bars)
        y_range = plot.handles['y_range']
        self.assertEqual(y_range.start, 0)
        self.assertEqual(y_range.end, 3.2)

    def test_bars_padding_logx(self):
        bars = Bars([(1, 1), (2, 2), (3,3)]).options(padding=0.2, logx=True)
        plot = bokeh_renderer.get_plot(bars)
        y_range = plot.handles['y_range']
        self.assertEqual(y_range.start, 0)
        self.assertEqual(y_range.end, 3.2)

    def test_bars_padding_logy(self):
        bars = Bars([(1, 2), (2, 1), (3, 3)]).options(padding=0.2, logy=True)
        plot = bokeh_renderer.get_plot(bars)
        y_range = plot.handles['y_range']
        self.assertEqual(y_range.start, 0.033483695221017122)
        self.assertEqual(y_range.end, 3.3483695221017129)

