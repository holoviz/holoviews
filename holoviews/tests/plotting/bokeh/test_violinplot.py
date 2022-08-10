from __future__ import absolute_import

from unittest import SkipTest

import numpy as np

from holoviews.element import Violin
from holoviews.operation.stats import univariate_kde
from holoviews.util.transform import dim

from .test_plot import TestBokehPlot, bokeh_renderer

from bokeh.models import LinearColorMapper, CategoricalColorMapper


class TestBokehViolinPlot(TestBokehPlot):

    def setUp(self):
        try:
            import scipy # noqa
        except:
            raise SkipTest('Violin plot requires SciPy to compute kde')
        super().setUp()

    def test_violin_simple(self):
        values = np.random.rand(100)
        violin = Violin(values).opts(violin_width=0.7)
        qmin, q1, q2, q3, qmax = (np.percentile(values, q=q)
                                  for q in range(0,125,25))
        iqr = q3 - q1
        upper = min(q3 + 1.5*iqr, np.nanmax(values))
        lower = max(q1 - 1.5*iqr, np.nanmin(values))
        r1, r2 = violin.range(0)
        kde = univariate_kde(violin, cut=5)
        xs, ys = (kde.dimension_values(i) for i in range(2))
        ys = (ys/ys.max())*(0.7/2.)
        ys = [('',)+(sign*y,) for sign, vs in ((-1, ys), (1, ys[::-1])) for y in vs]
        kde =  {'x': np.concatenate([xs, xs[::-1]]), 'y': ys}

        plot = bokeh_renderer.get_plot(violin)
        self.assertEqual(plot.handles['x_range'].factors, [''])
        self.assertEqual(plot.handles['y_range'].start, r1)
        self.assertEqual(plot.handles['y_range'].end, r2)
        self.assertIn('scatter_1_glyph_renderer', plot.handles)
        self.assertIn('vbar_1_glyph_renderer', plot.handles)
        seg_source = plot.handles['segment_1_source']
        self.assertEqual(seg_source.data['x'], [('', 0)])
        self.assertEqual(seg_source.data['y0'], [lower])
        self.assertEqual(seg_source.data['y1'], [upper])
        scatter_source = plot.handles['scatter_1_source']
        self.assertEqual(scatter_source.data['x'], [('', 0)])
        self.assertEqual(scatter_source.data['y'], [q2])
        patch_source = plot.handles['patches_1_source']
        self.assertEqual(patch_source.data['xs'], [kde['y']])
        self.assertEqual(patch_source.data['ys'], [kde['x']])

    def test_violin_multi_level(self):
        box= Violin((['A', 'B']*15, [3, 10, 1]*10, np.random.randn(30)),
                    ['Group', 'Category'], 'Value')
        plot = bokeh_renderer.get_plot(box)
        x_range = plot.handles['x_range']
        self.assertEqual(x_range.factors, [
            ('A', '1'), ('A', '3'), ('A', '10'), ('B', '1'), ('B', '3'), ('B', '10')])

    def test_violin_inner_quartiles(self):
        values = np.random.rand(100)
        violin = Violin(values).opts(inner='quartiles')
        kde = univariate_kde(violin, cut=5)
        xs = kde.dimension_values(0)
        plot = bokeh_renderer.get_plot(violin)
        self.assertIn('segment_1_glyph_renderer', plot.handles)
        seg_source = plot.handles['segment_1_source']
        q1, q2, q3 = (np.percentile(values, q=q) for q in range(25,100,25))
        y0, y1, y2 = [xs[np.argmin(np.abs(xs-v))] for v in (q1, q2, q3)]
        self.assertEqual(seg_source.data['x'], np.array([y0, y1, y2]))

    def test_violin_inner_stick(self):
        values = np.random.rand(100)
        violin = Violin(values).opts(inner='stick')
        kde = univariate_kde(violin, cut=5)
        xs = kde.dimension_values(0)
        plot = bokeh_renderer.get_plot(violin)
        self.assertIn('segment_1_glyph_renderer', plot.handles)
        segments = np.array([xs[np.argmin(np.abs(xs-v))] for v in values])
        self.assertEqual(plot.handles['segment_1_source'].data['x'],
                         segments)

    def test_violin_multi(self):
        violin = Violin((np.random.randint(0, 2, 100), np.random.rand(100)), kdims=['A']).sort()
        r1, r2 = violin.range(1)
        plot = bokeh_renderer.get_plot(violin)
        self.assertEqual(plot.handles['x_range'].factors, ['0', '1'])

    def test_violin_empty(self):
        violin = Violin([])
        plot = bokeh_renderer.get_plot(violin)
        patch_source = plot.handles['patches_1_source']
        self.assertEqual(patch_source.data['xs'], [[]])
        self.assertEqual(patch_source.data['ys'], [np.array([])])

    def test_violin_single_point(self):
        data = {'x': [1], 'y': [1]}
        violin = Violin(data=data, kdims='x', vdims='y').opts(inner='box')

        plot = bokeh_renderer.get_plot(violin)
        self.assertEqual(plot.handles['x_range'].factors, ['1'])

    ###########################
    #    Styling mapping      #
    ###########################

    def test_violin_linear_color_op(self):
        a = np.repeat(np.arange(5), 5)
        b = np.repeat(np.arange(5), 5)
        violin = Violin((a, b, np.arange(25)), ['a', 'b'], 'd').opts(violin_color='b')
        plot = bokeh_renderer.get_plot(violin)
        source = plot.handles['patches_1_source']
        cmapper = plot.handles['violin_color_color_mapper']
        glyph = plot.handles['patches_1_glyph']
        self.assertEqual(source.data['violin_color'], np.arange(5))
        self.assertTrue(cmapper, LinearColorMapper)
        self.assertEqual(cmapper.low, 0)
        self.assertEqual(cmapper.high, 4)
        self.assertEqual(glyph.fill_color, {'field': 'violin_color', 'transform': cmapper})

    def test_violin_categorical_color_op(self):
        a = np.repeat(np.arange(5), 5)
        b = np.repeat(['A', 'B', 'C', 'D', 'E'], 5)
        violin = Violin((a, b, np.arange(25)), ['a', 'b'], 'd').opts(violin_color='b')
        plot = bokeh_renderer.get_plot(violin)
        source = plot.handles['patches_1_source']
        glyph = plot.handles['patches_1_glyph']
        cmapper = plot.handles['violin_color_color_mapper']
        self.assertEqual(source.data['violin_color'], b[::5])
        self.assertTrue(cmapper, CategoricalColorMapper)
        self.assertEqual(cmapper.factors, ['A', 'B', 'C', 'D', 'E'])
        self.assertEqual(glyph.fill_color, {'field': 'violin_color', 'transform': cmapper})

    def test_violin_alpha_op(self):
        a = np.repeat(np.arange(5), 5)
        b = np.repeat(np.arange(5)/10., 5)
        violin = Violin((a, b, np.arange(25)), ['a', 'b'], 'd').opts(violin_alpha='b')
        plot = bokeh_renderer.get_plot(violin)
        source = plot.handles['patches_1_source']
        glyph = plot.handles['patches_1_glyph']
        self.assertEqual(source.data['violin_alpha'], np.arange(5)/10.)
        self.assertEqual(glyph.fill_alpha, {'field': 'violin_alpha'})

    def test_violin_line_width_op(self):
        a = np.repeat(np.arange(5), 5)
        b = np.repeat(np.arange(5), 5)
        violin = Violin((a, b, np.arange(25)), ['a', 'b'], 'd').opts(violin_line_width='b')
        plot = bokeh_renderer.get_plot(violin)
        source = plot.handles['multi_line_1_source']
        glyph = plot.handles['multi_line_1_glyph']
        self.assertEqual(source.data['outline_line_width'], np.arange(5))
        self.assertEqual(glyph.line_width, {'field': 'outline_line_width'})

    def test_violin_split_op_multi(self):
        a = np.repeat(np.arange(5), 5)
        b = np.repeat(np.arange(5), 5)
        violin = Violin((a, b, np.arange(25)), ['a', 'b'], 'd').opts(split=dim('b')>2)
        plot = bokeh_renderer.get_plot(violin)
        source = plot.handles['patches_1_source']
        glyph = plot.handles['patches_1_glyph']
        cmapper = plot.handles['violin_color_mapper']
        values = ['False', 'True', 'False', 'True', 'False', 'True', 'False', 'True', 'False', 'True']
        self.assertEqual(source.data["dim('b')>2"], values)
        self.assertEqual(glyph.fill_color, {'field': "dim('b')>2", 'transform': cmapper})

    def test_violin_split_op_single(self):
        a = np.repeat(np.arange(2), 5)
        violin = Violin((a, np.arange(10)), ['a'], 'd').opts(split='a')
        plot = bokeh_renderer.get_plot(violin)
        source = plot.handles['patches_1_source']
        glyph = plot.handles['patches_1_glyph']
        cmapper = plot.handles['violin_color_mapper']
        self.assertEqual(source.data["dim('a')"], ['0', '1'])
        self.assertEqual(glyph.fill_color, {'field': "dim('a')", 'transform': cmapper})

    def test_violin_box_linear_color_op(self):
        a = np.repeat(np.arange(5), 5)
        b = np.repeat(np.arange(5), 5)
        violin = Violin((a, b, np.arange(25)), ['a', 'b'], 'd').opts(box_color='b')
        plot = bokeh_renderer.get_plot(violin)
        source = plot.handles['vbar_1_source']
        cmapper = plot.handles['box_color_color_mapper']
        glyph = plot.handles['vbar_1_glyph']
        self.assertEqual(source.data['box_color'], np.arange(5))
        self.assertTrue(cmapper, LinearColorMapper)
        self.assertEqual(cmapper.low, 0)
        self.assertEqual(cmapper.high, 4)
        self.assertEqual(glyph.fill_color, {'field': 'box_color', 'transform': cmapper})

    def test_violin_box_categorical_color_op(self):
        a = np.repeat(np.arange(5), 5)
        b = np.repeat(['A', 'B', 'C', 'D', 'E'], 5)
        violin = Violin((a, b, np.arange(25)), ['a', 'b'], 'd').opts(box_color='b')
        plot = bokeh_renderer.get_plot(violin)
        source = plot.handles['vbar_1_source']
        glyph = plot.handles['vbar_1_glyph']
        cmapper = plot.handles['box_color_color_mapper']
        self.assertEqual(source.data['box_color'], b[::5])
        self.assertTrue(cmapper, CategoricalColorMapper)
        self.assertEqual(cmapper.factors, ['A', 'B', 'C', 'D', 'E'])
        self.assertEqual(glyph.fill_color, {'field': 'box_color', 'transform': cmapper})

    def test_violin_box_alpha_op(self):
        a = np.repeat(np.arange(5), 5)
        b = np.repeat(np.arange(5)/10., 5)
        violin = Violin((a, b, np.arange(25)), ['a', 'b'], 'd').opts(box_alpha='b')
        plot = bokeh_renderer.get_plot(violin)
        source = plot.handles['vbar_1_source']
        glyph = plot.handles['vbar_1_glyph']
        self.assertEqual(source.data['box_alpha'], np.arange(5)/10.)
        self.assertEqual(glyph.fill_alpha, {'field': 'box_alpha'})

    def test_violin_box_line_width_op(self):
        a = np.repeat(np.arange(5), 5)
        b = np.repeat(np.arange(5), 5)
        violin = Violin((a, b, np.arange(25)), ['a', 'b'], 'd').opts(box_line_width='b')
        plot = bokeh_renderer.get_plot(violin)
        source = plot.handles['vbar_1_source']
        glyph = plot.handles['vbar_1_glyph']
        self.assertEqual(source.data['box_line_width'], np.arange(5))
        self.assertEqual(glyph.line_width, {'field': 'box_line_width'})
