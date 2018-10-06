from __future__ import absolute_import

import numpy as np

from holoviews.element import Violin
from holoviews.operation.stats import univariate_kde

from .testplot import TestBokehPlot, bokeh_renderer


class TestBokehViolinPlot(TestBokehPlot):

    def test_violin_simple(self):
        values = np.random.rand(100)
        violin = Violin(values).opts(plot=dict(violin_width=0.7))
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
        patch_source = plot.handles['patch_0_source']
        self.assertEqual(patch_source.data['x'], kde['x'])
        self.assertEqual(patch_source.data['y'], kde['y'])

    def test_violin_inner_quartiles(self):
        values = np.random.rand(100)
        violin = Violin(values).opts(plot=dict(inner='quartiles'))
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
        violin = Violin(values).opts(plot=dict(inner='stick'))
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
