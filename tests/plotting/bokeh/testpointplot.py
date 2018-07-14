from unittest import SkipTest

import numpy as np

from holoviews.core import NdOverlay
from holoviews.core.options import Cycle
from holoviews.core.util import pd
from holoviews.element import Points

from .testplot import TestBokehPlot, bokeh_renderer
from ..utils import ParamLogStream

try:
    from bokeh.models import FactorRange, CategoricalColorMapper
except:
    pass


class TestPointPlot(TestBokehPlot):

    def test_points_colormapping(self):
        points = Points(np.random.rand(10, 4), vdims=['a', 'b']).opts(plot=dict(color_index=3))
        self._test_colormapping(points, 3)

    def test_points_colormapping_with_nonselection(self):
        opts = dict(plot=dict(color_index=3),
                    style=dict(nonselection_color='red'))
        points = Points(np.random.rand(10, 4), vdims=['a', 'b']).opts(**opts)
        self._test_colormapping(points, 3)

    def test_points_colormapping_categorical(self):
        points = Points([(i, i*2, i*3, chr(65+i)) for i in range(10)],
                         vdims=['a', 'b']).opts(plot=dict(color_index='b'))
        plot = bokeh_renderer.get_plot(points)
        plot.initialize_plot()
        cmapper = plot.handles['color_mapper']
        self.assertIsInstance(cmapper, CategoricalColorMapper)
        self.assertEqual(cmapper.factors, list(points['b']))

    def test_points_color_selection_nonselection(self):
        opts = dict(color='green', selection_color='red', nonselection_color='blue')
        points = Points([(i, i*2, i*3, chr(65+i)) for i in range(10)],
                         vdims=['a', 'b']).opts(style=opts)
        plot = bokeh_renderer.get_plot(points)
        glyph_renderer = plot.handles['glyph_renderer']
        self.assertEqual(glyph_renderer.glyph.fill_color, 'green')
        self.assertEqual(glyph_renderer.glyph.line_color, 'green')
        self.assertEqual(glyph_renderer.selection_glyph.fill_color, 'red')
        self.assertEqual(glyph_renderer.selection_glyph.line_color, 'red')
        self.assertEqual(glyph_renderer.nonselection_glyph.fill_color, 'blue')
        self.assertEqual(glyph_renderer.nonselection_glyph.line_color, 'blue')

    def test_points_alpha_selection_nonselection(self):
        opts = dict(alpha=0.8, selection_alpha=1.0, nonselection_alpha=0.2)
        points = Points([(i, i*2, i*3, chr(65+i)) for i in range(10)],
                         vdims=['a', 'b']).opts(style=opts)
        plot = bokeh_renderer.get_plot(points)
        glyph_renderer = plot.handles['glyph_renderer']
        self.assertEqual(glyph_renderer.glyph.fill_alpha, 0.8)
        self.assertEqual(glyph_renderer.glyph.line_alpha, 0.8)
        self.assertEqual(glyph_renderer.selection_glyph.fill_alpha, 1)
        self.assertEqual(glyph_renderer.selection_glyph.line_alpha, 1)
        self.assertEqual(glyph_renderer.nonselection_glyph.fill_alpha, 0.2)
        self.assertEqual(glyph_renderer.nonselection_glyph.line_alpha, 0.2)

    def test_points_alpha_selection_partial(self):
        opts = dict(selection_alpha=1.0, selection_fill_alpha=0.2)
        points = Points([(i, i*2, i*3, chr(65+i)) for i in range(10)],
                         vdims=['a', 'b']).opts(style=opts)
        plot = bokeh_renderer.get_plot(points)
        glyph_renderer = plot.handles['glyph_renderer']
        self.assertEqual(glyph_renderer.glyph.fill_alpha, 1.0)
        self.assertEqual(glyph_renderer.glyph.line_alpha, 1.0)
        self.assertEqual(glyph_renderer.selection_glyph.fill_alpha, 0.2)
        self.assertEqual(glyph_renderer.selection_glyph.line_alpha, 1)

    def test_batched_points(self):
        overlay = NdOverlay({i: Points(np.arange(i)) for i in range(1, 100)})
        plot = bokeh_renderer.get_plot(overlay)
        extents = plot.get_extents(overlay, {})
        self.assertEqual(extents, (0, 0, 98, 98))

    def test_batched_points_size_and_color(self):
        opts = {'NdOverlay': dict(plot=dict(legend_limit=0)),
                'Points': dict(style=dict(size=Cycle(values=[1, 2])))}
        overlay = NdOverlay({i: Points([(i, j) for j in range(2)])
                             for i in range(2)}).opts(opts)
        plot = bokeh_renderer.get_plot(overlay).subplots[()]
        size = np.array([1, 1, 2, 2])
        color = np.array(['#30a2da', '#30a2da', '#fc4f30', '#fc4f30'],
                         dtype='<U7')
        self.assertEqual(plot.handles['source'].data['color'], color)
        self.assertEqual(plot.handles['source'].data['size'], size)

    def test_batched_points_line_color_and_color(self):
        opts = {'NdOverlay': dict(plot=dict(legend_limit=0)),
                'Points': dict(style=dict(line_color=Cycle(values=['red', 'blue'])))}
        overlay = NdOverlay({i: Points([(i, j) for j in range(2)])
                             for i in range(2)}).opts(opts)
        plot = bokeh_renderer.get_plot(overlay).subplots[()]
        line_color = np.array(['red', 'red', 'blue', 'blue'])
        fill_color = np.array(['#30a2da', '#30a2da', '#fc4f30', '#fc4f30'],
                         dtype='<U7')
        self.assertEqual(plot.handles['source'].data['fill_color'], fill_color)
        self.assertEqual(plot.handles['source'].data['line_color'], line_color)

    def test_batched_points_alpha_and_color(self):
        opts = {'NdOverlay': dict(plot=dict(legend_limit=0)),
                'Points': dict(style=dict(alpha=Cycle(values=[0.5, 1])))}
        overlay = NdOverlay({i: Points([(i, j) for j in range(2)])
                             for i in range(2)}).opts(opts)
        plot = bokeh_renderer.get_plot(overlay).subplots[()]
        alpha = np.array([0.5, 0.5, 1., 1.])
        color = np.array(['#30a2da', '#30a2da', '#fc4f30', '#fc4f30'],
                         dtype='<U7')
        self.assertEqual(plot.handles['source'].data['alpha'], alpha)
        self.assertEqual(plot.handles['source'].data['color'], color)

    def test_batched_points_line_width_and_color(self):
        opts = {'NdOverlay': dict(plot=dict(legend_limit=0)),
                'Points': dict(style=dict(line_width=Cycle(values=[0.5, 1])))}
        overlay = NdOverlay({i: Points([(i, j) for j in range(2)])
                             for i in range(2)}).opts(opts)
        plot = bokeh_renderer.get_plot(overlay).subplots[()]
        line_width = np.array([0.5, 0.5, 1., 1.])
        color = np.array(['#30a2da', '#30a2da', '#fc4f30', '#fc4f30'],
                         dtype='<U7')
        self.assertEqual(plot.handles['source'].data['line_width'], line_width)
        self.assertEqual(plot.handles['source'].data['color'], color)

    def test_points_overlay_datetime_hover(self):
        if pd is None:
            raise SkipTest("Test requires pandas")
        obj = NdOverlay({i: Points((list(pd.date_range('2016-01-01', '2016-01-31')), range(31))) for i in range(5)},
                        kdims=['Test'])
        opts = {'Points': {'tools': ['hover']}}
        obj = obj(plot=opts)
        self._test_hover_info(obj, [('Test', '@{Test}'), ('x', '@{x_dt_strings}'), ('y', '@{y}')])

    def test_points_overlay_hover_batched(self):
        obj = NdOverlay({i: Points(np.random.rand(10,2)) for i in range(5)},
                        kdims=['Test'])
        opts = {'Points': {'tools': ['hover']},
                'NdOverlay': {'legend_limit': 0}}
        obj = obj(plot=opts)
        self._test_hover_info(obj, [('Test', '@{Test}'), ('x', '@{x}'), ('y', '@{y}')])

    def test_points_overlay_hover(self):
        obj = NdOverlay({i: Points(np.random.rand(10,2)) for i in range(5)},
                        kdims=['Test'])
        opts = {'Points': {'tools': ['hover']},
                'NdOverlay': {'legend_limit': 0}}
        obj = obj(plot=opts)
        self._test_hover_info(obj, [('Test', '@{Test}'), ('x', '@{x}'),
                                    ('y', '@{y}')])

    def test_points_no_single_item_legend(self):
        points = Points([('A', 1), ('B', 2)], label='A')
        plot = bokeh_renderer.get_plot(points)
        plot.initialize_plot()
        fig = plot.state
        self.assertEqual(len(fig.legend[0].items), 0)

    def test_points_non_numeric_size_warning(self):
        data = (np.arange(10), np.arange(10), list(map(chr, range(94,104))))
        points = Points(data, vdims=['z']).opts(plot=dict(size_index=2))
        with ParamLogStream() as log:
            plot = bokeh_renderer.get_plot(points)
        log_msg = log.stream.read()
        warning = ('%s: z dimension is not numeric, '
                   'cannot use to scale Points size.\n' % plot.name)
        self.assertEqual(log_msg, warning)

    def test_points_categorical_xaxis(self):
        points = Points((['A', 'B', 'C'], (1,2,3)))
        plot = bokeh_renderer.get_plot(points)
        x_range = plot.handles['x_range']
        self.assertIsInstance(x_range, FactorRange)
        self.assertEqual(x_range.factors, ['A', 'B', 'C'])

    def test_points_categorical_xaxis_mixed_type(self):
        points = Points(range(10))
        points2 = Points((['A', 'B', 'C', 1, 2.0], (1, 2, 3, 4, 5)))
        plot = bokeh_renderer.get_plot(points*points2)
        x_range = plot.handles['x_range']
        self.assertIsInstance(x_range, FactorRange)
        self.assertEqual(x_range.factors, list(map(str, range(10))) + ['A', 'B', 'C', '2.0'])

    def test_points_categorical_xaxis_invert_axes(self):
        points = Points((['A', 'B', 'C'], (1,2,3))).opts(plot=dict(invert_axes=True))
        plot = bokeh_renderer.get_plot(points)
        y_range = plot.handles['y_range']
        self.assertIsInstance(y_range, FactorRange)
        self.assertEqual(y_range.factors, ['A', 'B', 'C'])

    def test_points_overlay_categorical_xaxis(self):
        points = Points((['A', 'B', 'C'], (1,2,3)))
        points2 = Points((['B', 'C', 'D'], (1,2,3)))
        plot = bokeh_renderer.get_plot(points*points2)
        x_range = plot.handles['x_range']
        self.assertIsInstance(x_range, FactorRange)
        self.assertEqual(x_range.factors, ['A', 'B', 'C', 'D'])

    def test_points_overlay_categorical_xaxis_invert_axis(self):
        points = Points((['A', 'B', 'C'], (1,2,3))).opts(plot=dict(invert_xaxis=True))
        points2 = Points((['B', 'C', 'D'], (1,2,3)))
        plot = bokeh_renderer.get_plot(points*points2)
        x_range = plot.handles['x_range']
        self.assertIsInstance(x_range, FactorRange)
        self.assertEqual(x_range.factors, ['A', 'B', 'C', 'D'][::-1])

    def test_points_overlay_categorical_xaxis_invert_axes(self):
        points = Points((['A', 'B', 'C'], (1,2,3))).opts(plot=dict(invert_axes=True))
        points2 = Points((['B', 'C', 'D'], (1,2,3)))
        plot = bokeh_renderer.get_plot(points*points2)
        y_range = plot.handles['y_range']
        self.assertIsInstance(y_range, FactorRange)
        self.assertEqual(y_range.factors, ['A', 'B', 'C', 'D'])

    def test_points_padding_square(self):
        points = Points([1, 2, 3]).options(padding=0.2)
        plot = bokeh_renderer.get_plot(points)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, -0.2)
        self.assertEqual(x_range.end, 2.2)
        self.assertEqual(y_range.start, 0.8)
        self.assertEqual(y_range.end, 3.2)

    def test_points_padding_unequal(self):
        points = Points([1, 2, 3]).options(padding=(0.1, 0.2))
        plot = bokeh_renderer.get_plot(points)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, -0.1)
        self.assertEqual(x_range.end, 2.1)
        self.assertEqual(y_range.start, 0.8)
        self.assertEqual(y_range.end, 3.2)

    def test_points_padding_nonsquare(self):
        points = Points([1, 2, 3]).options(padding=0.2, width=600)
        plot = bokeh_renderer.get_plot(points)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, -0.1)
        self.assertEqual(x_range.end, 2.1)
        self.assertEqual(y_range.start, 0.8)
        self.assertEqual(y_range.end, 3.2)

    def test_points_padding_logx(self):
        points = Points([(1, 1), (2, 2), (3,3)]).options(padding=0.2, logx=True)
        plot = bokeh_renderer.get_plot(points)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 0.89595845984076228)
        self.assertEqual(x_range.end, 3.3483695221017129)
        self.assertEqual(y_range.start, 0.8)
        self.assertEqual(y_range.end, 3.2)
    
    def test_points_padding_logy(self):
        points = Points([1, 2, 3]).options(padding=0.2, logy=True)
        plot = bokeh_renderer.get_plot(points)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, -0.2)
        self.assertEqual(x_range.end, 2.2)
        self.assertEqual(y_range.start, 0.89595845984076228)
        self.assertEqual(y_range.end, 3.3483695221017129)
        
    def test_points_padding_datetime_square(self):
        points = Points([(np.datetime64('2016-04-0%d' % i), i) for i in range(1, 4)]).options(
            padding=0.2
        )
        plot = bokeh_renderer.get_plot(points)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, np.datetime64('2016-03-31T19:12:00.000000000'))
        self.assertEqual(x_range.end, np.datetime64('2016-04-03T04:48:00.000000000'))
        self.assertEqual(y_range.start, 0.8)
        self.assertEqual(y_range.end, 3.2)

    def test_points_padding_datetime_nonsquare(self):
        points = Points([(np.datetime64('2016-04-0%d' % i), i) for i in range(1, 4)]).options(
            padding=0.2, width=600
        )
        plot = bokeh_renderer.get_plot(points)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, np.datetime64('2016-03-31T21:36:00.000000000'))
        self.assertEqual(x_range.end, np.datetime64('2016-04-03T02:24:00.000000000'))
        self.assertEqual(y_range.start, 0.8)
        self.assertEqual(y_range.end, 3.2)

    def test_points_padding_hard_xrange(self):
        points = Points([1, 2, 3]).redim.range(x=(0, 3)).options(padding=0.2)
        plot = bokeh_renderer.get_plot(points)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 0)
        self.assertEqual(x_range.end, 3)
        self.assertEqual(y_range.start, 0.8)
        self.assertEqual(y_range.end, 3.2)

    def test_points_padding_soft_xrange(self):
        points = Points([1, 2, 3]).redim.soft_range(x=(0, 3)).options(padding=0.2)
        plot = bokeh_renderer.get_plot(points)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, -0.2)
        self.assertEqual(x_range.end, 3)
        self.assertEqual(y_range.start, 0.8)
        self.assertEqual(y_range.end, 3.2)
