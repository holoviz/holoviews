import datetime as dt

import numpy as np
import pandas as pd

from holoviews.core import NdOverlay
from holoviews.core.options import Cycle
from holoviews.element import Points
from holoviews.streams import Stream
from holoviews.plotting.bokeh.util import property_to_dict

from .test_plot import TestBokehPlot, bokeh_renderer
from ..utils import ParamLogStream

from bokeh.models import FactorRange, LinearColorMapper, CategoricalColorMapper
from bokeh.models import Scatter


class TestPointPlot(TestBokehPlot):

    def test_points_colormapping(self):
        points = Points(np.random.rand(10, 4), vdims=['a', 'b']).opts(color_index=3)
        self._test_colormapping(points, 3)

    def test_points_colormapping_with_nonselection(self):
        opts = dict(color_index=3, nonselection_color='red')
        points = Points(np.random.rand(10, 4), vdims=['a', 'b']).opts(**opts)
        self._test_colormapping(points, 3)

    def test_points_colormapping_categorical(self):
        points = Points([(i, i*2, i*3, chr(65+i)) for i in range(10)],
                         vdims=['a', 'b']).opts(color_index='b')
        plot = bokeh_renderer.get_plot(points)
        plot.initialize_plot()
        cmapper = plot.handles['color_mapper']
        self.assertIsInstance(cmapper, CategoricalColorMapper)
        self.assertEqual(cmapper.factors, list(points['b']))

    def test_points_color_selection_nonselection(self):
        opts = dict(color='green', selection_color='red', nonselection_color='blue')
        points = Points([(i, i*2, i*3, chr(65+i)) for i in range(10)],
                         vdims=['a', 'b']).opts(**opts)
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
                         vdims=['a', 'b']).opts(**opts)
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
                         vdims=['a', 'b']).opts(**opts)
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
        opts = {'NdOverlay': dict(legend_limit=0),
                'Points': dict(size=Cycle(values=[1, 2]))}
        overlay = NdOverlay({i: Points([(i, j) for j in range(2)])
                             for i in range(2)}).opts(opts)
        plot = bokeh_renderer.get_plot(overlay).subplots[()]
        size = np.array([1, 1, 2, 2])
        color = np.array(['#30a2da', '#30a2da', '#fc4f30', '#fc4f30'],
                         dtype='<U7')
        self.assertEqual(plot.handles['source'].data['color'], color)
        self.assertEqual(plot.handles['source'].data['size'], size)

    def test_batched_points_line_color_and_color(self):
        opts = {'NdOverlay': dict(legend_limit=0),
                'Points': dict(line_color=Cycle(values=['red', 'blue']))}
        overlay = NdOverlay({i: Points([(i, j) for j in range(2)])
                             for i in range(2)}).opts(opts)
        plot = bokeh_renderer.get_plot(overlay).subplots[()]
        line_color = np.array(['red', 'red', 'blue', 'blue'])
        fill_color = np.array(['#30a2da', '#30a2da', '#fc4f30', '#fc4f30'],
                         dtype='<U7')
        self.assertEqual(plot.handles['source'].data['fill_color'], fill_color)
        self.assertEqual(plot.handles['source'].data['line_color'], line_color)

    def test_batched_points_alpha_and_color(self):
        opts = {'NdOverlay': dict(legend_limit=0),
                'Points': dict(alpha=Cycle(values=[0.5, 1]))}
        overlay = NdOverlay({i: Points([(i, j) for j in range(2)])
                             for i in range(2)}).opts(opts)
        plot = bokeh_renderer.get_plot(overlay).subplots[()]
        alpha = np.array([0.5, 0.5, 1., 1.])
        color = np.array(['#30a2da', '#30a2da', '#fc4f30', '#fc4f30'],
                         dtype='<U7')
        self.assertEqual(plot.handles['source'].data['alpha'], alpha)
        self.assertEqual(plot.handles['source'].data['color'], color)

    def test_batched_points_line_width_and_color(self):
        opts = {'NdOverlay': dict(legend_limit=0),
                'Points': dict(line_width=Cycle(values=[0.5, 1]))}
        overlay = NdOverlay({i: Points([(i, j) for j in range(2)])
                             for i in range(2)}).opts(opts)
        plot = bokeh_renderer.get_plot(overlay).subplots[()]
        line_width = np.array([0.5, 0.5, 1., 1.])
        color = np.array(['#30a2da', '#30a2da', '#fc4f30', '#fc4f30'],
                         dtype='<U7')
        self.assertEqual(plot.handles['source'].data['line_width'], line_width)
        self.assertEqual(plot.handles['source'].data['color'], color)

    def test_points_overlay_datetime_hover(self):
        obj = NdOverlay({i: Points((list(pd.date_range('2016-01-01', '2016-01-31')), range(31))) for i in range(5)},
                        kdims=['Test'])
        opts = {'Points': {'tools': ['hover']}}
        obj = obj.opts(opts)
        self._test_hover_info(obj, [('Test', '@{Test}'), ('x', '@{x}{%F %T}'), ('y', '@{y}')],
                              formatters={'@{x}': "datetime"})

    def test_points_overlay_hover_batched(self):
        obj = NdOverlay({i: Points(np.random.rand(10,2)) for i in range(5)},
                        kdims=['Test'])
        opts = {'Points': {'tools': ['hover']},
                'NdOverlay': {'legend_limit': 0}}
        obj = obj.opts(opts)
        self._test_hover_info(obj, [('Test', '@{Test}'), ('x', '@{x}'), ('y', '@{y}')])

    def test_points_overlay_hover(self):
        obj = NdOverlay({i: Points(np.random.rand(10,2)) for i in range(5)},
                        kdims=['Test'])
        opts = {'Points': {'tools': ['hover']},
                'NdOverlay': {'legend_limit': 0}}
        obj = obj.opts(opts)
        self._test_hover_info(obj, [('Test', '@{Test}'), ('x', '@{x}'),
                                    ('y', '@{y}')])

    def test_points_no_single_item_legend(self):
        points = Points([('A', 1), ('B', 2)], label='A')
        plot = bokeh_renderer.get_plot(points)
        plot.initialize_plot()
        fig = plot.state
        self.assertEqual(len(fig.legend), 0)

    def test_points_non_numeric_size_warning(self):
        data = (np.arange(10), np.arange(10), list(map(chr, range(94,104))))
        points = Points(data, vdims=['z']).opts(size_index=2)
        with ParamLogStream() as log:
            bokeh_renderer.get_plot(points)
        log_msg = log.stream.read()
        warning = (
            "The `size_index` parameter is deprecated in favor of size style mapping, e.g. "
            "`size=dim('size')**2`.\nz dimension is not numeric, cannot use to scale Points size.\n"
        )
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
        points = Points((['A', 'B', 'C'], (1,2,3))).opts(invert_axes=True)
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
        points = Points((['A', 'B', 'C'], (1,2,3))).opts(invert_xaxis=True)
        points2 = Points((['B', 'C', 'D'], (1,2,3)))
        plot = bokeh_renderer.get_plot(points*points2)
        x_range = plot.handles['x_range']
        self.assertIsInstance(x_range, FactorRange)
        self.assertEqual(x_range.factors, ['A', 'B', 'C', 'D'][::-1])

    def test_points_overlay_categorical_xaxis_invert_axes(self):
        points = Points((['A', 'B', 'C'], (1,2,3))).opts(invert_axes=True)
        points2 = Points((['B', 'C', 'D'], (1,2,3)))
        plot = bokeh_renderer.get_plot(points*points2)
        y_range = plot.handles['y_range']
        self.assertIsInstance(y_range, FactorRange)
        self.assertEqual(y_range.factors, ['A', 'B', 'C', 'D'])

    def test_points_padding_square(self):
        points = Points([1, 2, 3]).opts(padding=0.1)
        plot = bokeh_renderer.get_plot(points)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, -0.2)
        self.assertEqual(x_range.end, 2.2)
        self.assertEqual(y_range.start, 0.8)
        self.assertEqual(y_range.end, 3.2)

    def test_curve_padding_square_per_axis(self):
        curve = Points([1, 2, 3]).opts(padding=((0, 0.1), (0.1, 0.2)))
        plot = bokeh_renderer.get_plot(curve)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 0)
        self.assertEqual(x_range.end, 2.2)
        self.assertEqual(y_range.start, 0.8)
        self.assertEqual(y_range.end, 3.4)

    def test_points_padding_unequal(self):
        points = Points([1, 2, 3]).opts(padding=(0.05, 0.1))
        plot = bokeh_renderer.get_plot(points)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, -0.1)
        self.assertEqual(x_range.end, 2.1)
        self.assertEqual(y_range.start, 0.8)
        self.assertEqual(y_range.end, 3.2)

    def test_points_padding_nonsquare(self):
        points = Points([1, 2, 3]).opts(padding=0.1, width=600)
        plot = bokeh_renderer.get_plot(points)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, -0.1)
        self.assertEqual(x_range.end, 2.1)
        self.assertEqual(y_range.start, 0.8)
        self.assertEqual(y_range.end, 3.2)

    def test_points_padding_logx(self):
        points = Points([(1, 1), (2, 2), (3,3)]).opts(padding=0.1, logx=True)
        plot = bokeh_renderer.get_plot(points)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 0.89595845984076228)
        self.assertEqual(x_range.end, 3.3483695221017129)
        self.assertEqual(y_range.start, 0.8)
        self.assertEqual(y_range.end, 3.2)

    def test_points_padding_logy(self):
        points = Points([1, 2, 3]).opts(padding=0.1, logy=True)
        plot = bokeh_renderer.get_plot(points)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, -0.2)
        self.assertEqual(x_range.end, 2.2)
        self.assertEqual(y_range.start, 0.89595845984076228)
        self.assertEqual(y_range.end, 3.3483695221017129)

    def test_points_padding_datetime_square(self):
        points = Points([(np.datetime64('2016-04-0%d' % i), i) for i in range(1, 4)]).opts(
            padding=0.1
        )
        plot = bokeh_renderer.get_plot(points)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, np.datetime64('2016-03-31T19:12:00.000000000'))
        self.assertEqual(x_range.end, np.datetime64('2016-04-03T04:48:00.000000000'))
        self.assertEqual(y_range.start, 0.8)
        self.assertEqual(y_range.end, 3.2)

    def test_points_padding_datetime_nonsquare(self):
        points = Points([(np.datetime64('2016-04-0%d' % i), i) for i in range(1, 4)]).opts(
            padding=0.1, width=600
        )
        plot = bokeh_renderer.get_plot(points)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, np.datetime64('2016-03-31T21:36:00.000000000'))
        self.assertEqual(x_range.end, np.datetime64('2016-04-03T02:24:00.000000000'))
        self.assertEqual(y_range.start, 0.8)
        self.assertEqual(y_range.end, 3.2)

    def test_points_padding_hard_xrange(self):
        points = Points([1, 2, 3]).redim.range(x=(0, 3)).opts(padding=0.1)
        plot = bokeh_renderer.get_plot(points)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 0)
        self.assertEqual(x_range.end, 3)
        self.assertEqual(y_range.start, 0.8)
        self.assertEqual(y_range.end, 3.2)

    def test_points_padding_soft_xrange(self):
        points = Points([1, 2, 3]).redim.soft_range(x=(0, 3)).opts(padding=0.1)
        plot = bokeh_renderer.get_plot(points)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 0)
        self.assertEqual(x_range.end, 3)
        self.assertEqual(y_range.start, 0.8)
        self.assertEqual(y_range.end, 3.2)

    def test_points_datetime_hover(self):
        points = Points([(0, 1, dt.datetime(2017, 1, 1))], vdims='date').opts(tools=['hover'])
        plot = bokeh_renderer.get_plot(points)
        cds = plot.handles['cds']
        self.assertEqual(cds.data['date'].astype('datetime64'), np.array([1483228800000000000]))
        hover = plot.handles['hover']
        self.assertEqual(hover.tooltips, [('x', '@{x}'), ('y', '@{y}'), ('date', '@{date}{%F %T}')],
                         {'@{date}': "datetime"})

    def test_points_selected(self):
        points = Points([(0, 0), (1, 1), (2, 2)]).opts(selected=[0, 2])
        plot = bokeh_renderer.get_plot(points)
        cds = plot.handles['cds']
        self.assertEqual(cds.selected.indices, [0, 2])

    def test_points_update_selected(self):
        stream = Stream.define('Selected', selected=[])()
        points = Points([(0, 0), (1, 1), (2, 2)]).apply.opts(selected=stream.param.selected)
        plot = bokeh_renderer.get_plot(points)
        cds = plot.handles['cds']
        self.assertEqual(cds.selected.indices, [])
        stream.event(selected=[0, 2])
        self.assertEqual(cds.selected.indices, [0, 2])

    ###########################
    #    Styling mapping      #
    ###########################

    def test_point_color_op(self):
        points = Points([(0, 0, '#000'), (0, 1, '#F00'), (0, 2, '#0F0')],
                        vdims='color').opts(color='color')
        plot = bokeh_renderer.get_plot(points)
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        self.assertEqual(cds.data['color'], np.array(['#000', '#F00', '#0F0']))
        self.assertEqual(property_to_dict(glyph.fill_color), {'field': 'color'})
        self.assertEqual(property_to_dict(glyph.line_color), {'field': 'color'})

    def test_point_linear_color_op(self):
        points = Points([(0, 0, 0), (0, 1, 1), (0, 2, 2)],
                        vdims='color').opts(color='color')
        plot = bokeh_renderer.get_plot(points)
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        cmapper = plot.handles['color_color_mapper']
        self.assertTrue(cmapper, LinearColorMapper)
        self.assertEqual(cmapper.low, 0)
        self.assertEqual(cmapper.high, 2)
        self.assertEqual(cds.data['color'], np.array([0, 1, 2]))
        self.assertEqual(property_to_dict(glyph.fill_color), {'field': 'color', 'transform': cmapper})
        self.assertEqual(property_to_dict(glyph.line_color), {'field': 'color', 'transform': cmapper})

    def test_point_categorical_color_op(self):
        points = Points([(0, 0, 'A'), (0, 1, 'B'), (0, 2, 'C')],
                        vdims='color').opts(color='color')
        plot = bokeh_renderer.get_plot(points)
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        cmapper = plot.handles['color_color_mapper']
        self.assertTrue(cmapper, CategoricalColorMapper)
        self.assertEqual(cmapper.factors, ['A', 'B', 'C'])
        self.assertEqual(cds.data['color'], np.array(['A', 'B', 'C']))
        self.assertEqual(property_to_dict(glyph.fill_color), {'field': 'color', 'transform': cmapper})
        self.assertEqual(property_to_dict(glyph.line_color), {'field': 'color', 'transform': cmapper})

    def test_point_categorical_color_op_legend_with_labels(self):
        labels = {'A': 'A point', 'B': 'B point', 'C': 'C point'}
        points = Points([(0, 0, 'A'), (0, 1, 'B'), (0, 2, 'C')],
                        vdims='color').opts(color='color', show_legend=True, legend_labels=labels)
        plot = bokeh_renderer.get_plot(points)
        cds = plot.handles['cds']
        legend = plot.state.legend[0].items[0]
        assert property_to_dict(legend.label) == {'field': '_color_labels'}
        assert cds.data['_color_labels'] == ['A point', 'B point', 'C point']

    def test_point_categorical_dtype_color_op(self):
        df = pd.DataFrame(dict(sample_id=['subject 1', 'subject 2', 'subject 3', 'subject 4'], category=['apple', 'pear', 'apple', 'pear'], value=[1, 2, 3, 4]))
        df['category'] = df['category'].astype('category')
        points = Points(df, ['sample_id', 'value']).opts(color='category')
        plot = bokeh_renderer.get_plot(points)
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        cmapper = plot.handles['color_color_mapper']
        self.assertTrue(cmapper, CategoricalColorMapper)
        self.assertEqual(cmapper.factors, ['apple', 'pear'])
        self.assertEqual(np.asarray(cds.data['color']), np.array(['apple', 'pear', 'apple', 'pear']))
        self.assertEqual(property_to_dict(glyph.fill_color), {'field': 'color', 'transform': cmapper})
        self.assertEqual(property_to_dict(glyph.line_color), {'field': 'color', 'transform': cmapper})

    def test_point_explicit_cmap_color_op(self):
        points = Points([(0, 0), (0, 1), (0, 2)]).opts(
            color='y', cmap={0: 'red', 1: 'green', 2: 'blue'})
        plot = bokeh_renderer.get_plot(points)
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        cmapper = plot.handles['color_color_mapper']
        self.assertTrue(cmapper, CategoricalColorMapper)
        self.assertEqual(cmapper.factors, ['0', '1', '2'])
        self.assertEqual(cmapper.palette, ['red', 'green', 'blue'])
        self.assertEqual(cds.data['color_str__'], ['0', '1', '2'])
        self.assertEqual(property_to_dict(glyph.fill_color), {'field': 'color_str__', 'transform': cmapper})
        self.assertEqual(property_to_dict(glyph.line_color), {'field': 'color_str__', 'transform': cmapper})

    def test_point_line_color_op(self):
        points = Points([(0, 0, '#000'), (0, 1, '#F00'), (0, 2, '#0F0')],
                        vdims='color').opts(line_color='color')
        plot = bokeh_renderer.get_plot(points)
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        self.assertEqual(cds.data['line_color'], np.array(['#000', '#F00', '#0F0']))
        self.assertNotEqual(property_to_dict(glyph.fill_color), {'field': 'line_color'})
        self.assertEqual(property_to_dict(glyph.line_color), {'field': 'line_color'})

    def test_point_fill_color_op(self):
        points = Points([(0, 0, '#000'), (0, 1, '#F00'), (0, 2, '#0F0')],
                        vdims='color').opts(fill_color='color')
        plot = bokeh_renderer.get_plot(points)
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        self.assertEqual(cds.data['fill_color'], np.array(['#000', '#F00', '#0F0']))
        self.assertEqual(property_to_dict(glyph.fill_color), {'field': 'fill_color'})
        self.assertNotEqual(property_to_dict(glyph.line_color), {'field': 'fill_color'})

    def test_point_angle_op(self):
        points = Points([(0, 0, 0), (0, 1, 45), (0, 2, 90)],
                        vdims='angle').opts(angle='angle')
        plot = bokeh_renderer.get_plot(points)
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        self.assertEqual(cds.data['angle'], np.array([0, 0.785398, 1.570796]))
        self.assertEqual(property_to_dict(glyph.angle), {'field': 'angle'})

    def test_point_alpha_op(self):
        points = Points([(0, 0, 0), (0, 1, 0.2), (0, 2, 0.7)],
                        vdims='alpha').opts(alpha='alpha')
        plot = bokeh_renderer.get_plot(points)
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        self.assertEqual(cds.data['alpha'], np.array([0, 0.2, 0.7]))
        self.assertEqual(property_to_dict(glyph.fill_alpha), {'field': 'alpha'})

    def test_point_line_alpha_op(self):
        points = Points([(0, 0, 0), (0, 1, 0.2), (0, 2, 0.7)],
                        vdims='alpha').opts(line_alpha='alpha')
        plot = bokeh_renderer.get_plot(points)
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        self.assertEqual(cds.data['line_alpha'], np.array([0, 0.2, 0.7]))
        self.assertEqual(property_to_dict(glyph.line_alpha), {'field': 'line_alpha'})
        self.assertNotEqual(property_to_dict(glyph.fill_alpha), {'field': 'line_alpha'})

    def test_point_fill_alpha_op(self):
        points = Points([(0, 0, 0), (0, 1, 0.2), (0, 2, 0.7)],
                        vdims='alpha').opts(fill_alpha='alpha')
        plot = bokeh_renderer.get_plot(points)
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        self.assertEqual(cds.data['fill_alpha'], np.array([0, 0.2, 0.7]))
        self.assertNotEqual(property_to_dict(glyph.line_alpha), {'field': 'fill_alpha'})
        self.assertEqual(property_to_dict(glyph.fill_alpha), {'field': 'fill_alpha'})

    def test_point_size_op(self):
        points = Points([(0, 0, 1), (0, 1, 4), (0, 2, 8)],
                        vdims='size').opts(size='size')
        plot = bokeh_renderer.get_plot(points)
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        self.assertEqual(cds.data['size'], np.array([1, 4, 8]))
        self.assertEqual(property_to_dict(glyph.size), {'field': 'size'})

    def test_point_line_width_op(self):
        points = Points([(0, 0, 1), (0, 1, 4), (0, 2, 8)],
                        vdims='line_width').opts(line_width='line_width')
        plot = bokeh_renderer.get_plot(points)
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        self.assertEqual(cds.data['line_width'], np.array([1, 4, 8]))
        self.assertEqual(property_to_dict(glyph.line_width), {'field': 'line_width'})

    def test_point_marker_op(self):
        points = Points([(0, 0, 'circle'), (0, 1, 'triangle'), (0, 2, 'square')],
                        vdims='marker').opts(marker='marker')
        plot = bokeh_renderer.get_plot(points)
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        self.assertEqual(cds.data['marker'], np.array(['circle', 'triangle', 'square']))
        self.assertEqual(property_to_dict(glyph.marker), {'field': 'marker'})

    def test_op_ndoverlay_value(self):
        markers = ['circle', 'triangle']
        overlay = NdOverlay({marker: Points(np.arange(i)) for i, marker in enumerate(markers)}, 'Marker').opts('Points', marker='Marker')
        plot = bokeh_renderer.get_plot(overlay)
        for subplot, glyph_type, marker in zip(plot.subplots.values(), [Scatter, Scatter], markers):
            self.assertIsInstance(subplot.handles['glyph'], glyph_type)
            self.assertEqual(subplot.handles['glyph'].marker, marker)

    def test_point_color_index_color_clash(self):
        points = Points([(0, 0, 0), (0, 1, 1), (0, 2, 2)],
                        vdims='color').opts(color='color', color_index='color')
        with ParamLogStream() as log:
            bokeh_renderer.get_plot(points)
        log_msg = log.stream.read()
        warning = (
            "The `color_index` parameter is deprecated in favor of color style mapping, e.g. "
            "`color=dim('color')` or `line_color=dim('color')`\nCannot declare style mapping "
            "for 'color' option and declare a color_index; ignoring the color_index.\n"
        )
        self.assertEqual(log_msg, warning)

    def test_point_color_index_color_no_clash(self):
        points = Points([(0, 0, 0), (0, 1, 1), (0, 2, 2)],
                        vdims='color').opts(fill_color='color', color_index='color')
        plot = bokeh_renderer.get_plot(points)
        glyph = plot.handles['glyph']
        cmapper = plot.handles['fill_color_color_mapper']
        cmapper2 = plot.handles['color_mapper']
        self.assertEqual(property_to_dict(glyph.fill_color), {'field': 'fill_color', 'transform': cmapper})
        self.assertEqual(property_to_dict(glyph.line_color), {'field': 'color', 'transform': cmapper2})

    def test_point_size_index_size_clash(self):
        points = Points([(0, 0, 0), (0, 1, 1), (0, 2, 2)],
                        vdims='size').opts(size='size', size_index='size')
        with ParamLogStream() as log:
            bokeh_renderer.get_plot(points)
        log_msg = log.stream.read()
        warning = (
            "The `size_index` parameter is deprecated in favor of size style mapping, e.g. "
            "`size=dim('size')**2`.\nCannot declare style mapping for 'size' option and declare a "
            "size_index; ignoring the size_index.\n"
        )
        self.assertEqual(log_msg, warning)
