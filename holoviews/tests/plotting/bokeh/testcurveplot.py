import datetime as dt
from unittest import skipIf

import numpy as np

from holoviews.core import NdOverlay, HoloMap, DynamicMap
from holoviews.core.options import Cycle, Palette
from holoviews.core.util import pd, basestring
from holoviews.element import Curve
from holoviews.plotting.util import rgb2hex
from holoviews.streams import PointerX
from holoviews.util.transform import dim

from .testplot import TestBokehPlot, bokeh_renderer

try:
    from bokeh.models import FactorRange, FixedTicker
    from holoviews.plotting.bokeh.callbacks import Callback, PointerXCallback
except:
    pass

pd_skip = skipIf(pd is None, 'Pandas not available')


class TestCurvePlot(TestBokehPlot):

    def test_batched_curve_subscribers_correctly_attached(self):
        posx = PointerX()
        opts = {'NdOverlay': dict(plot=dict(legend_limit=0)),
                'Curve': dict(style=dict(line_color=Cycle(values=['red', 'blue'])))}
        overlay = DynamicMap(lambda x: NdOverlay({i: Curve([(i, j) for j in range(2)])
                                                  for i in range(2)}).opts(opts), kdims=[],
                             streams=[posx])
        plot = bokeh_renderer.get_plot(overlay)
        self.assertIn(plot.refresh, posx.subscribers)
        self.assertNotIn(list(plot.subplots.values())[0].refresh, posx.subscribers)

    def test_batched_curve_subscribers_correctly_linked(self):
        # Checks if a stream callback is created to link batched plot
        # to the stream
        posx = PointerX()
        opts = {'NdOverlay': dict(plot=dict(legend_limit=0)),
                'Curve': dict(style=dict(line_color=Cycle(values=['red', 'blue'])))}
        overlay = DynamicMap(lambda x: NdOverlay({i: Curve([(i, j) for j in range(2)])
                                                  for i in range(2)}).opts(opts), kdims=[],
                             streams=[posx])
        plot = bokeh_renderer.get_plot(overlay)
        self.assertEqual(len(Callback._callbacks), 1)
        key = list(Callback._callbacks.keys())[0]
        self.assertEqual(key, (id(plot.handles['plot']), id(PointerXCallback)))

    def test_cyclic_palette_curves(self):
        palette = Palette('Set1')
        opts = dict(color=palette)
        hmap = HoloMap({i: NdOverlay({j: Curve(np.random.rand(3)).opts(style=opts)
                                      for j in range(3)})
                        for i in range(3)})
        colors = palette[3].values
        plot = bokeh_renderer.get_plot(hmap)
        for subp, color in zip(plot.subplots.values(), colors):
            color = color if isinstance(color, basestring) else rgb2hex(color)
            self.assertEqual(subp.handles['glyph'].line_color, color)

    def test_batched_curve_line_color_and_color(self):
        opts = {'NdOverlay': dict(plot=dict(legend_limit=0)),
                'Curve': dict(style=dict(line_color=Cycle(values=['red', 'blue'])))}
        overlay = NdOverlay({i: Curve([(i, j) for j in range(2)])
                             for i in range(2)}).opts(opts)
        plot = bokeh_renderer.get_plot(overlay).subplots[()]
        line_color = ['red', 'blue']
        self.assertEqual(plot.handles['source'].data['line_color'], line_color)

    def test_batched_curve_alpha_and_color(self):
        opts = {'NdOverlay': dict(plot=dict(legend_limit=0)),
                'Curve': dict(style=dict(alpha=Cycle(values=[0.5, 1])))}
        overlay = NdOverlay({i: Curve([(i, j) for j in range(2)])
                             for i in range(2)}).opts(opts)
        plot = bokeh_renderer.get_plot(overlay).subplots[()]
        alpha = [0.5, 1.]
        color = ['#30a2da', '#fc4f30']
        self.assertEqual(plot.handles['source'].data['alpha'], alpha)
        self.assertEqual(plot.handles['source'].data['color'], color)

    def test_batched_curve_line_width_and_color(self):
        opts = {'NdOverlay': dict(plot=dict(legend_limit=0)),
                'Curve': dict(style=dict(line_width=Cycle(values=[0.5, 1])))}
        overlay = NdOverlay({i: Curve([(i, j) for j in range(2)])
                             for i in range(2)}).opts(opts)
        plot = bokeh_renderer.get_plot(overlay).subplots[()]
        line_width = [0.5, 1.]
        color = ['#30a2da', '#fc4f30']
        self.assertEqual(plot.handles['source'].data['line_width'], line_width)
        self.assertEqual(plot.handles['source'].data['color'], color)

    def test_curve_overlay_datetime_hover(self):
        obj = NdOverlay({i: Curve([(dt.datetime(2016, 1, j+1), j) for j in range(31)]) for i in range(5)},
                        kdims=['Test'])
        opts = {'Curve': {'tools': ['hover']}}
        obj = obj.opts(plot=opts)
        self._test_hover_info(obj, [('Test', '@{Test}'), ('x', '@{x}{%F %T}'), ('y', '@{y}')],
                              formatters={'@{x}': "datetime"})

    def test_curve_overlay_hover_batched(self):
        obj = NdOverlay({i: Curve(np.random.rand(10,2)) for i in range(5)},
                        kdims=['Test'])
        opts = {'Curve': {'tools': ['hover']},
                'NdOverlay': {'legend_limit': 0}}
        obj = obj.opts(plot=opts)
        self._test_hover_info(obj, [('Test', '@{Test}')], 'prev')

    def test_curve_overlay_hover(self):
        obj = NdOverlay({i: Curve(np.random.rand(10,2)) for i in range(5)},
                        kdims=['Test'])
        opts = {'Curve': {'tools': ['hover']}}
        obj = obj.opts(plot=opts)
        self._test_hover_info(obj, [('Test', '@{Test}'), ('x', '@{x}'), ('y', '@{y}')], 'nearest')

    def test_curve_categorical_xaxis(self):
        curve = Curve((['A', 'B', 'C'], [1,2,3]))
        plot = bokeh_renderer.get_plot(curve)
        x_range = plot.handles['x_range']
        self.assertIsInstance(x_range, FactorRange)
        self.assertEqual(x_range.factors, ['A', 'B', 'C'])

    def test_curve_categorical_xaxis_invert_axes(self):
        curve = Curve((['A', 'B', 'C'], (1,2,3))).opts(plot=dict(invert_axes=True))
        plot = bokeh_renderer.get_plot(curve)
        y_range = plot.handles['y_range']
        self.assertIsInstance(y_range, FactorRange)
        self.assertEqual(y_range.factors, ['A', 'B', 'C'])

    def test_curve_datetime64(self):
        dates = [np.datetime64(dt.datetime(2016,1,i)) for i in range(1, 11)]
        curve = Curve((dates, np.random.rand(10)))
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.handles['x_range'].start, np.datetime64(dt.datetime(2016, 1, 1)))
        self.assertEqual(plot.handles['x_range'].end, np.datetime64(dt.datetime(2016, 1, 10)))

    @pd_skip
    def test_curve_pandas_timestamps(self):
        dates = pd.date_range('2016-01-01', '2016-01-10', freq='D')
        curve = Curve((dates, np.random.rand(10)))
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.handles['x_range'].start, np.datetime64(dt.datetime(2016, 1, 1)))
        self.assertEqual(plot.handles['x_range'].end, np.datetime64(dt.datetime(2016, 1, 10)))

    def test_curve_dt_datetime(self):
        dates = [dt.datetime(2016,1,i) for i in range(1, 11)]
        curve = Curve((dates, np.random.rand(10)))
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.handles['x_range'].start, np.datetime64(dt.datetime(2016, 1, 1)))
        self.assertEqual(plot.handles['x_range'].end, np.datetime64(dt.datetime(2016, 1, 10)))

    def test_curve_heterogeneous_datetime_types_overlay(self):
        dates64 = [np.datetime64(dt.datetime(2016,1,i)) for i in range(1, 11)]
        dates = [dt.datetime(2016,1,i) for i in range(2, 12)]
        curve_dt64 = Curve((dates64, np.random.rand(10)))
        curve_dt = Curve((dates, np.random.rand(10)))
        plot = bokeh_renderer.get_plot(curve_dt*curve_dt64)
        self.assertEqual(plot.handles['x_range'].start, np.datetime64(dt.datetime(2016, 1, 1)))
        self.assertEqual(plot.handles['x_range'].end, np.datetime64(dt.datetime(2016, 1, 11)))

    @pd_skip
    def test_curve_heterogeneous_datetime_types_with_pd_overlay(self):
        dates_pd = pd.date_range('2016-01-04', '2016-01-13', freq='D')
        dates64 = [np.datetime64(dt.datetime(2016,1,i)) for i in range(1, 11)]
        dates = [dt.datetime(2016,1,i) for i in range(2, 12)]
        curve_dt64 = Curve((dates64, np.random.rand(10)))
        curve_dt = Curve((dates, np.random.rand(10)))
        curve_pd = Curve((dates_pd, np.random.rand(10)))
        plot = bokeh_renderer.get_plot(curve_dt*curve_dt64*curve_pd)
        self.assertEqual(plot.handles['x_range'].start, np.datetime64(dt.datetime(2016, 1, 1)))
        self.assertEqual(plot.handles['x_range'].end, np.datetime64(dt.datetime(2016, 1, 13)))

    def test_curve_fontsize_xlabel(self):
        curve = Curve(range(10)).opts(plot=dict(fontsize={'xlabel': '14pt'}))
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.handles['xaxis'].axis_label_text_font_size,
                         '14pt')

    def test_curve_fontsize_ylabel(self):
        curve = Curve(range(10)).opts(plot=dict(fontsize={'ylabel': '14pt'}))
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.handles['yaxis'].axis_label_text_font_size,
                         '14pt')

    def test_curve_fontsize_both_labels(self):
        curve = Curve(range(10)).opts(plot=dict(fontsize={'labels': '14pt'}))
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.handles['xaxis'].axis_label_text_font_size,
                         '14pt')
        self.assertEqual(plot.handles['yaxis'].axis_label_text_font_size,
                         '14pt')

    def test_curve_fontsize_xticks(self):
        curve = Curve(range(10)).opts(plot=dict(fontsize={'xticks': '14pt'}))
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.handles['xaxis'].major_label_text_font_size,
                         '14pt')

    def test_curve_fontsize_yticks(self):
        curve = Curve(range(10)).opts(plot=dict(fontsize={'yticks': '14pt'}))
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.handles['yaxis'].major_label_text_font_size,
                         '14pt')

    def test_curve_fontsize_both_ticks(self):
        curve = Curve(range(10)).opts(plot=dict(fontsize={'ticks': '14pt'}))
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.handles['xaxis'].major_label_text_font_size,
                         '14pt')
        self.assertEqual(plot.handles['yaxis'].major_label_text_font_size,
                         '14pt')

    def test_curve_fontsize_xticks_and_both_ticks(self):
        curve = Curve(range(10)).opts(plot=dict(fontsize={'xticks': '18pt', 'ticks': '14pt'}))
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.handles['xaxis'].major_label_text_font_size,
                         '18pt')
        self.assertEqual(plot.handles['yaxis'].major_label_text_font_size,
                         '14pt')

    def test_curve_xticks_list(self):
        curve = Curve(range(10)).opts(plot=dict(xticks=[0, 5, 10]))
        plot = bokeh_renderer.get_plot(curve).state
        self.assertIsInstance(plot.xaxis[0].ticker, FixedTicker)
        self.assertEqual(plot.xaxis[0].ticker.ticks, [0, 5, 10])

    def test_curve_xticks_list_of_tuples_xaxis(self):
        ticks = [(0, 'zero'), (5, 'five'), (10, 'ten')]
        curve = Curve(range(10)).opts(plot=dict(xticks=ticks))
        plot = bokeh_renderer.get_plot(curve).state
        self.assertIsInstance(plot.xaxis[0].ticker, FixedTicker)
        self.assertEqual(plot.xaxis[0].major_label_overrides, dict(ticks))

    def test_curve_yticks_list(self):
        curve = Curve(range(10)).opts(plot=dict(yticks=[0, 5, 10]))
        plot = bokeh_renderer.get_plot(curve).state
        self.assertIsInstance(plot.yaxis[0].ticker, FixedTicker)
        self.assertEqual(plot.yaxis[0].ticker.ticks, [0, 5, 10])

    def test_curve_xticks_list_of_tuples_yaxis(self):
        ticks = [(0, 'zero'), (5, 'five'), (10, 'ten')]
        curve = Curve(range(10)).opts(plot=dict(yticks=ticks))
        plot = bokeh_renderer.get_plot(curve).state
        self.assertIsInstance(plot.yaxis[0].ticker, FixedTicker)
        self.assertEqual(plot.yaxis[0].major_label_overrides, dict(ticks))

    def test_curve_padding_square(self):
        curve = Curve([1, 2, 3]).options(padding=0.1)
        plot = bokeh_renderer.get_plot(curve)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, -0.2)
        self.assertEqual(x_range.end, 2.2)
        self.assertEqual(y_range.start, 0.8)
        self.assertEqual(y_range.end, 3.2)

    def test_curve_padding_square_per_axis(self):
        curve = Curve([1, 2, 3]).options(padding=((0, 0.1), (0.1, 0.2)))
        plot = bokeh_renderer.get_plot(curve)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 0)
        self.assertEqual(x_range.end, 2.2)
        self.assertEqual(y_range.start, 0.8)
        self.assertEqual(y_range.end, 3.4)

    def test_curve_padding_hard_xrange(self):
        curve = Curve([1, 2, 3]).redim.range(x=(0, 3)).options(padding=0.1)
        plot = bokeh_renderer.get_plot(curve)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 0)
        self.assertEqual(x_range.end, 3)
        self.assertEqual(y_range.start, 0.8)
        self.assertEqual(y_range.end, 3.2)

    def test_curve_padding_soft_xrange(self):
        curve = Curve([1, 2, 3]).redim.soft_range(x=(0, 3)).options(padding=0.1)
        plot = bokeh_renderer.get_plot(curve)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 0)
        self.assertEqual(x_range.end, 3)
        self.assertEqual(y_range.start, 0.8)
        self.assertEqual(y_range.end, 3.2)

    def test_curve_padding_unequal(self):
        curve = Curve([1, 2, 3]).options(padding=(0.05, 0.1))
        plot = bokeh_renderer.get_plot(curve)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, -0.1)
        self.assertEqual(x_range.end, 2.1)
        self.assertEqual(y_range.start, 0.8)
        self.assertEqual(y_range.end, 3.2)

    def test_curve_padding_nonsquare(self):
        curve = Curve([1, 2, 3]).options(padding=0.1, width=600)
        plot = bokeh_renderer.get_plot(curve)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, -0.1)
        self.assertEqual(x_range.end, 2.1)
        self.assertEqual(y_range.start, 0.8)
        self.assertEqual(y_range.end, 3.2)

    def test_curve_padding_logx(self):
        curve = Curve([(1, 1), (2, 2), (3,3)]).options(padding=0.1, logx=True)
        plot = bokeh_renderer.get_plot(curve)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 0.89595845984076228)
        self.assertEqual(x_range.end, 3.3483695221017129)
        self.assertEqual(y_range.start, 0.8)
        self.assertEqual(y_range.end, 3.2)

    def test_curve_padding_logy(self):
        curve = Curve([1, 2, 3]).options(padding=0.1, logy=True)
        plot = bokeh_renderer.get_plot(curve)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, -0.2)
        self.assertEqual(x_range.end, 2.2)
        self.assertEqual(y_range.start, 0.89595845984076228)
        self.assertEqual(y_range.end, 3.3483695221017129)

    def test_curve_padding_datetime_square(self):
        curve = Curve([(np.datetime64('2016-04-0%d' % i), i) for i in range(1, 4)]).options(
            padding=0.1
        )
        plot = bokeh_renderer.get_plot(curve)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, np.datetime64('2016-03-31T19:12:00.000000000'))
        self.assertEqual(x_range.end, np.datetime64('2016-04-03T04:48:00.000000000'))
        self.assertEqual(y_range.start, 0.8)
        self.assertEqual(y_range.end, 3.2)

    def test_curve_padding_datetime_nonsquare(self):
        curve = Curve([(np.datetime64('2016-04-0%d' % i), i) for i in range(1, 4)]).options(
            padding=0.1, width=600
        )
        plot = bokeh_renderer.get_plot(curve)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, np.datetime64('2016-03-31T21:36:00.000000000'))
        self.assertEqual(x_range.end, np.datetime64('2016-04-03T02:24:00.000000000'))
        self.assertEqual(y_range.start, 0.8)
        self.assertEqual(y_range.end, 3.2)

    ###########################
    #    Styling mapping      #
    ###########################

    def test_curve_scalar_color_op(self):
        curve = Curve([(0, 0, 'red'), (0, 1, 'red'), (0, 2, 'red')],
                       vdims=['y', 'color']).options(color='color')
        plot = bokeh_renderer.get_plot(curve)
        glyph = plot.handles['glyph']
        self.assertEqual(glyph.line_color, 'red')

    def test_op_ndoverlay_color_value(self):
        colors = ['blue', 'red']
        overlay = NdOverlay({color: Curve(np.arange(i))
                             for i, color in enumerate(colors)},
                            'color').options('Curve', color='color')
        plot = bokeh_renderer.get_plot(overlay)
        for subplot, color in zip(plot.subplots.values(), colors):
            style = dict(subplot.style[subplot.cyclic_index])
            style = subplot._apply_transforms(subplot.current_frame, {}, {}, style)
            self.assertEqual(style['color'], color)

    def test_curve_color_op(self):
        curve = Curve([(0, 0, 'red'), (0, 1, 'blue'), (0, 2, 'red')],
                       vdims=['y', 'color']).options(color='color')
        with self.assertRaises(Exception):
            bokeh_renderer.get_plot(curve)

    def test_curve_alpha_op(self):
        curve = Curve([(0, 0, 0.1), (0, 1, 0.3), (0, 2, 1)],
                       vdims=['y', 'alpha']).options(alpha='alpha')
        with self.assertRaises(Exception):
            bokeh_renderer.get_plot(curve)

    def test_curve_line_width_op(self):
        curve = Curve([(0, 0, 0.1), (0, 1, 0.3), (0, 2, 1)],
                       vdims=['y', 'linewidth']).options(line_width='linewidth')
        with self.assertRaises(Exception):
            bokeh_renderer.get_plot(curve)

    def test_curve_style_mapping_ndoverlay_dimensions(self):
        ndoverlay = NdOverlay({
            (0, 'A'): Curve([1, 2, 0]), (0, 'B'): Curve([1, 2, 1]),
            (1, 'A'): Curve([1, 2, 2]), (1, 'B'): Curve([1, 2, 3])},
                              ['num', 'cat']
        ).opts({
            'Curve': dict(
                color=dim('num').categorize({0: 'red', 1: 'blue'}),
                line_dash=dim('cat').categorize({'A': 'solid', 'B': 'dashed'})
            )
        })
        plot = bokeh_renderer.get_plot(ndoverlay)
        for (num, cat), sp in plot.subplots.items():
            glyph = sp.handles['glyph']
            color = glyph.line_color
            if num == 0:
                self.assertEqual(color, 'red')
            else:
                self.assertEqual(color, 'blue')
            linestyle = glyph.line_dash
            if cat == 'A':
                self.assertEqual(linestyle, [])
            else:
                self.assertEqual(linestyle, [6])

    def test_curve_style_mapping_constant_value_dimensions(self):
        vdims = ['y', 'num', 'cat']
        ndoverlay = NdOverlay({
            0: Curve([(0, 1, 0, 'A'), (1, 0, 0, 'A')], vdims=vdims),
            1: Curve([(0, 1, 0, 'B'), (1, 1, 0, 'B')], vdims=vdims),
            2: Curve([(0, 1, 1, 'A'), (1, 2, 1, 'A')], vdims=vdims),
            3: Curve([(0, 1, 1, 'B'), (1, 3, 1, 'B')], vdims=vdims)}
        ).opts({
            'Curve': dict(
                color=dim('num').categorize({0: 'red', 1: 'blue'}),
                line_dash=dim('cat').categorize({'A': 'solid', 'B': 'dashed'})
            )
        })
        plot = bokeh_renderer.get_plot(ndoverlay)
        for k, sp in plot.subplots.items():
            glyph = sp.handles['glyph']
            color = glyph.line_color
            if ndoverlay[k].iloc[0, 2] == 0:
                self.assertEqual(color, 'red')
            else:
                self.assertEqual(color, 'blue')
            linestyle = glyph.line_dash
            if ndoverlay[k].iloc[0, 3] == 'A':
                self.assertEqual(linestyle, [])
            else:
                self.assertEqual(linestyle, [6])
