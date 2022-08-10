import datetime as dt

from unittest import SkipTest
from collections import OrderedDict

import numpy as np

from holoviews.core import Dimension, DynamicMap, NdOverlay, HoloMap
from holoviews.core.util import dt_to_int
from holoviews.element import Curve, Image, Scatter, Labels
from holoviews.streams import Stream, PointDraw
from holoviews.plotting.util import process_cmap
from holoviews.util import render

from .test_plot import TestBokehPlot, bokeh_renderer
from ...utils import LoggingComparisonTestCase

import panel as pn

from bokeh.document import Document
from bokeh.models import tools
from bokeh.models import (FuncTickFormatter, PrintfTickFormatter,
                            NumeralTickFormatter, LogTicker,
                            LinearColorMapper, LogColorMapper)
from holoviews.plotting.bokeh.util import LooseVersion, bokeh_version


class TestElementPlot(LoggingComparisonTestCase, TestBokehPlot):

    def test_element_show_frame_disabled(self):
        curve = Curve(range(10)).opts(show_frame=False)
        plot = bokeh_renderer.get_plot(curve).state
        self.assertEqual(plot.outline_line_alpha, 0)

    def test_element_font_scaling(self):
        curve = Curve(range(10)).opts(fontscale=2, title='A title')
        plot = bokeh_renderer.get_plot(curve)
        fig = plot.state
        xaxis = plot.handles['xaxis']
        yaxis = plot.handles['yaxis']
        if bokeh_version > LooseVersion('2.2.3'):
            self.assertEqual(fig.title.text_font_size, '24pt')
        else:
            self.assertEqual(fig.title.text_font_size, {'value': '24pt'})
        if bokeh_version < LooseVersion('2.0.2'):
            self.assertEqual(xaxis.axis_label_text_font_size, '20pt')
            self.assertEqual(yaxis.axis_label_text_font_size, '20pt')
            self.assertEqual(xaxis.major_label_text_font_size, '16pt')
            self.assertEqual(yaxis.major_label_text_font_size, '16pt')
        else:
            self.assertEqual(xaxis.axis_label_text_font_size, '26px')
            self.assertEqual(yaxis.axis_label_text_font_size, '26px')
            self.assertEqual(xaxis.major_label_text_font_size, '22px')
            self.assertEqual(yaxis.major_label_text_font_size, '22px')

    def test_element_font_scaling_fontsize_override_common(self):
        curve = Curve(range(10)).opts(fontscale=2, fontsize='14pt', title='A title')
        plot = bokeh_renderer.get_plot(curve)
        fig = plot.state
        xaxis = plot.handles['xaxis']
        yaxis = plot.handles['yaxis']
        if bokeh_version > LooseVersion('2.2.3'):
            self.assertEqual(fig.title.text_font_size, '28pt')
        else:
            self.assertEqual(fig.title.text_font_size, {'value': '28pt'})
        self.assertEqual(xaxis.axis_label_text_font_size, '28pt')
        self.assertEqual(yaxis.axis_label_text_font_size, '28pt')
        if bokeh_version < LooseVersion('2.0.2'):
            self.assertEqual(xaxis.major_label_text_font_size, '16pt')
            self.assertEqual(yaxis.major_label_text_font_size, '16pt')
        else:
            self.assertEqual(xaxis.major_label_text_font_size, '22px')
            self.assertEqual(yaxis.major_label_text_font_size, '22px')

    def test_element_font_scaling_fontsize_override_specific(self):
        curve = Curve(range(10)).opts(
            fontscale=2, fontsize={'title': '100%', 'xlabel': '12pt', 'xticks': '1.2em'},
            title='A title')
        plot = bokeh_renderer.get_plot(curve)
        fig = plot.state
        xaxis = plot.handles['xaxis']
        yaxis = plot.handles['yaxis']
        if bokeh_version > LooseVersion('2.2.3'):
            self.assertEqual(fig.title.text_font_size, '200%')
        else:
            self.assertEqual(fig.title.text_font_size, {'value': '200%'})
        self.assertEqual(xaxis.axis_label_text_font_size, '24pt')
        self.assertEqual(xaxis.major_label_text_font_size, '2.4em')
        if bokeh_version < LooseVersion('2.0.2'):
            self.assertEqual(yaxis.axis_label_text_font_size, '20pt')
            self.assertEqual(yaxis.major_label_text_font_size, '16pt')
        else:
            self.assertEqual(yaxis.axis_label_text_font_size, '26px')
            self.assertEqual(yaxis.major_label_text_font_size, '22px')

    def test_element_xaxis_top(self):
        curve = Curve(range(10)).opts(xaxis='top')
        plot = bokeh_renderer.get_plot(curve)
        xaxis = plot.handles['xaxis']
        self.assertTrue(xaxis in plot.state.above)

    def test_element_xaxis_bare(self):
        curve = Curve(range(10)).opts(xaxis='bare')
        plot = bokeh_renderer.get_plot(curve)
        xaxis = plot.handles['xaxis']
        self.assertEqual(xaxis.axis_label_text_font_size, '0pt')
        self.assertEqual(xaxis.major_label_text_font_size, '0pt')
        self.assertEqual(xaxis.minor_tick_line_color, None)
        self.assertEqual(xaxis.major_tick_line_color, None)
        self.assertTrue(xaxis in plot.state.below)

    def test_element_xaxis_bottom_bare(self):
        curve = Curve(range(10)).opts(xaxis='bottom-bare')
        plot = bokeh_renderer.get_plot(curve)
        xaxis = plot.handles['xaxis']
        self.assertEqual(xaxis.axis_label_text_font_size, '0pt')
        self.assertEqual(xaxis.major_label_text_font_size, '0pt')
        self.assertEqual(xaxis.minor_tick_line_color, None)
        self.assertEqual(xaxis.major_tick_line_color, None)
        self.assertTrue(xaxis in plot.state.below)

    def test_element_xaxis_top_bare(self):
        curve = Curve(range(10)).opts(xaxis='top-bare')
        plot = bokeh_renderer.get_plot(curve)
        xaxis = plot.handles['xaxis']
        self.assertEqual(xaxis.axis_label_text_font_size, '0pt')
        self.assertEqual(xaxis.major_label_text_font_size, '0pt')
        self.assertEqual(xaxis.minor_tick_line_color, None)
        self.assertEqual(xaxis.major_tick_line_color, None)
        self.assertTrue(xaxis in plot.state.above)

    def test_element_yaxis_right(self):
        curve = Curve(range(10)).opts(yaxis='right')
        plot = bokeh_renderer.get_plot(curve)
        yaxis = plot.handles['yaxis']
        self.assertTrue(yaxis in plot.state.right)

    def test_element_yaxis_bare(self):
        curve = Curve(range(10)).opts(yaxis='bare')
        plot = bokeh_renderer.get_plot(curve)
        yaxis = plot.handles['yaxis']
        self.assertEqual(yaxis.axis_label_text_font_size, '0pt')
        self.assertEqual(yaxis.major_label_text_font_size, '0pt')
        self.assertEqual(yaxis.minor_tick_line_color, None)
        self.assertEqual(yaxis.major_tick_line_color, None)
        self.assertTrue(yaxis in plot.state.left)

    def test_element_yaxis_left_bare(self):
        curve = Curve(range(10)).opts(yaxis='left-bare')
        plot = bokeh_renderer.get_plot(curve)
        yaxis = plot.handles['yaxis']
        self.assertEqual(yaxis.axis_label_text_font_size, '0pt')
        self.assertEqual(yaxis.major_label_text_font_size, '0pt')
        self.assertEqual(yaxis.minor_tick_line_color, None)
        self.assertEqual(yaxis.major_tick_line_color, None)
        self.assertTrue(yaxis in plot.state.left)

    def test_element_yaxis_right_bare(self):
        curve = Curve(range(10)).opts(yaxis='right-bare')
        plot = bokeh_renderer.get_plot(curve)
        yaxis = plot.handles['yaxis']
        self.assertEqual(yaxis.axis_label_text_font_size, '0pt')
        self.assertEqual(yaxis.major_label_text_font_size, '0pt')
        self.assertEqual(yaxis.minor_tick_line_color, None)
        self.assertEqual(yaxis.major_tick_line_color, None)
        self.assertTrue(yaxis in plot.state.right)

    def test_element_title_format(self):
        title_str = ('Label: {label}, group: {group}, '
                     'dims: {dimensions}, type: {type}')
        e = Scatter(
            [],
            label='the_label',
            group='the_group',
        ).opts(title=title_str)
        title = 'Label: the_label, group: the_group, dims: , type: Scatter'
        self.assertEqual(render(e).title.text, title)

    def test_element_hooks(self):
        def hook(plot, element):
            plot.handles['plot'].title.text = 'Called'
        curve = Curve(range(10), label='Not Called').opts(hooks=[hook])
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.title.text, 'Called')

    def test_element_update_visible(self):
        checkbox = pn.widgets.Checkbox(value=True)
        scatter = Scatter([]).apply.opts(visible=checkbox)
        plot = bokeh_renderer.get_plot(scatter)
        assert plot.handles['glyph_renderer'].visible
        checkbox.value = False
        assert not plot.handles['glyph_renderer'].visible
        checkbox.value = True
        assert plot.handles['glyph_renderer'].visible
        
    def test_element_xformatter_string(self):
        curve = Curve(range(10)).opts(xformatter='%d')
        plot = bokeh_renderer.get_plot(curve)
        xaxis = plot.handles['xaxis']
        self.assertIsInstance(xaxis.formatter, PrintfTickFormatter)
        self.assertEqual(xaxis.formatter.format, '%d')

    def test_element_yformatter_string(self):
        curve = Curve(range(10)).opts(yformatter='%d')
        plot = bokeh_renderer.get_plot(curve)
        yaxis = plot.handles['yaxis']
        self.assertIsInstance(yaxis.formatter, PrintfTickFormatter)
        self.assertEqual(yaxis.formatter.format, '%d')

    def test_element_xformatter_function(self):
        try:
            import pscript # noqa
        except:
            raise SkipTest('Test requires pscript')
        def formatter(value):
            return str(value) + ' %'
        curve = Curve(range(10)).opts(xformatter=formatter)
        plot = bokeh_renderer.get_plot(curve)
        xaxis = plot.handles['xaxis']
        self.assertIsInstance(xaxis.formatter, FuncTickFormatter)

    def test_element_yformatter_function(self):
        try:
            import pscript # noqa
        except:
            raise SkipTest('Test requires pscript')
        def formatter(value):
            return str(value) + ' %'
        curve = Curve(range(10)).opts(yformatter=formatter)
        plot = bokeh_renderer.get_plot(curve)
        yaxis = plot.handles['yaxis']
        self.assertIsInstance(yaxis.formatter, FuncTickFormatter)

    def test_element_xformatter_instance(self):
        formatter = NumeralTickFormatter()
        curve = Curve(range(10)).opts(xformatter=formatter)
        plot = bokeh_renderer.get_plot(curve)
        xaxis = plot.handles['xaxis']
        self.assertIs(xaxis.formatter, formatter)

    def test_element_yformatter_instance(self):
        formatter = NumeralTickFormatter()
        curve = Curve(range(10)).opts(yformatter=formatter)
        plot = bokeh_renderer.get_plot(curve)
        yaxis = plot.handles['yaxis']
        self.assertIs(yaxis.formatter, formatter)

    def test_empty_element_visibility(self):
        curve = Curve([])
        plot = bokeh_renderer.get_plot(curve)
        self.assertTrue(plot.handles['glyph_renderer'].visible)

    def test_element_no_xaxis(self):
        curve = Curve(range(10)).opts(xaxis=None)
        plot = bokeh_renderer.get_plot(curve).state
        self.assertFalse(plot.xaxis[0].visible)

    def test_element_no_yaxis(self):
        curve = Curve(range(10)).opts(yaxis=None)
        plot = bokeh_renderer.get_plot(curve).state
        self.assertFalse(plot.yaxis[0].visible)

    def test_element_xrotation(self):
        curve = Curve(range(10)).opts(xrotation=90)
        plot = bokeh_renderer.get_plot(curve).state
        self.assertEqual(plot.xaxis[0].major_label_orientation, np.pi/2)

    def test_element_yrotation(self):
        curve = Curve(range(10)).opts(yrotation=90)
        plot = bokeh_renderer.get_plot(curve).state
        self.assertEqual(plot.yaxis[0].major_label_orientation, np.pi/2)

    def test_element_xlabel_override(self):
        curve = Curve(range(10)).opts(xlabel='custom x-label')
        plot = bokeh_renderer.get_plot(curve).state
        self.assertEqual(plot.xaxis[0].axis_label, 'custom x-label')

    def test_element_ylabel_override(self):
        curve = Curve(range(10)).opts(ylabel='custom y-label')
        plot = bokeh_renderer.get_plot(curve).state
        self.assertEqual(plot.yaxis[0].axis_label, 'custom y-label')

    def test_element_labelled_x_disabled(self):
        curve = Curve(range(10)).opts(labelled=['y'])
        plot = bokeh_renderer.get_plot(curve).state
        self.assertEqual(plot.xaxis[0].axis_label, '')
        self.assertEqual(plot.yaxis[0].axis_label, 'y')

    def test_element_labelled_y_disabled(self):
        curve = Curve(range(10)).opts(labelled=['x'])
        plot = bokeh_renderer.get_plot(curve).state
        self.assertEqual(plot.xaxis[0].axis_label, 'x')
        self.assertEqual(plot.yaxis[0].axis_label, '')

    def test_element_labelled_both_disabled(self):
        curve = Curve(range(10)).opts(labelled=[])
        plot = bokeh_renderer.get_plot(curve).state
        self.assertEqual(plot.xaxis[0].axis_label, '')
        self.assertEqual(plot.yaxis[0].axis_label, '')

    def test_static_source_optimization(self):
        global data
        data = np.ones((5, 5))
        img = Image(data)
        def get_img(test):
            global data
            data *= test
            return img
        stream = Stream.define(str('Test'), test=1)()
        dmap = DynamicMap(get_img, streams=[stream])
        plot = bokeh_renderer.get_plot(dmap, doc=Document())
        source = plot.handles['source']
        self.assertEqual(source.data['image'][0].mean(), 1)
        stream.event(test=2)
        self.assertTrue(plot.static_source)
        self.assertEqual(source.data['image'][0].mean(), 2)
        self.assertNotIn(source, plot.current_handles)

    def test_stream_cleanup(self):
        stream = Stream.define(str('Test'), test=1)()
        dmap = DynamicMap(lambda test: Curve([]), streams=[stream])
        plot = bokeh_renderer.get_plot(dmap)
        self.assertTrue(bool(stream._subscribers))
        plot.cleanup()
        self.assertFalse(bool(stream._subscribers))

    def test_element_formatter_xaxis(self):
        try:
            import pscript # noqa
        except:
            raise SkipTest('Test requires pscript')
        def formatter(x):
            return '%s' % x
        curve = Curve(range(10), kdims=[Dimension('x', value_format=formatter)])
        plot = bokeh_renderer.get_plot(curve).state
        self.assertIsInstance(plot.xaxis[0].formatter, FuncTickFormatter)

    def test_element_formatter_yaxis(self):
        try:
            import pscript # noqa
        except:
            raise SkipTest('Test requires pscript')
        def formatter(x):
            return '%s' % x
        curve = Curve(range(10), vdims=[Dimension('y', value_format=formatter)])
        plot = bokeh_renderer.get_plot(curve).state
        self.assertIsInstance(plot.yaxis[0].formatter, FuncTickFormatter)

    def test_element_xticks_datetime(self):
        dates = [(dt.datetime(2016, 1, i), i) for i in range(1, 4)]
        tick = dt.datetime(2016, 1, 1, 12)
        curve = Curve(dates).opts(xticks=[tick])
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.xaxis.ticker.ticks, [dt_to_int(tick, 'ms')])

    def test_element_xticks_datetime_label_override(self):
        dates = [(dt.datetime(2016, 1, i), i) for i in range(1, 4)]
        tick = dt.datetime(2016, 1, 1, 12)
        curve = Curve(dates).opts(xticks=[(tick, 'A')])
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.xaxis.ticker.ticks, [dt_to_int(tick, 'ms')])
        self.assertEqual(plot.state.xaxis.major_label_overrides, {dt_to_int(tick, 'ms'): 'A'})

    def test_element_grid_custom_xticker(self):
        curve = Curve([1, 2, 3]).opts(xticks=[0.5, 1.5], show_grid=True)
        plot = bokeh_renderer.get_plot(curve)
        self.assertIs(plot.state.xgrid[0].ticker, plot.state.xaxis[0].ticker)

    def test_element_grid_custom_yticker(self):
        curve = Curve([1, 2, 3]).opts(yticks=[0.5, 2.5], show_grid=True)
        plot = bokeh_renderer.get_plot(curve)
        self.assertIs(plot.state.ygrid[0].ticker, plot.state.yaxis[0].ticker)

    def test_element_grid_options(self):
        grid_style = {'grid_line_color': 'blue', 'grid_line_width': 1.5, 'ygrid_bounds': (0.3, 0.7),
                      'minor_xgrid_line_color': 'lightgray', 'xgrid_line_dash': [4, 4]}
        curve = Curve(range(10)).opts(show_grid=True, gridstyle=grid_style)
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.xgrid[0].grid_line_color, 'blue')
        self.assertEqual(plot.state.xgrid[0].grid_line_width, 1.5)
        self.assertEqual(plot.state.xgrid[0].grid_line_dash, [4, 4])
        self.assertEqual(plot.state.xgrid[0].minor_grid_line_color, 'lightgray')
        self.assertEqual(plot.state.ygrid[0].grid_line_color, 'blue')
        self.assertEqual(plot.state.ygrid[0].grid_line_width, 1.5)
        self.assertEqual(plot.state.ygrid[0].bounds, (0.3, 0.7))

    def test_change_cds_columns(self):
        lengths = {'a': 1, 'b': 2, 'c': 3}
        curve = DynamicMap(lambda a: Curve(range(lengths[a]), a), kdims=['a']).redim.values(a=['a', 'b', 'c'])
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(sorted(plot.handles['source'].data.keys()), ['a', 'y'])
        self.assertEqual(plot.state.xaxis[0].axis_label, 'a')
        plot.update(('b',))
        self.assertEqual(sorted(plot.handles['source'].data.keys()), ['b', 'y'])
        self.assertEqual(plot.state.xaxis[0].axis_label, 'b')

    def test_update_cds_columns(self):
        curve = DynamicMap(lambda a: Curve(range(10), a), kdims=['a']).redim.values(a=['a', 'b', 'c'])
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(sorted(plot.handles['source'].data.keys()), ['a', 'y'])
        self.assertEqual(plot.state.xaxis[0].axis_label, 'a')
        plot.update(('b',))
        self.assertEqual(sorted(plot.handles['source'].data.keys()), ['a', 'b', 'y'])
        self.assertEqual(plot.state.xaxis[0].axis_label, 'b')

    def test_categorical_axis_fontsize(self):
        curve = Curve([('A', 1), ('B', 2)]).opts(fontsize={'minor_xticks': '6pt', 'xticks': 18})
        plot = bokeh_renderer.get_plot(curve)
        xaxis = plot.handles['xaxis']
        self.assertEqual(xaxis.major_label_text_font_size, '6pt')
        self.assertEqual(xaxis.group_text_font_size, '18pt')

    def test_categorical_axis_fontsize_both(self):
        curve = Curve([('A', 1), ('B', 2)]).opts(fontsize={'xticks': 18})
        plot = bokeh_renderer.get_plot(curve)
        xaxis = plot.handles['xaxis']
        self.assertEqual(xaxis.major_label_text_font_size, '18pt')
        self.assertEqual(xaxis.group_text_font_size, '18pt')

    def test_cftime_transform_gregorian_no_warn(self):
        try:
            import cftime
        except:
            raise SkipTest('Test requires cftime library')
        gregorian_dates = [cftime.DatetimeGregorian(2000, 2, 28),
                           cftime.DatetimeGregorian(2000, 3, 1),
                           cftime.DatetimeGregorian(2000, 3, 2)]
        curve = Curve((gregorian_dates, [1, 2, 3]))
        plot = bokeh_renderer.get_plot(curve)
        xs = plot.handles['cds'].data['x']
        self.assertEqual(xs.astype('int64'),
                         np.array([951696000000, 951868800000, 951955200000]))

    def test_cftime_transform_noleap_warn(self):
        try:
            import cftime
        except:
            raise SkipTest('Test requires cftime library')
        gregorian_dates = [cftime.DatetimeNoLeap(2000, 2, 28),
                           cftime.DatetimeNoLeap(2000, 3, 1),
                           cftime.DatetimeNoLeap(2000, 3, 2)]
        curve = Curve((gregorian_dates, [1, 2, 3]))
        plot = bokeh_renderer.get_plot(curve)
        xs = plot.handles['cds'].data['x']
        self.assertEqual(xs.astype('int64'),
                         np.array([951696000000, 951868800000, 951955200000]))
        substr = (
            "Converting cftime.datetime from a non-standard calendar "
            "(noleap) to a standard calendar for plotting. This may "
            "lead to subtle errors in formatting dates, for accurate "
            "tick formatting switch to the matplotlib backend.")
        self.log_handler.assertEndsWith('WARNING', substr)

    def test_active_tools_drag(self):
        curve = Curve([1, 2, 3]).opts(active_tools=['box_zoom'])
        plot = bokeh_renderer.get_plot(curve)
        toolbar = plot.state.toolbar
        self.assertIsInstance(toolbar.active_drag, tools.BoxZoomTool)

    def test_active_tools_scroll(self):
        curve = Curve([1, 2, 3]).opts(active_tools=['wheel_zoom'])
        plot = bokeh_renderer.get_plot(curve)
        toolbar = plot.state.toolbar
        self.assertIsInstance(toolbar.active_scroll, tools.WheelZoomTool)

    def test_active_tools_tap(self):
        curve = Curve([1, 2, 3]).opts(active_tools=['tap'], tools=['tap'])
        plot = bokeh_renderer.get_plot(curve)
        toolbar = plot.state.toolbar
        self.assertIsInstance(toolbar.active_tap, tools.TapTool)

    def test_active_tools_draw_stream(self):
        scatter = Scatter([1, 2, 3]).opts(active_tools=['point_draw'])
        PointDraw(source=scatter)
        plot = bokeh_renderer.get_plot(scatter)
        toolbar = plot.state.toolbar
        self.assertIsInstance(toolbar.active_tap, tools.PointDrawTool)
        self.assertIsInstance(toolbar.active_drag, tools.PointDrawTool)

    def test_hover_tooltip_update(self):
        hmap = HoloMap({'a': Curve([1, 2, 3], vdims='a'), 'b': Curve([1, 2, 3], vdims='b')}).opts(
            tools=['hover'])
        plot = bokeh_renderer.get_plot(hmap)
        self.assertEqual(plot.handles['hover'].tooltips, [('x', '@{x}'), ('a', '@{a}')])
        plot.update(('b',))
        self.assertEqual(plot.handles['hover'].tooltips, [('x', '@{x}'), ('b', '@{b}')])

    def test_categorical_dimension_values(self):
        curve = Curve([('C', 1), ('B', 3)]).redim.values(x=['A', 'B', 'C'])
        plot = bokeh_renderer.get_plot(curve)
        x_range = plot.handles['x_range']
        self.assertEqual(x_range.factors, ['A', 'B', 'C'])

    def test_categorical_dimension_type(self):
        curve = Curve([]).redim.type(x=str)
        plot = bokeh_renderer.get_plot(curve)
        x_range = plot.handles['x_range']
        self.assertEqual(x_range.factors, [])

    #################################################################
    # Aspect tests
    #################################################################

    def test_element_aspect(self):
        curve = Curve([1, 2, 3]).opts(aspect=2)
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.plot_height, None)
        self.assertEqual(plot.state.plot_width, None)
        self.assertEqual(plot.state.frame_height, 300)
        self.assertEqual(plot.state.frame_width, 600)
        self.assertEqual(plot.state.aspect_ratio, None)

    def test_element_aspect_width(self):
        curve = Curve([1, 2, 3]).opts(aspect=2, width=400)
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.plot_height, None)
        self.assertEqual(plot.state.plot_width, None)
        self.assertEqual(plot.state.frame_height, 200)
        self.assertEqual(plot.state.frame_width, 400)
        self.assertEqual(plot.state.aspect_ratio, None)
        self.log_handler.assertContains('WARNING', "uses those values as frame_width/frame_height instead")

    def test_element_aspect_height(self):
        curve = Curve([1, 2, 3]).opts(aspect=2, height=400)
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.plot_height, None)
        self.assertEqual(plot.state.plot_width, None)
        self.assertEqual(plot.state.frame_height, 400)
        self.assertEqual(plot.state.frame_width, 800)
        self.assertEqual(plot.state.aspect_ratio, None)
        self.log_handler.assertContains('WARNING', "uses those values as frame_width/frame_height instead")

    def test_element_aspect_width_height(self):
        curve = Curve([1, 2, 3]).opts(aspect=2, height=400, width=400)
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.plot_height, 400)
        self.assertEqual(plot.state.plot_width, 400)
        self.assertEqual(plot.state.frame_height, None)
        self.assertEqual(plot.state.frame_width, None)
        self.assertEqual(plot.state.aspect_ratio, None)
        self.log_handler.assertContains('WARNING', "aspect value was ignored")

    def test_element_aspect_frame_width(self):
        curve = Curve([1, 2, 3]).opts(aspect=2, frame_width=400)
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.plot_height, None)
        self.assertEqual(plot.state.plot_width, None)
        self.assertEqual(plot.state.frame_height, 200)
        self.assertEqual(plot.state.frame_width, 400)
        self.assertEqual(plot.state.aspect_ratio, None)

    def test_element_aspect_frame_height(self):
        curve = Curve([1, 2, 3]).opts(aspect=2, frame_height=400)
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.plot_height, None)
        self.assertEqual(plot.state.plot_width, None)
        self.assertEqual(plot.state.frame_height, 400)
        self.assertEqual(plot.state.frame_width, 800)
        self.assertEqual(plot.state.aspect_ratio, None)

    def test_element_aspect_frame_width_frame_height(self):
        curve = Curve([1, 2, 3]).opts(aspect=2, frame_height=400, frame_width=400)
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.plot_height, None)
        self.assertEqual(plot.state.plot_width, None)
        self.assertEqual(plot.state.frame_height, 400)
        self.assertEqual(plot.state.frame_width, 400)
        self.assertEqual(plot.state.aspect_ratio, None)
        self.log_handler.assertContains('WARNING', "aspect value was ignored")

    def test_element_data_aspect(self):
        curve = Curve([0, 0.5, 1, 1.5]).opts(data_aspect=1.5)
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.plot_height, None)
        self.assertEqual(plot.state.plot_width, None)
        self.assertEqual(plot.state.frame_height, 300)
        self.assertEqual(plot.state.frame_width, 400)
        self.assertEqual(plot.state.aspect_scale, 1.5)

    def test_element_data_aspect_width(self):
        curve = Curve([0, 0.5, 1, 1.5]).opts(data_aspect=2, width=400)
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.plot_height, None)
        self.assertEqual(plot.state.plot_width, None)
        self.assertEqual(plot.state.frame_height, 400)
        self.assertEqual(plot.state.frame_width, 400)
        self.assertEqual(plot.state.aspect_scale, 2)
        self.log_handler.assertContains('WARNING', "uses those values as frame_width/frame_height instead")

    def test_element_data_aspect_height(self):
        curve = Curve([0, 0.5, 1, 1.5]).opts(data_aspect=2, height=400)
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.plot_height, None)
        self.assertEqual(plot.state.plot_width, None)
        self.assertEqual(plot.state.frame_height, 400)
        self.assertEqual(plot.state.frame_width, 400)
        self.assertEqual(plot.state.aspect_scale, 2)
        self.log_handler.assertContains('WARNING', "uses those values as frame_width/frame_height instead")

    def test_element_data_aspect_width_height(self):
        curve = Curve([0, 2, 3]).opts(data_aspect=2, height=400, width=400)
        plot = bokeh_renderer.get_plot(curve)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(plot.state.plot_height, 400)
        self.assertEqual(plot.state.plot_width, 400)
        self.assertEqual(plot.state.aspect_scale, 2)
        self.assertEqual(x_range.start, -2)
        self.assertEqual(x_range.end, 4)
        self.assertEqual(y_range.start, 0)
        self.assertEqual(y_range.end, 3)

    def test_element_data_aspect_frame_width(self):
        curve = Curve([1, 2, 3]).opts(data_aspect=2, frame_width=400)
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.plot_height, None)
        self.assertEqual(plot.state.plot_width, None)
        self.assertEqual(plot.state.frame_height, 800)
        self.assertEqual(plot.state.frame_width, 400)
        self.assertEqual(plot.state.aspect_scale, 2)

    def test_element_data_aspect_frame_height(self):
        curve = Curve([1, 2, 3]).opts(data_aspect=2, frame_height=400)
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.plot_height, None)
        self.assertEqual(plot.state.plot_width, None)
        self.assertEqual(plot.state.frame_height, 400)
        self.assertEqual(plot.state.frame_width, 200)
        self.assertEqual(plot.state.aspect_scale, 2)

    def test_element_data_aspect_frame_width_frame_height(self):
        curve = Curve([1, 2, 3]).opts(data_aspect=2, frame_height=400, frame_width=400)
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.plot_height, None)
        self.assertEqual(plot.state.plot_width, None)
        self.assertEqual(plot.state.frame_height, 400)
        self.assertEqual(plot.state.frame_width, 400)
        self.assertEqual(plot.state.aspect_scale, 2)

    #################################################################
    # Aspect tests
    #################################################################

    def test_element_responsive(self):
        curve = Curve([1, 2, 3]).opts(responsive=True)
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.plot_height, None)
        self.assertEqual(plot.state.plot_width, None)
        self.assertEqual(plot.state.frame_height, None)
        self.assertEqual(plot.state.frame_width, None)
        self.assertEqual(plot.state.sizing_mode, 'stretch_both')

    def test_element_width_responsive(self):
        curve = Curve([1, 2, 3]).opts(width=400, responsive=True)
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.plot_height, None)
        self.assertEqual(plot.state.plot_width, 400)
        self.assertEqual(plot.state.frame_height, None)
        self.assertEqual(plot.state.frame_width, None)
        self.assertEqual(plot.state.sizing_mode, 'stretch_height')

    def test_element_height_responsive(self):
        curve = Curve([1, 2, 3]).opts(height=400, responsive=True)
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.plot_height, 400)
        self.assertEqual(plot.state.plot_width, None)
        self.assertEqual(plot.state.frame_height, None)
        self.assertEqual(plot.state.frame_width, None)
        self.assertEqual(plot.state.sizing_mode, 'stretch_width')

    def test_element_frame_width_responsive(self):
        curve = Curve([1, 2, 3]).opts(frame_width=400, responsive=True)
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.plot_height, None)
        self.assertEqual(plot.state.plot_width, None)
        self.assertEqual(plot.state.frame_height, None)
        self.assertEqual(plot.state.frame_width, 400)
        self.assertEqual(plot.state.sizing_mode, 'stretch_height')

    def test_element_frame_height_responsive(self):
        curve = Curve([1, 2, 3]).opts(frame_height=400, responsive=True)
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.plot_height, None)
        self.assertEqual(plot.state.plot_width, None)
        self.assertEqual(plot.state.frame_height, 400)
        self.assertEqual(plot.state.frame_width, None)
        self.assertEqual(plot.state.sizing_mode, 'stretch_width')

    def test_element_aspect_responsive(self):
        curve = Curve([1, 2, 3]).opts(aspect=2, responsive=True)
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.plot_height, None)
        self.assertEqual(plot.state.plot_width, None)
        self.assertEqual(plot.state.frame_height, None)
        self.assertEqual(plot.state.frame_width, None)
        self.assertEqual(plot.state.sizing_mode, 'scale_both')

    def test_element_aspect_width_responsive(self):
        curve = Curve([1, 2, 3]).opts(aspect=2, width=400, responsive=True)
        plot = bokeh_renderer.get_plot(curve)
        self.log_handler.assertContains('WARNING', "responsive mode could not be enabled")
        self.assertEqual(plot.state.plot_height, None)
        self.assertEqual(plot.state.plot_width, None)
        self.assertEqual(plot.state.frame_height, 200)
        self.assertEqual(plot.state.frame_width, 400)
        self.assertEqual(plot.state.sizing_mode, 'fixed')
        self.log_handler.assertContains('WARNING', "uses those values as frame_width/frame_height instead")

    def test_element_aspect_height_responsive(self):
        curve = Curve([1, 2, 3]).opts(aspect=2, height=400, responsive=True)
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.frame_height, 400)
        self.assertEqual(plot.state.frame_width, 800)
        self.log_handler.assertContains('WARNING', "responsive mode could not be enabled")
        self.assertEqual(plot.state.plot_height, None)
        self.assertEqual(plot.state.plot_width, None)
        self.assertEqual(plot.state.sizing_mode, 'fixed')
        self.log_handler.assertContains('WARNING', "uses those values as frame_width/frame_height instead")

    def test_element_width_height_responsive(self):
        curve = Curve([1, 2, 3]).opts(height=400, width=400, responsive=True)
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.plot_height, 400)
        self.assertEqual(plot.state.plot_width, 400)
        self.log_handler.assertContains('WARNING', "responsive mode could not be enabled")
        self.assertEqual(plot.state.frame_height, None)
        self.assertEqual(plot.state.frame_width, None)
        self.assertEqual(plot.state.sizing_mode, 'fixed')

    def test_element_aspect_frame_width_responsive(self):
        curve = Curve([1, 2, 3]).opts(aspect=2, frame_width=400, responsive=True)
        plot = bokeh_renderer.get_plot(curve)
        self.log_handler.assertContains('WARNING', "responsive mode could not be enabled")
        self.assertEqual(plot.state.plot_height, None)
        self.assertEqual(plot.state.plot_width, None)
        self.assertEqual(plot.state.frame_height, 200)
        self.assertEqual(plot.state.frame_width, 400)
        self.assertEqual(plot.state.sizing_mode, 'fixed')

    def test_element_aspect_frame_height_responsive(self):
        curve = Curve([1, 2, 3]).opts(aspect=2, frame_height=400, responsive=True)
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.frame_height, 400)
        self.assertEqual(plot.state.frame_width, 800)
        self.assertEqual(plot.state.sizing_mode, 'fixed')
        self.log_handler.assertContains('WARNING', "responsive mode could not be enabled")

    def test_element_frame_width_frame_height_responsive(self):
        curve = Curve([1, 2, 3]).opts(frame_height=400, frame_width=400, responsive=True)
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.frame_height, 400)
        self.assertEqual(plot.state.frame_width, 400)
        self.assertEqual(plot.state.sizing_mode, 'fixed')
        self.log_handler.assertContains('WARNING', "responsive mode could not be enabled")

    def test_element_data_aspect_responsive(self):
        curve = Curve([0, 2]).opts(data_aspect=1, responsive=True)
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.aspect_ratio, 0.5)
        self.assertEqual(plot.state.aspect_scale, 1)
        self.assertEqual(plot.state.sizing_mode, 'scale_both')

    def test_element_data_aspect_and_aspect_responsive(self):
        curve = Curve([0, 2]).opts(data_aspect=1, aspect=2, responsive=True)
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.aspect_ratio, 0.5)
        self.assertEqual(plot.state.aspect_scale, 1)
        self.assertEqual(plot.state.sizing_mode, 'scale_both')
        x_range = plot.handles['x_range']
        y_range = plot.handles['y_range']
        self.assertEqual(x_range.start, 0)
        self.assertEqual(x_range.end, 1)
        self.assertEqual(y_range.start, 0)
        self.assertEqual(y_range.end, 2)

    def test_element_data_aspect_width_responsive(self):
        curve = Curve([0, 0.5, 1, 1.5]).opts(data_aspect=2, width=400, responsive=True)
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.frame_height, 400)
        self.assertEqual(plot.state.frame_width, 400)
        self.assertEqual(plot.state.sizing_mode, 'fixed')
        self.log_handler.assertContains('WARNING', "responsive mode could not be enabled")
        self.log_handler.assertContains('WARNING', "uses those values as frame_width/frame_height instead")

    def test_element_data_aspect_height_responsive(self):
        curve = Curve([0, 0.5, 1, 1.5]).opts(data_aspect=2, height=400, responsive=True)
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.frame_height, 400)
        self.assertEqual(plot.state.frame_width, 400)
        self.assertEqual(plot.state.sizing_mode, 'fixed')
        self.log_handler.assertContains('WARNING', "responsive mode could not be enabled")
        self.log_handler.assertContains('WARNING', "uses those values as frame_width/frame_height instead")

    def test_element_data_aspect_frame_width_responsive(self):
        curve = Curve([1, 2, 3]).opts(data_aspect=2, frame_width=400, responsive=True)
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.frame_height, 800)
        self.assertEqual(plot.state.frame_width, 400)
        self.assertEqual(plot.state.sizing_mode, 'fixed')
        self.log_handler.assertContains('WARNING', "responsive mode could not be enabled")

    def test_element_data_aspect_frame_height_responsive(self):
        curve = Curve([1, 2, 3]).opts(data_aspect=2, frame_height=400, responsive=True)
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.frame_height, 400)
        self.assertEqual(plot.state.frame_width, 200)
        self.assertEqual(plot.state.sizing_mode, 'fixed')
        self.log_handler.assertContains('WARNING', "responsive mode could not be enabled")



class TestColorbarPlot(LoggingComparisonTestCase, TestBokehPlot):

    def test_colormapper_symmetric(self):
        img = Image(np.array([[0, 1], [2, 3]])).opts(symmetric=True)
        plot = bokeh_renderer.get_plot(img)
        cmapper = plot.handles['color_mapper']
        self.assertEqual(cmapper.low, -3)
        self.assertEqual(cmapper.high, 3)

    def test_colormapper_logz_int_zero_bound(self):
        img = Image(np.array([[0, 1], [2, 3]])).opts(logz=True)
        plot = bokeh_renderer.get_plot(img)
        cmapper = plot.handles['color_mapper']
        self.assertEqual(cmapper.low, 1)
        self.assertEqual(cmapper.high, 3)

    def test_colormapper_logz_float_zero_bound(self):
        img = Image(np.array([[0, 1], [2, 3.]])).opts(logz=True)
        plot = bokeh_renderer.get_plot(img)
        cmapper = plot.handles['color_mapper']
        self.assertEqual(cmapper.low, 0)
        self.assertEqual(cmapper.high, 3)
        self.log_handler.assertContains('WARNING', "Log color mapper lower bound <= 0")

    def test_colormapper_color_levels(self):
        cmap = process_cmap('viridis', provider='bokeh')
        img = Image(np.array([[0, 1], [2, 3]])).opts(color_levels=5, cmap=cmap)
        plot = bokeh_renderer.get_plot(img)
        cmapper = plot.handles['color_mapper']
        self.assertEqual(len(cmapper.palette), 5)
        self.assertEqual(cmapper.palette, ['#440154', '#440255', '#440357', '#450558', '#45065A'])

    def test_colormapper_transparent_nan(self):
        img = Image(np.array([[0, 1], [2, 3]])).opts(clipping_colors={'NaN': 'transparent'})
        plot = bokeh_renderer.get_plot(img)
        cmapper = plot.handles['color_mapper']
        self.assertEqual(cmapper.nan_color, 'rgba(0, 0, 0, 0)')

    def test_colormapper_cnorm_linear(self):
        img = Image(np.array([[0, 1], [2, 3]])).opts(cnorm='linear')
        plot = bokeh_renderer.get_plot(img)
        cmapper = plot.handles['color_mapper']
        self.assertTrue(cmapper, LinearColorMapper)

    def test_colormapper_cnorm_log(self):
        img = Image(np.array([[0, 1], [2, 3]])).opts(cnorm='log')
        plot = bokeh_renderer.get_plot(img)
        cmapper = plot.handles['color_mapper']
        self.assertTrue(cmapper, LogColorMapper)

    def test_colormapper_cnorm_eqhist(self):
        try:
            from bokeh.models import EqHistColorMapper
        except:
            raise SkipTest("Option cnorm='eq_hist' requires EqHistColorMapper")
        img = Image(np.array([[0, 1], [2, 3]])).opts(cnorm='eq_hist')
        plot = bokeh_renderer.get_plot(img)
        cmapper = plot.handles['color_mapper']
        self.assertTrue(cmapper, EqHistColorMapper)


    def test_colormapper_min_max_colors(self):
        img = Image(np.array([[0, 1], [2, 3]])).opts(clipping_colors={'min': 'red', 'max': 'blue'})
        plot = bokeh_renderer.get_plot(img)
        cmapper = plot.handles['color_mapper']
        self.assertEqual(cmapper.low_color, 'red')
        self.assertEqual(cmapper.high_color, 'blue')

    def test_custom_colorbar_ticker(self):
        ticker = LogTicker()
        img = Image(np.array([[0, 1], [2, 3]])).opts(colorbar=True, colorbar_opts=dict(ticker=ticker))
        plot = bokeh_renderer.get_plot(img)
        colorbar = plot.handles['colorbar']
        self.assertIs(colorbar.ticker, ticker)

    def test_colorbar_fontsize_scaling(self):
        img = Image(np.array([[0, 1], [2, 3]])).opts(colorbar=True, fontscale=2)
        plot = bokeh_renderer.get_plot(img)
        colorbar = plot.handles['colorbar']
        if bokeh_version < LooseVersion('2.0.2'):
            self.assertEqual(colorbar.title_text_font_size, '20pt')
            self.assertEqual(colorbar.major_label_text_font_size, '16pt')
        else:
            self.assertEqual(colorbar.title_text_font_size, '26px')
            self.assertEqual(colorbar.major_label_text_font_size, '22px')

    def test_explicit_categorical_cmap_on_integer_data(self):
        explicit_mapping = OrderedDict([(0, 'blue'), (1, 'red'), (2, 'green'), (3, 'purple')])
        points = Scatter(([0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]), vdims=['y', 'Category']).opts(
            color_index='Category', cmap=explicit_mapping
        )
        plot = bokeh_renderer.get_plot(points)
        cmapper = plot.handles['color_mapper']
        cds = plot.handles['cds']
        self.assertEqual(cds.data['Category_str__'], ['0', '1', '2', '3'])
        self.assertEqual(cmapper.factors, ['0', '1', '2', '3'])
        self.assertEqual(cmapper.palette, ['blue', 'red', 'green', 'purple'])


class TestOverlayPlot(TestBokehPlot):

    def test_overlay_projection_clashing(self):
        overlay = Curve([]).opts(projection='polar') * Curve([]).opts(projection='custom')
        with self.assertRaises(Exception):
            bokeh_renderer.get_plot(overlay)

    def test_overlay_projection_propagates(self):
        overlay = Curve([]) * Curve([]).opts(projection='custom')
        plot = bokeh_renderer.get_plot(overlay)
        self.assertEqual([p.projection for p in plot.subplots.values()], ['custom', 'custom'])

    def test_overlay_propagates_batched(self):
        overlay = NdOverlay({
            i: Curve([1, 2, 3]).opts(yformatter='%.1f') for i in range(10)
        }).opts(yformatter='%.3f', legend_limit=1)
        plot = bokeh_renderer.get_plot(overlay)
        self.assertEqual(plot.state.yaxis.formatter.format, '%.3f')

    def test_overlay_gridstyle_applies(self):
        grid_style = {'grid_line_color': 'blue', 'grid_line_width': 2}
        overlay = (Scatter([(10,10)]).opts(gridstyle=grid_style, show_grid=True, size=20)
                   * Labels([(10, 10, 'A')]))
        plot = bokeh_renderer.get_plot(overlay)
        self.assertEqual(plot.state.xgrid[0].grid_line_color, 'blue')
        self.assertEqual(plot.state.xgrid[0].grid_line_width, 2)

    def test_ndoverlay_legend_muted(self):
        overlay = NdOverlay({i: Curve(np.random.randn(10).cumsum()) for i in range(5)}).opts(legend_muted=True)
        plot = bokeh_renderer.get_plot(overlay)
        for sp in plot.subplots.values():
            self.assertTrue(sp.handles['glyph_renderer'].muted)

    def test_overlay_legend_muted(self):
        overlay = (Curve(np.random.randn(10).cumsum(), label='A') *
                   Curve(np.random.randn(10).cumsum(), label='B')).opts(legend_muted=True)
        plot = bokeh_renderer.get_plot(overlay)
        for sp in plot.subplots.values():
            self.assertTrue(sp.handles['glyph_renderer'].muted)

    def test_overlay_legend_opts(self):
        overlay = (
            Curve(np.random.randn(10).cumsum(), label='A') *
            Curve(np.random.randn(10).cumsum(), label='B')
        ).opts(legend_opts={'background_fill_alpha': 0.5, 'background_fill_color': 'red'})
        plot = bokeh_renderer.get_plot(overlay)
        legend = plot.state.legend
        self.assertEqual(legend.background_fill_alpha, 0.5)
        self.assertEqual(legend.background_fill_color, 'red')

    def test_active_tools_drag(self):
        curve = Curve([1, 2, 3])
        scatter = Scatter([1, 2, 3])
        overlay = (scatter * curve).opts(active_tools=['box_zoom'])
        plot = bokeh_renderer.get_plot(overlay)
        toolbar = plot.state.toolbar
        self.assertIsInstance(toolbar.active_drag, tools.BoxZoomTool)

    def test_active_tools_scroll(self):
        curve = Curve([1, 2, 3])
        scatter = Scatter([1, 2, 3])
        overlay = (scatter * curve).opts(active_tools=['wheel_zoom'])
        plot = bokeh_renderer.get_plot(overlay)
        toolbar = plot.state.toolbar
        self.assertIsInstance(toolbar.active_scroll, tools.WheelZoomTool)

    def test_active_tools_tap(self):
        curve = Curve([1, 2, 3])
        scatter = Scatter([1, 2, 3]).opts(tools=['tap'])
        overlay = (scatter * curve).opts(active_tools=['tap'])
        plot = bokeh_renderer.get_plot(overlay)
        toolbar = plot.state.toolbar
        self.assertIsInstance(toolbar.active_tap, tools.TapTool)

    def test_active_tools_draw_stream(self):
        curve = Curve([1, 2, 3])
        scatter = Scatter([1, 2, 3]).opts(active_tools=['point_draw'])
        PointDraw(source=scatter)
        overlay = (scatter * curve)
        plot = bokeh_renderer.get_plot(overlay)
        toolbar = plot.state.toolbar
        self.assertIsInstance(toolbar.active_tap, tools.PointDrawTool)
        self.assertIsInstance(toolbar.active_drag, tools.PointDrawTool)

    def test_categorical_overlay_dimension_values(self):
        curve = Curve([('C', 1), ('B', 3)]).redim.values(x=['A', 'B', 'C'])
        scatter = Scatter([('A', 2)])
        plot = bokeh_renderer.get_plot(curve*scatter)
        x_range = plot.handles['x_range']
        self.assertEqual(x_range.factors, ['A', 'B', 'C'])

    def test_categorical_overlay_dimension_values_skip_factor(self):
        curve = Curve([('C', 1), ('B', 3)])
        scatter = Scatter([('A', 2)])
        plot = bokeh_renderer.get_plot((curve*scatter).redim.values(x=['A', 'C']))
        x_range = plot.handles['x_range']
        self.assertEqual(x_range.factors, ['A', 'C'])
