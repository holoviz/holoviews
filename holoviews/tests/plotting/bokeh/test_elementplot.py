import datetime as dt
from unittest import SkipTest

import numpy as np
import panel as pn
import param
import pytest
from bokeh.document import Document
from bokeh.models import (
    EqHistColorMapper,
    FixedTicker,
    LinearColorMapper,
    LogColorMapper,
    LogTicker,
    NumeralTickFormatter,
    PrintfTickFormatter,
    tools,
)

from holoviews import opts
from holoviews.core import Dimension, DynamicMap, HoloMap, NdOverlay, Overlay
from holoviews.core.util import dt_to_int
from holoviews.element import Curve, HeatMap, Image, Labels, Scatter
from holoviews.plotting.bokeh.util import BOKEH_GE_3_4_0, BOKEH_GE_3_6_0
from holoviews.plotting.util import process_cmap
from holoviews.streams import PointDraw, Stream
from holoviews.util import render

from ...utils import LoggingComparisonTestCase
from .test_plot import TestBokehPlot, bokeh_renderer


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
        self.assertEqual(fig.title.text_font_size, '24pt')
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
        self.assertEqual(fig.title.text_font_size, '28pt')
        self.assertEqual(xaxis.axis_label_text_font_size, '28pt')
        self.assertEqual(yaxis.axis_label_text_font_size, '28pt')
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
        self.assertEqual(fig.title.text_font_size, '200%')
        self.assertEqual(xaxis.axis_label_text_font_size, '24pt')
        self.assertEqual(xaxis.major_label_text_font_size, '2.4em')
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

    def test_element_yaxis_true(self):
        curve = Curve(range(10)).opts(yaxis=True)
        plot = bokeh_renderer.get_plot(curve)
        yaxis = plot.handles['yaxis']
        assert yaxis in plot.state.left

    def test_element_yaxis_false(self):
        curve = Curve(range(10)).opts(yaxis=False)
        plot = bokeh_renderer.get_plot(curve)
        yaxis = plot.handles['yaxis']
        assert yaxis in plot.state.left
        assert not yaxis.visible

    def test_element_yaxis_none(self):
        curve = Curve(range(10)).opts(yaxis=None)
        plot = bokeh_renderer.get_plot(curve)
        yaxis = plot.handles['yaxis']
        assert yaxis in plot.state.left
        assert not yaxis.visible

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
        data = np.ones((5, 5))
        img = Image(data)

        def get_img(test):
            get_img.data *= test
            return img

        get_img.data = data

        stream = Stream.define('Test', test=1)()
        dmap = DynamicMap(get_img, streams=[stream])
        plot = bokeh_renderer.get_plot(dmap, doc=Document())
        source = plot.handles['source']
        self.assertEqual(source.data['image'][0].mean(), 1)
        stream.event(test=2)
        self.assertTrue(plot.static_source)
        self.assertEqual(source.data['image'][0].mean(), 2)
        self.assertNotIn(source, plot.current_handles)

    def test_stream_cleanup(self):
        stream = Stream.define('Test', test=1)()
        dmap = DynamicMap(lambda test: Curve([]), streams=[stream])
        plot = bokeh_renderer.get_plot(dmap)
        self.assertTrue(bool(stream._subscribers))
        plot.cleanup()
        self.assertFalse(bool(stream._subscribers))

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
        except ImportError:
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
        except ImportError:
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

    def test_style_map_dimension_object(self):
        x = Dimension('x')
        y = Dimension('y')
        scatter = Scatter([1, 2, 3], kdims=[x], vdims=[y]).opts(color=x)
        self._test_colormapping(scatter, 'x', prefix='color_')

    #################################################################
    # Aspect tests
    #################################################################

    def test_element_aspect(self):
        curve = Curve([1, 2, 3]).opts(aspect=2)
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.height, None)
        self.assertEqual(plot.state.width, None)
        self.assertEqual(plot.state.frame_height, 300)
        self.assertEqual(plot.state.frame_width, 600)
        self.assertEqual(plot.state.aspect_ratio, None)

    def test_element_aspect_width(self):
        curve = Curve([1, 2, 3]).opts(aspect=2, width=400)
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.height, None)
        self.assertEqual(plot.state.width, None)
        self.assertEqual(plot.state.frame_height, 200)
        self.assertEqual(plot.state.frame_width, 400)
        self.assertEqual(plot.state.aspect_ratio, None)
        self.log_handler.assertContains('WARNING', "uses those values as frame_width/frame_height instead")

    def test_element_aspect_height(self):
        curve = Curve([1, 2, 3]).opts(aspect=2, height=400)
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.height, None)
        self.assertEqual(plot.state.width, None)
        self.assertEqual(plot.state.frame_height, 400)
        self.assertEqual(plot.state.frame_width, 800)
        self.assertEqual(plot.state.aspect_ratio, None)
        self.log_handler.assertContains('WARNING', "uses those values as frame_width/frame_height instead")

    def test_element_aspect_width_height(self):
        curve = Curve([1, 2, 3]).opts(aspect=2, height=400, width=400)
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.height, 400)
        self.assertEqual(plot.state.width, 400)
        self.assertEqual(plot.state.frame_height, None)
        self.assertEqual(plot.state.frame_width, None)
        self.assertEqual(plot.state.aspect_ratio, None)
        self.log_handler.assertContains('WARNING', "aspect value was ignored")

    def test_element_aspect_frame_width(self):
        curve = Curve([1, 2, 3]).opts(aspect=2, frame_width=400)
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.height, None)
        self.assertEqual(plot.state.width, None)
        self.assertEqual(plot.state.frame_height, 200)
        self.assertEqual(plot.state.frame_width, 400)
        self.assertEqual(plot.state.aspect_ratio, None)

    def test_element_aspect_frame_height(self):
        curve = Curve([1, 2, 3]).opts(aspect=2, frame_height=400)
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.height, None)
        self.assertEqual(plot.state.width, None)
        self.assertEqual(plot.state.frame_height, 400)
        self.assertEqual(plot.state.frame_width, 800)
        self.assertEqual(plot.state.aspect_ratio, None)

    def test_element_aspect_frame_width_frame_height(self):
        curve = Curve([1, 2, 3]).opts(aspect=2, frame_height=400, frame_width=400)
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.height, None)
        self.assertEqual(plot.state.width, None)
        self.assertEqual(plot.state.frame_height, 400)
        self.assertEqual(plot.state.frame_width, 400)
        self.assertEqual(plot.state.aspect_ratio, None)
        self.log_handler.assertContains('WARNING', "aspect value was ignored")

    def test_element_data_aspect(self):
        curve = Curve([0, 0.5, 1, 1.5]).opts(data_aspect=1.5)
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.height, None)
        self.assertEqual(plot.state.width, None)
        self.assertEqual(plot.state.frame_height, 300)
        self.assertEqual(plot.state.frame_width, 400)
        self.assertEqual(plot.state.aspect_scale, 1.5)

    def test_element_data_aspect_width(self):
        curve = Curve([0, 0.5, 1, 1.5]).opts(data_aspect=2, width=400)
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.height, None)
        self.assertEqual(plot.state.width, None)
        self.assertEqual(plot.state.frame_height, 400)
        self.assertEqual(plot.state.frame_width, 400)
        self.assertEqual(plot.state.aspect_scale, 2)
        self.log_handler.assertContains('WARNING', "uses those values as frame_width/frame_height instead")

    def test_element_data_aspect_height(self):
        curve = Curve([0, 0.5, 1, 1.5]).opts(data_aspect=2, height=400)
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.height, None)
        self.assertEqual(plot.state.width, None)
        self.assertEqual(plot.state.frame_height, 400)
        self.assertEqual(plot.state.frame_width, 400)
        self.assertEqual(plot.state.aspect_scale, 2)
        self.log_handler.assertContains('WARNING', "uses those values as frame_width/frame_height instead")

    def test_element_data_aspect_width_height(self):
        curve = Curve([0, 2, 3]).opts(data_aspect=2, height=400, width=400)
        plot = bokeh_renderer.get_plot(curve)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(plot.state.height, 400)
        self.assertEqual(plot.state.width, 400)
        self.assertEqual(plot.state.aspect_scale, 2)
        self.assertEqual(x_range.start, -2)
        self.assertEqual(x_range.end, 4)
        self.assertEqual(y_range.start, 0)
        self.assertEqual(y_range.end, 3)

    def test_element_data_aspect_frame_width(self):
        curve = Curve([1, 2, 3]).opts(data_aspect=2, frame_width=400)
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.height, None)
        self.assertEqual(plot.state.width, None)
        self.assertEqual(plot.state.frame_height, 800)
        self.assertEqual(plot.state.frame_width, 400)
        self.assertEqual(plot.state.aspect_scale, 2)

    def test_element_data_aspect_frame_height(self):
        curve = Curve([1, 2, 3]).opts(data_aspect=2, frame_height=400)
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.height, None)
        self.assertEqual(plot.state.width, None)
        self.assertEqual(plot.state.frame_height, 400)
        self.assertEqual(plot.state.frame_width, 200)
        self.assertEqual(plot.state.aspect_scale, 2)

    def test_element_data_aspect_frame_width_frame_height(self):
        curve = Curve([1, 2, 3]).opts(data_aspect=2, frame_height=400, frame_width=400)
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.height, None)
        self.assertEqual(plot.state.width, None)
        self.assertEqual(plot.state.frame_height, 400)
        self.assertEqual(plot.state.frame_width, 400)
        self.assertEqual(plot.state.aspect_scale, 2)

    #################################################################
    # Aspect tests
    #################################################################

    def test_element_responsive(self):
        curve = Curve([1, 2, 3]).opts(responsive=True)
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.height, None)
        self.assertEqual(plot.state.width, None)
        self.assertEqual(plot.state.frame_height, None)
        self.assertEqual(plot.state.frame_width, None)
        self.assertEqual(plot.state.sizing_mode, 'stretch_both')

    def test_element_width_responsive(self):
        curve = Curve([1, 2, 3]).opts(width=400, responsive=True)
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.height, None)
        self.assertEqual(plot.state.width, 400)
        self.assertEqual(plot.state.frame_height, None)
        self.assertEqual(plot.state.frame_width, None)
        self.assertEqual(plot.state.sizing_mode, 'stretch_height')

    def test_element_height_responsive(self):
        curve = Curve([1, 2, 3]).opts(height=400, responsive=True)
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.height, 400)
        self.assertEqual(plot.state.width, None)
        self.assertEqual(plot.state.frame_height, None)
        self.assertEqual(plot.state.frame_width, None)
        self.assertEqual(plot.state.sizing_mode, 'stretch_width')

    def test_element_frame_width_responsive(self):
        curve = Curve([1, 2, 3]).opts(frame_width=400, responsive=True)
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.height, None)
        self.assertEqual(plot.state.width, None)
        self.assertEqual(plot.state.frame_height, None)
        self.assertEqual(plot.state.frame_width, 400)
        self.assertEqual(plot.state.sizing_mode, 'stretch_height')

    def test_element_frame_height_responsive(self):
        curve = Curve([1, 2, 3]).opts(frame_height=400, responsive=True)
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.height, None)
        self.assertEqual(plot.state.width, None)
        self.assertEqual(plot.state.frame_height, 400)
        self.assertEqual(plot.state.frame_width, None)
        self.assertEqual(plot.state.sizing_mode, 'stretch_width')

    def test_element_aspect_responsive(self):
        curve = Curve([1, 2, 3]).opts(aspect=2, responsive=True)
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.height, None)
        self.assertEqual(plot.state.width, None)
        self.assertEqual(plot.state.frame_height, None)
        self.assertEqual(plot.state.frame_width, None)
        self.assertEqual(plot.state.sizing_mode, 'scale_both')

    def test_element_aspect_width_responsive(self):
        curve = Curve([1, 2, 3]).opts(aspect=2, width=400, responsive=True)
        plot = bokeh_renderer.get_plot(curve)
        self.log_handler.assertContains('WARNING', "responsive mode could not be enabled")
        self.assertEqual(plot.state.height, None)
        self.assertEqual(plot.state.width, None)
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
        self.assertEqual(plot.state.height, None)
        self.assertEqual(plot.state.width, None)
        self.assertEqual(plot.state.sizing_mode, 'fixed')
        self.log_handler.assertContains('WARNING', "uses those values as frame_width/frame_height instead")

    def test_element_width_height_responsive(self):
        curve = Curve([1, 2, 3]).opts(height=400, width=400, responsive=True)
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.state.height, 400)
        self.assertEqual(plot.state.width, 400)
        self.log_handler.assertContains('WARNING', "responsive mode could not be enabled")
        self.assertEqual(plot.state.frame_height, None)
        self.assertEqual(plot.state.frame_width, None)
        self.assertEqual(plot.state.sizing_mode, 'fixed')

    def test_element_aspect_frame_width_responsive(self):
        curve = Curve([1, 2, 3]).opts(aspect=2, frame_width=400, responsive=True)
        plot = bokeh_renderer.get_plot(curve)
        self.log_handler.assertContains('WARNING', "responsive mode could not be enabled")
        self.assertEqual(plot.state.height, None)
        self.assertEqual(plot.state.width, None)
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

    #################################################################
    # Custom opts tests
    #################################################################

    def test_element_backend_opts(self):
        heat_map = HeatMap([(1, 2, 3), (2, 3, 4), (3, 4, 5)]).opts(
            colorbar=True,
            backend_opts={
                "colorbar.title": "Testing",
                "colorbar.ticker": FixedTicker(ticks=(3.5, 5)),
                "colorbar.major_label_overrides": {3.5: "A", 5: "B"},
            },
        )
        plot = bokeh_renderer.get_plot(heat_map)
        colorbar = plot.handles['colorbar']
        self.assertEqual(colorbar.title, "Testing")
        self.assertEqual(colorbar.ticker.ticks, (3.5, 5))
        self.assertEqual(colorbar.major_label_overrides, {3.5: "A", 5: "B"})

    def test_element_backend_opts_alias(self):
        heat_map = HeatMap([(1, 2, 3), (2, 3, 4), (3, 4, 5)]).opts(
            colorbar=True,
            backend_opts={
                "cbar.title": "Testing",
                "cbar.ticker": FixedTicker(ticks=(3.5, 5)),
                "cbar.major_label_overrides": {3.5: "A", 5: "B"},
            },
        )
        plot = bokeh_renderer.get_plot(heat_map)
        colorbar = plot.handles['colorbar']
        self.assertEqual(colorbar.title, "Testing")
        self.assertEqual(colorbar.ticker.ticks, (3.5, 5))
        self.assertEqual(colorbar.major_label_overrides, {3.5: "A", 5: "B"})

    def test_element_backend_opts_two_accessors(self):
        heat_map = HeatMap([(1, 2, 3), (2, 3, 4), (3, 4, 5)]).opts(
            colorbar=True, backend_opts={"colorbar": "Testing"},
        )
        bokeh_renderer.get_plot(heat_map)
        self.log_handler.assertContains(
            "WARNING", "Custom option 'colorbar' expects at least two"
        )

    def test_element_backend_opts_model_not_resolved(self):
        heat_map = HeatMap([(1, 2, 3), (2, 3, 4), (3, 4, 5)]).opts(
            colorbar=True, backend_opts={"cb.title": "Testing"},
        )
        bokeh_renderer.get_plot(heat_map)
        self.log_handler.assertContains(
            "WARNING", "cb model could not be"
        )


@pytest.mark.usefixtures("bokeh_backend")
@pytest.mark.skipif(not BOKEH_GE_3_4_0, reason="requires Bokeh >= 3.4")
class TestScalebarPlot:

    def get_scalebar(self, element):
        plot = bokeh_renderer.get_plot(element)
        return plot.handles.get('scalebar')

    def test_scalebar(self):
        curve = Curve([1, 2, 3]).opts(scalebar=True)
        scalebar = self.get_scalebar(curve)
        assert scalebar.visible
        assert scalebar.location == 'bottom_right'
        assert scalebar.background_fill_alpha == 0.8
        assert scalebar.unit == "m"

    def test_no_scalebar(self):
        curve = Curve([1, 2, 3])
        scalebar = self.get_scalebar(curve)
        assert scalebar is None

    def test_scalebar_unit(self):
        curve = Curve([1, 2, 3]).opts(scalebar=True, scalebar_unit='cm')
        scalebar = self.get_scalebar(curve)
        assert scalebar.visible
        assert scalebar.unit == "cm"

    def test_dim_unit(self):
        dim = Dimension("dim", unit="cm")
        curve = Curve([1, 2, 3], kdims=dim).opts(scalebar=True)
        scalebar = self.get_scalebar(curve)
        assert scalebar.visible
        assert scalebar.unit == "cm"

    def test_scalebar_custom_opts(self):
        curve = Curve([1, 2, 3]).opts(scalebar=True, scalebar_opts={'background_fill_alpha': 1})
        scalebar = self.get_scalebar(curve)
        assert scalebar.visible
        assert scalebar.background_fill_alpha == 1

    def test_scalebar_label(self):
        curve = Curve([1, 2, 3]).opts(scalebar=True, scalebar_label='Test')
        scalebar = self.get_scalebar(curve)
        assert scalebar.visible
        assert scalebar.label == 'Test'

    def test_scalebar_icon(self):
        curve = Curve([1, 2, 3]).opts(scalebar=True)
        plot = bokeh_renderer.get_plot(curve)
        toolbar = plot.handles['plot'].toolbar
        scalebar_icon = [tool for tool in toolbar.tools if tool.description == "Toggle ScaleBar"]
        assert len(scalebar_icon) == 1

    def test_scalebar_no_icon(self):
        curve = Curve([1, 2, 3]).opts(scalebar=False)
        plot = bokeh_renderer.get_plot(curve)
        toolbar = plot.handles['plot'].toolbar
        scalebar_icon = [tool for tool in toolbar.tools if tool.description == "Toggle ScaleBar"]
        assert len(scalebar_icon) == 0

    def test_scalebar_icon_multiple_overlay(self):
        curve1 = Curve([1, 2, 3]).opts(scalebar=True)
        curve2 = Curve([1, 2, 3]).opts(scalebar=True)
        plot = bokeh_renderer.get_plot(curve1 * curve2)
        toolbar = plot.handles['plot'].toolbar
        scalebar_icon = [tool for tool in toolbar.tools if tool.description == "Toggle ScaleBar"]
        assert len(scalebar_icon) == 1

    @pytest.mark.skipif(not BOKEH_GE_3_6_0, reason="requires Bokeh >= 3.6")
    @pytest.mark.parametrize("enabled1", [True, False])
    @pytest.mark.parametrize("enabled2", [True, False])
    @pytest.mark.parametrize("enabled3", [True, False])
    def test_scalebar_with_subcoordinate_y(self, enabled1, enabled2, enabled3):
        from bokeh.models import ScaleBar

        enabled = [enabled1, enabled2, enabled3]
        curve1 = Curve([1, 2, 3], label='c1').opts(scalebar=enabled1, subcoordinate_y=True)
        curve2 = Curve([1, 2, 3], label='c2').opts(scalebar=enabled2, subcoordinate_y=True)
        curve3 = Curve([1, 2, 3], label='c3').opts(scalebar=enabled3, subcoordinate_y=True)
        curves = curve1 * curve2 * curve3

        plot = bokeh_renderer.get_plot(curves).handles["plot"]
        coordinates = [r.coordinates for r in plot.renderers][::-1]
        sb = (c for c in plot.center if isinstance(c, ScaleBar))
        scalebars = [next(sb) if e else None for e in enabled]
        assert sum(map(bool, scalebars)) == sum(enabled)

        for coordinate, scalebar, idx in zip(coordinates, scalebars, "123", strict=None):
            assert coordinate.y_source.name == f"c{idx}"
            if scalebar is None:
                continue
            assert coordinate.y_source is scalebar.range


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
        self.assertEqual(colorbar.title_text_font_size, '26px')
        self.assertEqual(colorbar.major_label_text_font_size, '22px')

    def test_explicit_categorical_cmap_on_integer_data(self):
        explicit_mapping = dict([(0, 'blue'), (1, 'red'), (2, 'green'), (3, 'purple')])
        points = Scatter(([0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]), vdims=['y', 'Category']).opts(
            color_index='Category', cmap=explicit_mapping
        )
        plot = bokeh_renderer.get_plot(points)
        cmapper = plot.handles['color_mapper']
        cds = plot.handles['cds']
        self.assertEqual(cds.data['Category_str__'], ['0', '1', '2', '3'])
        self.assertEqual(cmapper.factors, ['0', '1', '2', '3'])
        self.assertEqual(cmapper.palette, ['blue', 'red', 'green', 'purple'])

    def test_cticks_int(self):
        img = Image(np.array([[0, 1], [2, 3]])).opts(cticks=3, colorbar=True)
        plot = bokeh_renderer.get_plot(img)
        colorbar = plot.handles["colorbar"]
        ticker = colorbar.ticker
        assert ticker.desired_num_ticks == 3

    def test_cticks_list(self):
        img = Image(np.array([[0, 1], [2, 3]])).opts(cticks=[1, 2], colorbar=True)
        plot = bokeh_renderer.get_plot(img)
        colorbar = plot.handles["colorbar"]
        ticker = colorbar.ticker
        assert ticker.ticks == [1, 2]

    def test_cticks_tuple(self):
        img = Image(np.array([[0, 1], [2, 3]])).opts(cticks=(1, 2), colorbar=True)
        plot = bokeh_renderer.get_plot(img)
        colorbar = plot.handles["colorbar"]
        ticker = colorbar.ticker
        assert ticker.ticks == (1, 2)

    def test_cticks_np_array(self):
        img = Image(np.array([[0, 1], [2, 3]])).opts(cticks=np.array([1, 2]), colorbar=True)
        plot = bokeh_renderer.get_plot(img)
        colorbar = plot.handles["colorbar"]
        ticker = colorbar.ticker
        assert ticker.ticks == [1, 2]

    def test_cticks_labels(self):
        img = Image(np.array([[0, 1], [2, 3]])).opts(cticks=[(1, "A"), (2, "B")], colorbar=True)
        plot = bokeh_renderer.get_plot(img)
        colorbar = plot.handles["colorbar"]
        assert colorbar.major_label_overrides == {1: "A", 2: "B"}
        ticker = colorbar.ticker
        assert ticker.ticks == [1, 2]

    def test_cticks_ticker(self):
        img = Image(np.array([[0, 1], [2, 3]])).opts(
            cticks=FixedTicker(ticks=[0, 1]), colorbar=True
        )
        plot = bokeh_renderer.get_plot(img)
        colorbar = plot.handles["colorbar"]
        ticker = colorbar.ticker
        assert ticker.ticks == [0, 1]


class TestOverlayPlot(TestBokehPlot):

    def test_overlay_projection_clashing(self):
        overlay = Curve([]).opts(projection='polar') * Curve([]).opts(projection='custom')
        msg = "An axis may only be assigned one projection type"
        with pytest.raises(ValueError, match=msg):
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

    def test_clim_percentile(self):
        arr = np.random.rand(10,10)
        arr[0, 0] = -100
        arr[-1, -1] = 100
        im = Image(arr).opts(clim_percentile=True)

        plot = bokeh_renderer.get_plot(im)
        low, high = plot.ranges[('Image',)]['z']['robust']
        assert low > 0
        assert high < 1

    def test_propagate_tools(self):
        scatter = lambda: Scatter([]).opts(default_tools=[])
        overlay = scatter() * scatter()
        plot = bokeh_renderer.get_plot(overlay)
        assert plot.default_tools == []

class TestApplyHardBounds(TestBokehPlot):
    def test_apply_hard_bounds(self):
        """Test `apply_hard_bounds` with a single element."""
        x_values = np.linspace(10, 50, 5)
        y_values = np.array([10, 20, 30, 40, 50])
        curve = Curve((x_values, y_values)).opts(apply_hard_bounds=True)
        plot = bokeh_renderer.get_plot(curve)
        assert plot.handles['x_range'].bounds == (10, 50)

    def test_apply_hard_bounds_overlay(self):
        """Test `apply_hard_bounds` with an overlay of curves."""
        x1_values = np.linspace(10, 50, 5)
        x2_values = np.linspace(10, 90, 5)
        y_values = np.array([10, 20, 30, 40, 50])
        curve1 = Curve((x1_values, y_values))
        curve2 = Curve((x2_values, y_values))
        overlay = Overlay([curve1, curve2]).opts(opts.Curve(apply_hard_bounds=True))
        plot = bokeh_renderer.get_plot(overlay)
        # Check if the large of the data range can be navigated to
        assert plot.handles['x_range'].bounds == (10, 90)

    def test_apply_hard_bounds_with_xlim(self):
        """Test `apply_hard_bounds` with `xlim` set. Initial view should be within xlim but allow panning to data range."""
        x_values = np.linspace(10, 50, 5)
        y_values = np.array([10, 20, 30, 40, 50])
        curve = Curve((x_values, y_values)).opts(apply_hard_bounds=True, xlim=(15, 35))
        plot = bokeh_renderer.get_plot(curve)
        initial_view_range = (plot.handles['x_range'].start, plot.handles['x_range'].end)
        assert initial_view_range == (15, 35)
        # Check if data beyond xlim can be navigated to
        assert plot.handles['x_range'].bounds == (10, 50)

    def test_apply_hard_bounds_with_redim_range(self):
        """Test `apply_hard_bounds` with `.redim.range(x=...)`. Hard bounds should strictly apply."""
        x_values = np.linspace(10, 50, 5)
        y_values = np.array([10, 20, 30, 40, 50])
        curve = Curve((x_values, y_values)).redim.range(x=(25, None)).opts(apply_hard_bounds=True)
        plot = bokeh_renderer.get_plot(curve)
        # Expected to strictly adhere to any redim.range bounds, otherwise the data range
        assert (plot.handles['x_range'].start, plot.handles['x_range'].end)  == (25, 50)
        assert plot.handles['x_range'].bounds == (25, 50)

    def test_apply_hard_bounds_datetime(self):
        """Test datetime axes with hard bounds."""
        target_xlim_l = dt.datetime(2020, 1, 3)
        target_xlim_h = dt.datetime(2020, 1, 7)
        dates = [dt.datetime(2020, 1, i) for i in range(1, 11)]
        values = np.linspace(0, 100, 10)
        curve = Curve((dates, values)).opts(
            apply_hard_bounds=True,
            xlim=(target_xlim_l, target_xlim_h)
        )
        plot = bokeh_renderer.get_plot(curve)
        initial_view_range = (dt_to_int(plot.handles['x_range'].start), dt_to_int(plot.handles['x_range'].end))
        assert initial_view_range == (dt_to_int(target_xlim_l), dt_to_int(target_xlim_h))
        # Validate navigation bounds include entire data range
        hard_bounds = (dt_to_int(plot.handles['x_range'].bounds[0]), dt_to_int(plot.handles['x_range'].bounds[1]))
        assert hard_bounds == (dt_to_int(dt.datetime(2020, 1, 1)), dt_to_int(dt.datetime(2020, 1, 10)))

    def test_dynamic_map_bounds_update(self):
        """Test that `apply_hard_bounds` applies correctly when DynamicMap is updated."""

        def curve_data(choice):
            datasets = {
                'set1': (np.linspace(0, 5, 100), np.random.rand(100)),
                'set2': (np.linspace(0, 20, 100), np.random.rand(100)),
            }
            x, y = datasets[choice]
            return Curve((x, y))

        ChoiceStream = Stream.define(
            'Choice',
            choice=param.Selector(default='set1', objects=['set1', 'set2'])
        )
        choice_stream = ChoiceStream()
        dmap = DynamicMap(curve_data, kdims=[], streams=[choice_stream])
        dmap = dmap.opts(opts.Curve(apply_hard_bounds=True, xlim=(2,3), framewise=True))
        dmap = dmap.redim.values(choice=['set1', 'set2'])
        plot = bokeh_renderer.get_plot(dmap)

        # Keeping the xlim consistent between updates, and change data range bounds
        # Initially select 'set1'
        dmap.event(choice='set1')
        assert plot.handles['x_range'].start == 2
        assert plot.handles['x_range'].end == 3
        assert plot.handles['x_range'].bounds == (0, 5)

        # Update to 'set2'
        dmap.event(choice='set2')
        assert plot.handles['x_range'].start == 2
        assert plot.handles['x_range'].end == 3
        assert plot.handles['x_range'].bounds == (0, 20)
