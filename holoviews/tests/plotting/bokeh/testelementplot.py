from unittest import SkipTest
from collections import OrderedDict

import numpy as np

from bokeh.core.properties import value
from holoviews.core import Dimension, DynamicMap, NdOverlay
from holoviews.element import Curve, Image, Scatter, Labels
from holoviews.streams import Stream, PointDraw
from holoviews.plotting.util import process_cmap

from .testplot import TestBokehPlot, bokeh_renderer
from ...utils import LoggingComparisonTestCase

try:
    from bokeh.document import Document
    from bokeh.models import tools
    from bokeh.models import FuncTickFormatter, PrintfTickFormatter, NumeralTickFormatter
except:
    pass



class TestElementPlot(LoggingComparisonTestCase, TestBokehPlot):

    def test_element_show_frame_disabled(self):
        curve = Curve(range(10)).opts(plot=dict(show_frame=False))
        plot = bokeh_renderer.get_plot(curve).state
        self.assertEqual(plot.outline_line_alpha, 0)

    def test_element_xaxis_top(self):
        curve = Curve(range(10)).options(xaxis='top')
        plot = bokeh_renderer.get_plot(curve)
        xaxis = plot.handles['xaxis']
        self.assertTrue(xaxis in plot.state.above)

    def test_element_xaxis_bare(self):
        curve = Curve(range(10)).options(xaxis='bare')
        plot = bokeh_renderer.get_plot(curve)
        xaxis = plot.handles['xaxis']
        self.assertEqual(xaxis.axis_label_text_font_size, value('0pt'))
        self.assertEqual(xaxis.major_label_text_font_size, value('0pt'))
        self.assertEqual(xaxis.minor_tick_line_color, None)
        self.assertEqual(xaxis.major_tick_line_color, None)
        self.assertTrue(xaxis in plot.state.below)

    def test_element_xaxis_bottom_bare(self):
        curve = Curve(range(10)).options(xaxis='bottom-bare')
        plot = bokeh_renderer.get_plot(curve)
        xaxis = plot.handles['xaxis']
        self.assertEqual(xaxis.axis_label_text_font_size, value('0pt'))
        self.assertEqual(xaxis.major_label_text_font_size, value('0pt'))
        self.assertEqual(xaxis.minor_tick_line_color, None)
        self.assertEqual(xaxis.major_tick_line_color, None)
        self.assertTrue(xaxis in plot.state.below)

    def test_element_xaxis_top_bare(self):
        curve = Curve(range(10)).options(xaxis='top-bare')
        plot = bokeh_renderer.get_plot(curve)
        xaxis = plot.handles['xaxis']
        self.assertEqual(xaxis.axis_label_text_font_size, value('0pt'))
        self.assertEqual(xaxis.major_label_text_font_size, value('0pt'))
        self.assertEqual(xaxis.minor_tick_line_color, None)
        self.assertEqual(xaxis.major_tick_line_color, None)
        self.assertTrue(xaxis in plot.state.above)

    def test_element_yaxis_right(self):
        curve = Curve(range(10)).options(yaxis='right')
        plot = bokeh_renderer.get_plot(curve)
        yaxis = plot.handles['yaxis']
        self.assertTrue(yaxis in plot.state.right)

    def test_element_yaxis_bare(self):
        curve = Curve(range(10)).options(yaxis='bare')
        plot = bokeh_renderer.get_plot(curve)
        yaxis = plot.handles['yaxis']
        self.assertEqual(yaxis.axis_label_text_font_size, value('0pt'))
        self.assertEqual(yaxis.major_label_text_font_size, value('0pt'))
        self.assertEqual(yaxis.minor_tick_line_color, None)
        self.assertEqual(yaxis.major_tick_line_color, None)
        self.assertTrue(yaxis in plot.state.left)

    def test_element_yaxis_left_bare(self):
        curve = Curve(range(10)).options(yaxis='left-bare')
        plot = bokeh_renderer.get_plot(curve)
        yaxis = plot.handles['yaxis']
        self.assertEqual(yaxis.axis_label_text_font_size, value('0pt'))
        self.assertEqual(yaxis.major_label_text_font_size, value('0pt'))
        self.assertEqual(yaxis.minor_tick_line_color, None)
        self.assertEqual(yaxis.major_tick_line_color, None)
        self.assertTrue(yaxis in plot.state.left)

    def test_element_yaxis_right_bare(self):
        curve = Curve(range(10)).options(yaxis='right-bare')
        plot = bokeh_renderer.get_plot(curve)
        yaxis = plot.handles['yaxis']
        self.assertEqual(yaxis.axis_label_text_font_size, value('0pt'))
        self.assertEqual(yaxis.major_label_text_font_size, value('0pt'))
        self.assertEqual(yaxis.minor_tick_line_color, None)
        self.assertEqual(yaxis.major_tick_line_color, None)
        self.assertTrue(yaxis in plot.state.right)

    def test_element_xformatter_string(self):
        curve = Curve(range(10)).options(xformatter='%d')
        plot = bokeh_renderer.get_plot(curve)
        xaxis = plot.handles['xaxis']
        self.assertIsInstance(xaxis.formatter, PrintfTickFormatter)
        self.assertEqual(xaxis.formatter.format, '%d')

    def test_element_yformatter_string(self):
        curve = Curve(range(10)).options(yformatter='%d')
        plot = bokeh_renderer.get_plot(curve)
        yaxis = plot.handles['yaxis']
        self.assertIsInstance(yaxis.formatter, PrintfTickFormatter)
        self.assertEqual(yaxis.formatter.format, '%d')

    def test_element_xformatter_function(self):
        try:
            import flexx # noqa
        except:
            raise SkipTest('Test requires flexx')
        def formatter(value):
            return str(value) + ' %'
        curve = Curve(range(10)).options(xformatter=formatter)
        plot = bokeh_renderer.get_plot(curve)
        xaxis = plot.handles['xaxis']
        self.assertIsInstance(xaxis.formatter, FuncTickFormatter)

    def test_element_yformatter_function(self):
        try:
            import flexx # noqa
        except:
            raise SkipTest('Test requires flexx')
        def formatter(value):
            return str(value) + ' %'
        curve = Curve(range(10)).options(yformatter=formatter)
        plot = bokeh_renderer.get_plot(curve)
        yaxis = plot.handles['yaxis']
        self.assertIsInstance(yaxis.formatter, FuncTickFormatter)

    def test_element_xformatter_instance(self):
        formatter = NumeralTickFormatter()
        curve = Curve(range(10)).options(xformatter=formatter)
        plot = bokeh_renderer.get_plot(curve)
        xaxis = plot.handles['xaxis']
        self.assertIs(xaxis.formatter, formatter)

    def test_element_yformatter_instance(self):
        formatter = NumeralTickFormatter()
        curve = Curve(range(10)).options(yformatter=formatter)
        plot = bokeh_renderer.get_plot(curve)
        yaxis = plot.handles['yaxis']
        self.assertIs(yaxis.formatter, formatter)

    def test_empty_element_visibility(self):
        curve = Curve([])
        plot = bokeh_renderer.get_plot(curve)
        self.assertTrue(plot.handles['glyph_renderer'].visible)

    def test_element_no_xaxis(self):
        curve = Curve(range(10)).opts(plot=dict(xaxis=None))
        plot = bokeh_renderer.get_plot(curve).state
        self.assertFalse(plot.xaxis[0].visible)

    def test_element_no_yaxis(self):
        curve = Curve(range(10)).opts(plot=dict(yaxis=None))
        plot = bokeh_renderer.get_plot(curve).state
        self.assertFalse(plot.yaxis[0].visible)

    def test_element_xrotation(self):
        curve = Curve(range(10)).opts(plot=dict(xrotation=90))
        plot = bokeh_renderer.get_plot(curve).state
        self.assertEqual(plot.xaxis[0].major_label_orientation, np.pi/2)

    def test_element_yrotation(self):
        curve = Curve(range(10)).opts(plot=dict(yrotation=90))
        plot = bokeh_renderer.get_plot(curve).state
        self.assertEqual(plot.yaxis[0].major_label_orientation, np.pi/2)

    def test_element_xlabel_override(self):
        curve = Curve(range(10)).options(xlabel='custom x-label')
        plot = bokeh_renderer.get_plot(curve).state
        self.assertEqual(plot.xaxis[0].axis_label, 'custom x-label')

    def test_element_ylabel_override(self):
        curve = Curve(range(10)).options(ylabel='custom y-label')
        plot = bokeh_renderer.get_plot(curve).state
        self.assertEqual(plot.yaxis[0].axis_label, 'custom y-label')

    def test_element_labelled_x_disabled(self):
        curve = Curve(range(10)).options(labelled=['y'])
        plot = bokeh_renderer.get_plot(curve).state
        self.assertEqual(plot.xaxis[0].axis_label, '')
        self.assertEqual(plot.yaxis[0].axis_label, 'y')

    def test_element_labelled_y_disabled(self):
        curve = Curve(range(10)).options(labelled=['x'])
        plot = bokeh_renderer.get_plot(curve).state
        self.assertEqual(plot.xaxis[0].axis_label, 'x')
        self.assertEqual(plot.yaxis[0].axis_label, '')

    def test_element_labelled_both_disabled(self):
        curve = Curve(range(10)).options(labelled=[])
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
            import flexx # noqa
        except:
            raise SkipTest('Test requires flexx')
        def formatter(x):
            return '%s' % x
        curve = Curve(range(10), kdims=[Dimension('x', value_format=formatter)])
        plot = bokeh_renderer.get_plot(curve).state
        self.assertIsInstance(plot.xaxis[0].formatter, FuncTickFormatter)

    def test_element_formatter_yaxis(self):
        try:
            import flexx # noqa
        except:
            raise SkipTest('Test requires flexx')
        def formatter(x):
            return '%s' % x
        curve = Curve(range(10), vdims=[Dimension('y', value_format=formatter)])
        plot = bokeh_renderer.get_plot(curve).state
        self.assertIsInstance(plot.yaxis[0].formatter, FuncTickFormatter)

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
        curve = Curve(range(10)).options(show_grid=True, gridstyle=grid_style)
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
        curve = Curve([('A', 1), ('B', 2)]).options(fontsize={'minor_xticks': '6pt', 'xticks': 18})
        plot = bokeh_renderer.get_plot(curve)
        xaxis = plot.handles['xaxis']
        self.assertEqual(xaxis.major_label_text_font_size, '6pt')
        self.assertEqual(xaxis.group_text_font_size, {'value': '18pt'})

    def test_categorical_axis_fontsize_both(self):
        curve = Curve([('A', 1), ('B', 2)]).options(fontsize={'xticks': 18})
        plot = bokeh_renderer.get_plot(curve)
        xaxis = plot.handles['xaxis']
        self.assertEqual(xaxis.major_label_text_font_size, {'value': '18pt'})
        self.assertEqual(xaxis.group_text_font_size, {'value': '18pt'})

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
        curve = Curve([1, 2, 3]).options(active_tools=['box_zoom'])
        plot = bokeh_renderer.get_plot(curve)
        toolbar = plot.state.toolbar
        self.assertIsInstance(toolbar.active_drag, tools.BoxZoomTool)

    def test_active_tools_scroll(self):
        curve = Curve([1, 2, 3]).options(active_tools=['wheel_zoom'])
        plot = bokeh_renderer.get_plot(curve)
        toolbar = plot.state.toolbar
        self.assertIsInstance(toolbar.active_scroll, tools.WheelZoomTool)

    def test_active_tools_tap(self):
        curve = Curve([1, 2, 3]).options(active_tools=['tap'], tools=['tap'])
        plot = bokeh_renderer.get_plot(curve)
        toolbar = plot.state.toolbar
        self.assertIsInstance(toolbar.active_tap, tools.TapTool)

    def test_active_tools_draw_stream(self):
        scatter = Scatter([1, 2, 3]).options(active_tools=['point_draw'])
        PointDraw(source=scatter)
        plot = bokeh_renderer.get_plot(scatter)
        toolbar = plot.state.toolbar
        self.assertIsInstance(toolbar.active_tap, tools.PointDrawTool)
        self.assertIsInstance(toolbar.active_drag, tools.PointDrawTool)


class TestColorbarPlot(TestBokehPlot):

    def test_colormapper_symmetric(self):
        img = Image(np.array([[0, 1], [2, 3]])).options(symmetric=True)
        plot = bokeh_renderer.get_plot(img)
        cmapper = plot.handles['color_mapper']
        self.assertEqual(cmapper.low, -3)
        self.assertEqual(cmapper.high, 3)

    def test_colormapper_color_levels(self):
        cmap = process_cmap('viridis', provider='bokeh')
        img = Image(np.array([[0, 1], [2, 3]])).options(color_levels=5, cmap=cmap)
        plot = bokeh_renderer.get_plot(img)
        cmapper = plot.handles['color_mapper']
        self.assertEqual(len(cmapper.palette), 5)
        self.assertEqual(cmapper.palette, ['#440154', '#440255', '#440357', '#450558', '#45065A'])

    def test_colormapper_transparent_nan(self):
        img = Image(np.array([[0, 1], [2, 3]])).options(clipping_colors={'NaN': 'transparent'})
        plot = bokeh_renderer.get_plot(img)
        cmapper = plot.handles['color_mapper']
        self.assertEqual(cmapper.nan_color, 'rgba(0, 0, 0, 0)')

    def test_colormapper_min_max_colors(self):
        img = Image(np.array([[0, 1], [2, 3]])).options(clipping_colors={'min': 'red', 'max': 'blue'})
        plot = bokeh_renderer.get_plot(img)
        cmapper = plot.handles['color_mapper']
        self.assertEqual(cmapper.low_color, 'red')
        self.assertEqual(cmapper.high_color, 'blue')

    def test_explicit_categorical_cmap_on_integer_data(self):
        explicit_mapping = OrderedDict([(0, 'blue'), (1, 'red'), (2, 'green'), (3, 'purple')])
        points = Scatter(([0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]), vdims=['y', 'Category']).options(
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
        overlay = Curve([]).options(projection='polar') * Curve([]).options(projection='custom')
        with self.assertRaises(Exception):
            bokeh_renderer.get_plot(overlay)

    def test_overlay_projection_propagates(self):
        overlay = Curve([]) * Curve([]).options(projection='custom')
        plot = bokeh_renderer.get_plot(overlay)
        self.assertEqual([p.projection for p in plot.subplots.values()], ['custom', 'custom'])

    def test_overlay_gridstyle_applies(self):
        grid_style = {'grid_line_color': 'blue', 'grid_line_width': 2}
        overlay = (Scatter([(10,10)]).options(gridstyle=grid_style, show_grid=True, size=20)
                   * Labels([(10, 10, 'A')]))
        plot = bokeh_renderer.get_plot(overlay)
        self.assertEqual(plot.state.xgrid[0].grid_line_color, 'blue')
        self.assertEqual(plot.state.xgrid[0].grid_line_width, 2)

    def test_ndoverlay_legend_muted(self):
        overlay = NdOverlay({i: Curve(np.random.randn(10).cumsum()) for i in range(5)}).options(legend_muted=True)
        plot = bokeh_renderer.get_plot(overlay)
        for sp in plot.subplots.values():
            self.assertTrue(sp.handles['glyph_renderer'].muted)

    def test_overlay_legend_muted(self):
        overlay = (Curve(np.random.randn(10).cumsum(), label='A') *
                   Curve(np.random.randn(10).cumsum(), label='B')).options(legend_muted=True)
        plot = bokeh_renderer.get_plot(overlay)
        for sp in plot.subplots.values():
            self.assertTrue(sp.handles['glyph_renderer'].muted)

    def test_active_tools_drag(self):
        curve = Curve([1, 2, 3])
        scatter = Scatter([1, 2, 3])
        overlay = (scatter * curve).options(active_tools=['box_zoom'])
        plot = bokeh_renderer.get_plot(overlay)
        toolbar = plot.state.toolbar
        self.assertIsInstance(toolbar.active_drag, tools.BoxZoomTool)

    def test_active_tools_scroll(self):
        curve = Curve([1, 2, 3])
        scatter = Scatter([1, 2, 3])
        overlay = (scatter * curve).options(active_tools=['wheel_zoom'])
        plot = bokeh_renderer.get_plot(overlay)
        toolbar = plot.state.toolbar
        self.assertIsInstance(toolbar.active_scroll, tools.WheelZoomTool)

    def test_active_tools_tap(self):
        curve = Curve([1, 2, 3])
        scatter = Scatter([1, 2, 3]).options(tools=['tap'])
        overlay = (scatter * curve).options(active_tools=['tap'])
        plot = bokeh_renderer.get_plot(overlay)
        toolbar = plot.state.toolbar
        self.assertIsInstance(toolbar.active_tap, tools.TapTool)

    def test_active_tools_draw_stream(self):
        curve = Curve([1, 2, 3])
        scatter = Scatter([1, 2, 3]).options(active_tools=['point_draw'])
        PointDraw(source=scatter)
        overlay = (scatter * curve)
        plot = bokeh_renderer.get_plot(overlay)
        toolbar = plot.state.toolbar
        self.assertIsInstance(toolbar.active_tap, tools.PointDrawTool)
        self.assertIsInstance(toolbar.active_drag, tools.PointDrawTool)
