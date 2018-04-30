from nose.plugins.attrib import attr

import numpy as np

from holoviews.core import Dimension, DynamicMap
from holoviews.element import Curve, Image
from holoviews.streams import Stream

from .testplot import TestBokehPlot, bokeh_renderer

try:
    from bokeh.document import Document
    from bokeh.models import FuncTickFormatter
except:
    pass



class TestElementPlot(TestBokehPlot):

    def test_element_show_frame_disabled(self):
        curve = Curve(range(10)).opts(plot=dict(show_frame=False))
        plot = bokeh_renderer.get_plot(curve).state
        self.assertEqual(plot.outline_line_alpha, 0)

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

    @attr(optional=1)  # Requires Flexx
    def test_element_formatter_xaxis(self):
        def formatter(x):
            return '%s' % x
        curve = Curve(range(10), kdims=[Dimension('x', value_format=formatter)])
        plot = bokeh_renderer.get_plot(curve).state
        self.assertIsInstance(plot.xaxis[0].formatter, FuncTickFormatter)

    @attr(optional=1)  # Requires Flexx
    def test_element_formatter_yaxis(self):
        def formatter(x):
            return '%s' % x
        curve = Curve(range(10), vdims=[Dimension('y', value_format=formatter)])
        plot = bokeh_renderer.get_plot(curve).state
        self.assertIsInstance(plot.yaxis[0].formatter, FuncTickFormatter)

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


class TestColorbarPlot(TestBokehPlot):

    def test_colormapper_symmetric(self):
        img = Image(np.array([[0, 1], [2, 3]])).options(symmetric=True)
        plot = bokeh_renderer.get_plot(img)
        cmapper = plot.handles['color_mapper']
        self.assertEqual(cmapper.low, -3)
        self.assertEqual(cmapper.high, 3)

    def test_colormapper_color_levels(self):
        img = Image(np.array([[0, 1], [2, 3]])).options(color_levels=5)
        plot = bokeh_renderer.get_plot(img)
        cmapper = plot.handles['color_mapper']
        self.assertEqual(len(cmapper.palette), 5)

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
