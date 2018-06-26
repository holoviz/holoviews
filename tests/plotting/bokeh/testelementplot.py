from nose.plugins.attrib import attr

import numpy as np

from holoviews.core import Dimension, DynamicMap, NdOverlay
from holoviews.element import Curve, Image, Scatter, Labels
from holoviews.streams import Stream
from holoviews.plotting.util import process_cmap

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
