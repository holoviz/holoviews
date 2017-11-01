"""
Tests of plot instantiation (not display tests, just instantiation)
"""
from __future__ import unicode_literals

import logging
import datetime as dt
from collections import deque
from unittest import SkipTest
from nose.plugins.attrib import attr
from io import StringIO

import param
import numpy as np
from holoviews import (Dimension, Overlay, DynamicMap, Store, Dataset,
                       NdOverlay, GridSpace, HoloMap, Layout, Cycle,
                       Palette, Element, Empty)
from holoviews.core.util import pd
from holoviews.element import (Curve, Scatter, Image, VLine, Points,
                               HeatMap, QuadMesh, Spikes, ErrorBars,
                               Scatter3D, Path, Polygons, Bars, Text,
                               BoxWhisker, HLine, RGB, Raster, Contours)
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import Stream, PointerXY, PointerX
from holoviews.operation import gridmatrix
from holoviews.plotting import comms
from holoviews.plotting.util import rgb2hex

# Standardize backend due to random inconsistencies
try:
    from matplotlib import pyplot
    pyplot.switch_backend('agg')
    from matplotlib.colors import ListedColormap
    from holoviews.plotting.mpl import OverlayPlot
    mpl_renderer = Store.renderers['matplotlib']
except:
    mpl_renderer = None

try:
    from holoviews.plotting.bokeh.util import bokeh_version
    bokeh_renderer = Store.renderers['bokeh']
    from holoviews.plotting.bokeh.callbacks import Callback, PointerXCallback
    from bokeh.document import Document
    from bokeh.models import (
        Div, ColumnDataSource, FactorRange, Range1d, Row, Column,
        ToolbarBox, FixedTicker, FuncTickFormatter
    )
    from bokeh.models.mappers import (LinearColorMapper, LogColorMapper,
                                      CategoricalColorMapper)
    from bokeh.models.tools import HoverTool
    from bokeh.plotting import Figure
except:
    bokeh_renderer = None

try:
    import holoviews.plotting.plotly # noqa (Activate backend)
    plotly_renderer = Store.renderers['plotly']
except:
    plotly_renderer = None


class ParamLogStream(object):
    """
    Context manager that replaces the param logger and captures
    log messages in a StringIO stream.
    """

    def __enter__(self):
        self.stream = StringIO()
        self._handler = logging.StreamHandler(self.stream)
        self._logger = logging.getLogger('testlogger')
        for handler in self._logger.handlers:
            self._logger.removeHandler(handler)
        self._logger.addHandler(self._handler)
        self._param_logger = param.parameterized.logger
        param.parameterized.logger = self._logger
        return self

    def __exit__(self, *args):
        param.parameterized.logger = self._param_logger
        self._handler.close()
        self.stream.seek(0)



class TestMPLPlotInstantiation(ComparisonTestCase):

    def setUp(self):
        self.previous_backend = Store.current_backend
        Store.current_backend = 'matplotlib'
        if mpl_renderer is None:
            raise SkipTest("Matplotlib required to test plot instantiation")
        self.default_comm = mpl_renderer.comms['default']
        mpl_renderer.comms['default'] = (comms.Comm, '')

    def tearDown(self):
        mpl_renderer.comms['default'] = self.default_comm
        Store.current_backend = self.previous_backend

    def test_interleaved_overlay(self):
        """
        Test to avoid regression after fix of https://github.com/ioam/holoviews/issues/41
        """
        o = Overlay([Curve(np.array([[0, 1]])) , Scatter([[1,1]]) , Curve(np.array([[0, 1]]))])
        OverlayPlot(o)

    @attr(optional=1) # Requires jinja2
    def test_dynamic_nonoverlap(self):
        kdims = [Dimension('File', range=(0.01, 1)),
                 Dimension('SliceDimension', range=(0.01, 1)),
                 Dimension('Coordinates', range=(0.01, 1))]
        dmap1 = DynamicMap(lambda x, y, z: Image(np.random.rand(10,10)), kdims=kdims)
        dmap2 = DynamicMap(lambda x: Curve(np.random.rand(10,2))*VLine(x),
                           kdims=kdims[:1])
        mpl_renderer.get_widget(dmap1 + dmap2, 'selection')

    @attr(optional=1) # Requires jinja2
    def test_dynamic_values_partial_overlap(self):
        kdims = [Dimension('File', range=(0.01, 1)),
                 Dimension('SliceDimension', values=['x', 'y', 'z']),
                 Dimension('Coordinates', range=(0.01, 1))]
        dmap1 = DynamicMap(lambda x, y, z: Image(np.random.rand(10,10)), kdims=kdims)
        dmap2 = DynamicMap(lambda x: Curve(np.random.rand(10,2))*VLine(x),
                           kdims=kdims[:1])
        mpl_renderer.get_widget(dmap1 + dmap2, 'selection')

    def test_dynamic_streams_refresh(self):
        stream = PointerXY(x=0, y=0)
        dmap = DynamicMap(lambda x, y: Points([(x, y)]),
                             kdims=[], streams=[stream])
        plot = mpl_renderer.get_plot(dmap)
        pre = mpl_renderer(plot, fmt='png')
        plot.state.set_dpi(72)
        stream.event(x=1, y=1)
        post = mpl_renderer(plot, fmt='png')
        self.assertNotEqual(pre, post)

    def test_errorbar_test(self):
        errorbars = ErrorBars(([0,1],[1,2],[0.1,0.2]))
        plot = mpl_renderer.get_plot(errorbars)
        plot.initialize_plot()

    def test_stream_callback_single_call(self):
        def history_callback(x, history=deque(maxlen=10)):
            history.append(x)
            return Curve(list(history))
        stream = PointerX(x=0)
        dmap = DynamicMap(history_callback, kdims=[], streams=[stream])
        plot = mpl_renderer.get_plot(dmap)
        mpl_renderer(plot)
        for i in range(20):
            plot.state.set_dpi(72)
            stream.event(x=i)
        x, y = plot.handles['artist'].get_data()
        self.assertEqual(x, np.arange(10))
        self.assertEqual(y, np.arange(10, 20))

    def test_points_non_numeric_size_warning(self):
        data = (np.arange(10), np.arange(10), list(map(chr, range(94,104))))
        points = Points(data, vdims=['z']).opts(plot=dict(size_index=2))
        with ParamLogStream() as log:
            plot = mpl_renderer.get_plot(points)
        log_msg = log.stream.read()
        warning = ('%s: z dimension is not numeric, '
                   'cannot use to scale Points size.\n' % plot.name)
        self.assertEqual(log_msg, warning)

    def test_curve_datetime64(self):
        dates = [np.datetime64(dt.datetime(2016,1,i)) for i in range(1, 11)]
        curve = Curve((dates, np.random.rand(10)))
        plot = mpl_renderer.get_plot(curve)
        self.assertEqual(plot.handles['axis'].get_xlim(), (735964.0, 735973.0))

    def test_curve_pandas_timestamps(self):
        if not pd:
            raise SkipTest("Pandas not available")
        dates = pd.date_range('2016-01-01', '2016-01-10', freq='D')
        curve = Curve((dates, np.random.rand(10)))
        plot = mpl_renderer.get_plot(curve)
        self.assertEqual(plot.handles['axis'].get_xlim(), (735964.0, 735973.0))

    def test_curve_dt_datetime(self):
        dates = [dt.datetime(2016,1,i) for i in range(1, 11)]
        curve = Curve((dates, np.random.rand(10)))
        plot = mpl_renderer.get_plot(curve)
        self.assertEqual(plot.handles['axis'].get_xlim(), (735964.0, 735973.0))

    def test_curve_heterogeneous_datetime_types_overlay(self):
        dates64 = [np.datetime64(dt.datetime(2016,1,i)) for i in range(1, 11)]
        dates = [dt.datetime(2016,1,i) for i in range(2, 12)]
        curve_dt64 = Curve((dates64, np.random.rand(10)))
        curve_dt = Curve((dates, np.random.rand(10)))
        plot = mpl_renderer.get_plot(curve_dt*curve_dt64)
        self.assertEqual(plot.handles['axis'].get_xlim(), (735964.0, 735974.0))

    def test_curve_heterogeneous_datetime_types_with_pd_overlay(self):
        if not pd:
            raise SkipTest("Pandas not available")
        dates_pd = pd.date_range('2016-01-04', '2016-01-13', freq='D')
        dates64 = [np.datetime64(dt.datetime(2016,1,i)) for i in range(1, 11)]
        dates = [dt.datetime(2016,1,i) for i in range(2, 12)]
        curve_dt64 = Curve((dates64, np.random.rand(10)))
        curve_dt = Curve((dates, np.random.rand(10)))
        curve_pd = Curve((dates_pd, np.random.rand(10)))
        plot = mpl_renderer.get_plot(curve_dt*curve_dt64*curve_pd)
        self.assertEqual(plot.handles['axis'].get_xlim(), (735964.0, 735976.0))

    def test_image_cbar_extend_both(self):
        img = Image(np.array([[0, 1], [2, 3]])).redim(z=dict(range=(1,2)))
        plot = mpl_renderer.get_plot(img(plot=dict(colorbar=True)))
        self.assertEqual(plot.handles['cbar'].extend, 'both')

    def test_image_cbar_extend_min(self):
        img = Image(np.array([[0, 1], [2, 3]])).redim(z=dict(range=(1, None)))
        plot = mpl_renderer.get_plot(img(plot=dict(colorbar=True)))
        self.assertEqual(plot.handles['cbar'].extend, 'min')

    def test_image_cbar_extend_max(self):
        img = Image(np.array([[0, 1], [2, 3]])).redim(z=dict(range=(None, 2)))
        plot = mpl_renderer.get_plot(img(plot=dict(colorbar=True)))
        self.assertEqual(plot.handles['cbar'].extend, 'max')

    def test_image_cbar_extend_clime(self):
        img = Image(np.array([[0, 1], [2, 3]])).opts(style=dict(clim=(None, None)))
        plot = mpl_renderer.get_plot(img(plot=dict(colorbar=True, color_index=1)))
        self.assertEqual(plot.handles['cbar'].extend, 'neither')

    def test_points_cbar_extend_both(self):
        img = Points(([0, 1], [0, 3])).redim(y=dict(range=(1,2)))
        plot = mpl_renderer.get_plot(img(plot=dict(colorbar=True, color_index=1)))
        self.assertEqual(plot.handles['cbar'].extend, 'both')

    def test_points_cbar_extend_min(self):
        img = Points(([0, 1], [0, 3])).redim(y=dict(range=(1, None)))
        plot = mpl_renderer.get_plot(img(plot=dict(colorbar=True, color_index=1)))
        self.assertEqual(plot.handles['cbar'].extend, 'min')

    def test_points_cbar_extend_max(self):
        img = Points(([0, 1], [0, 3])).redim(y=dict(range=(None, 2)))
        plot = mpl_renderer.get_plot(img(plot=dict(colorbar=True, color_index=1)))
        self.assertEqual(plot.handles['cbar'].extend, 'max')

    def test_points_cbar_extend_clime(self):
        img = Points(([0, 1], [0, 3])).opts(style=dict(clim=(None, None)))
        plot = mpl_renderer.get_plot(img(plot=dict(colorbar=True, color_index=1)))
        self.assertEqual(plot.handles['cbar'].extend, 'neither')

    def test_layout_instantiate_subplots(self):
        layout = (Curve(range(10)) + Curve(range(10)) + Image(np.random.rand(10,10)) +
                  Curve(range(10)) + Curve(range(10)))
        plot = mpl_renderer.get_plot(layout)
        positions = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0)]
        self.assertEqual(sorted(plot.subplots.keys()), positions)
        for i, pos in enumerate(positions):
            adjoint = plot.subplots[pos]
            if 'main' in adjoint.subplots:
                self.assertEqual(adjoint.subplots['main'].layout_num, i+1)

    def test_layout_empty_subplots(self):
        layout = Curve(range(10)) + NdOverlay() + HoloMap() + HoloMap({1: Image(np.random.rand(10,10))})
        plot = mpl_renderer.get_plot(layout)
        self.assertEqual(len(plot.subplots.values()), 2)

    def test_overlay_empty_layers(self):
        overlay = Curve(range(10)) * NdOverlay()
        plot = mpl_renderer.get_plot(overlay)
        self.assertEqual(len(plot.subplots), 1)

    def test_layout_instantiate_subplots_transposed(self):
        layout = (Curve(range(10)) + Curve(range(10)) + Image(np.random.rand(10,10)) +
                  Curve(range(10)) + Curve(range(10)))
        plot = mpl_renderer.get_plot(layout(plot=dict(transpose=True)))
        positions = [(0, 0), (0, 1), (1, 0), (2, 0), (3, 0)]
        self.assertEqual(sorted(plot.subplots.keys()), positions)
        nums = [1, 5, 2, 3, 4]
        for pos, num in zip(positions, nums):
            adjoint = plot.subplots[pos]
            if 'main' in adjoint.subplots:
                self.assertEqual(adjoint.subplots['main'].layout_num, num)

    def test_points_rcparams_do_not_persist(self):
        opts = dict(fig_rcparams={'text.usetex': True})
        points = Points(([0, 1], [0, 3])).opts(plot=opts)
        mpl_renderer.get_plot(points)
        self.assertFalse(pyplot.rcParams['text.usetex'])

    def test_points_rcparams_used(self):
        opts = dict(fig_rcparams={'grid.color': 'red'})
        points = Points(([0, 1], [0, 3])).opts(plot=opts)
        plot = mpl_renderer.get_plot(points)
        ax = plot.state.axes[0]
        lines = ax.get_xgridlines()
        self.assertEqual(lines[0].get_color(), 'red')

    def test_polygons_colored(self):
        polygons = NdOverlay({j: Polygons([[(i**j, i) for i in range(10)]], level=j)
                              for j in range(5)})
        plot = mpl_renderer.get_plot(polygons)
        for j, splot in enumerate(plot.subplots.values()):
            artist = splot.handles['artist']
            self.assertEqual(artist.get_array(), np.array([j]))
            self.assertEqual(artist.get_clim(), (0, 4))

    def test_raster_invert_axes(self):
        arr = np.array([[0, 1, 2], [3, 4,  5]])
        raster = Raster(arr).opts(plot=dict(invert_axes=True))
        plot = mpl_renderer.get_plot(raster)
        artist = plot.handles['artist']
        self.assertEqual(artist.get_array().data, arr.T[::-1])
        self.assertEqual(artist.get_extent(), [0, 2, 0, 3])

    def test_image_invert_axes(self):
        arr = np.array([[0, 1, 2], [3, 4,  5]])
        raster = Image(arr).opts(plot=dict(invert_axes=True))
        plot = mpl_renderer.get_plot(raster)
        artist = plot.handles['artist']
        self.assertEqual(artist.get_array().data, arr.T[::-1, ::-1])
        self.assertEqual(artist.get_extent(), [-0.5, 0.5, -0.5, 0.5])

    def test_quadmesh_invert_axes(self):
        arr = np.array([[0, 1, 2], [3, 4,  5]])
        qmesh = QuadMesh(Image(arr)).opts(plot=dict(invert_axes=True))
        plot = mpl_renderer.get_plot(qmesh)
        artist = plot.handles['artist']
        self.assertEqual(artist.get_array().data, arr.T[:, ::-1].flatten())

    def test_heatmap_invert_axes(self):
        arr = np.array([[0, 1, 2], [3, 4, 5]])
        hm = HeatMap(Image(arr)).opts(plot=dict(invert_axes=True))
        plot = mpl_renderer.get_plot(hm)
        artist = plot.handles['artist']
        self.assertEqual(artist.get_array().data, arr.T[::-1, ::-1])
        self.assertEqual(artist.get_extent(), (0, 2, 0, 3))

    def test_image_listed_cmap(self):
        colors = ['#ffffff','#000000']
        img = Image(np.array([[0, 1, 2], [3, 4, 5]])).opts(style=dict(cmap=colors))
        plot = mpl_renderer.get_plot(img)
        artist = plot.handles['artist']
        cmap = artist.get_cmap()
        self.assertIsInstance(cmap, ListedColormap)
        self.assertEqual(cmap.colors, colors)


class TestBokehPlotInstantiation(ComparisonTestCase):

    def setUp(self):
        self.previous_backend = Store.current_backend
        if not bokeh_renderer:
            raise SkipTest("Bokeh required to test plot instantiation")
        Store.current_backend = 'bokeh'
        Callback._comm_type = comms.Comm
        self.default_comm = bokeh_renderer.comms['default']
        bokeh_renderer.comms['default'] = (comms.Comm, '')

    def tearDown(self):
        Store.current_backend = self.previous_backend
        Callback._comm_type = comms.JupyterCommJS
        mpl_renderer.comms['default'] = self.default_comm
        Callback._callbacks = {}

    def test_overlay_legend(self):
        overlay = Curve(range(10), label='A') * Curve(range(10), label='B')
        plot = bokeh_renderer.get_plot(overlay)
        legend_labels = [l.label['value'] for l in plot.state.legend[0].items]
        self.assertEqual(legend_labels, ['A', 'B'])

    def test_overlay_update_sources(self):
        hmap = HoloMap({i: (Curve(np.arange(i), label='A') *
                            Curve(np.arange(i)*2, label='B'))
                        for i in range(10, 13)})
        plot = bokeh_renderer.get_plot(hmap)
        plot.update((12,))
        subplot1, subplot2 = plot.subplots.values()
        self.assertEqual(subplot1.handles['source'].data['y'], np.arange(12))
        self.assertEqual(subplot2.handles['source'].data['y'], np.arange(12)*2)

    def test_overlay_update_visible(self):
        hmap = HoloMap({i: Curve(np.arange(i), label='A') for i in range(1, 3)})
        hmap2 = HoloMap({i: Curve(np.arange(i), label='B') for i in range(3, 5)})
        plot = bokeh_renderer.get_plot(hmap*hmap2)
        subplot1, subplot2 = plot.subplots.values()
        self.assertTrue(subplot1.handles['glyph_renderer'].visible)
        self.assertFalse(subplot2.handles['glyph_renderer'].visible)
        plot.update((4,))
        self.assertFalse(subplot1.handles['glyph_renderer'].visible)
        self.assertTrue(subplot2.handles['glyph_renderer'].visible)

    def test_batched_empty_update_invisible(self):
        hmap = HoloMap({i: NdOverlay({j: Curve(np.arange(i), label='A') for j in range(i%2)})
                        for i in range(1, 4)})
        opts = {'NdOverlay': {'legend_limit': 0}}
        plot = list(bokeh_renderer.get_plot(hmap(plot=opts)).subplots.values())[0]
        self.assertTrue(plot.handles['glyph_renderer'].visible)
        plot.update((2,))
        self.assertFalse(plot.handles['glyph_renderer'].visible)

    def test_layout_update_visible(self):
        hmap = HoloMap({i: Curve(np.arange(i), label='A') for i in range(1, 3)})
        hmap2 = HoloMap({i: Curve(np.arange(i), label='B') for i in range(3, 5)})
        plot = bokeh_renderer.get_plot(hmap+hmap2)
        subplot1, subplot2 = [p for k, p in sorted(plot.subplots.items())]
        subplot1 = subplot1.subplots['main']
        subplot2 = subplot2.subplots['main']
        self.assertTrue(subplot1.handles['glyph_renderer'].visible)
        self.assertFalse(subplot2.handles['glyph_renderer'].visible)
        plot.update((4,))
        self.assertFalse(subplot1.handles['glyph_renderer'].visible)
        self.assertTrue(subplot2.handles['glyph_renderer'].visible)

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

    def test_batched_plot(self):
        overlay = NdOverlay({i: Points(np.arange(i)) for i in range(1, 100)})
        plot = bokeh_renderer.get_plot(overlay)
        extents = plot.get_extents(overlay, {})
        self.assertEqual(extents, (0, 0, 98, 98))

    def test_batched_spike_plot(self):
        overlay = NdOverlay({i: Spikes([i], kdims=['Time']).opts(plot=dict(position=0.1*i,
                                                                      spike_length=0.1,
                                                                      show_legend=False))
                             for i in range(10)})
        plot = bokeh_renderer.get_plot(overlay)
        extents = plot.get_extents(overlay, {})
        self.assertEqual(extents, (0, 0, 9, 1))

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

    def test_cyclic_palette_curves(self):
        palette = Palette('Set1')
        opts = dict(color=palette)
        hmap = HoloMap({i: NdOverlay({j: Curve(np.random.rand(3)).opts(style=opts)
                                      for j in range(3)})
                        for i in range(3)})
        colors = palette[3].values
        plot = bokeh_renderer.get_plot(hmap)
        for subp, color in zip(plot.subplots.values(), colors):
            self.assertEqual(subp.handles['glyph'].line_color, rgb2hex(color))


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

    def test_batched_path_line_color_and_color(self):
        opts = {'NdOverlay': dict(plot=dict(legend_limit=0)),
                'Path': dict(style=dict(line_color=Cycle(values=['red', 'blue'])))}
        overlay = NdOverlay({i: Path([[(i, j) for j in range(2)]])
                             for i in range(2)}).opts(opts)
        plot = bokeh_renderer.get_plot(overlay).subplots[()]
        line_color = ['red', 'blue']
        self.assertEqual(plot.handles['source'].data['line_color'], line_color)

    def test_batched_path_alpha_and_color(self):
        opts = {'NdOverlay': dict(plot=dict(legend_limit=0)),
                'Path': dict(style=dict(alpha=Cycle(values=[0.5, 1])))}
        overlay = NdOverlay({i: Path([[(i, j) for j in range(2)]])
                             for i in range(2)}).opts(opts)
        plot = bokeh_renderer.get_plot(overlay).subplots[()]
        alpha = [0.5, 1.]
        color = ['#30a2da', '#fc4f30']
        self.assertEqual(plot.handles['source'].data['alpha'], alpha)
        self.assertEqual(plot.handles['source'].data['color'], color)

    def test_batched_path_line_width_and_color(self):
        opts = {'NdOverlay': dict(plot=dict(legend_limit=0)),
                'Path': dict(style=dict(line_width=Cycle(values=[0.5, 1])))}
        overlay = NdOverlay({i: Path([[(i, j) for j in range(2)]])
                             for i in range(2)}).opts(opts)
        plot = bokeh_renderer.get_plot(overlay).subplots[()]
        line_width = [0.5, 1.]
        color = ['#30a2da', '#fc4f30']
        self.assertEqual(plot.handles['source'].data['line_width'], line_width)
        self.assertEqual(plot.handles['source'].data['color'], color)

    def _test_hover_info(self, element, tooltips, line_policy='nearest'):
        plot = bokeh_renderer.get_plot(element)
        plot.initialize_plot()
        fig = plot.state
        renderers = [r for r in plot.traverse(lambda x: x.handles.get('glyph_renderer'))
                     if r is not None]
        hover = fig.select(dict(type=HoverTool))
        self.assertTrue(len(hover))
        self.assertEqual(hover[0].tooltips, tooltips)
        self.assertEqual(hover[0].line_policy, line_policy)

        if isinstance(element, Element):
            cds = fig.select_one(dict(type=ColumnDataSource))
            for label, lookup in hover[0].tooltips:
                if label in element.dimensions():
                    self.assertIn(lookup[2:-1], cds.data)

        # Ensure all the glyph renderers have a hover tool
        for renderer in renderers:
            self.assertTrue(any(renderer in h.renderers for h in hover))

    def test_bars_hover_ensure_kdims_sanitized(self):
        obj = Bars(np.random.rand(10,2), kdims=['Dim with spaces'])
        obj = obj(plot={'tools': ['hover']})
        self._test_hover_info(obj, [('Dim with spaces', '@{Dim_with_spaces}'), ('y', '@{y}')])

    def test_bars_hover_ensure_vdims_sanitized(self):
        obj = Bars(np.random.rand(10,2), vdims=['Dim with spaces'])
        obj = obj(plot={'tools': ['hover']})
        self._test_hover_info(obj, [('x', '@{x}'), ('Dim with spaces', '@{Dim_with_spaces}')])

    def test_heatmap_hover_ensure_kdims_sanitized(self):
        hm = HeatMap([(1,1,1), (2,2,0)], kdims=['x with space', 'y with $pecial symbol'])
        hm = hm(plot={'tools': ['hover']})
        self._test_hover_info(hm, [('x with space', '@{x_with_space}'),
                                   ('y with $pecial symbol', '@{y_with_pecial_symbol}'),
                                   ('z', '@{z}')])

    def test_heatmap_hover_ensure_vdims_sanitized(self):
        hm = HeatMap([(1,1,1), (2,2,0)], vdims=['z with $pace'])
        hm = hm(plot={'tools': ['hover']})
        self._test_hover_info(hm, [('x', '@{x}'), ('y', '@{y}'),
                                   ('z with $pace', '@{z_with_pace}')])

    def test_points_overlay_datetime_hover(self):
        obj = NdOverlay({i: Points((list(pd.date_range('2016-01-01', '2016-01-31')), range(31))) for i in range(5)},
                        kdims=['Test'])
        opts = {'Points': {'tools': ['hover']}}
        obj = obj(plot=opts)
        self._test_hover_info(obj, [('Test', '@{Test}'), ('x', '@{x_dt_strings}'), ('y', '@{y}')])

    def test_curve_overlay_datetime_hover(self):
        obj = NdOverlay({i: Curve((list(pd.date_range('2016-01-01', '2016-01-31')), range(31))) for i in range(5)},
                        kdims=['Test'])
        opts = {'Curve': {'tools': ['hover']}}
        obj = obj(plot=opts)
        self._test_hover_info(obj, [('Test', '@{Test}'), ('x', '@{x_dt_strings}'), ('y', '@{y}')])

    def test_points_overlay_hover_batched(self):
        obj = NdOverlay({i: Points(np.random.rand(10,2)) for i in range(5)},
                        kdims=['Test'])
        opts = {'Points': {'tools': ['hover']},
                'NdOverlay': {'legend_limit': 0}}
        obj = obj(plot=opts)
        self._test_hover_info(obj, [('Test', '@{Test}'), ('x', '@{x}'), ('y', '@{y}')])

    def test_curve_overlay_hover_batched(self):
        obj = NdOverlay({i: Curve(np.random.rand(10,2)) for i in range(5)},
                        kdims=['Test'])
        opts = {'Curve': {'tools': ['hover']},
                'NdOverlay': {'legend_limit': 0}}
        obj = obj(plot=opts)
        self._test_hover_info(obj, [('Test', '@{Test}')], 'prev')

    def test_curve_overlay_hover(self):
        obj = NdOverlay({i: Curve(np.random.rand(10,2)) for i in range(5)},
                        kdims=['Test'])
        opts = {'Curve': {'tools': ['hover']}}
        obj = obj(plot=opts)
        self._test_hover_info(obj, [('Test', '@{Test}'), ('x', '@{x}'), ('y', '@{y}')], 'nearest')

    def test_points_overlay_hover(self):
        obj = NdOverlay({i: Points(np.random.rand(10,2)) for i in range(5)},
                        kdims=['Test'])
        opts = {'Points': {'tools': ['hover']},
                'NdOverlay': {'legend_limit': 0}}
        obj = obj(plot=opts)
        self._test_hover_info(obj, [('Test', '@{Test}'), ('x', '@{x}'),
                                    ('y', '@{y}')])

    def test_path_overlay_hover(self):
        obj = NdOverlay({i: Path([np.random.rand(10,2)]) for i in range(5)},
                        kdims=['Test'])
        opts = {'Path': {'tools': ['hover']},
                'NdOverlay': {'legend_limit': 0}}
        obj = obj(plot=opts)
        self._test_hover_info(obj, [('Test', '@{Test}')])

    def test_polygons_overlay_hover(self):
        obj = NdOverlay({i: Polygons([np.random.rand(10,2)], vdims=['z'], level=0)
                         for i in range(5)}, kdims=['Test'])
        opts = {'Polygons': {'tools': ['hover']},
                'NdOverlay': {'legend_limit': 0}}
        obj = obj(plot=opts)
        self._test_hover_info(obj, [('Test', '@{Test}'), ('z', '@{z}')])

    def test_hover_tool_instance_renderer_association(self):
        tooltips = [("index", "$index")]
        hover = HoverTool(tooltips=tooltips)
        opts = dict(tools=[hover])
        overlay = Curve(np.random.rand(10,2)).opts(plot=opts) * Points(np.random.rand(10,2))
        plot = bokeh_renderer.get_plot(overlay)
        curve_plot = plot.subplots[('Curve', 'I')]
        self.assertEqual(len(curve_plot.handles['hover'].renderers), 1)
        self.assertIn(curve_plot.handles['glyph_renderer'], curve_plot.handles['hover'].renderers)
        self.assertEqual(plot.handles['hover'].tooltips, tooltips)

    def test_hover_tool_nested_overlay_renderers(self):
        overlay1 = NdOverlay({0: Curve(range(2)), 1: Curve(range(3))}, kdims=['Test'])
        overlay2 = NdOverlay({0: Curve(range(4)), 1: Curve(range(5))}, kdims=['Test'])
        nested_overlay = (overlay1 * overlay2).opts(plot={'Curve': dict(tools=['hover'])})
        plot = bokeh_renderer.get_plot(nested_overlay)
        self.assertEqual(len(plot.handles['hover'].renderers), 4)
        self.assertEqual(plot.handles['hover'].tooltips,
                         [('Test', '@{Test}'), ('x', '@{x}'), ('y', '@{y}')])

    def _test_colormapping(self, element, dim, log=False):
        plot = bokeh_renderer.get_plot(element)
        plot.initialize_plot()
        cmapper = plot.handles['color_mapper']
        low, high = element.range(dim)
        self.assertEqual(cmapper.low, low)
        self.assertEqual(cmapper.high, high)
        mapper_type = LogColorMapper if log else LinearColorMapper
        self.assertTrue(isinstance(cmapper, mapper_type))

    def test_polygons_colored(self):
        polygons = NdOverlay({j: Polygons([[(i**j, i) for i in range(10)]], level=j)
                              for j in range(5)})
        plot = bokeh_renderer.get_plot(polygons)
        for i, splot in enumerate(plot.subplots.values()):
            cmapper = splot.handles['color_mapper']
            self.assertEqual(cmapper.low, 0)
            self.assertEqual(cmapper.high, 4)
            source = splot.handles['source']
            self.assertEqual(source.data['Value'], np.array([i]))

    def test_polygons_colored_batched(self):
        polygons = NdOverlay({j: Polygons([[(i**j, i) for i in range(10)]], level=j)
                              for j in range(5)}).opts(plot=dict(legend_limit=0))
        plot = list(bokeh_renderer.get_plot(polygons).subplots.values())[0]
        cmapper = plot.handles['color_mapper']
        self.assertEqual(cmapper.low, 0)
        self.assertEqual(cmapper.high, 4)
        source = plot.handles['source']
        self.assertEqual(plot.handles['glyph'].fill_color['transform'], cmapper)
        self.assertEqual(source.data['Value'], list(range(5)))

    def test_polygons_colored_batched_unsanitized(self):
        polygons = NdOverlay({j: Polygons([[(i**j, i) for i in range(10)] for i in range(2)],
                                          level=j, vdims=['some ? unescaped name'])
                              for j in range(5)}).opts(plot=dict(legend_limit=0))
        plot = list(bokeh_renderer.get_plot(polygons).subplots.values())[0]
        cmapper = plot.handles['color_mapper']
        self.assertEqual(cmapper.low, 0)
        self.assertEqual(cmapper.high, 4)
        source = plot.handles['source']
        self.assertEqual(source.data['some_question_mark_unescaped_name'],
                         [j for i in range(5) for j in [i, i]])

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

    def test_image_colormapping(self):
        img = Image(np.random.rand(10, 10)).opts(plot=dict(logz=True))
        self._test_colormapping(img, 2, True)

    def test_heatmap_colormapping(self):
        hm = HeatMap([(1,1,1), (2,2,0)])
        self._test_colormapping(hm, 2)

    def test_quadmesh_colormapping(self):
        n = 21
        xs = np.logspace(1, 3, n)
        ys = np.linspace(1, 10, n)
        qmesh = QuadMesh((xs, ys, np.random.rand(n-1, n-1)))
        self._test_colormapping(qmesh, 2)

    def test_spikes_colormapping(self):
        spikes = Spikes(np.random.rand(20, 2), vdims=['Intensity'])
        color_spikes = spikes.opts(plot=dict(color_index=1))
        self._test_colormapping(color_spikes, 1)

    def test_empty_spikes_plot(self):
        spikes = Spikes([], vdims=['Intensity'])
        plot = bokeh_renderer.get_plot(spikes)
        source = plot.handles['source']
        self.assertEqual(len(source.data['x']), 0)
        self.assertEqual(len(source.data['y0']), 0)
        self.assertEqual(len(source.data['y1']), 0)

    def test_empty_path_plot(self):
        path = Path([], vdims=['Intensity']).opts(plot=dict(color_index=2))
        plot = bokeh_renderer.get_plot(path)
        source = plot.handles['source']
        self.assertEqual(len(source.data['xs']), 0)
        self.assertEqual(len(source.data['ys']), 0)
        self.assertEqual(len(source.data['Intensity']), 0)

    def test_empty_contours_plot(self):
        contours = Contours([], vdims=['Intensity'])
        plot = bokeh_renderer.get_plot(contours)
        source = plot.handles['source']
        self.assertEqual(len(source.data['xs']), 0)
        self.assertEqual(len(source.data['ys']), 0)
        self.assertEqual(len(source.data['Intensity']), 0)

    def test_empty_polygons_plot(self):
        poly = Polygons([], vdims=['Intensity'])
        plot = bokeh_renderer.get_plot(poly)
        source = plot.handles['source']
        self.assertEqual(len(source.data['xs']), 0)
        self.assertEqual(len(source.data['ys']), 0)
        self.assertEqual(len(source.data['Intensity']), 0)

    def test_side_histogram_no_cmapper(self):
        points = Points(np.random.rand(100, 2))
        plot = bokeh_renderer.get_plot(points.hist())
        plot.initialize_plot()
        adjoint_plot = list(plot.subplots.values())[0]
        main_plot = adjoint_plot.subplots['main']
        right_plot = adjoint_plot.subplots['right']
        self.assertTrue('color_mapper' not in main_plot.handles)
        self.assertTrue('color_mapper' not in right_plot.handles)

    def test_side_histogram_cmapper(self):
        """Assert histogram shares colormapper"""
        x,y = np.mgrid[-50:51, -50:51] * 0.1
        img = Image(np.sin(x**2+y**2), bounds=(-1,-1,1,1))
        plot = bokeh_renderer.get_plot(img.hist())
        plot.initialize_plot()
        adjoint_plot = list(plot.subplots.values())[0]
        main_plot = adjoint_plot.subplots['main']
        right_plot = adjoint_plot.subplots['right']
        self.assertIs(main_plot.handles['color_mapper'],
                      right_plot.handles['color_mapper'])
        self.assertEqual(main_plot.handles['color_dim'], img.vdims[0])

    def test_side_histogram_cmapper_weighted(self):
        """Assert weighted histograms share colormapper"""
        x,y = np.mgrid[-50:51, -50:51] * 0.1
        img = Image(np.sin(x**2+y**2), bounds=(-1,-1,1,1))
        adjoint = img.hist(dimension=['x', 'y'], weight_dimension='z',
                           mean_weighted=True)
        plot = bokeh_renderer.get_plot(adjoint)
        plot.initialize_plot()
        adjoint_plot = list(plot.subplots.values())[0]
        main_plot = adjoint_plot.subplots['main']
        right_plot = adjoint_plot.subplots['right']
        top_plot = adjoint_plot.subplots['top']
        self.assertIs(main_plot.handles['color_mapper'],
                      right_plot.handles['color_mapper'])
        self.assertIs(main_plot.handles['color_mapper'],
                      top_plot.handles['color_mapper'])
        self.assertEqual(main_plot.handles['color_dim'], img.vdims[0])

    def test_stream_callback(self):
        if bokeh_version < str('0.12.5'):
            raise SkipTest("Bokeh >= 0.12.5 required to test streams")
        dmap = DynamicMap(lambda x, y: Points([(x, y)]), kdims=[], streams=[PointerXY()])
        plot = bokeh_renderer.get_plot(dmap)
        bokeh_renderer(plot)
        plot.callbacks[0].on_msg({"x": 10, "y": -10})
        data = plot.handles['source'].data
        self.assertEqual(data['x'], np.array([10]))
        self.assertEqual(data['y'], np.array([-10]))

    def test_stream_callback_with_ids(self):
        if bokeh_version < str('0.12.5'):
            raise SkipTest("Bokeh >= 0.12.5 required to test streams")

        dmap = DynamicMap(lambda x, y: Points([(x, y)]), kdims=[], streams=[PointerXY()])
        plot = bokeh_renderer.get_plot(dmap)
        bokeh_renderer(plot)
        model = plot.state
        plot.callbacks[0].on_msg({"x": {'id': model.ref['id'], 'value': 10},
                                  "y": {'id': model.ref['id'], 'value': -10}})
        data = plot.handles['source'].data
        self.assertEqual(data['x'], np.array([10]))
        self.assertEqual(data['y'], np.array([-10]))

    def test_stream_callback_single_call(self):
        if bokeh_version < str('0.12.5'):
            raise SkipTest("Bokeh >= 0.12.5 required to test streams")

        def history_callback(x, history=deque(maxlen=10)):
            history.append(x)
            return Curve(list(history))
        stream = PointerX(x=0)
        dmap = DynamicMap(history_callback, kdims=[], streams=[stream])
        plot = bokeh_renderer.get_plot(dmap)
        bokeh_renderer(plot)
        for i in range(20):
            stream.event(x=i)
        data = plot.handles['source'].data
        self.assertEqual(data['x'], np.arange(10))
        self.assertEqual(data['y'], np.arange(10, 20))

    def test_bars_suppress_legend(self):
        bars = Bars([('A', 1), ('B', 2)]).opts(plot=dict(show_legend=False))
        plot = bokeh_renderer.get_plot(bars)
        plot.initialize_plot()
        fig = plot.state
        self.assertEqual(len(fig.legend), 0)

    def test_empty_bars(self):
        bars = Bars([], kdims=['x', 'y'], vdims=['z']).opts(plot=dict(group_index=1))
        plot = bokeh_renderer.get_plot(bars)
        plot.initialize_plot()
        source = plot.handles['source']
        for v in source.data.values():
            self.assertEqual(len(v), 0)

    def test_bars_grouped_categories(self):
        bars = Bars([('A', 0, 1), ('A', 1, -1), ('B', 0, 2)],
                    kdims=['Index', 'Category'], vdims=['Value'])
        plot = bokeh_renderer.get_plot(bars)
        source = plot.handles['source']
        self.assertEqual([tuple(x) for x in source.data['xoffsets']],
                         [('A', '0'), ('B', '0'), ('A', '1')])
        self.assertEqual(list(source.data['Category']), [0, 0, 1])
        self.assertEqual(source.data['Value'], np.array([1, 2, -1]))
        x_range = plot.handles['x_range']
        self.assertEqual(x_range.factors, [('A', '0'), ('A', '1'), ('B', '0'), ('B', '1')])

    def test_bars_positive_negative_mixed(self):
        bars = Bars([('A', 0, 1), ('A', 1, -1), ('B', 0, 2)],
                    kdims=['Index', 'Category'], vdims=['Value'])
        plot = bokeh_renderer.get_plot(bars.opts(plot=dict(stack_index=1)))
        source = plot.handles['source']
        self.assertEqual(list(source.data['Category']), [1, 0, 0])
        self.assertEqual(list(source.data['Index']), ['A', 'A', 'B'])
        self.assertEqual(source.data['top'], np.array([0, 1, 2]))
        self.assertEqual(source.data['bottom'], np.array([-1, 0, 0]))

    def test_bars_logy(self):
        bars = Bars([('A', 1), ('B', 2), ('C', 3)],
                    kdims=['Index'], vdims=['Value'])
        plot = bokeh_renderer.get_plot(bars.opts(plot=dict(logy=True)))
        source = plot.handles['source']
        glyph = plot.handles['glyph']
        y_range = plot.handles['y_range']
        self.assertEqual(list(source.data['Index']), ['A', 'B', 'C'])
        self.assertEqual(source.data['Value'], np.array([1, 2, 3]))
        self.assertEqual(glyph.bottom, 10**(np.log10(3)-2))
        self.assertEqual(y_range.start, 10**(np.log10(3)-2))
        self.assertEqual(y_range.end, 3.)

    def test_bars_logy_explicit_range(self):
        bars = Bars([('A', 1), ('B', 2), ('C', 3)],
                    kdims=['Index'], vdims=['Value']).redim.range(Value=(0.001, 3))
        plot = bokeh_renderer.get_plot(bars.opts(plot=dict(logy=True)))
        source = plot.handles['source']
        glyph = plot.handles['glyph']
        y_range = plot.handles['y_range']
        self.assertEqual(list(source.data['Index']), ['A', 'B', 'C'])
        self.assertEqual(source.data['Value'], np.array([1, 2, 3]))
        self.assertEqual(glyph.bottom, 0.001)
        self.assertEqual(y_range.start, 0.001)
        self.assertEqual(y_range.end, 3.)

    def test_points_no_single_item_legend(self):
        points = Points([('A', 1), ('B', 2)], label='A')
        plot = bokeh_renderer.get_plot(points)
        plot.initialize_plot()
        fig = plot.state
        self.assertEqual(len(fig.legend[0].items), 0)

    def test_image_boolean_array(self):
        img = Image(np.array([[True, False], [False, True]]))
        plot = bokeh_renderer.get_plot(img)
        cmapper = plot.handles['color_mapper']
        source = plot.handles['source']
        self.assertEqual(cmapper.low, 0)
        self.assertEqual(cmapper.high, 1)
        self.assertEqual(source.data['image'][0],
                         np.array([[0, 1], [1, 0]]))

    def test_layout_title(self):
        hmap1 = HoloMap({a: Image(np.random.rand(10,10)) for a in range(3)})
        hmap2 = HoloMap({a: Image(np.random.rand(10,10)) for a in range(3)})
        plot = bokeh_renderer.get_plot(hmap1+hmap2)
        title = plot.handles['title']
        self.assertIsInstance(title, Div)
        text = "<span style='font-size: 16pt'><b>Default: 0</b></font>"
        self.assertEqual(title.text, text)

    def test_layout_title_fontsize(self):
        hmap1 = HoloMap({a: Image(np.random.rand(10,10)) for a in range(3)})
        hmap2 = HoloMap({a: Image(np.random.rand(10,10)) for a in range(3)})
        layout = Layout([hmap1, hmap2]).opts(plot=dict(fontsize={'title': '12pt'}))
        plot = bokeh_renderer.get_plot(layout)
        title = plot.handles['title']
        self.assertIsInstance(title, Div)
        text = "<span style='font-size: 12pt'><b>Default: 0</b></font>"
        self.assertEqual(title.text, text)

    def test_layout_title_show_title_false(self):
        hmap1 = HoloMap({a: Image(np.random.rand(10,10)) for a in range(3)})
        hmap2 = HoloMap({a: Image(np.random.rand(10,10)) for a in range(3)})
        layout = Layout([hmap1, hmap2]).opts(plot=dict(show_title=False))
        plot = bokeh_renderer.get_plot(layout)
        self.assertTrue('title' not in plot.handles)

    def test_layout_title_update(self):
        hmap1 = HoloMap({a: Image(np.random.rand(10,10)) for a in range(3)})
        hmap2 = HoloMap({a: Image(np.random.rand(10,10)) for a in range(3)})
        plot = bokeh_renderer.get_plot(hmap1+hmap2)
        plot.update(1)
        title = plot.handles['title']
        self.assertIsInstance(title, Div)
        text = "<span style='font-size: 16pt'><b>Default: 1</b></font>"
        self.assertEqual(title.text, text)

    def test_grid_title(self):
        grid = GridSpace({(i, j): HoloMap({a: Image(np.random.rand(10,10))
                                           for a in range(3)}, kdims=['X'])
                          for i in range(2) for j in range(3)})
        plot = bokeh_renderer.get_plot(grid)
        title = plot.handles['title']
        self.assertIsInstance(title, Div)
        text = "<span style='font-size: 16pt'><b>X: 0</b></font>"
        self.assertEqual(title.text, text)

    def test_grid_title_update(self):
        grid = GridSpace({(i, j): HoloMap({a: Image(np.random.rand(10,10))
                                           for a in range(3)}, kdims=['X'])
                          for i in range(2) for j in range(3)})
        plot = bokeh_renderer.get_plot(grid)
        plot.update(1)
        title = plot.handles['title']
        self.assertIsInstance(title, Div)
        text = "<span style='font-size: 16pt'><b>X: 1</b></font>"
        self.assertEqual(title.text, text)

    def test_points_non_numeric_size_warning(self):
        data = (np.arange(10), np.arange(10), list(map(chr, range(94,104))))
        points = Points(data, vdims=['z']).opts(plot=dict(size_index=2))
        with ParamLogStream() as log:
            plot = bokeh_renderer.get_plot(points)
        log_msg = log.stream.read()
        warning = ('%s: z dimension is not numeric, '
                   'cannot use to scale Points size.\n' % plot.name)
        self.assertEqual(log_msg, warning)

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

    def test_heatmap_categorical_axes_string_int(self):
        hmap = HeatMap([('A',1, 1), ('B', 2, 2)])
        plot = bokeh_renderer.get_plot(hmap)
        x_range = plot.handles['x_range']
        y_range = plot.handles['y_range']
        self.assertIsInstance(x_range, FactorRange)
        self.assertEqual(x_range.factors, ['A', 'B'])
        self.assertIsInstance(y_range, FactorRange)
        self.assertEqual(y_range.factors, ['1', '2'])

    def test_heatmap_categorical_axes_string_int_invert_xyaxis(self):
        opts = dict(invert_xaxis=True, invert_yaxis=True)
        hmap = HeatMap([('A',1, 1), ('B', 2, 2)]).opts(plot=opts)
        plot = bokeh_renderer.get_plot(hmap)
        x_range = plot.handles['x_range']
        y_range = plot.handles['y_range']
        self.assertIsInstance(x_range, FactorRange)
        self.assertEqual(x_range.factors, ['A', 'B'][::-1])
        self.assertIsInstance(y_range, FactorRange)
        self.assertEqual(y_range.factors, ['1', '2'][::-1])

    def test_heatmap_categorical_axes_string_int_inverted(self):
        hmap = HeatMap([('A',1, 1), ('B', 2, 2)]).opts(plot=dict(invert_axes=True))
        plot = bokeh_renderer.get_plot(hmap)
        x_range = plot.handles['x_range']
        y_range = plot.handles['y_range']
        self.assertIsInstance(x_range, FactorRange)
        self.assertEqual(x_range.factors, ['1', '2'])
        self.assertIsInstance(y_range, FactorRange)
        self.assertEqual(y_range.factors, ['A', 'B'])

    def test_heatmap_points_categorical_axes_string_int(self):
        hmap = HeatMap([('A',1, 1), ('B', 2, 2)])
        points = Points([('A', 2), ('B', 1),  ('C', 3)])
        plot = bokeh_renderer.get_plot(hmap*points)
        x_range = plot.handles['x_range']
        y_range = plot.handles['y_range']
        self.assertIsInstance(x_range, FactorRange)
        self.assertEqual(x_range.factors, ['A', 'B', 'C'])
        self.assertIsInstance(y_range, FactorRange)
        self.assertEqual(y_range.factors, ['1', '2', '3'])

    def test_heatmap_points_categorical_axes_string_int_inverted(self):
        hmap = HeatMap([('A',1, 1), ('B', 2, 2)]).opts(plot=dict(invert_axes=True))
        points = Points([('A', 2), ('B', 1),  ('C', 3)])
        plot = bokeh_renderer.get_plot(hmap*points)
        x_range = plot.handles['x_range']
        y_range = plot.handles['y_range']
        self.assertIsInstance(x_range, FactorRange)
        self.assertEqual(x_range.factors, ['1', '2', '3'])
        self.assertIsInstance(y_range, FactorRange)
        self.assertEqual(y_range.factors, ['A', 'B', 'C'])

    def test_points_errorbars_text_ndoverlay_categorical_xaxis(self):
        overlay = NdOverlay({i: Points(([chr(65+i)]*10,np.random.randn(10)))
                             for i in range(5)})
        error = ErrorBars([(el['x'][0], np.mean(el['y']), np.std(el['y']))
                           for el in overlay])
        text = Text('C', 0, 'Test')
        plot = bokeh_renderer.get_plot(overlay*error*text)
        x_range = plot.handles['x_range']
        y_range = plot.handles['y_range']
        self.assertIsInstance(x_range, FactorRange)
        factors = ['A', 'B', 'C', 'D', 'E']
        self.assertEqual(x_range.factors, ['A', 'B', 'C', 'D', 'E'])
        self.assertIsInstance(y_range, Range1d)
        error_plot = plot.subplots[('ErrorBars', 'I')]
        for xs, factor in zip(error_plot.handles['source'].data['base'], factors):
            self.assertEqual(factor, xs)

    def test_points_errorbars_text_ndoverlay_categorical_xaxis_invert_axes(self):
        overlay = NdOverlay({i: Points(([chr(65+i)]*10,np.random.randn(10)))
                             for i in range(5)})
        error = ErrorBars([(el['x'][0], np.mean(el['y']), np.std(el['y']))
                           for el in overlay]).opts(plot=dict(invert_axes=True))
        text = Text('C', 0, 'Test')
        plot = bokeh_renderer.get_plot(overlay*error*text)
        x_range = plot.handles['x_range']
        y_range = plot.handles['y_range']
        self.assertIsInstance(x_range, Range1d)
        self.assertIsInstance(y_range, FactorRange)
        self.assertEqual(y_range.factors, ['A', 'B', 'C', 'D', 'E'])

    def test_hline_invert_axes(self):
        hline = HLine(1.1).opts(plot=dict(invert_axes=True))
        plot = bokeh_renderer.get_plot(hline)
        span = plot.handles['glyph']
        self.assertEqual(span.dimension, 'height')
        self.assertEqual(span.location, 1.1)

    def test_hline_plot(self):
        hline = HLine(1.1)
        plot = bokeh_renderer.get_plot(hline)
        span = plot.handles['glyph']
        self.assertEqual(span.dimension, 'width')
        self.assertEqual(span.location, 1.1)

    def test_vline_invert_axes(self):
        vline = VLine(1.1).opts(plot=dict(invert_axes=True))
        plot = bokeh_renderer.get_plot(vline)
        span = plot.handles['glyph']
        self.assertEqual(span.dimension, 'width')
        self.assertEqual(span.location, 1.1)

    def test_vline_plot(self):
        vline = VLine(1.1)
        plot = bokeh_renderer.get_plot(vline)
        span = plot.handles['glyph']
        self.assertEqual(span.dimension, 'height')
        self.assertEqual(span.location, 1.1)

    def test_raster_invert_axes(self):
        arr = np.array([[0, 1, 2], [3, 4,  5]])
        raster = Raster(arr).opts(plot=dict(invert_axes=True))
        plot = bokeh_renderer.get_plot(raster)
        source = plot.handles['source']
        self.assertEqual(source.data['image'][0], np.rot90(arr))
        self.assertEqual(source.data['x'][0], 0)
        self.assertEqual(source.data['y'][0], 3)
        self.assertEqual(source.data['dw'][0], 2)
        self.assertEqual(source.data['dh'][0], -3)

    def test_image_invert_axes(self):
        arr = np.array([[0, 1, 2], [3, 4,  5]])
        raster = Image(arr).opts(plot=dict(invert_axes=True))
        plot = bokeh_renderer.get_plot(raster)
        source = plot.handles['source']
        self.assertEqual(source.data['image'][0], np.rot90(arr)[::-1, ::-1])
        self.assertEqual(source.data['x'][0], -.5)
        self.assertEqual(source.data['y'][0], -.5)
        self.assertEqual(source.data['dw'][0], 1)
        self.assertEqual(source.data['dh'][0], 1)

    def test_quadmesh_invert_axes(self):
        arr = np.array([[0, 1, 2], [3, 4,  5]])
        qmesh = QuadMesh(Image(arr)).opts(plot=dict(invert_axes=True))
        plot = bokeh_renderer.get_plot(qmesh)
        source = plot.handles['source']
        self.assertEqual(source.data['z'], qmesh.dimension_values(2, flat=False).flatten())
        self.assertEqual(source.data['x'], qmesh.dimension_values(0))
        self.assertEqual(source.data['y'], qmesh.dimension_values(1))

    def test_heatmap_invert_axes(self):
        arr = np.array([[0, 1, 2], [3, 4,  5]])
        hm = HeatMap(Image(arr)).opts(plot=dict(invert_axes=True))
        plot = bokeh_renderer.get_plot(hm)
        xdim, ydim = hm.kdims
        source = plot.handles['source']
        self.assertEqual(source.data['zvalues'], hm.dimension_values(2, flat=False).T.flatten())
        self.assertEqual(source.data['x'], [xdim.pprint_value(v) for v in hm.dimension_values(0)])
        self.assertEqual(source.data['y'], [ydim.pprint_value(v) for v in hm.dimension_values(1)])

    def test_box_whisker_datetime(self):
        times = np.arange(dt.datetime(2017,1,1), dt.datetime(2017,2,1),
                          dt.timedelta(days=1))
        box = BoxWhisker((times, np.random.rand(len(times))), kdims=['Date'])
        plot = bokeh_renderer.get_plot(box)
        formatted = [box.kdims[0].pprint_value(t) for t in times]
        if bokeh_version < str('0.12.7'):
            formatted = [f.replace(':', ';') for f in formatted]
        self.assertTrue(all(cds.data['index'][0] in formatted for cds in
                            plot.state.select(ColumnDataSource)
                            if len(cds.data.get('index', []))))

    def test_curve_datetime64(self):
        dates = [np.datetime64(dt.datetime(2016,1,i)) for i in range(1, 11)]
        curve = Curve((dates, np.random.rand(10)))
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.handles['x_range'].start, np.datetime64(dt.datetime(2016, 1, 1)))
        self.assertEqual(plot.handles['x_range'].end, np.datetime64(dt.datetime(2016, 1, 10)))

    def test_curve_pandas_timestamps(self):
        if not pd:
            raise SkipTest("Pandas not available")
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

    def test_curve_heterogeneous_datetime_types_with_pd_overlay(self):
        if not pd:
            raise SkipTest("Pandas not available")
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
                         {'value': '14pt'})

    def test_curve_fontsize_ylabel(self):
        curve = Curve(range(10)).opts(plot=dict(fontsize={'ylabel': '14pt'}))
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.handles['yaxis'].axis_label_text_font_size,
                         {'value': '14pt'})

    def test_curve_fontsize_both_labels(self):
        curve = Curve(range(10)).opts(plot=dict(fontsize={'labels': '14pt'}))
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.handles['xaxis'].axis_label_text_font_size,
                         {'value': '14pt'})
        self.assertEqual(plot.handles['yaxis'].axis_label_text_font_size,
                         {'value': '14pt'})

    def test_curve_fontsize_xticks(self):
        curve = Curve(range(10)).opts(plot=dict(fontsize={'xticks': '14pt'}))
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.handles['xaxis'].major_label_text_font_size,
                         {'value': '14pt'})

    def test_curve_fontsize_yticks(self):
        curve = Curve(range(10)).opts(plot=dict(fontsize={'yticks': '14pt'}))
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.handles['yaxis'].major_label_text_font_size,
                         {'value': '14pt'})

    def test_curve_fontsize_both_ticks(self):
        curve = Curve(range(10)).opts(plot=dict(fontsize={'ticks': '14pt'}))
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.handles['xaxis'].major_label_text_font_size,
                         {'value': '14pt'})
        self.assertEqual(plot.handles['yaxis'].major_label_text_font_size,
                         {'value': '14pt'})

    def test_curve_fontsize_xticks_and_both_ticks(self):
        curve = Curve(range(10)).opts(plot=dict(fontsize={'xticks': '18pt', 'ticks': '14pt'}))
        plot = bokeh_renderer.get_plot(curve)
        self.assertEqual(plot.handles['xaxis'].major_label_text_font_size,
                         {'value': '18pt'})
        self.assertEqual(plot.handles['yaxis'].major_label_text_font_size,
                         {'value': '14pt'})

    def test_gridmatrix_overlaid_batched(self):
        ds = Dataset((['A']*5+['B']*5, np.random.rand(10), np.random.rand(10)),
                     kdims=['a', 'b', 'c'])
        gmatrix = gridmatrix(ds.groupby('a', container_type=NdOverlay))
        plot = bokeh_renderer.get_plot(gmatrix)

        sp1 = plot.subplots[('b', 'c')]
        self.assertEqual(sp1.state.xaxis[0].visible, False)
        self.assertEqual(sp1.state.yaxis[0].visible, True)
        sp2 = plot.subplots[('b', 'b')]
        self.assertEqual(sp2.state.xaxis[0].visible, True)
        self.assertEqual(sp2.state.yaxis[0].visible, True)
        sp3 = plot.subplots[('c', 'b')]
        self.assertEqual(sp3.state.xaxis[0].visible, True)
        self.assertEqual(sp3.state.yaxis[0].visible, False)
        sp4 = plot.subplots[('c', 'c')]
        self.assertEqual(sp4.state.xaxis[0].visible, False)
        self.assertEqual(sp4.state.yaxis[0].visible, False)

    def test_layout_empty_subplots(self):
        layout = Curve(range(10)) + NdOverlay() + HoloMap() + HoloMap({1: Image(np.random.rand(10,10))})
        plot = bokeh_renderer.get_plot(layout)
        self.assertEqual(len(plot.subplots.values()), 2)

    def test_overlay_empty_layers(self):
        overlay = Curve(range(10)) * NdOverlay()
        plot = bokeh_renderer.get_plot(overlay)
        self.assertEqual(len(plot.subplots), 1)

    def test_gridspace_sparse(self):
        grid = GridSpace({(i, j): Curve(range(i+j)) for i in range(1, 3)
                            for j in range(2,4) if not (i==1 and j == 2)})
        plot = bokeh_renderer.get_plot(grid)
        size = bokeh_renderer.get_size(plot.state)
        self.assertEqual(size, (299, 293))

    def test_layout_gridspaces(self):
        layout = (GridSpace({(i, j): Curve(range(i+j)) for i in range(1, 3)
                             for j in range(2,4)}) +
                  GridSpace({(i, j): Curve(range(i+j)) for i in range(1, 3)
                             for j in range(2,4)}) +
                  Curve(range(10))).cols(2)
        layout_plot = bokeh_renderer.get_plot(layout)
        plot = layout_plot.state

        # Unpack until getting down to two rows
        self.assertIsInstance(plot, Column)
        self.assertEqual(len(plot.children), 2)
        toolbar, column = plot.children
        self.assertIsInstance(toolbar, ToolbarBox)
        self.assertIsInstance(column, Column)
        self.assertEqual(len(column.children), 2)
        row1, row2 = column.children
        self.assertIsInstance(row1, Row)
        self.assertIsInstance(row2, Row)

        # Check the row of GridSpaces
        self.assertEqual(len(row1.children), 2)
        grid1, grid2 = row1.children
        self.assertIsInstance(grid1, Column)
        self.assertIsInstance(grid2, Column)
        self.assertEqual(len(grid1.children), 1)
        self.assertEqual(len(grid2.children), 1)
        grid1, grid2 = grid1.children[0], grid2.children[0]
        self.assertIsInstance(grid1, Column)
        self.assertIsInstance(grid2, Column)
        for grid in [grid1, grid2]:
            self.assertEqual(len(grid.children), 2)
            grow1, grow2 = grid.children
            self.assertIsInstance(grow1, Row)
            self.assertIsInstance(grow2, Row)
            self.assertEqual(len(grow1.children), 2)
            self.assertEqual(len(grow2.children), 2)
            ax_row, grid_row = grow1.children
            grow1, grow2 = grid_row.children[0].children
            gfig1, gfig2 = grow1.children
            gfig3, gfig4 = grow2.children
            self.assertIsInstance(gfig1, Figure)
            self.assertIsInstance(gfig2, Figure)
            self.assertIsInstance(gfig3, Figure)
            self.assertIsInstance(gfig4, Figure)

        # Check the row of Curve and a spacer
        self.assertEqual(len(row2.children), 2)
        fig, spacer = row2.children
        self.assertIsInstance(fig, Figure)
        self.assertIsInstance(spacer, Figure)

    def test_layout_instantiate_subplots(self):
        layout = (Curve(range(10)) + Curve(range(10)) + Image(np.random.rand(10,10)) +
                  Curve(range(10)) + Curve(range(10)))
        plot = bokeh_renderer.get_plot(layout)
        positions = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0)]
        self.assertEqual(sorted(plot.subplots.keys()), positions)

    def test_layout_instantiate_subplots_transposed(self):
        layout = (Curve(range(10)) + Curve(range(10)) + Image(np.random.rand(10,10)) +
                  Curve(range(10)) + Curve(range(10)))
        plot = bokeh_renderer.get_plot(layout(plot=dict(transpose=True)))
        positions = [(0, 0), (0, 1), (1, 0), (2, 0), (3, 0)]
        self.assertEqual(sorted(plot.subplots.keys()), positions)

    def test_element_show_frame_disabled(self):
        curve = Curve(range(10)).opts(plot=dict(show_frame=False))
        plot = bokeh_renderer.get_plot(curve).state
        self.assertEqual(plot.outline_line_alpha, 0)

    def test_overlay_show_frame_disabled(self):
        overlay = (Curve(range(10)) * Curve(range(10))).opts(plot=dict(show_frame=False))
        plot = bokeh_renderer.get_plot(overlay).state
        self.assertEqual(plot.outline_line_alpha, 0)

    def test_element_no_xaxis(self):
        curve = Curve(range(10)).opts(plot=dict(xaxis=None))
        plot = bokeh_renderer.get_plot(curve).state
        self.assertFalse(plot.xaxis[0].visible)

    def test_element_no_yaxis(self):
        curve = Curve(range(10)).opts(plot=dict(yaxis=None))
        plot = bokeh_renderer.get_plot(curve).state
        self.assertFalse(plot.yaxis[0].visible)

    def test_overlay_no_xaxis(self):
        overlay = (Curve(range(10)) * Curve(range(10))).opts(plot=dict(xaxis=None))
        plot = bokeh_renderer.get_plot(overlay).state
        self.assertFalse(plot.xaxis[0].visible)

    def test_overlay_no_yaxis(self):
        overlay = (Curve(range(10)) * Curve(range(10))).opts(plot=dict(yaxis=None))
        plot = bokeh_renderer.get_plot(overlay).state
        self.assertFalse(plot.yaxis[0].visible)

    def test_element_xrotation(self):
        curve = Curve(range(10)).opts(plot=dict(xrotation=90))
        plot = bokeh_renderer.get_plot(curve).state
        self.assertEqual(plot.xaxis[0].major_label_orientation, np.pi/2)

    def test_element_yrotation(self):
        curve = Curve(range(10)).opts(plot=dict(yrotation=90))
        plot = bokeh_renderer.get_plot(curve).state
        self.assertEqual(plot.yaxis[0].major_label_orientation, np.pi/2)

    def test_overlay_xrotation(self):
        overlay = (Curve(range(10)) * Curve(range(10))).opts(plot=dict(xrotation=90))
        plot = bokeh_renderer.get_plot(overlay).state
        self.assertEqual(plot.xaxis[0].major_label_orientation, np.pi/2)

    def test_overlay_yrotation(self):
        overlay = (Curve(range(10)) * Curve(range(10))).opts(plot=dict(yrotation=90))
        plot = bokeh_renderer.get_plot(overlay).state
        self.assertEqual(plot.yaxis[0].major_label_orientation, np.pi/2)

    def test_element_xticks_list(self):
        curve = Curve(range(10)).opts(plot=dict(xticks=[0, 5, 10]))
        plot = bokeh_renderer.get_plot(curve).state
        self.assertIsInstance(plot.xaxis[0].ticker, FixedTicker)
        self.assertEqual(plot.xaxis[0].ticker.ticks, [0, 5, 10])

    def test_element_xticks_list_of_tuples_xaxis(self):
        if bokeh_version < str('0.12.6'):
            raise SkipTest('Bokeh 0.12.6 required for specifying explicit tick labels')
        ticks = [(0, 'zero'), (5, 'five'), (10, 'ten')]
        curve = Curve(range(10)).opts(plot=dict(xticks=ticks))
        plot = bokeh_renderer.get_plot(curve).state
        self.assertIsInstance(plot.xaxis[0].ticker, FixedTicker)
        self.assertEqual(plot.xaxis[0].major_label_overrides, dict(ticks))

    def test_element_yticks_list(self):
        curve = Curve(range(10)).opts(plot=dict(yticks=[0, 5, 10]))
        plot = bokeh_renderer.get_plot(curve).state
        self.assertIsInstance(plot.yaxis[0].ticker, FixedTicker)
        self.assertEqual(plot.yaxis[0].ticker.ticks, [0, 5, 10])

    def test_element_xticks_list_of_tuples_yaxis(self):
        if bokeh_version < str('0.12.6'):
            raise SkipTest('Bokeh 0.12.6 required for specifying explicit tick labels')
        ticks = [(0, 'zero'), (5, 'five'), (10, 'ten')]
        curve = Curve(range(10)).opts(plot=dict(yticks=ticks))
        plot = bokeh_renderer.get_plot(curve).state
        self.assertIsInstance(plot.yaxis[0].ticker, FixedTicker)
        self.assertEqual(plot.yaxis[0].major_label_overrides, dict(ticks))

    def test_overlay_xticks_list(self):
        overlay = (Curve(range(10)) * Curve(range(10))).opts(plot=dict(xticks=[0, 5, 10]))
        plot = bokeh_renderer.get_plot(overlay).state
        self.assertIsInstance(plot.xaxis[0].ticker, FixedTicker)
        self.assertEqual(plot.xaxis[0].ticker.ticks, [0, 5, 10])

    def test_overlay_yticks_list(self):
        overlay = (Curve(range(10)) * Curve(range(10))).opts(plot=dict(yticks=[0, 5, 10]))
        plot = bokeh_renderer.get_plot(overlay).state
        self.assertIsInstance(plot.yaxis[0].ticker, FixedTicker)
        self.assertEqual(plot.yaxis[0].ticker.ticks, [0, 5, 10])

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

    def test_image_invert_xaxis(self):
        arr = np.random.rand(10, 10)
        img = Image(arr).opts(plot=dict(invert_xaxis=True))
        plot = bokeh_renderer.get_plot(img)
        x_range = plot.handles['x_range']
        self.assertEqual(x_range.start, 0.5)
        self.assertEqual(x_range.end, -0.5)
        cdata = plot.handles['source'].data
        self.assertEqual(cdata['x'], [0.5])
        self.assertEqual(cdata['y'], [-0.5])
        self.assertEqual(cdata['dh'], [1.0])
        self.assertEqual(cdata['dw'], [-1.0])
        self.assertEqual(cdata['image'][0], arr[::-1, ::-1])

    def test_image_invert_yaxis(self):
        arr = np.random.rand(10, 10)
        img = Image(arr).opts(plot=dict(invert_yaxis=True))
        plot = bokeh_renderer.get_plot(img)
        y_range = plot.handles['y_range']
        self.assertEqual(y_range.start, 0.5)
        self.assertEqual(y_range.end, -0.5)
        cdata = plot.handles['source'].data
        self.assertEqual(cdata['x'], [-0.5])
        self.assertEqual(cdata['y'], [0.5])
        self.assertEqual(cdata['dh'], [-1.0])
        self.assertEqual(cdata['dw'], [1.0])
        self.assertEqual(cdata['image'][0], arr)

    def test_rgb_invert_xaxis(self):
        rgb = RGB(np.random.rand(10, 10, 3)).opts(plot=dict(invert_xaxis=True))
        plot = bokeh_renderer.get_plot(rgb)
        x_range = plot.handles['x_range']
        self.assertEqual(x_range.start, 0.5)
        self.assertEqual(x_range.end, -0.5)
        cdata = plot.handles['source'].data
        self.assertEqual(cdata['x'], [0.5])
        self.assertEqual(cdata['y'], [-0.5])
        self.assertEqual(cdata['dh'], [1.0])
        self.assertEqual(cdata['dw'], [-1.0])

    def test_rgb_invert_yaxis(self):
        rgb = RGB(np.random.rand(10, 10, 3)).opts(plot=dict(invert_yaxis=True))
        plot = bokeh_renderer.get_plot(rgb)
        y_range = plot.handles['y_range']
        self.assertEqual(y_range.start, 0.5)
        self.assertEqual(y_range.end, -0.5)
        cdata = plot.handles['source'].data
        self.assertEqual(cdata['x'], [-0.5])
        self.assertEqual(cdata['y'], [0.5])
        self.assertEqual(cdata['dh'], [-1.0])
        self.assertEqual(cdata['dw'], [1.0])

    def test_shared_axes(self):
        curve = Curve(range(10))
        img = Image(np.random.rand(10,10))
        plot = bokeh_renderer.get_plot(curve+img)
        plot = plot.subplots[(0, 1)].subplots['main']
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual((x_range.start, x_range.end), (-.5, 9))
        self.assertEqual((y_range.start, y_range.end), (-.5, 9))

    def test_shared_axes_disable(self):
        curve = Curve(range(10))
        img = Image(np.random.rand(10,10)).opts(plot=dict(shared_axes=False))
        plot = bokeh_renderer.get_plot(curve+img)
        plot = plot.subplots[(0, 1)].subplots['main']
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual((x_range.start, x_range.end), (-.5, .5))
        self.assertEqual((y_range.start, y_range.end), (-.5, .5))

    def test_empty_adjoint_plot(self):
        adjoint = Curve([0,1,1,2,3]) << Empty() << Curve([0,1,1,0,1])
        plot = bokeh_renderer.get_plot(adjoint)
        adjoint_plot = plot.subplots[(0, 0)]
        self.assertEqual(len(adjoint_plot.subplots), 3)
        column = plot.state.children[1]
        row1, row2 = column.children
        self.assertEqual(row1.children[0].plot_height, row1.children[1].plot_height)
        self.assertEqual(row1.children[1].plot_width, 0)
        self.assertEqual(row2.children[1].plot_width, 0)
        self.assertEqual(row2.children[0].plot_height, row2.children[1].plot_height)

    def test_layout_shared_source_synced_update(self):
        hmap = HoloMap({i: Dataset({chr(65+j): np.random.rand(i+2)
                                    for j in range(4)}, kdims=['A', 'B', 'C', 'D'])
                        for i in range(3)})

        # Create two holomaps of points sharing the same data source
        hmap1=  hmap.map(lambda x: Points(x.clone(kdims=['A', 'B'])), Dataset)
        hmap2 = hmap.map(lambda x: Points(x.clone(kdims=['D', 'C'])), Dataset)

        # Pop key (1,) for one of the HoloMaps and make Layout
        hmap2.pop((1,))
        layout = (hmap1 + hmap2).opts(plot=dict(shared_datasource=True))

        # Get plot
        plot = bokeh_renderer.get_plot(layout)

        # Check plot created shared data source and recorded expected columns
        sources = plot.handles.get('shared_sources', [])
        source_cols = plot.handles.get('source_cols', {})
        self.assertEqual(len(sources), 1)
        source = sources[0]
        data = source.data
        cols = source_cols[id(source)]
        self.assertEqual(set(cols), {'A', 'B', 'C', 'D'})

        # Ensure the source contains the expected columns
        self.assertEqual(set(data.keys()), {'A', 'B', 'C', 'D'})

        # Update to key (1,) and check the source contains data
        # corresponding to hmap1 and filled in NaNs for hmap2,
        # which was popped above
        plot.update((1,))
        self.assertEqual(data['A'], hmap1[1].dimension_values(0))
        self.assertEqual(data['B'], hmap1[1].dimension_values(1))
        self.assertEqual(data['C'], np.full_like(hmap1[1].dimension_values(0), np.NaN))
        self.assertEqual(data['D'], np.full_like(hmap1[1].dimension_values(0), np.NaN))

    def test_grid_shared_source_synced_update(self):
        hmap = HoloMap({i: Dataset({chr(65+j): np.random.rand(i+2)
                                    for j in range(4)}, kdims=['A', 'B', 'C', 'D'])
                        for i in range(3)})

        # Create two holomaps of points sharing the same data source
        hmap1=  hmap.map(lambda x: Points(x.clone(kdims=['A', 'B'])), Dataset)
        hmap2 = hmap.map(lambda x: Points(x.clone(kdims=['D', 'C'])), Dataset)

        # Pop key (1,) for one of the HoloMaps and make GridSpace
        hmap2.pop(1)
        grid = GridSpace({0: hmap1, 2: hmap2}, kdims=['X']).opts(plot=dict(shared_datasource=True))

        # Get plot
        plot = bokeh_renderer.get_plot(grid)

        # Check plot created shared data source and recorded expected columns
        sources = plot.handles.get('shared_sources', [])
        source_cols = plot.handles.get('source_cols', {})
        self.assertEqual(len(sources), 1)
        source = sources[0]
        data = source.data
        cols = source_cols[id(source)]
        self.assertEqual(set(cols), {'A', 'B', 'C', 'D'})

        # Ensure the source contains the expected columns
        self.assertEqual(set(data.keys()), {'A', 'B', 'C', 'D'})

        # Update to key (1,) and check the source contains data
        # corresponding to hmap1 and filled in NaNs for hmap2,
        # which was popped above
        plot.update((1,))
        self.assertEqual(data['A'], hmap1[1].dimension_values(0))
        self.assertEqual(data['B'], hmap1[1].dimension_values(1))
        self.assertEqual(data['C'], np.full_like(hmap1[1].dimension_values(0), np.NaN))
        self.assertEqual(data['D'], np.full_like(hmap1[1].dimension_values(0), np.NaN))


@attr(optional=1)
class TestPlotlyPlotInstantiation(ComparisonTestCase):

    def setUp(self):
        self.previous_backend = Store.current_backend
        Store.current_backend = 'plotly'
        self.default_comm = plotly_renderer.comms['default']
        if not plotly_renderer:
            raise SkipTest("Plotly required to test plot instantiation")
        plotly_renderer.comms['default'] = (comms.Comm, '')


    def tearDown(self):
        Store.current_backend = self.previous_backend
        plotly_renderer.comms['default'] = self.default_comm

    def _get_plot_state(self, element):
        plot = plotly_renderer.get_plot(element)
        plot.initialize_plot()
        return plot.state

    def test_curve_state(self):
        curve = Curve([1, 2, 3])
        state = self._get_plot_state(curve)
        self.assertEqual(state['data'][0]['y'], np.array([1, 2, 3]))
        self.assertEqual(state['layout']['yaxis']['range'], [1, 3])

    def test_scatter3d_state(self):
        scatter = Scatter3D(([0,1], [2,3], [4,5]))
        state = self._get_plot_state(scatter)
        self.assertEqual(state['data'][0]['x'], np.array([0, 1]))
        self.assertEqual(state['data'][0]['y'], np.array([2, 3]))
        self.assertEqual(state['data'][0]['z'], np.array([4, 5]))
        self.assertEqual(state['layout']['scene']['xaxis']['range'], [0, 1])
        self.assertEqual(state['layout']['scene']['yaxis']['range'], [2, 3])
        self.assertEqual(state['layout']['scene']['zaxis']['range'], [4, 5])

    def test_overlay_state(self):
        layout = Curve([1, 2, 3]) * Curve([2, 4, 6])
        state = self._get_plot_state(layout)
        self.assertEqual(state['data'][0]['y'], np.array([1, 2, 3]))
        self.assertEqual(state['data'][1]['y'], np.array([2, 4, 6]))
        self.assertEqual(state['layout']['yaxis']['range'], [1, 6])

    def test_layout_state(self):
        layout = Curve([1, 2, 3]) + Curve([2, 4, 6])
        state = self._get_plot_state(layout)
        self.assertEqual(state['data'][0]['y'], np.array([1, 2, 3]))
        self.assertEqual(state['data'][0]['yaxis'], 'y1')
        self.assertEqual(state['data'][1]['y'], np.array([2, 4, 6]))
        self.assertEqual(state['data'][1]['yaxis'], 'y2')

    def test_grid_state(self):
        grid = GridSpace({(i, j): Curve([i, j]) for i in [0, 1]
                          for j in [0, 1]})
        state = self._get_plot_state(grid)
        self.assertEqual(state['data'][0]['y'], np.array([0, 0]))
        self.assertEqual(state['data'][0]['xaxis'], 'x1')
        self.assertEqual(state['data'][0]['yaxis'], 'y1')
        self.assertEqual(state['data'][1]['y'], np.array([1, 0]))
        self.assertEqual(state['data'][1]['xaxis'], 'x2')
        self.assertEqual(state['data'][1]['yaxis'], 'y1')
        self.assertEqual(state['data'][2]['y'], np.array([0, 1]))
        self.assertEqual(state['data'][2]['xaxis'], 'x1')
        self.assertEqual(state['data'][2]['yaxis'], 'y2')
        self.assertEqual(state['data'][3]['y'], np.array([1, 1]))
        self.assertEqual(state['data'][3]['xaxis'], 'x2')
        self.assertEqual(state['data'][3]['yaxis'], 'y2')

    def test_stream_callback_single_call(self):
        def history_callback(x, history=deque(maxlen=10)):
            history.append(x)
            return Curve(list(history))
        stream = PointerX(x=0)
        dmap = DynamicMap(history_callback, kdims=[], streams=[stream])
        plot = plotly_renderer.get_plot(dmap)
        plotly_renderer(plot)
        for i in range(20):
            stream.event(x=i)
        state = plot.state
        self.assertEqual(state['data'][0]['x'], np.arange(10))
        self.assertEqual(state['data'][0]['y'], np.arange(10, 20))

    def test_layout_instantiate_subplots(self):
        layout = (Curve(range(10)) + Curve(range(10)) + Image(np.random.rand(10,10)) +
                  Curve(range(10)) + Curve(range(10)))
        plot = plotly_renderer.get_plot(layout)
        positions = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
        self.assertEqual(sorted(plot.subplots.keys()), positions)

    def test_layout_instantiate_subplots_transposed(self):
        layout = (Curve(range(10)) + Curve(range(10)) + Image(np.random.rand(10,10)) +
                  Curve(range(10)) + Curve(range(10)))
        plot = plotly_renderer.get_plot(layout(plot=dict(transpose=True)))
        positions = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]
        self.assertEqual(sorted(plot.subplots.keys()), positions)
