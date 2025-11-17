import numpy as np
import pyviz_comms as comms
from bokeh.models import (
    ColumnDataSource,
    CrosshairTool,
    CustomJS,
    HoverTool,
    LinearColorMapper,
    LogColorMapper,
    Span,
)
from param import concrete_descendents

from holoviews import Curve
from holoviews.core.element import Element
from holoviews.core.options import Store
from holoviews.core.spaces import DynamicMap
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting.bokeh.callbacks import Callback
from holoviews.plotting.bokeh.element import ElementPlot
from holoviews.streams import Pipe

bokeh_renderer = Store.renderers['bokeh']

from .. import option_intersections


class TestPlotDefinitions(ComparisonTestCase):

    known_clashes = []

    def test_bokeh_option_definitions(self):
        # Check option definitions do not introduce new clashes
        self.assertEqual(option_intersections('bokeh'), self.known_clashes)


class TestBokehPlot(ComparisonTestCase):

    def setUp(self):
        self.previous_backend = Store.current_backend
        self.comm_manager = bokeh_renderer.comm_manager
        bokeh_renderer.comm_manager = comms.CommManager
        Store.set_current_backend('bokeh')
        self._padding = {}
        for plot in concrete_descendents(ElementPlot).values():
            self._padding[plot] = plot.padding
            plot.padding = 0

    def tearDown(self):
        Store.current_backend = self.previous_backend
        bokeh_renderer.comm_manager = self.comm_manager
        Callback._callbacks = {}
        for plot, padding in self._padding.items():
            plot.padding = padding

    def _test_colormapping(self, element, dim, log=False, prefix=''):
        plot = bokeh_renderer.get_plot(element)
        plot.initialize_plot()
        cmapper = plot.handles[f'{prefix}color_mapper']
        low, high = element.range(dim)
        self.assertEqual(cmapper.low, low)
        self.assertEqual(cmapper.high, high)
        mapper_type = LogColorMapper if log else LinearColorMapper
        self.assertTrue(isinstance(cmapper, mapper_type))

    def _test_hover_info(self, element, tooltips, line_policy='nearest', formatters=None):
        if formatters is None:
            formatters = {}
        plot = bokeh_renderer.get_plot(element)
        plot.initialize_plot()
        fig = plot.state
        renderers = [r for r in plot.traverse(lambda x: x.handles.get('glyph_renderer'))
                     if r is not None]
        hover = fig.select(dict(type=HoverTool))
        self.assertTrue(len(hover))
        self.assertEqual(hover[0].tooltips, tooltips)
        self.assertEqual(hover[0].formatters, formatters)
        self.assertEqual(hover[0].line_policy, line_policy)

        if isinstance(element, Element):
            cds = fig.select_one(dict(type=ColumnDataSource))
            for label, lookup in hover[0].tooltips:
                if label in element.dimensions():
                    self.assertIn(lookup[2:-1], cds.data)

        # Ensure all the glyph renderers have a hover tool
        for renderer in renderers:
            self.assertTrue(any(renderer in h.renderers for h in hover))


def test_element_plot_stream_cleanup():
    stream = Pipe()

    dmap = DynamicMap(Curve, streams=[stream])

    plot = bokeh_renderer.get_plot(dmap)

    assert len(stream._subscribers) == 1

    plot.cleanup()

    assert not stream._subscribers


def test_overlay_plot_stream_cleanup():
    stream1 = Pipe()
    stream2 = Pipe()

    dmap1 = DynamicMap(Curve, streams=[stream1])
    dmap2 = DynamicMap(Curve, streams=[stream2])

    plot = bokeh_renderer.get_plot(dmap1 * dmap2)

    assert len(stream1._subscribers) == 4
    assert len(stream2._subscribers) == 4

    plot.cleanup()

    assert not stream1._subscribers
    assert not stream2._subscribers


def test_layout_plot_stream_cleanup():
    stream1 = Pipe()
    stream2 = Pipe()

    dmap1 = DynamicMap(Curve, streams=[stream1])
    dmap2 = DynamicMap(Curve, streams=[stream2])

    plot = bokeh_renderer.get_plot(dmap1 + dmap2)

    assert len(stream1._subscribers) == 2
    assert len(stream2._subscribers) == 2

    plot.cleanup()

    assert not stream1._subscribers
    assert not stream2._subscribers


def test_sync_two_plots():
    curve = lambda i: Curve(np.arange(10) * i, label="ABC"[i])
    plot1 = curve(0) * curve(1)
    plot2 = curve(0) * curve(1) * curve(2)
    combined_plot = plot1 + plot2

    grid_bkplot = bokeh_renderer.get_plot(combined_plot).handles["plot"]
    for p, *_ in grid_bkplot.children:
        for r in p.renderers:
            if r.name == "C":
                assert r.js_property_callbacks == {}
            else:
                k, v = next(iter(r.js_property_callbacks.items()))
                assert k == "change:muted"
                assert len(v) == 1
                assert isinstance(v[0], CustomJS)
                assert v[0].code == "dst.muted = src.muted"


def test_sync_three_plots():
    curve = lambda i: Curve(np.arange(10) * i, label="ABC"[i])
    plot1 = curve(0) * curve(1)
    plot2 = curve(0) * curve(1) * curve(2)
    plot3 = curve(0) * curve(1)
    combined_plot = plot1 + plot2 + plot3

    grid_bkplot = bokeh_renderer.get_plot(combined_plot).handles["plot"]
    for p, *_ in grid_bkplot.children:
        for r in p.renderers:
            if r.name == "C":
                assert r.js_property_callbacks == {}
            else:
                k, v = next(iter(r.js_property_callbacks.items()))
                assert k == "change:muted"
                assert len(v) == 2
                assert isinstance(v[0], CustomJS)
                assert v[0].code == "dst.muted = src.muted"
                assert isinstance(v[1], CustomJS)
                assert v[1].code == "dst.muted = src.muted"


def test_span_not_cloned_crosshair():
    # See https://github.com/holoviz/holoviews/issues/6386
    height = Span(dimension="height")
    cht = CrosshairTool(overlay=height)

    layout = Curve([]).opts(tools=[cht]) + Curve([]).opts(tools=[cht])

    (fig1, *_), (fig2, *_) = bokeh_renderer.get_plot(layout).handles["plot"].children
    tool1 = next(t for t in fig1.tools if isinstance(t, CrosshairTool))
    tool2 = next(t for t in fig2.tools if isinstance(t, CrosshairTool))

    assert tool1.overlay is tool2.overlay


def test_autohide_toolbar_single_plot():
    curve = Curve([1, 2, 3])

    # Test with autohide_toolbar=False (default)
    plot = bokeh_renderer.get_plot(curve)
    assert plot.autohide_toolbar is False
    assert plot.state.toolbar.autohide is False

    # Test with autohide_toolbar=True
    plot = bokeh_renderer.get_plot(curve.opts(autohide_toolbar=True))
    assert plot.autohide_toolbar is True
    assert plot.state.toolbar.autohide is True


def test_autohide_toolbar_overlay():
    overlay = Curve([1, 2, 3]) * Curve([2, 3, 4])

    # Test with autohide_toolbar=False (default)
    plot = bokeh_renderer.get_plot(overlay)
    assert plot.autohide_toolbar is False
    assert plot.state.toolbar.autohide is False

    # Test with autohide_toolbar=True
    plot = bokeh_renderer.get_plot(overlay.opts(autohide_toolbar=True))
    assert plot.autohide_toolbar is True
    assert plot.state.toolbar.autohide is True


def test_autohide_toolbar_layout():
    layout = Curve([1, 2, 3]) + Curve([2, 3, 4])

    # Test with autohide_toolbar=False (default)
    plot = bokeh_renderer.get_plot(layout)
    assert plot.autohide_toolbar is False
    grid = plot.handles['plot']
    for child, *_ in grid.children:
        assert child.toolbar.autohide is False

    # Test with autohide_toolbar=True
    plot = bokeh_renderer.get_plot(layout.opts(autohide_toolbar=True))
    assert plot.autohide_toolbar is True
    grid = plot.handles['plot']
    assert grid.toolbar.autohide is True

def test_autohide_toolbar_nested():
    overlay = Curve([1, 2, 3]) * Curve([2, 3, 4])

    # Test with different settings for components
    mixed_layout = overlay.opts(autohide_toolbar=True) + Curve([3, 4, 5]).opts(autohide_toolbar=False)
    plot = bokeh_renderer.get_plot(mixed_layout)
    grid = plot.handles['plot']
    child1, child2 = [child for child, *_ in grid.children]
    assert child1.toolbar.autohide is True
    assert child2.toolbar.autohide is False
