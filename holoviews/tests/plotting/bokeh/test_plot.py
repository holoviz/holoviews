import numpy as np
import pytest
import pyviz_comms as comms

from param import concrete_descendents

from holoviews.core.element import Element
from holoviews.core.options import Store
from holoviews.element.comparison import ComparisonTestCase
from holoviews import Curve

from bokeh.models import (
    ColumnDataSource, CustomJS, LinearColorMapper, LogColorMapper, HoverTool
)
from holoviews.plotting.bokeh.callbacks import Callback
from holoviews.plotting.bokeh.element import ElementPlot
from holoviews.plotting.bokeh.util import bokeh3

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

    def _test_colormapping(self, element, dim, log=False):
        plot = bokeh_renderer.get_plot(element)
        plot.initialize_plot()
        cmapper = plot.handles['color_mapper']
        low, high = element.range(dim)
        self.assertEqual(cmapper.low, low)
        self.assertEqual(cmapper.high, high)
        mapper_type = LogColorMapper if log else LinearColorMapper
        self.assertTrue(isinstance(cmapper, mapper_type))

    def _test_hover_info(self, element, tooltips, line_policy='nearest', formatters={}):
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
        print(renderers, hover)
        for renderer in renderers:
            self.assertTrue(any(renderer in h.renderers for h in hover))


@pytest.mark.skipif(not bokeh3, reason="Bokeh>=3.0 required")
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


@pytest.mark.skipif(not bokeh3, reason="Bokeh>=3.0 required")
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
