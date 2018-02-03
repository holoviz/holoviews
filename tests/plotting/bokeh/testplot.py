from unittest import SkipTest

import param
import numpy as np

from holoviews.core.element import Element
from holoviews.core.options import Store
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting import comms

try:
    from bokeh.models import (
        ColumnDataSource, LinearColorMapper, LogColorMapper, HoverTool
    )
    from holoviews.plotting.bokeh.callbacks import Callback
    from holoviews.plotting.bokeh.util import bokeh_version
    bokeh_renderer = Store.renderers['bokeh']
except:
    bokeh_renderer = None


class TestBokehPlot(ComparisonTestCase):

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
        bokeh_renderer.comms['default'] = self.default_comm
        Callback._callbacks = {}

    def _test_colormapping(self, element, dim, log=False):
        plot = bokeh_renderer.get_plot(element)
        plot.initialize_plot()
        cmapper = plot.handles['color_mapper']
        low, high = element.range(dim)
        self.assertEqual(cmapper.low, low)
        self.assertEqual(cmapper.high, high)
        mapper_type = LogColorMapper if log else LinearColorMapper
        self.assertTrue(isinstance(cmapper, mapper_type))

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
