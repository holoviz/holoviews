from unittest import SkipTest


from holoviews.core.element import Element
from holoviews.core.options import Store
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting import comms

try:
    from bokeh.models import (
        ColumnDataSource, LinearColorMapper, LogColorMapper, HoverTool
    )
    from holoviews.plotting.bokeh.callbacks import Callback
    bokeh_renderer = Store.renderers['bokeh']
except:
    bokeh_renderer = None

from .. import option_intersections


class TestPlotDefinitions(ComparisonTestCase):

    known_clashes = [(('Bars',), {'width'}), (('BoxWhisker',), {'width'})]

    def test_bokeh_option_definitions(self):
        # Check option definitions do not introduce new clashes
        self.assertEqual(option_intersections('bokeh'), self.known_clashes)


class TestBokehPlot(ComparisonTestCase):

    def setUp(self):
        self.previous_backend = Store.current_backend
        self.comm_manager = bokeh_renderer.comm_manager
        bokeh_renderer.comm_manager = comms.CommManager
        if not bokeh_renderer:
            raise SkipTest("Bokeh required to test plot instantiation")
        Store.current_backend = 'bokeh'

    def tearDown(self):
        Store.current_backend = self.previous_backend
        bokeh_renderer.comm_manager = self.comm_manager
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
