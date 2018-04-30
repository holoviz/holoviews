from unittest import SkipTest
from nose.plugins.attrib import attr

from holoviews.core import NdOverlay
from holoviews.core.options import Store
from holoviews.element import Curve
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import Selection1D

try:
    from holoviews.plotting.bokeh.callbacks import Callback, Selection1DCallback
    from holoviews.plotting.bokeh.util import bokeh_version

    from bokeh.events import Tap
    from bokeh.models import Range1d, Plot, ColumnDataSource, Selection
    bokeh_renderer = Store.renderers['bokeh']
except:
    bokeh_renderer = None

@attr(optional=1)
class TestBokehCustomJSCallbacks(ComparisonTestCase):

    def setUp(self):
        if bokeh_version < str('0.12.5'):
            raise SkipTest("Bokeh >= 0.12.5 required to test callbacks")


    def test_customjs_callback_attributes_js_for_model(self):
        js_code = Callback.attributes_js({'x0': 'x_range.attributes.start',
                                          'x1': 'x_range.attributes.end'})

        code = (
            'if ((x_range != undefined)) { data["x0"] = {id: x_range["id"], value: '
            'x_range["attributes"]["start"]};\n }'
            'if ((x_range != undefined)) { data["x1"] = {id: x_range["id"], value: '
            'x_range["attributes"]["end"]};\n }'
        )
        self.assertEqual(js_code, code)

    def test_customjs_callback_attributes_js_for_cb_obj(self):
        js_code = Callback.attributes_js({'x': 'cb_obj.x',
                                          'y': 'cb_obj.y'})
        code = 'data["x"] = cb_obj["x"];\ndata["y"] = cb_obj["y"];\n'
        self.assertEqual(js_code, code)

    def test_customjs_callback_attributes_js_for_cb_data(self):
        js_code = Callback.attributes_js({'x0': 'cb_data.geometry.x0',
                                          'x1': 'cb_data.geometry.x1',
                                          'y0': 'cb_data.geometry.y0',
                                          'y1': 'cb_data.geometry.y1'})
        code = ('data["x0"] = cb_data["geometry"]["x0"];\n'
                'data["x1"] = cb_data["geometry"]["x1"];\n'
                'data["y0"] = cb_data["geometry"]["y0"];\n'
                'data["y1"] = cb_data["geometry"]["y1"];\n')
        self.assertEqual(js_code, code)


    def test_callback_on_ndoverlay_is_attached(self):
        ndoverlay = NdOverlay({i: Curve([i]) for i in range(5)})
        selection = Selection1D(source=ndoverlay)
        plot = bokeh_renderer.get_plot(ndoverlay)
        self.assertEqual(len(plot.callbacks), 1)
        self.assertIsInstance(plot.callbacks[0], Selection1DCallback)
        self.assertIn(selection, plot.callbacks[0].streams)


@attr(optional=1)
class TestBokehServerJSCallbacks(ComparisonTestCase):

    def setUp(self):
        if bokeh_version < str('0.12.5'):
            raise SkipTest("Bokeh >= 0.12.5 required to test callbacks")

    def test_server_callback_resolve_attr_spec_range1d_start(self):
        range1d = Range1d(start=0, end=10)
        msg = Callback.resolve_attr_spec('x_range.attributes.start', range1d)
        self.assertEqual(msg, {'id': range1d.ref['id'], 'value': 0})

    def test_server_callback_resolve_attr_spec_range1d_end(self):
        range1d = Range1d(start=0, end=10)
        msg = Callback.resolve_attr_spec('x_range.attributes.end', range1d)
        self.assertEqual(msg, {'id': range1d.ref['id'], 'value': 10})

    def test_server_callback_resolve_attr_spec_source_selected(self):
        source = ColumnDataSource()
        source.selected = Selection(indices=[1, 2, 3])
        msg = Callback.resolve_attr_spec('cb_obj.selected.1d.indices', source)
        self.assertEqual(msg, {'id': source.ref['id'], 'value': [1, 2, 3]})

    def test_server_callback_resolve_attr_spec_tap_event(self):
        plot = Plot()
        event = Tap(plot, x=42)
        msg = Callback.resolve_attr_spec('cb_obj.x', event, plot)
        self.assertEqual(msg, {'id': plot.ref['id'], 'value': 42})
