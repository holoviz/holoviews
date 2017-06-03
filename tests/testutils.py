# -*- coding: utf-8 -*-
"""
Unit tests of the helper functions in utils
"""
from unittest import SkipTest

from holoviews import notebook_extension
from holoviews.element.comparison import ComparisonTestCase
from holoviews import Store
from holoviews.util import output, OutputSettings
from holoviews.core import OrderedDict

from holoviews.plotting import mpl
try:
    from holoviews.plotting import bokeh
except:
    bokeh = None

BACKENDS = ['matplotlib'] + (['bokeh'] if bokeh else [])
notebook_extension(*BACKENDS)


class TestOutputUtil(ComparisonTestCase):

    def setUp(self):
        Store.current_backend = 'matplotlib'
        super(TestOutputUtil, self).setUp()

    def tearDown(self):
        Store.renderers['matplotlib'] = mpl.MPLRenderer.instance()
        if bokeh:
            Store.renderers['bokeh'] = bokeh.BokehRenderer.instance()
        OutputSettings.options =  OrderedDict(OutputSettings.defaults.items())
        super(TestOutputUtil, self).tearDown()

    def test_output_util_svg_string(self):
        self.assertEqual(OutputSettings.options.get('fig', None), None)
        output("fig='svg'")
        self.assertEqual(OutputSettings.options.get('fig', None), 'svg')

    def test_output_util_png_kwargs(self):
        self.assertEqual(OutputSettings.options.get('fig', None), None)
        output(fig='png')
        self.assertEqual(OutputSettings.options.get('fig', None), 'png')

    def test_output_util_backend_string(self):
        if bokeh is None:
            raise SkipTest('Bokeh needed to test backend switch')
        self.assertEqual(OutputSettings.options.get('backend', None), None)
        output("backend='bokeh'")
        self.assertEqual(OutputSettings.options.get('backend', None), 'bokeh')

    def test_output_util_backend_kwargs(self):
        if bokeh is None:
            raise SkipTest('Bokeh needed to test backend switch')
        self.assertEqual(OutputSettings.options.get('backend', None), None)
        output(backend='bokeh')
        self.assertEqual(OutputSettings.options.get('backend', None), 'bokeh')
