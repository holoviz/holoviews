# -*- coding: utf-8 -*-
"""
Unit tests of the helper functions in utils
"""
from unittest import SkipTest
import numpy as np

import holoviews as hv
from holoviews import notebook_extension
from holoviews.element.comparison import ComparisonTestCase
from holoviews import Store
from holoviews.util import output, opts, OutputSettings, Options
from holoviews.core import OrderedDict

from holoviews.core.options import OptionTree
from pyviz_comms import CommManager

try:
    from holoviews.plotting import mpl
except:
    mpl = None

try:
    from holoviews.plotting import bokeh
except:
    bokeh = None

BACKENDS = ['matplotlib'] + (['bokeh'] if bokeh else [])

from ..utils import LoggingComparisonTestCase


class TestOutputUtil(ComparisonTestCase):

    def setUp(self):
        notebook_extension(*BACKENDS)
        Store.current_backend = 'matplotlib'
        Store.renderers['matplotlib'] = mpl.MPLRenderer.instance()
        if bokeh:
            Store.renderers['bokeh'] = bokeh.BokehRenderer.instance()
        OutputSettings.options =  OrderedDict(OutputSettings.defaults.items())

        super(TestOutputUtil, self).setUp()

    def tearDown(self):
        Store.renderers['matplotlib'] = mpl.MPLRenderer.instance()
        if bokeh:
            Store.renderers['bokeh'] = bokeh.BokehRenderer.instance()
        OutputSettings.options =  OrderedDict(OutputSettings.defaults.items())
        for renderer in Store.renderers.values():
            renderer.comm_manager = CommManager
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

    def test_output_util_object_noop(self):
        self.assertEqual(output("fig='svg'",3), 3)


class TestOptsUtil(LoggingComparisonTestCase):
    """
    Mirrors the magic tests in TestOptsMagic
    """

    def setUp(self):
        self.backend = Store.current_backend
        Store.current_backend = 'matplotlib'
        self.store_copy = OptionTree(sorted(Store.options().items()),
                                     groups=Options._option_groups)
        super(TestOptsUtil, self).setUp()

    def tearDown(self):
        Store.current_backend = self.backend
        Store.options(val=self.store_copy)
        Store._custom_options = {k:{} for k in Store._custom_options.keys()}
        super(TestOptsUtil, self).tearDown()

    def test_cell_opts_util_style(self):
        mat1 = hv.Image(np.random.rand(5,5), name='mat1')
        self.assertEqual(mat1.id, None)
        opts("Image (cmap='hot')", mat1)
        self.assertNotEqual(mat1.id, None)

        self.assertEqual(
             Store.lookup_options('matplotlib',
                                  mat1, 'style').options.get('cmap',None),'hot')
        self.log_handler.assertContains('WARNING', 'Double positional argument signature of opts is deprecated')

    def test_cell_opts_util_plot(self):

        mat1 = hv.Image(np.random.rand(5,5), name='mat1')

        self.assertEqual(mat1.id, None)
        opts("Image [show_title=False]", mat1)
        self.assertNotEqual(mat1.id, None)
        self.assertEqual(
            Store.lookup_options('matplotlib',
                                 mat1, 'plot').options.get('show_title',True), False)
        self.log_handler.assertContains('WARNING', 'Double positional argument signature of opts is deprecated')

    def test_cell_opts_util_norm(self):
        mat1 = hv.Image(np.random.rand(5,5), name='mat1')
        self.assertEqual(mat1.id, None)
        opts("Image {+axiswise}", mat1)
        self.assertNotEqual(mat1.id, None)

        self.assertEqual(
            Store.lookup_options('matplotlib',
                                 mat1, 'norm').options.get('axiswise',True), True)
        self.log_handler.assertContains('WARNING', 'Double positional argument signature of opts is deprecated')

    def test_opts_builder_repr(self):
        magic= "Bivariate [bandwidth=0.5] (cmap='jet') Points [logx=True] (size=2)"
        expected= ["opts.Bivariate(bandwidth=0.5, cmap='jet')",
                   "opts.Points(logx=True, size=2)"]
        reprs = opts._builder_reprs(magic)
        self.assertEqual(reprs, expected)

    def test_opts_builder_repr_line_magic(self):
        magic= "%opts Bivariate [bandwidth=0.5] (cmap='jet') Points [logx=True] (size=2)"
        expected= ["opts.Bivariate(bandwidth=0.5, cmap='jet')",
                   "opts.Points(logx=True, size=2)"]
        reprs = opts._builder_reprs(magic)
        self.assertEqual(reprs, expected)

    def test_opts_builder_repr_cell_magic(self):
        magic= "%%opts Bivariate [bandwidth=0.5] (cmap='jet') Points [logx=True] (size=2)"
        expected= ["opts.Bivariate(bandwidth=0.5, cmap='jet')",
                   "opts.Points(logx=True, size=2)"]
        reprs = opts._builder_reprs(magic)
        self.assertEqual(reprs, expected)

    def test_opts_builder_repr_options_dotted(self):
        options = [Options('Bivariate.Test.Example', bandwidth=0.5, cmap='Blues'),
                   Options('Points', size=2, logx=True)]
        expected= ["opts.Bivariate('Test.Example', bandwidth=0.5, cmap='Blues')",
                   "opts.Points(logx=True, size=2)"]
        reprs = opts._builder_reprs(options)
        self.assertEqual(reprs, expected)
