from __future__ import unicode_literals

from io import BytesIO
from unittest import SkipTest
from nose.plugins.attrib import attr

import numpy as np

from holoviews import HoloMap, Image, GridSpace, Table, Curve, Store
from holoviews.plotting import Renderer
from holoviews.element.comparison import ComparisonTestCase

try:
    from bokeh.io import curdoc
    from holoviews.plotting.bokeh import BokehRenderer
except:
    pass


@attr(optional=1)
class BokehRendererTest(ComparisonTestCase):

    def setUp(self):
        if 'bokeh' not in Store.renderers:
            raise SkipTest("Bokeh required to test widgets")
        self.image1 = Image(np.array([[0,1],[2,3]]), label='Image1')
        self.image2 = Image(np.array([[1,0],[4,-2]]), label='Image2')
        self.map1 = HoloMap({1:self.image1, 2:self.image2}, label='TestMap')
        self.renderer = BokehRenderer.instance()
        self.nbcontext = Renderer.notebook_context 
        Renderer.notebook_context = False

    def tearDown(self):
        Renderer.notebook_context = self.nbcontext

    def test_save_html(self):
        bytesio = BytesIO()
        self.renderer.save(self.image1, bytesio)

    def test_export_widgets(self):
        bytesio = BytesIO()
        self.renderer.export_widgets(self.map1, bytesio, fmt='widgets')

    def test_render_get_plot_server_doc(self):
        renderer = self.renderer.instance(mode='server')
        plot = renderer.get_plot(self.image1)
        self.assertIs(plot.document, curdoc())

    def test_get_size_single_plot(self):
        plot = self.renderer.get_plot(self.image1)
        w, h = self.renderer.get_size(plot)
        self.assertEqual((w, h), (300, 300))

    def test_get_size_row_plot(self):
        plot = self.renderer.get_plot(self.image1+self.image2)
        w, h = self.renderer.get_size(plot)
        self.assertEqual((w, h), (600, 300))

    def test_get_size_column_plot(self):
        plot = self.renderer.get_plot((self.image1+self.image2).cols(1))
        w, h = self.renderer.get_size(plot)
        self.assertEqual((w, h), (300, 600))

    def test_get_size_grid_plot(self):
        grid = GridSpace({(i, j): self.image1 for i in range(3) for j in range(3)})
        plot = self.renderer.get_plot(grid)
        w, h = self.renderer.get_size(plot)
        self.assertEqual((w, h), (419, 413))

    def test_get_size_table(self):
        table = Table(range(10), kdims=['x'])
        plot = self.renderer.get_plot(table)
        w, h = self.renderer.get_size(plot)
        self.assertEqual((w, h), (400, 300))

    def test_get_size_tables_in_layout(self):
        table = Table(range(10), kdims=['x'])
        plot = self.renderer.get_plot(table+table)
        w, h = self.renderer.get_size(plot)
        self.assertEqual((w, h), (680, 300))

    def test_render_to_png(self):
        curve = Curve([])
        renderer = BokehRenderer.instance(fig='png')
        png, info = renderer(curve)
        self.assertIsInstance(png, bytes)
        self.assertEqual(info['file-ext'], 'png')
