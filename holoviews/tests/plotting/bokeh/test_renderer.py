import sys
from io import BytesIO

import numpy as np
import panel as pn
import param
import pytest
from bokeh.io import curdoc
from bokeh.themes.theme import Theme
from panel.widgets import DiscreteSlider, FloatSlider, Player
from pyviz_comms import CommManager

from holoviews import Curve, DynamicMap, GridSpace, HoloMap, Image, Table
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting import Renderer
from holoviews.plotting.bokeh import BokehRenderer
from holoviews.streams import Stream


@pytest.mark.usefixtures("bokeh_backend")
class BokehRendererTest(ComparisonTestCase):

    def setUp(self):
        self.image1 = Image(np.array([[0,1],[2,3]]), label='Image1')
        self.image2 = Image(np.array([[1,0],[4,-2]]), label='Image2')
        self.map1 = HoloMap({1:self.image1, 2:self.image2}, label='TestMap')
        self.renderer = BokehRenderer.instance()
        self.nbcontext = Renderer.notebook_context
        self.comm_manager = Renderer.comm_manager
        with param.logging_level('ERROR'):
            Renderer.notebook_context = False
            Renderer.comm_manager = CommManager

    def tearDown(self):
        with param.logging_level('ERROR'):
            Renderer.notebook_context = self.nbcontext
            Renderer.comm_manager = self.comm_manager

    def test_save_html(self):
        bytesio = BytesIO()
        self.renderer.save(self.image1, bytesio)

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
        self.assertEqual((w, h), (446, 437))

    def test_get_size_table(self):
        table = Table(range(10), kdims=['x'])
        plot = self.renderer.get_plot(table)
        w, h = self.renderer.get_size(plot)
        self.assertEqual((w, h), (400, 300))

    def test_get_size_tables_in_layout(self):
        table = Table(range(10), kdims=['x'])
        plot = self.renderer.get_plot(table+table)
        w, h = self.renderer.get_size(plot)
        self.assertEqual((w, h), (800, 300))

    def test_theme_rendering(self):
        attrs = {'figure': {'outline_line_color': '#444444'}}
        theme = Theme(json={'attrs' : attrs})
        self.renderer.theme = theme
        plot = self.renderer.get_plot(Curve([]))
        self.renderer.components(plot, 'html')
        self.assertEqual(plot.state.outline_line_color, '#444444')

    @pytest.mark.skipif(sys.platform == "linux", reason="2025-03: Flaky test on Linux")
    def test_render_to_png(self):
        pytest.importorskip("selenium")
        curve = Curve([])
        renderer = BokehRenderer.instance(fig='png')
        png, info = renderer(curve)
        self.assertIsInstance(png, bytes)
        self.assertEqual(info['file-ext'], 'png')

    def test_render_static(self):
        curve = Curve([])
        obj, _ = self.renderer._validate(curve, None)
        self.assertIsInstance(obj, pn.pane.HoloViews)
        self.assertEqual(obj.center, True)
        self.assertIs(obj.renderer, self.renderer)
        self.assertEqual(obj.backend, 'bokeh')

    def test_render_holomap_individual(self):
        hmap = HoloMap({i: Curve([1, 2, i]) for i in range(5)})
        obj, _ = self.renderer._validate(hmap, None)
        self.assertIsInstance(obj, pn.pane.HoloViews)
        self.assertEqual(obj.center, True)
        self.assertEqual(obj.widget_location, 'right')
        self.assertEqual(obj.widget_type, 'individual')
        widgets = obj.layout.select(DiscreteSlider)
        self.assertEqual(len(widgets), 1)
        slider = widgets[0]
        self.assertEqual(slider.options, dict([(str(i), i) for i in range(5)]))

    def test_render_holomap_embedded(self):
        hmap = HoloMap({i: Curve([1, 2, i]) for i in range(5)})
        data, _ = self.renderer.components(hmap)
        self.assertIn('State"', data['text/html'])

    # def test_render_holomap_not_embedded(self):
    #     hmap = HoloMap({i: Curve([1, 2, i]) for i in range(5)})
    #     data, _ = self.renderer.instance(widget_mode='live').components(hmap)
    #     self.assertNotIn('State"', data['text/html'])

    def test_render_holomap_scrubber(self):
        hmap = HoloMap({i: Curve([1, 2, i]) for i in range(5)})
        obj, _ = self.renderer._validate(hmap, 'scrubber')
        self.assertIsInstance(obj, pn.pane.HoloViews)
        self.assertEqual(obj.center, True)
        self.assertEqual(obj.widget_location, 'bottom')
        self.assertEqual(obj.widget_type, 'scrubber')
        widgets = obj.layout.select(Player)
        self.assertEqual(len(widgets), 1)
        player = widgets[0]
        self.assertEqual(player.start, 0)
        self.assertEqual(player.end, 4)

    def test_render_holomap_scrubber_fps(self):
        hmap = HoloMap({i: Curve([1, 2, i]) for i in range(5)})
        obj, _ = self.renderer.instance(fps=2)._validate(hmap, 'scrubber')
        self.assertIsInstance(obj, pn.pane.HoloViews)
        widgets = obj.layout.select(Player)
        self.assertEqual(len(widgets), 1)
        player = widgets[0]
        self.assertEqual(player.interval, 500)

    def test_render_holomap_individual_widget_position(self):
        hmap = HoloMap({i: Curve([1, 2, i]) for i in range(5)})
        obj, _ = self.renderer.instance(widget_location='top')._validate(hmap, None)
        self.assertIsInstance(obj, pn.pane.HoloViews)
        self.assertEqual(obj.center, True)
        self.assertEqual(obj.widget_location, 'top')
        self.assertEqual(obj.widget_type, 'individual')

    @pytest.mark.filterwarnings('ignore:Attempted to send message over Jupyter Comm')
    def test_render_dynamicmap_with_dims(self):
        dmap = DynamicMap(lambda y: Curve([1, 2, y]), kdims=['y']).redim.range(y=(0.1, 5))
        obj, _ = self.renderer._validate(dmap, None)
        self.renderer.components(obj)
        [(plot, pane)] = obj._plots.values()
        cds = plot.handles['cds']

        self.assertEqual(cds.data['y'][2], 0.1)
        slider = obj.layout.select(FloatSlider)[0]
        slider.value = 3.1
        self.assertEqual(cds.data['y'][2], 3.1)

    @pytest.mark.filterwarnings('ignore:Attempted to send message over Jupyter Comm')
    def test_render_dynamicmap_with_stream(self):
        stream = Stream.define('Custom', y=2)()
        dmap = DynamicMap(lambda y: Curve([1, 2, y]), kdims=['y'], streams=[stream])
        obj, _ = self.renderer._validate(dmap, None)
        self.renderer.components(obj)
        [(plot, pane)] = obj._plots.values()
        cds = plot.handles['cds']

        self.assertEqual(cds.data['y'][2], 2)
        stream.event(y=3)
        self.assertEqual(cds.data['y'][2], 3)

    @pytest.mark.filterwarnings('ignore:Attempted to send message over Jupyter Comm')
    def test_render_dynamicmap_with_stream_dims(self):
        stream = Stream.define('Custom', y=2)()
        dmap = DynamicMap(lambda x, y: Curve([x, 1, y]), kdims=['x', 'y'],
                          streams=[stream]).redim.values(x=[1, 2, 3])
        obj, _ = self.renderer._validate(dmap, None)
        self.renderer.components(obj)
        [(plot, pane)] = obj._plots.values()
        cds = plot.handles['cds']

        self.assertEqual(cds.data['y'][2], 2)
        stream.event(y=3)
        self.assertEqual(cds.data['y'][2], 3)

        self.assertEqual(cds.data['y'][0], 1)
        slider = obj.layout.select(DiscreteSlider)[0]
        slider.value = 3
        self.assertEqual(cds.data['y'][0], 3)
