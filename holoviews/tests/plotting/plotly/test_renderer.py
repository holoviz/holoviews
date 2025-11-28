"""
Test cases for rendering exporters
"""
import panel as pn
import param
import pytest
from panel.widgets import DiscreteSlider, FloatSlider, Player
from pyviz_comms import CommManager

from holoviews import Curve, DynamicMap, HoloMap, Store
from holoviews.plotting.plotly import PlotlyRenderer
from holoviews.plotting.renderer import Renderer
from holoviews.streams import Stream


class PlotlyRendererTest:

    def setup_method(self):
        self.previous_backend = Store.current_backend
        Store.current_backend = 'plotly'
        self.renderer = PlotlyRenderer.instance()
        self.nbcontext = Renderer.notebook_context
        self.comm_manager = Renderer.comm_manager
        with param.logging_level('ERROR'):
            Renderer.notebook_context = False
            Renderer.comm_manager = CommManager

    def teardown_method(self):
        with param.logging_level('ERROR'):
            Renderer.notebook_context = self.nbcontext
            Renderer.comm_manager = self.comm_manager
        Store.current_backend = self.previous_backend

    def test_render_static(self):
        curve = Curve([])
        obj, _ = self.renderer._validate(curve, None)
        assert isinstance(obj, pn.pane.HoloViews)
        assert obj.center is True
        assert obj.renderer is self.renderer
        assert obj.backend == 'plotly'

    def test_render_holomap_individual(self):
        hmap = HoloMap({i: Curve([1, 2, i]) for i in range(5)})
        obj, _ = self.renderer._validate(hmap, None)
        assert isinstance(obj, pn.pane.HoloViews)
        assert obj.center is True
        assert obj.widget_location == 'right'
        assert obj.widget_type == 'individual'
        widgets = obj.layout.select(DiscreteSlider)
        assert len(widgets) == 1
        slider = widgets[0]
        assert slider.options == dict([(str(i), i) for i in range(5)])

    def test_render_holomap_embedded(self):
        hmap = HoloMap({i: Curve([1, 2, i]) for i in range(5)})
        data, _ = self.renderer.components(hmap)
        assert 'State"' in data['text/html']

    # def test_render_holomap_not_embedded(self):
    #     hmap = HoloMap({i: Curve([1, 2, i]) for i in range(5)})
    #     data, _ = self.renderer.instance(widget_mode='live').components(hmap)
    #     self.assertNotIn('State"', data['text/html'])

    def test_render_holomap_scrubber(self):
        hmap = HoloMap({i: Curve([1, 2, i]) for i in range(5)})
        obj, _ = self.renderer._validate(hmap, 'scrubber')
        assert isinstance(obj, pn.pane.HoloViews)
        assert obj.center is True
        assert obj.widget_location == 'bottom'
        assert obj.widget_type == 'scrubber'
        widgets = obj.layout.select(Player)
        assert len(widgets) == 1
        player = widgets[0]
        assert player.start == 0
        assert player.end == 4

    def test_render_holomap_scrubber_fps(self):
        hmap = HoloMap({i: Curve([1, 2, i]) for i in range(5)})
        obj, _ = self.renderer.instance(fps=2)._validate(hmap, 'scrubber')
        assert isinstance(obj, pn.pane.HoloViews)
        widgets = obj.layout.select(Player)
        assert len(widgets) == 1
        player = widgets[0]
        assert player.interval == 500

    def test_render_holomap_individual_widget_position(self):
        hmap = HoloMap({i: Curve([1, 2, i]) for i in range(5)})
        obj, _ = self.renderer.instance(widget_location='top')._validate(hmap, None)
        assert isinstance(obj, pn.pane.HoloViews)
        assert obj.center is True
        assert obj.widget_location == 'top'
        assert obj.widget_type == 'individual'

    @pytest.mark.filterwarnings('ignore:Attempted to send message over Jupyter Comm:UserWarning')
    def test_render_dynamicmap_with_dims(self):
        dmap = DynamicMap(lambda y: Curve([1, 2, y]), kdims=['y']).redim.range(y=(0.1, 5))
        obj, _ = self.renderer._validate(dmap, None)
        self.renderer.components(obj)
        [(plot, _pane)] = obj._plots.values()

        y = plot.handles['fig']['data'][0]['y']
        assert y[2] == 0.1
        slider = obj.layout.select(FloatSlider)[0]
        slider.value = 3.1
        y = plot.handles['fig']['data'][0]['y']
        assert y[2] == 3.1

    @pytest.mark.filterwarnings('ignore:Attempted to send message over Jupyter Comm:UserWarning')
    def test_render_dynamicmap_with_stream(self):
        stream = Stream.define('Custom', y=2)()
        dmap = DynamicMap(lambda y: Curve([1, 2, y]), kdims=['y'], streams=[stream])
        obj, _ = self.renderer._validate(dmap, None)
        self.renderer.components(obj)
        [(plot, _pane)] = obj._plots.values()

        y = plot.handles['fig']['data'][0]['y']
        assert y[2] == 2
        stream.event(y=3)
        y = plot.handles['fig']['data'][0]['y']
        assert y[2] == 3

    @pytest.mark.filterwarnings('ignore:Attempted to send message over Jupyter Comm:UserWarning')
    def test_render_dynamicmap_with_stream_dims(self):
        stream = Stream.define('Custom', y=2)()
        dmap = DynamicMap(lambda x, y: Curve([x, 1, y]), kdims=['x', 'y'],
                          streams=[stream]).redim.values(x=[1, 2, 3])
        obj, _ = self.renderer._validate(dmap, None)
        self.renderer.components(obj)
        [(plot, _pane)] = obj._plots.values()

        y = plot.handles['fig']['data'][0]['y']
        assert y[2] == 2
        stream.event(y=3)
        y = plot.handles['fig']['data'][0]['y']
        assert y[2] == 3

        y = plot.handles['fig']['data'][0]['y']
        assert y[0] == 1
        slider = obj.layout.select(DiscreteSlider)[0]
        slider.value = 3
        y = plot.handles['fig']['data'][0]['y']
        assert y[0] == 3
