from collections import deque

import numpy as np

from holoviews.core.spaces import DynamicMap
from holoviews.element import Curve, Scatter3D, Path3D
from holoviews.streams import PointerX

from .testplot import TestPlotlyPlot, plotly_renderer


class TestElementPlot(TestPlotlyPlot):

    def test_stream_callback_single_call(self):
        def history_callback(x, history=deque(maxlen=10)):
            history.append(x)
            return Curve(list(history))
        stream = PointerX(x=0)
        dmap = DynamicMap(history_callback, kdims=[], streams=[stream])
        plot = plotly_renderer.get_plot(dmap)
        plotly_renderer(plot)
        for i in range(20):
            stream.event(x=i)
        state = plot.state
        self.assertEqual(state['data'][0]['x'], np.arange(10))
        self.assertEqual(state['data'][0]['y'], np.arange(10, 20))

    ### Axis labelling ###
        
    def test_element_plot_xlabel(self):
        curve = Curve([(10, 1), (100, 2), (1000, 3)]).options(xlabel='X-Axis')
        state = self._get_plot_state(curve)
        self.assertEqual(state['layout']['xaxis']['title'], 'X-Axis')

    def test_element_plot_ylabel(self):
        curve = Curve([(10, 1), (100, 2), (1000, 3)]).options(ylabel='Y-Axis')
        state = self._get_plot_state(curve)
        self.assertEqual(state['layout']['yaxis']['title'], 'Y-Axis')

    def test_element_plot_zlabel(self):
        scatter = Scatter3D([(10, 1, 2), (100, 2, 3), (1000, 3, 5)]).options(zlabel='Z-Axis')
        state = self._get_plot_state(scatter)
        self.assertEqual(state['layout']['scene']['zaxis']['title'], 'Z-Axis')

    ### Axis ranges ###
        
    def test_element_plot_xrange(self):
        curve = Curve([(10, 1), (100, 2), (1000, 3)])
        state = self._get_plot_state(curve)
        self.assertEqual(state['layout']['xaxis']['range'], [10, 1000])

    def test_element_plot_xlim(self):
        curve = Curve([(1, 1), (2, 10), (3, 100)]).options(xlim=(0, 1010))
        state = self._get_plot_state(curve)
        self.assertEqual(state['layout']['xaxis']['range'], [0, 1010])

    def test_element_plot_yrange(self):
        curve = Curve([(10, 1), (100, 2), (1000, 3)])
        state = self._get_plot_state(curve)
        self.assertEqual(state['layout']['yaxis']['range'], [1, 3])

    def test_element_plot_ylim(self):
        curve = Curve([(1, 1), (2, 10), (3, 100)]).options(ylim=(0, 8))
        state = self._get_plot_state(curve)
        self.assertEqual(state['layout']['yaxis']['range'], [0, 8])

    def test_element_plot_zrange(self):
        scatter = Scatter3D([(10, 1, 2), (100, 2, 3), (1000, 3, 5)])
        state = self._get_plot_state(scatter)
        self.assertEqual(state['layout']['scene']['zaxis']['range'], [2, 5])

    def test_element_plot_zlim(self):
        scatter = Scatter3D([(10, 1, 2), (100, 2, 3), (1000, 3, 5)]).options(zlim=(1, 6))
        state = self._get_plot_state(scatter)
        self.assertEqual(state['layout']['scene']['zaxis']['range'], [1, 6])

    def test_element_plot_xpadding(self):
        curve = Curve([(0, 1), (1, 2), (2, 3)]).options(padding=(0.1, 0))
        state = self._get_plot_state(curve)
        self.assertEqual(state['layout']['xaxis']['range'], [-0.2, 2.2])
        self.assertEqual(state['layout']['yaxis']['range'], [1, 3])

    def test_element_plot_ypadding(self):
        curve = Curve([(0, 1), (1, 2), (2, 3)]).options(padding=(0, 0.1))
        state = self._get_plot_state(curve)
        self.assertEqual(state['layout']['xaxis']['range'], [0, 2])
        self.assertEqual(state['layout']['yaxis']['range'], [0.8, 3.2])

    def test_element_plot_zpadding(self):
        scatter = Scatter3D([(10, 1, 2), (100, 2, 3), (1000, 3, 5)]).options(padding=(0, 0, 0.1))
        state = self._get_plot_state(scatter)
        self.assertEqual(state['layout']['scene']['zaxis']['range'], [1.7, 5.3])

    def test_element_plot_padding(self):
        curve = Curve([(0, 1), (1, 2), (2, 3)]).options(padding=0.1)
        state = self._get_plot_state(curve)
        self.assertEqual(state['layout']['xaxis']['range'], [-0.2, 2.2])
        self.assertEqual(state['layout']['yaxis']['range'], [0.8, 3.2])

    def test_element_plot3d_padding(self):
        scatter = Scatter3D([(0, 1, 2), (1, 2, 3), (2, 3, 5)]).options(padding=0.1)
        state = self._get_plot_state(scatter)
        self.assertEqual(state['layout']['scene']['xaxis']['range'], [-0.2, 2.2])
        self.assertEqual(state['layout']['scene']['yaxis']['range'], [0.8, 3.2])
        self.assertEqual(state['layout']['scene']['zaxis']['range'], [1.7, 5.3])

    ### Axis log ###
        
    def test_element_plot_logx(self):
        curve = Curve([(10, 1), (100, 2), (1000, 3)]).options(logx=True)
        state = self._get_plot_state(curve)
        self.assertEqual(state['layout']['xaxis']['type'], 'log')

    def test_element_plot_logy(self):
        curve = Curve([(1, 1), (2, 10), (3, 100)]).options(logy=True)
        state = self._get_plot_state(curve)
        self.assertEqual(state['layout']['yaxis']['type'], 'log')

    def test_element_plot_logz(self):
        scatter = Scatter3D([(0, 1, 10), (1, 2, 100), (2, 3, 1000)]).options(logz=True)
        state = self._get_plot_state(scatter)
        self.assertEqual(state['layout']['scene']['zaxis']['type'], 'log')


class TestOverlayPlot(TestPlotlyPlot):
    
    def test_overlay_state(self):
        layout = Curve([1, 2, 3]) * Curve([2, 4, 6])
        state = self._get_plot_state(layout)
        self.assertEqual(state['data'][0]['y'], np.array([1, 2, 3]))
        self.assertEqual(state['data'][1]['y'], np.array([2, 4, 6]))
        self.assertEqual(state['layout']['yaxis']['range'], [1, 6])

    ### Axis log ###
        
    def test_overlay_plot_logx(self):
        curve = (Curve([(10, 1), (100, 2), (1000, 3)]) * Curve([])).options(logx=True)
        state = self._get_plot_state(curve)
        self.assertEqual(state['layout']['xaxis']['type'], 'log')

    def test_overlay_plot_logy(self):
        curve = (Curve([(1, 1), (2, 10), (3, 100)]) * Curve([])).options(logy=True)
        state = self._get_plot_state(curve)
        self.assertEqual(state['layout']['yaxis']['type'], 'log')

    def test_overlay_plot_logz(self):
        scatter = (Scatter3D([(0, 1, 10), (1, 2, 100), (2, 3, 1000)]) * Path3D([])).options(logz=True)
        state = self._get_plot_state(scatter)
        self.assertEqual(state['layout']['scene']['zaxis']['type'], 'log')

    ### Axis labelling ###
        
    def test_overlay_plot_xlabel(self):
        overlay = Curve([]) * Curve([(10, 1), (100, 2), (1000, 3)]).options(xlabel='X-Axis')
        state = self._get_plot_state(overlay)
        self.assertEqual(state['layout']['xaxis']['title'], 'X-Axis')

    def test_overlay_plot_ylabel(self):
        overlay = Curve([]) * Curve([(10, 1), (100, 2), (1000, 3)]).options(ylabel='Y-Axis')
        state = self._get_plot_state(overlay)
        self.assertEqual(state['layout']['yaxis']['title'], 'Y-Axis')

    def test_overlay_plot_zlabel(self):
        scatter = Path3D([]) * Scatter3D([(10, 1, 2), (100, 2, 3), (1000, 3, 5)]).options(zlabel='Z-Axis')
        state = self._get_plot_state(scatter)
        self.assertEqual(state['layout']['scene']['zaxis']['title'], 'Z-Axis')
