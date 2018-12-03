from collections import deque
from unittest import SkipTest
from nose.plugins.attrib import attr

import numpy as np

from holoviews.core import Store, DynamicMap, GridSpace
from holoviews.element import Curve, Scatter3D, Image, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import PointerX
import pyviz_comms as comms

try:
    import holoviews.plotting.plotly # noqa (Activate backend)
    plotly_renderer = Store.renderers['plotly']
    from holoviews.plotting.plotly.util import figure_grid
    import plotly.graph_objs as go
except:
    plotly_renderer = None

from .. import option_intersections


class TestPlotDefinitions(ComparisonTestCase):

    known_clashes = [(('Curve',), {'width'}), (('ErrorBars',), {'width'})]

    def test_plotly_option_definitions(self):
        # Check option definitions do not introduce new clashes
        self.assertEqual(option_intersections('plotly'), self.known_clashes)


@attr(optional=1)
class TestPlotlyPlotInstantiation(ComparisonTestCase):

    def setUp(self):
        self.previous_backend = Store.current_backend
        Store.current_backend = 'plotly'
        self.comm_manager = plotly_renderer.comm_manager
        plotly_renderer.comm_manager = comms.CommManager
        if not plotly_renderer:
            raise SkipTest("Plotly required to test plot instantiation")

    def tearDown(self):
        Store.current_backend = self.previous_backend
        plotly_renderer.comm_manager = self.comm_manager

    def _get_plot_state(self, element):
        plot = plotly_renderer.get_plot(element)
        plot.initialize_plot()

        # Pass to plotly.py for full property validation
        go.Figure(plot.state)

        return plot.state

    def test_curve_state(self):
        curve = Curve([1, 2, 3])
        state = self._get_plot_state(curve)
        self.assertEqual(state['data'][0]['y'], np.array([1, 2, 3]))
        self.assertEqual(state['data'][0]['mode'], 'lines')
        self.assertEqual(state['layout']['yaxis']['range'], [1, 3])

    def test_scatter_state(self):
        scatter = Scatter([3, 2, 1])
        state = self._get_plot_state(scatter)
        self.assertEqual(state['data'][0]['y'], np.array([3, 2, 1]))
        self.assertEqual(state['data'][0]['mode'], 'markers')
        self.assertEqual(state['layout']['yaxis']['range'], [1, 3])

    def test_scatter3d_state(self):
        scatter = Scatter3D(([0,1], [2,3], [4,5]))
        state = self._get_plot_state(scatter)
        self.assertEqual(state['data'][0]['x'], np.array([0, 1]))
        self.assertEqual(state['data'][0]['y'], np.array([2, 3]))
        self.assertEqual(state['data'][0]['z'], np.array([4, 5]))
        self.assertEqual(state['layout']['scene']['xaxis']['range'], [0, 1])
        self.assertEqual(state['layout']['scene']['yaxis']['range'], [2, 3])
        self.assertEqual(state['layout']['scene']['zaxis']['range'], [4, 5])

    def test_overlay_state(self):
        layout = Curve([1, 2, 3]) * Curve([2, 4, 6])
        state = self._get_plot_state(layout)
        self.assertEqual(state['data'][0]['y'], np.array([1, 2, 3]))
        self.assertEqual(state['data'][1]['y'], np.array([2, 4, 6]))
        self.assertEqual(state['layout']['yaxis']['range'], [1, 6])

    def test_layout_state(self):
        layout = Curve([1, 2, 3]) + Curve([2, 4, 6])
        state = self._get_plot_state(layout)
        self.assertEqual(state['data'][0]['y'], np.array([1, 2, 3]))
        self.assertEqual(state['data'][0]['yaxis'], 'y')
        self.assertEqual(state['data'][1]['y'], np.array([2, 4, 6]))
        self.assertEqual(state['data'][1]['yaxis'], 'y2')

    def test_grid_state(self):
        grid = GridSpace({(i, j): Curve([i, j]) for i in [0, 1]
                          for j in [0, 1]})
        state = self._get_plot_state(grid)
        self.assertEqual(state['data'][0]['y'], np.array([0, 0]))
        self.assertEqual(state['data'][0]['xaxis'], 'x')
        self.assertEqual(state['data'][0]['yaxis'], 'y')
        self.assertEqual(state['data'][1]['y'], np.array([1, 0]))
        self.assertEqual(state['data'][1]['xaxis'], 'x2')
        self.assertEqual(state['data'][1]['yaxis'], 'y')
        self.assertEqual(state['data'][2]['y'], np.array([0, 1]))
        self.assertEqual(state['data'][2]['xaxis'], 'x')
        self.assertEqual(state['data'][2]['yaxis'], 'y2')
        self.assertEqual(state['data'][3]['y'], np.array([1, 1]))
        self.assertEqual(state['data'][3]['xaxis'], 'x2')
        self.assertEqual(state['data'][3]['yaxis'], 'y2')

    def test_layout_with_grid(self):
        # Create GridSpace
        grid = GridSpace({(i, j): Curve([i, j]) for i in [0, 1]
                          for j in [0, 1]})
        grid = grid.options(vspacing=0, hspacing=0)

        # Create Scatter
        scatter = Scatter([-10, 0])

        # Create Horizontal Layout
        layout = (scatter + grid).options(vspacing=0, hspacing=0)

        state = self._get_plot_state(layout)

        # Check the scatter plot on the left
        self.assertEqual(state['data'][0]['y'], np.array([-10, 0]))
        self.assertEqual(state['data'][0]['mode'], 'markers')
        self.assertEqual(state['data'][0]['xaxis'], 'x')
        self.assertEqual(state['data'][0]['yaxis'], 'y')
        self.assertEqual(state['layout']['xaxis']['range'], [0, 1])
        self.assertEqual(state['layout']['xaxis']['domain'], [0, 0.5])
        self.assertEqual(state['layout']['yaxis']['range'], [-10, 0])
        self.assertEqual(state['layout']['yaxis']['domain'], [0, 1])

        # Check the grid plot on the right

        # (0, 0) - bottom-left
        self.assertEqual(state['data'][1]['y'], np.array([0, 0]))
        self.assertEqual(state['data'][1]['mode'], 'lines')
        self.assertEqual(state['data'][1]['xaxis'], 'x2')
        self.assertEqual(state['data'][1]['yaxis'], 'y2')

        # (1, 0) - bottom-right
        self.assertEqual(state['data'][2]['y'], np.array([1, 0]))
        self.assertEqual(state['data'][2]['mode'], 'lines')
        self.assertEqual(state['data'][2]['xaxis'], 'x3')
        self.assertEqual(state['data'][2]['yaxis'], 'y2')

        # (0, 1) - top-left
        self.assertEqual(state['data'][3]['y'], np.array([0, 1]))
        self.assertEqual(state['data'][3]['mode'], 'lines')
        self.assertEqual(state['data'][3]['xaxis'], 'x2')
        self.assertEqual(state['data'][3]['yaxis'], 'y3')

        # (1, 1) - top-right
        self.assertEqual(state['data'][4]['y'], np.array([1, 1]))
        self.assertEqual(state['data'][4]['mode'], 'lines')
        self.assertEqual(state['data'][4]['xaxis'], 'x3')
        self.assertEqual(state['data'][4]['yaxis'], 'y3')

        # Axes
        self.assertEqual(state['layout']['xaxis2']['domain'], [0.5, 0.75])
        self.assertEqual(state['layout']['xaxis3']['domain'], [0.75, 1.0])
        self.assertEqual(state['layout']['yaxis2']['domain'], [0, 0.5])
        self.assertEqual(state['layout']['yaxis3']['domain'], [0.5, 1.0])

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

    def test_layout_instantiate_subplots(self):
        layout = (Curve(range(10)) + Curve(range(10)) + Image(np.random.rand(10,10)) +
                  Curve(range(10)) + Curve(range(10)))
        plot = plotly_renderer.get_plot(layout)
        positions = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
        self.assertEqual(sorted(plot.subplots.keys()), positions)

    def test_layout_instantiate_subplots_transposed(self):
        layout = (Curve(range(10)) + Curve(range(10)) + Image(np.random.rand(10,10)) +
                  Curve(range(10)) + Curve(range(10)))
        plot = plotly_renderer.get_plot(layout(plot=dict(transpose=True)))
        positions = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]
        self.assertEqual(sorted(plot.subplots.keys()), positions)


@attr(optional=1)
class TestPlotlyFigureGrid(ComparisonTestCase):

    def test_figure_grid_solo_traces(self):

        fig = figure_grid([[
            {'data': [{'type': 'table',
                       'header': {'values': [['One', 'Two']]}}]},
            {'data': [{'type': 'parcoords',
                       'dimensions': [{'values': [1, 2]}]}]}
        ]], column_widths=[0.4, 0.6],
            column_spacing=0)

        # Validate resulting figure object
        go.Figure(fig)

        # Check domains
        self.assertEqual(fig['data'][0]['type'], 'table')
        self.assertEqual(fig['data'][0]['domain'],
                         {'x': [0, 0.4], 'y': [0, 1.0]})

        self.assertEqual(fig['data'][1]['type'], 'parcoords')
        self.assertEqual(fig['data'][1]['domain'],
                         {'x': [0.4, 1.0], 'y': [0, 1.0]})

    def test_figure_grid_polar_subplots(self):
        fig = figure_grid([[
            {'data': [{'type': 'scatterpolar',
                       'theta': [0, 90], 'r': [0.5, 1.0]}]}
        ], [
            {'data': [{'type': 'barpolar',
                       'theta': [90, 180], 'r': [1.0, 10.0]}],
             'layout': {'polar': {'radialaxis': {'title': 'radial'}}}}
        ]], row_spacing=0.1)

        # Validate resulting figure object
        go.Figure(fig)

        # Check domains
        self.assertEqual(fig['data'][0]['type'], 'scatterpolar')
        self.assertEqual(fig['data'][0]['subplot'], 'polar')
        self.assertEqual(fig['layout']['polar']['domain'],
                         {'y': [0, 0.45], 'x': [0, 1.0]})

        self.assertEqual(fig['data'][1]['type'], 'barpolar')
        self.assertEqual(fig['data'][1]['subplot'], 'polar2')
        self.assertEqual(fig['layout']['polar2']['domain'],
                         {'y': [0.55, 1.0], 'x': [0, 1.0]})

        # Check that radial axis title stayed with the barpolar trace's polar
        # subplot
        self.assertEqual(fig['layout']['polar2']['radialaxis'],
                         {'title': 'radial'})

    def test_titles_converted_to_annotations(self):
        fig = figure_grid([[
            {'data': [{'type': 'scatter',
                       'y': [1, 3, 2]}],
             'layout': {'title': 'Scatter!'}}
        ], [
            {'data': [{'type': 'bar',
                       'y': [2, 3, 1]}],
             'layout': {'title': 'Bar!'}}
        ]])

        # Validate resulting figure object
        go.Figure(fig)

        self.assertNotIn('title', fig['layout'])
        self.assertEqual(len(fig['layout']['annotations']), 2)
        self.assertEqual(fig['layout']['annotations'][0]['text'], 'Scatter!')
        self.assertEqual(fig['layout']['annotations'][1]['text'], 'Bar!')


    def test_annotations_stick_with_axis(self):
        fig = figure_grid([[
            {'data': [{'type': 'scatter',
                       'y': [1, 3, 2]}],
             'layout': {
                 'annotations': [
                     {'text': 'One',
                      'xref': 'x', 'yref': 'y',
                      'x': 0, 'y': 0},
                     {'text': 'Two',
                      'xref': 'x', 'yref': 'y',
                      'x': 1, 'y': 0}
                 ]}}
            ,
            {'data': [{'type': 'bar',
                       'y': [2, 3, 1]}],
             'layout': {
                 'annotations': [
                     {'text': 'Three',
                      'xref': 'x', 'yref': 'y',
                      'x': 2, 'y': 0},
                     {'text': 'Four',
                      'xref': 'x', 'yref': 'y',
                      'x': 3, 'y': 0}
                 ]}}
        ]])

        # Validate resulting figure object
        go.Figure(fig)

        annotations = fig['layout']['annotations']
        self.assertEqual(len(annotations), 4)
        self.assertEqual(annotations[0]['text'], 'One')
        self.assertEqual(annotations[0]['xref'], 'x')
        self.assertEqual(annotations[0]['yref'], 'y')

        self.assertEqual(annotations[1]['text'], 'Two')
        self.assertEqual(annotations[1]['xref'], 'x')
        self.assertEqual(annotations[1]['yref'], 'y')

        self.assertEqual(annotations[2]['text'], 'Three')
        self.assertEqual(annotations[2]['xref'], 'x2')
        self.assertEqual(annotations[2]['yref'], 'y2')

        self.assertEqual(annotations[3]['text'], 'Four')
        self.assertEqual(annotations[3]['xref'], 'x2')
        self.assertEqual(annotations[3]['yref'], 'y2')
