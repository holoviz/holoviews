from unittest import SkipTest

from holoviews.core import Store
from holoviews.element.comparison import ComparisonTestCase
import pyviz_comms as comms

try:
    import holoviews.plotting.plotly # noqa (Activate backend)
    plotly_renderer = Store.renderers['plotly']
    from holoviews.plotting.plotly.util import figure_grid
    import plotly.graph_objs as go
except:
    plotly_renderer = None

from .. import option_intersections



class TestPlotlyPlot(ComparisonTestCase):

    def setUp(self):
        if not plotly_renderer:
            raise SkipTest("Plotly required to test plot instantiation")
        self.previous_backend = Store.current_backend
        Store.current_backend = 'plotly'
        self.comm_manager = plotly_renderer.comm_manager
        plotly_renderer.comm_manager = comms.CommManager

    def tearDown(self):
        Store.current_backend = self.previous_backend
        plotly_renderer.comm_manager = self.comm_manager

    def _get_plot_state(self, element):
        plot = plotly_renderer.get_plot(element)
        plot.initialize_plot()

        # Pass to plotly.py for full property validation
        go.Figure(plot.state)

        return plot.state


class TestPlotDefinitions(TestPlotlyPlot):

    known_clashes = []

    def test_plotly_option_definitions(self):
        # Check option definitions do not introduce new clashes
        self.assertEqual(option_intersections('plotly'), self.known_clashes)


class TestPlotlyFigureGrid(TestPlotlyPlot):

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
