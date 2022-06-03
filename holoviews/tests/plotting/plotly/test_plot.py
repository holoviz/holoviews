from param import concrete_descendents

import plotly.graph_objs as go
import pyviz_comms as comms

from holoviews.core import Store
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting.plotly.element import ElementPlot
from holoviews.plotting.plotly.util import figure_grid

from .. import option_intersections

plotly_renderer = Store.renderers['plotly']


class TestPlotlyPlot(ComparisonTestCase):

    def setUp(self):
        self.previous_backend = Store.current_backend
        Store.set_current_backend('plotly')
        self.comm_manager = plotly_renderer.comm_manager
        plotly_renderer.comm_manager = comms.CommManager
        self._padding = {}
        for plot in concrete_descendents(ElementPlot).values():
            self._padding[plot] = plot.padding
            plot.padding = 0

    def tearDown(self):
        Store.current_backend = self.previous_backend
        plotly_renderer.comm_manager = self.comm_manager
        for plot, padding in self._padding.items():
            plot.padding = padding

    def _get_plot_state(self, element):
        fig_dict = plotly_renderer.get_plot_state(element)
        return fig_dict

    def assert_property_values(self, obj, props):
        """
        Assert that a dictionary has the specified properties, handling magic underscore
        notation

        For example

        self.assert_property_values(
            {'a': {'b': 23}, 'c': 42},
            {'a_b': 23, 'c': 42}
        )

        will pass this test
        """

        for prop, val in props.items():
            prop_parts = prop.split('_')
            prop_parent = obj
            for prop_part in prop_parts[:-1]:
                prop_parent = prop_parent.get(prop_part, {})

            self.assertEqual(val, prop_parent[prop_parts[-1]])


class TestPlotDefinitions(TestPlotlyPlot):

    known_clashes = []

    def test_plotly_option_definitions(self):
        # Check option definitions do not introduce new clashes
        self.assertEqual(option_intersections('plotly'), self.known_clashes)


class TestPlotlyFigureGrid(TestPlotlyPlot):

    def test_figure_grid_solo_traces(self):

        fig = figure_grid([[
            {'data': [{'type': 'table',
                       'header': {'values': [['One', 'Two']]}}],
             'layout': {
                 'width': 400,
                 'height': 1000,
             }},
            {'data': [{'type': 'parcoords',
                       'dimensions': [{'values': [1, 2]}]}],
             'layout': {
                 'width': 600,
                 'height': 1000,
             }}
        ]], row_spacing=0, column_spacing=0)

        # Validate resulting figure object
        go.Figure(fig)

        # Check domains
        self.assertEqual(fig['data'][0]['type'], 'table')
        self.assertEqual(fig['data'][0]['domain'],
                         {'x': [0, 0.4], 'y': [0.0, 1.0]})

        self.assertEqual(fig['data'][1]['type'], 'parcoords')
        self.assertEqual(fig['data'][1]['domain'],
                         {'x': [0.4, 1.0], 'y': [0, 1.0]})

        # Check width and height
        self.assertEqual(fig['layout']['width'], 1000)
        self.assertEqual(fig['layout']['height'], 1000)

    def test_figure_grid_solo_traces_fig_width_height(self):

        fig = figure_grid([[
            {'data': [{'type': 'table',
                       'header': {'values': [['One', 'Two']]}}],
             'layout': {'width': 400, 'height': 1000}
             },
            {'data': [{'type': 'parcoords',
                       'dimensions': [{'values': [1, 2]}]}],
             'layout': {'width': 600, 'height': 1000}}
        ]], column_spacing=0)

        # Validate resulting figure object
        go.Figure(fig)

        # Check domains
        self.assertEqual(fig['data'][0]['type'], 'table')
        self.assertEqual(fig['data'][0]['domain'],
                         {'x': [0, 0.4], 'y': [0, 1.0]})

        self.assertEqual(fig['data'][1]['type'], 'parcoords')
        self.assertEqual(fig['data'][1]['domain'],
                         {'x': [0.4, 1.0], 'y': [0, 1.0]})

        # Check width and height
        self.assertEqual(fig['layout']['width'], 1000)
        self.assertEqual(fig['layout']['height'], 1000)

    def test_figure_grid_polar_subplots(self):
        fig = figure_grid([[
            {'data': [{'type': 'scatterpolar',
                       'theta': [0, 90], 'r': [0.5, 1.0]}],
             'layout': {
                 'width': 450,
                 'height': 450,
             }}
        ], [
            {'data': [{'type': 'barpolar',
                       'theta': [90, 180], 'r': [1.0, 10.0]}],
             'layout': {
                 'width': 450,
                 'height': 450,
                 'polar': {'radialaxis': {'title': 'radial'}}
             }}
        ]], row_spacing=100)

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

        # Check width and height
        self.assertEqual(fig['layout']['width'], 450)
        self.assertEqual(fig['layout']['height'], 1000)

        # Check that radial axis title stayed with the barpolar trace's polar
        # subplot
        self.assertEqual(fig['layout']['polar2']['radialaxis'],
                         {'title': 'radial'})

    def test_titles_converted_to_annotations(self):
        fig = figure_grid([[
            {'data': [{'type': 'scatter',
                       'y': [1, 3, 2]}],
             'layout': {
                 'width': 400,
                 'height': 400,
                 'title': 'Scatter!'
             }}
        ], [
            {'data': [{'type': 'bar',
                       'y': [2, 3, 1]}],
             'layout': {
                 'width': 400,
                 'height': 400,
                 'title': 'Bar!'
             }}
        ]], row_spacing=100)

        # Validate resulting figure object
        go.Figure(fig)

        self.assertNotIn('title', fig['layout'])
        self.assertEqual(len(fig['layout']['annotations']), 2)
        self.assertEqual(fig['layout']['annotations'][0]['text'], 'Scatter!')
        self.assertEqual(fig['layout']['annotations'][1]['text'], 'Bar!')

        # Check width and height
        self.assertEqual(fig['layout']['width'], 400)
        self.assertEqual(fig['layout']['height'], 900)

    def test_annotations_stick_with_axis(self):
        fig = figure_grid([[
            {'data': [{'type': 'scatter',
                       'y': [1, 3, 2]}],
             'layout': {
                 'width': 400,
                 'height': 400,
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
                 'width': 400,
                 'height': 400,
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

    def test_shapes_stick_with_axis(self):
        fig = figure_grid([[
            {'data': [{'type': 'scatter',
                       'y': [1, 3, 2]}],
             'layout': {
                 'width': 400,
                 'height': 400,
                 'shapes': [
                     {'type': 'rect',
                      'xref': 'x', 'yref': 'y',
                      'x0': 0, 'y0': 0, 'x1': 1, 'y1': 1},
                     {'type': 'circle',
                      'xref': 'x', 'yref': 'y',
                      'x0': 1, 'y0': 1, 'x1': 2, 'y1': 2}
                 ]}}
            ,
            {'data': [{'type': 'bar',
                       'y': [2, 3, 1]}],
             'layout': {
                 'width': 400,
                 'height': 400,
                 'shapes': [
                     {'type': 'line',
                      'xref': 'x', 'yref': 'y',
                      'x0': 2, 'y0': 0, 'x1': 1, 'y1': 3},
                     {'type': 'path',
                      'xref': 'x', 'yref': 'y',
                      'x0': 3, 'y0': 0, 'x1': 3, 'y1': 6}
                 ]}}
        ]])

        # Validate resulting figure object
        go.Figure(fig)

        shapes = fig['layout']['shapes']
        self.assertEqual(len(shapes), 4)
        self.assertEqual(shapes[0]['type'], 'rect')
        self.assertEqual(shapes[0]['xref'], 'x')
        self.assertEqual(shapes[0]['yref'], 'y')

        self.assertEqual(shapes[1]['type'], 'circle')
        self.assertEqual(shapes[1]['xref'], 'x')
        self.assertEqual(shapes[1]['yref'], 'y')

        self.assertEqual(shapes[2]['type'], 'line')
        self.assertEqual(shapes[2]['xref'], 'x2')
        self.assertEqual(shapes[2]['yref'], 'y2')

        self.assertEqual(shapes[3]['type'], 'path')
        self.assertEqual(shapes[3]['xref'], 'x2')
        self.assertEqual(shapes[3]['yref'], 'y2')

    def test_images_stick_with_axis(self):
        fig = figure_grid([[
            {'data': [{'type': 'scatter',
                       'y': [1, 3, 2]}],
             'layout': {
                 'width': 400,
                 'height': 400,
                 'images': [
                     {'source': 'One.png',
                      'xref': 'x', 'yref': 'y',
                      'x': 0, 'y': 0},
                     {'source': 'Two.png',
                      'xref': 'x', 'yref': 'y',
                      'x': 1, 'y': 0}
                 ]}
             }
            ,
            {'data': [{'type': 'bar',
                       'y': [2, 3, 1]}],
             'layout': {
                 'width': 400,
                 'height': 400,
                 'images': [
                         {'source': 'Three.png',
                          'xref': 'x', 'yref': 'y',
                          'x': 2, 'y': 0},
                         {'source': 'Four.png',
                          'xref': 'x', 'yref': 'y',
                          'x': 3, 'y': 0}
                 ]}}
        ]])

        # Validate resulting figure object
        go.Figure(fig)

        images = fig['layout']['images']
        self.assertEqual(len(images), 4)
        self.assertEqual(images[0]['source'], 'One.png')
        self.assertEqual(images[0]['xref'], 'x')
        self.assertEqual(images[0]['yref'], 'y')

        self.assertEqual(images[1]['source'], 'Two.png')
        self.assertEqual(images[1]['xref'], 'x')
        self.assertEqual(images[1]['yref'], 'y')

        self.assertEqual(images[2]['source'], 'Three.png')
        self.assertEqual(images[2]['xref'], 'x2')
        self.assertEqual(images[2]['yref'], 'y2')

        self.assertEqual(images[3]['source'], 'Four.png')
        self.assertEqual(images[3]['xref'], 'x2')
        self.assertEqual(images[3]['yref'], 'y2')

    def test_width_height_with_spacing(self):
        fig = figure_grid([[
            {'data': [{'type': 'scatter',
                       'y': [1, 3, 2]}],
             'layout': {
                 'width': 380,
                 'height': 380}}
            ,
            {'data': [{'type': 'bar',
                       'y': [2, 3, 1]}],
             'layout': {
                 'width': 380,
                 'height': 380}}
        ], [
            {'data': [{'type': 'scatterpolar',
                       'theta': [0, 90], 'r': [0.5, 1.0]}],
             'layout': {
                 'width': 380,
                 'height': 1140,
             }},
            {'data': [{'type': 'table',
                       'header': {'values': [['One', 'Two']]}}],
             'layout': {
                 'width': 380,
                 'height': 1140}
             }
        ]], column_spacing=40, row_spacing=80,
            width=400, height=800)

        # Check domains
        expected_x_domains = [[0, 0.45], [0.55, 1]]
        expected_y_domains = [[0, 0.225], [0.325, 1]]

        # scatter
        self.assertEqual(fig['layout']['xaxis']['domain'], expected_x_domains[0])
        self.assertEqual(fig['layout']['yaxis']['domain'], expected_y_domains[0])

        # bar
        self.assertEqual(fig['layout']['xaxis2']['domain'], expected_x_domains[1])
        self.assertEqual(fig['layout']['yaxis2']['domain'], expected_y_domains[0])

        # scatterpolar
        self.assertEqual(fig['layout']['polar']['domain']['x'], expected_x_domains[0])
        self.assertEqual(fig['layout']['polar']['domain']['y'], expected_y_domains[1])

        # table
        self.assertEqual(fig['data'][3]['domain']['x'], expected_x_domains[1])
        self.assertEqual(fig['data'][3]['domain']['y'], expected_y_domains[1])

        # Check width and height
        self.assertEqual(fig['layout']['width'], 400)
        self.assertEqual(fig['layout']['height'], 800)

    def test_axis_matching_offset(self):
        fig = figure_grid([[
            {'data': [{'type': 'scatter',
                       'y': [1, 3, 2],
                       'xaxis': 'x',
                       'yaxis': 'y'},
                      {'type': 'bar',
                       'y': [2, 3, 1],
                       'xaxis': 'x2',
                       'yaxis': 'y2',
                       }],
             'layout': {
                 'width': 400,
                 'height': 400,
                 'xaxis2': {'matches': 'x'},
                 'yaxis2': {'matches': 'y'},
             }},
        ], [
            {'data': [{'type': 'scatter',
                       'y': [1, 3, 2],
                       'xaxis': 'x',
                       'yaxis': 'y',
                       },
                      {'type': 'bar',
                       'y': [2, 3, 1],
                       'xaxis': 'x2',
                       'yaxis': 'y2',
                       }],
             'layout': {
                 'width': 400,
                 'height': 400,
                 'xaxis2': {'matches': 'y'},
                 'yaxis2': {'matches': 'x'},
             }},
        ]], column_spacing=0, row_spacing=0)

        # Check axes that traces are associated with
        self.assertEqual(fig['data'][0]['xaxis'], 'x')
        self.assertEqual(fig['data'][0]['yaxis'], 'y')

        self.assertEqual(fig['data'][1]['xaxis'], 'x2')
        self.assertEqual(fig['data'][1]['yaxis'], 'y2')

        self.assertEqual(fig['data'][2]['xaxis'], 'x3')
        self.assertEqual(fig['data'][2]['yaxis'], 'y3')

        self.assertEqual(fig['data'][3]['xaxis'], 'x4')
        self.assertEqual(fig['data'][3]['yaxis'], 'y4')

        # Check matches references
        self.assertEqual(fig['layout']['xaxis'].get('matches', None), None)
        self.assertEqual(fig['layout']['yaxis'].get('matches', None), None)
        self.assertEqual(fig['layout']['xaxis2'].get('matches', None), 'x')
        self.assertEqual(fig['layout']['yaxis2'].get('matches', None), 'y')
        self.assertEqual(fig['layout']['xaxis3'].get('matches', None), None)
        self.assertEqual(fig['layout']['yaxis3'].get('matches', None), None)
        self.assertEqual(fig['layout']['xaxis4'].get('matches', None), 'y3')
        self.assertEqual(fig['layout']['yaxis4'].get('matches', None), 'x3')
