from holoviews.element import (
    VLine, HLine, Bounds, Box, Rectangles, Segments, Tiles, Path
)
import numpy as np
from .test_plot import TestPlotlyPlot

default_shape_color = '#2a3f5f'

class TestShape(TestPlotlyPlot):
    def assert_shape_element_styling(self, element):
        props = dict(
            fillcolor='orange',
            line_color='yellow',
            line_dash='dot',
            line_width=5,
            opacity=0.7
        )

        element = element.clone().opts(**props)
        state = self._get_plot_state(element)
        shapes = state['layout']['shapes']
        self.assert_property_values(shapes[0], props)


class TestMapboxShape(TestPlotlyPlot):
    def setUp(self):
        super().setUp()

        # Precompute coordinates
        self.xs = [3000000, 2000000, 1000000]
        self.ys = [-3000000, -2000000, -1000000]
        self.x_range = (-5000000, 4000000)
        self.x_center = sum(self.x_range) / 2.0
        self.y_range = (-3000000, 2000000)
        self.y_center = sum(self.y_range) / 2.0
        self.lon_range, self.lat_range = Tiles.easting_northing_to_lon_lat(self.x_range, self.y_range)
        self.lon_centers, self.lat_centers = Tiles.easting_northing_to_lon_lat(
            [self.x_center], [self.y_center]
        )
        self.lon_center, self.lat_center = self.lon_centers[0], self.lat_centers[0]
        self.lons, self.lats = Tiles.easting_northing_to_lon_lat(self.xs, self.ys)


class TestVLineHLine(TestShape):

    def assert_vline(self, shape, x, xref='x', ydomain=(0, 1)):
        self.assertEqual(shape['type'], 'line')
        self.assertEqual(shape['x0'], x)
        self.assertEqual(shape['x1'], x)
        self.assertEqual(shape['xref'], xref)

        self.assertEqual(shape['y0'], ydomain[0])
        self.assertEqual(shape['y1'], ydomain[1])
        self.assertEqual(shape['yref'], 'paper')

    def assert_hline(self, shape, y, yref='y', xdomain=(0, 1)):
        self.assertEqual(shape['type'], 'line')
        self.assertEqual(shape['y0'], y)
        self.assertEqual(shape['y1'], y)
        self.assertEqual(shape['yref'], yref)

        self.assertEqual(shape['x0'], xdomain[0])
        self.assertEqual(shape['x1'], xdomain[1])
        self.assertEqual(shape['xref'], 'paper')

    def test_single_vline(self):
        vline = VLine(3)
        state = self._get_plot_state(vline)

        shapes = state['layout']['shapes']
        self.assertEqual(len(shapes), 1)
        self.assert_vline(shapes[0], 3)

    def test_single_hline(self):
        hline = HLine(3)
        state = self._get_plot_state(hline)

        shapes = state['layout']['shapes']
        self.assertEqual(len(shapes), 1)
        self.assert_hline(shapes[0], 3)

    def test_vline_layout(self):
        layout = (VLine(1) + VLine(2) +
                  VLine(3) + VLine(4)).cols(2).opts(vspacing=0, hspacing=0)
        state = self._get_plot_state(layout)

        shapes = state['layout']['shapes']
        self.assertEqual(len(shapes), 4)

        # Check shapes
        self.assert_vline(shapes[0], 3, xref='x', ydomain=[0.0, 0.5])
        self.assert_vline(shapes[1], 4, xref='x2', ydomain=[0.0, 0.5])
        self.assert_vline(shapes[2], 1, xref='x3', ydomain=[0.5, 1.0])
        self.assert_vline(shapes[3], 2, xref='x4', ydomain=[0.5, 1.0])

    def test_hline_layout(self):
        layout = (HLine(1) + HLine(2) +
                  HLine(3) + HLine(4)).cols(2).opts(vspacing=0, hspacing=0)
        state = self._get_plot_state(layout)

        shapes = state['layout']['shapes']
        self.assertEqual(len(shapes), 4)

        # Check shapes
        self.assert_hline(shapes[0], 3, yref='y', xdomain=[0.0, 0.5])
        self.assert_hline(shapes[1], 4, yref='y2', xdomain=[0.5, 1.0])
        self.assert_hline(shapes[2], 1, yref='y3', xdomain=[0.0, 0.5])
        self.assert_hline(shapes[3], 2, yref='y4', xdomain=[0.5, 1.0])

    def test_vline_styling(self):
        self.assert_shape_element_styling(VLine(3))

    def test_hline_styling(self):
        self.assert_shape_element_styling(HLine(3))


class TestPathShape(TestShape):
    def assert_path_shape_element(self, shape, element, xref='x', yref='y'):
        # Check type
        self.assertEqual(shape['type'], 'path')

        # Check svg path
        expected_path = 'M' + 'L'.join([
            f'{x} {y}' for x, y in
            zip(element.dimension_values(0), element.dimension_values(1))]) + 'Z'

        self.assertEqual(shape['path'], expected_path)

        # Check axis references
        self.assertEqual(shape['xref'], xref)
        self.assertEqual(shape['yref'], yref)

    def test_simple_path(self):
        path = Path([(0, 0), (1, 1), (0, 1), (0, 0)])
        state = self._get_plot_state(path)
        shapes = state['layout']['shapes']
        self.assertEqual(len(shapes), 1)
        self.assert_path_shape_element(shapes[0], path)
        self.assert_shape_element_styling(path)


class TestMapboxPathShape(TestMapboxShape):
    def test_simple_path(self):
        path = Tiles("") * Path([
            (self.x_range[0], self.y_range[0]),
            (self.x_range[1], self.y_range[1]),
            (self.x_range[0], self.y_range[1]),
            (self.x_range[0], self.y_range[0]),
        ]).redim.range(
            x=self.x_range, y=self.y_range
        )

        state = self._get_plot_state(path)
        self.assertEqual(state["data"][1]["type"], "scattermapbox")
        self.assertEqual(state["data"][1]["mode"], "lines")
        self.assertEqual(state["data"][1]["lon"], np.array([
            self.lon_range[i] for i in (0, 1, 0, 0)
        ] + [np.nan]))
        self.assertEqual(state["data"][1]["lat"], np.array([
            self.lat_range[i] for i in (0, 1, 1, 0)
        ] + [np.nan]))
        self.assertEqual(state["data"][1]["showlegend"], False)
        self.assertEqual(state["data"][1]["line"]["color"], default_shape_color)
        self.assertEqual(
            state['layout']['mapbox']['center'], {
                'lat': self.lat_center, 'lon': self.lon_center
            }
        )


class TestBounds(TestPathShape):

    def test_single_bounds(self):
        bounds = Bounds((1, 2, 3, 4))

        state = self._get_plot_state(bounds)

        shapes = state['layout']['shapes']
        self.assertEqual(len(shapes), 1)
        self.assert_path_shape_element(shapes[0], bounds)

    def test_bounds_layout(self):
        bounds1 = Bounds((0, 0, 1, 1))
        bounds2 = Bounds((0, 0, 2, 2))
        bounds3 = Bounds((0, 0, 3, 3))
        bounds4 = Bounds((0, 0, 4, 4))

        layout = (bounds1 + bounds2 +
                  bounds3 + bounds4).cols(2)

        state = self._get_plot_state(layout)

        shapes = state['layout']['shapes']
        self.assertEqual(len(shapes), 4)

        # Check shapes
        self.assert_path_shape_element(shapes[0], bounds3, xref='x', yref='y')
        self.assert_path_shape_element(shapes[1], bounds4, xref='x2', yref='y2')
        self.assert_path_shape_element(shapes[2], bounds1, xref='x3', yref='y3')
        self.assert_path_shape_element(shapes[3], bounds2, xref='x4', yref='y4')

    def test_bounds_styling(self):
        self.assert_shape_element_styling(Bounds((1, 2, 3, 4)))


class TestMapboxBounds(TestMapboxShape):
    def test_single_bounds(self):
        bounds = Tiles("") * Bounds(
            (self.x_range[0], self.y_range[0], self.x_range[1], self.y_range[1])
        ).redim.range(
            x=self.x_range, y=self.y_range
        )

        state = self._get_plot_state(bounds)
        self.assertEqual(state["data"][1]["type"], "scattermapbox")
        self.assertEqual(state["data"][1]["mode"], "lines")
        self.assertEqual(state["data"][1]["lon"], np.array([
            self.lon_range[i] for i in (0, 0, 1, 1, 0)
        ]))
        self.assertEqual(state["data"][1]["lat"], np.array([
            self.lat_range[i] for i in (0, 1, 1, 0, 0)
        ]))
        self.assertEqual(state["data"][1]["showlegend"], False)
        self.assertEqual(state["data"][1]["line"]["color"], default_shape_color)
        self.assertEqual(
            state['layout']['mapbox']['center'], {
                'lat': self.lat_center, 'lon': self.lon_center
            }
        )

    def test_bounds_layout(self):
        bounds1 = Bounds((0, 0, 1, 1))
        bounds2 = Bounds((0, 0, 2, 2))
        bounds3 = Bounds((0, 0, 3, 3))
        bounds4 = Bounds((0, 0, 4, 4))

        layout = (Tiles("") * bounds1 + Tiles("") * bounds2 +
                  Tiles("") * bounds3 + Tiles("") * bounds4).cols(2)

        state = self._get_plot_state(layout)
        self.assertEqual(state['data'][1]["subplot"], "mapbox")
        self.assertEqual(state['data'][3]["subplot"], "mapbox2")
        self.assertEqual(state['data'][5]["subplot"], "mapbox3")
        self.assertEqual(state['data'][7]["subplot"], "mapbox4")
        self.assertNotIn("xaxis", state['layout'])
        self.assertNotIn("yaxis", state['layout'])


class TestBox(TestPathShape):

    def test_single_box(self):
        box = Box(0, 0, (1, 2), orientation=1)
        state = self._get_plot_state(box)
        shapes = state['layout']['shapes']
        self.assertEqual(len(shapes), 1)
        self.assert_path_shape_element(shapes[0], box)

    def test_bounds_layout(self):
        box1 = Box(0, 0, (1, 1), orientation=0)
        box2 = Box(0, 0, (2, 2), orientation=0.5)
        box3 = Box(0, 0, (3, 3), orientation=1.0)
        box4 = Box(0, 0, (4, 4), orientation=1.5)

        layout = (box1 + box2 +
                  box3 + box4).cols(2)

        state = self._get_plot_state(layout)

        shapes = state['layout']['shapes']
        self.assertEqual(len(shapes), 4)

        # Check shapes
        self.assert_path_shape_element(shapes[0], box3, xref='x', yref='y')
        self.assert_path_shape_element(shapes[1], box4, xref='x2', yref='y2')
        self.assert_path_shape_element(shapes[2], box1, xref='x3', yref='y3')
        self.assert_path_shape_element(shapes[3], box2, xref='x4', yref='y4')

    def test_box_styling(self):
        self.assert_shape_element_styling(Box(0, 0, (1, 1)))


class TestMapboxBox(TestMapboxShape):
    def test_single_box(self):
        box = Tiles("") * Box(0, 0, (1000000, 2000000)).redim.range(
            x=self.x_range, y=self.y_range
        )

        x_box_range = [-500000, 500000]
        y_box_range = [-1000000, 1000000]
        lon_box_range, lat_box_range = Tiles.easting_northing_to_lon_lat(x_box_range, y_box_range)

        state = self._get_plot_state(box)
        self.assertEqual(state["data"][1]["type"], "scattermapbox")
        self.assertEqual(state["data"][1]["mode"], "lines")
        self.assertEqual(state["data"][1]["showlegend"], False)
        self.assertEqual(state["data"][1]["line"]["color"], default_shape_color)
        self.assertEqual(state["data"][1]["lon"], np.array([
            lon_box_range[i] for i in (0, 0, 1, 1, 0)
        ]))
        self.assertEqual(state["data"][1]["lat"], np.array([
            lat_box_range[i] for i in (0, 1, 1, 0, 0)
        ]))

        self.assertEqual(
            state['layout']['mapbox']['center'], {
                'lat': self.lat_center, 'lon': self.lon_center
            }
        )


class TestRectangles(TestPathShape):

    def test_boxes_simple(self):
        boxes = Rectangles([(0, 0, 1, 1), (2, 2, 4, 3)])
        state = self._get_plot_state(boxes)
        shapes = state['layout']['shapes']
        self.assertEqual(len(shapes), 2)
        self.assertEqual(shapes[0], {'type': 'rect', 'x0': 0, 'y0': 0, 'x1': 1,
                                     'y1': 1, 'xref': 'x', 'yref': 'y', 'name': '',
                                     'line': {'color': default_shape_color}})
        self.assertEqual(shapes[1], {'type': 'rect', 'x0': 2, 'y0': 2, 'x1': 4,
                                     'y1': 3, 'xref': 'x', 'yref': 'y', 'name': '',
                                     'line': {'color': default_shape_color}})
        self.assertEqual(state['layout']['xaxis']['range'], [0, 4])
        self.assertEqual(state['layout']['yaxis']['range'], [0, 3])


class TestMapboxRectangles(TestMapboxShape):
    def test_rectangles_simple(self):
        rectangles = Tiles("") * Rectangles([
            (0, 0, self.x_range[1], self.y_range[1]),
            (self.x_range[0], self.y_range[0], 0, 0),
        ]).redim.range(
            x=self.x_range, y=self.y_range
        )

        state = self._get_plot_state(rectangles)
        self.assertEqual(state["data"][1]["type"], "scattermapbox")
        self.assertEqual(state["data"][1]["mode"], "lines")
        self.assertEqual(state["data"][1]["showlegend"], False)
        self.assertEqual(state["data"][1]["line"]["color"], default_shape_color)
        self.assertEqual(state["data"][1]["lon"], np.array([
            0, 0, self.lon_range[1], self.lon_range[1], 0, np.nan,
            self.lon_range[0], self.lon_range[0], 0, 0, self.lon_range[0], np.nan
        ]))
        self.assertEqual(state["data"][1]["lat"], np.array([
            0, self.lat_range[1], self.lat_range[1], 0, 0, np.nan,
            self.lat_range[0], 0, 0, self.lat_range[0], self.lat_range[0], np.nan
        ]))

        self.assertEqual(
            state['layout']['mapbox']['center'], {
                'lat': self.lat_center, 'lon': self.lon_center
            }
        )


class TestSegments(TestPathShape):

    def test_segments_simple(self):
        boxes = Segments([(0, 0, 1, 1), (2, 2, 4, 3)])
        state = self._get_plot_state(boxes)
        shapes = state['layout']['shapes']
        self.assertEqual(len(shapes), 2)
        self.assertEqual(shapes[0], {'type': 'line', 'x0': 0, 'y0': 0, 'x1': 1,
                                     'y1': 1, 'xref': 'x', 'yref': 'y', 'name': '',
                                     'line': {'color': default_shape_color}})
        self.assertEqual(shapes[1], {'type': 'line', 'x0': 2, 'y0': 2, 'x1': 4,
                                     'y1': 3, 'xref': 'x', 'yref': 'y', 'name': '',
                                     'line': {'color': default_shape_color}})
        self.assertEqual(state['layout']['xaxis']['range'], [0, 4])
        self.assertEqual(state['layout']['yaxis']['range'], [0, 3])


class TestMapboxSegments(TestMapboxShape):
    def test_segments_simple(self):
        rectangles = Tiles("") * Segments([
            (0, 0, self.x_range[1], self.y_range[1]),
            (self.x_range[0], self.y_range[0], 0, 0),
        ]).redim.range(
            x=self.x_range, y=self.y_range
        )

        state = self._get_plot_state(rectangles)
        self.assertEqual(state["data"][1]["type"], "scattermapbox")
        self.assertEqual(state["data"][1]["mode"], "lines")
        self.assertEqual(state["data"][1]["showlegend"], False)
        self.assertEqual(state["data"][1]["line"]["color"], default_shape_color)
        self.assertEqual(state["data"][1]["lon"], np.array([
            0, self.lon_range[1], np.nan,
            self.lon_range[0], 0, np.nan
        ]))
        self.assertEqual(state["data"][1]["lat"], np.array([
            0, self.lat_range[1], np.nan,
            self.lat_range[0], 0, np.nan
        ]))

        self.assertEqual(
            state['layout']['mapbox']['center'], {
                'lat': self.lat_center, 'lon': self.lon_center
            }
        )
