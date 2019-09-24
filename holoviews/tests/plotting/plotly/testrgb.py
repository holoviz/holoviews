import numpy as np

import PIL.Image

try:
    import plotly.graph_objs as go
except:
    go = None

from holoviews.element import RGB

from .testplot import TestPlotlyPlot, plotly_renderer


class TestRGBPlot(TestPlotlyPlot):

    @staticmethod
    def rgb_element_to_pil_img(rgb_data):
        return PIL.Image.fromarray(np.clip(rgb_data * 255, 0, 255).astype('uint8'))

    def test_rgb(self):
        rgb_data = np.random.rand(10, 10, 3)
        rgb = RGB(rgb_data)
        fig_dict = plotly_renderer.get_plot_state(rgb)
        x_range = fig_dict['layout']['xaxis']['range']
        self.assertEqual(x_range[0], -0.5)
        self.assertEqual(x_range[1], 0.5)

        y_range = fig_dict['layout']['yaxis']['range']
        self.assertEqual(y_range[0], -0.5)
        self.assertEqual(y_range[1], 0.5)

        # Check layout.image object
        images = fig_dict['layout']['images']
        self.assertEqual(len(images), 1)
        image = images[0]

        # Check location properties
        self.assert_property_values(image, {
            'xref': 'x',
            'yref': 'y',
            'x': -0.5,
            'y': 0.5,
            'sizex': 1.0,
            'sizey': 1.0,
            'sizing': 'stretch',
            'layer': 'above',
        })

        # Check image itself
        pil_img = self.rgb_element_to_pil_img(rgb.data)
        expected_source = go.layout.Image(source=pil_img).source
        self.assertEqual(image['source'], expected_source)

    def test_rgb_invert_xaxis(self):
        rgb_data = np.random.rand(10, 10, 3)
        rgb = RGB(rgb_data).opts(invert_xaxis=True)

        fig_dict = plotly_renderer.get_plot_state(rgb)
        x_range = fig_dict['layout']['xaxis']['range']
        self.assertEqual(x_range[0], 0.5)
        self.assertEqual(x_range[1], -0.5)

        y_range = fig_dict['layout']['yaxis']['range']
        self.assertEqual(y_range[0], -0.5)
        self.assertEqual(y_range[1], 0.5)

        # Check layout.image object
        images = fig_dict['layout']['images']
        self.assertEqual(len(images), 1)
        image = images[0]

        # Check location properties
        self.assert_property_values(image, {
            'xref': 'x',
            'yref': 'y',
            'x': 0.5,
            'y': 0.5,
            'sizex': -1.0,
            'sizey': 1.0,
            'sizing': 'stretch',
            'layer': 'above',
        })

        # Check image itself
        pil_img = self.rgb_element_to_pil_img(rgb.data)

        # Flip left-to-right since x-axis is inverted
        pil_img = pil_img.transpose(PIL.Image.FLIP_LEFT_RIGHT)

        expected_source = go.layout.Image(source=pil_img).source

        self.assertEqual(image['source'], expected_source)

    def test_rgb_invert_xaxis_and_yaxis(self):
        rgb_data = np.random.rand(10, 10, 3)
        rgb = RGB(rgb_data).opts(invert_xaxis=True, invert_yaxis=True)

        fig_dict = plotly_renderer.get_plot_state(rgb)
        x_range = fig_dict['layout']['xaxis']['range']
        self.assertEqual(x_range[0], 0.5)
        self.assertEqual(x_range[1], -0.5)

        y_range = fig_dict['layout']['yaxis']['range']
        self.assertEqual(y_range[0], 0.5)
        self.assertEqual(y_range[1], -0.5)

        # Check layout.image object
        images = fig_dict['layout']['images']
        self.assertEqual(len(images), 1)
        image = images[0]

        # Check location properties
        self.assert_property_values(image, {
            'xref': 'x',
            'yref': 'y',
            'x': 0.5,
            'y': -0.5,
            'sizex': -1.0,
            'sizey': -1.0,
            'sizing': 'stretch',
            'layer': 'above',
        })

        # Check image itself
        pil_img = self.rgb_element_to_pil_img(rgb.data)

        # Flip left-to-right and top-to-bottom since both x-axis and y-axis are inverted
        pil_img = (pil_img
                   .transpose(PIL.Image.FLIP_LEFT_RIGHT)
                   .transpose(PIL.Image.FLIP_TOP_BOTTOM))

        expected_source = go.layout.Image(source=pil_img).source

        self.assertEqual(image['source'], expected_source)

    def test_rgb_invert_axes(self):
        rgb_data = np.random.rand(10, 10, 3)
        rgb = RGB(rgb_data).opts(invert_axes=True)

        fig_dict = plotly_renderer.get_plot_state(rgb)
        x_range = fig_dict['layout']['xaxis']['range']
        self.assertEqual(x_range[0], -0.5)
        self.assertEqual(x_range[1], 0.5)

        y_range = fig_dict['layout']['yaxis']['range']
        self.assertEqual(y_range[0], -0.5)
        self.assertEqual(y_range[1], 0.5)

        # Check layout.image object
        images = fig_dict['layout']['images']
        self.assertEqual(len(images), 1)
        image = images[0]

        # Check location properties
        self.assert_property_values(image, {
            'xref': 'x',
            'yref': 'y',
            'x': -0.5,
            'y': 0.5,
            'sizex': 1.0,
            'sizey': 1.0,
            'sizing': 'stretch',
            'layer': 'above',
        })

        # Check image itself
        pil_img = self.rgb_element_to_pil_img(rgb.data)

        # Flip left-to-right and top-to-bottom since both x-axis and y-axis are inverted
        pil_img = (pil_img
                   .transpose(PIL.Image.ROTATE_90)
                   .transpose(PIL.Image.FLIP_LEFT_RIGHT))

        expected_source = go.layout.Image(source=pil_img).source

        self.assertEqual(image['source'], expected_source)

    def test_rgb_invert_xaxis_and_yaxis_and_axes(self):
        rgb_data = np.random.rand(10, 10, 3)
        rgb = RGB(rgb_data).opts(invert_xaxis=True, invert_yaxis=True, invert_axes=True)

        fig_dict = plotly_renderer.get_plot_state(rgb)
        x_range = fig_dict['layout']['xaxis']['range']
        self.assertEqual(x_range[0], 0.5)
        self.assertEqual(x_range[1], -0.5)

        y_range = fig_dict['layout']['yaxis']['range']
        self.assertEqual(y_range[0], 0.5)
        self.assertEqual(y_range[1], -0.5)

        # Check layout.image object
        images = fig_dict['layout']['images']
        self.assertEqual(len(images), 1)
        image = images[0]

        # Check location properties
        self.assert_property_values(image, {
            'xref': 'x',
            'yref': 'y',
            'x': 0.5,
            'y': -0.5,
            'sizex': -1.0,
            'sizey': -1.0,
            'sizing': 'stretch',
            'layer': 'above',
        })

        # Check image itself
        pil_img = self.rgb_element_to_pil_img(rgb.data)

        # Flip left-to-right and top-to-bottom since both x-axis and y-axis are inverted
        pil_img = (pil_img
                   .transpose(PIL.Image.ROTATE_270)
                   .transpose(PIL.Image.FLIP_LEFT_RIGHT))

        expected_source = go.layout.Image(source=pil_img).source

        self.assertEqual(image['source'], expected_source)

    def test_rgb_opacity(self):
        rgb_data = np.random.rand(10, 10, 3)
        rgb = RGB(rgb_data).opts(opacity=0.5)
        fig_dict = plotly_renderer.get_plot_state(rgb)

        # Check layout.image object
        images = fig_dict['layout']['images']
        self.assertEqual(len(images), 1)
        image = images[0]

        # Check location properties
        self.assert_property_values(image, {
            'xref': 'x',
            'yref': 'y',
            'x': -0.5,
            'y': 0.5,
            'sizex': 1.0,
            'sizey': 1.0,
            'sizing': 'stretch',
            'layer': 'above',
            'opacity': 0.5,
        })

        # Check image itself
        pil_img = self.rgb_element_to_pil_img(rgb_data)
        expected_source = go.layout.Image(source=pil_img).source
        self.assertEqual(image['source'], expected_source)
