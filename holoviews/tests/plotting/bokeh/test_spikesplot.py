import datetime as dt

import numpy as np
from holoviews.core import NdOverlay
from holoviews.element import Spikes

from bokeh.models import CategoricalColorMapper, LinearColorMapper
from holoviews.plotting.bokeh.util import property_to_dict

from ..utils import ParamLogStream
from .test_plot import TestBokehPlot, bokeh_renderer


class TestSpikesPlot(TestBokehPlot):

    def test_spikes_colormapping(self):
        spikes = Spikes(np.random.rand(20, 2), vdims=['Intensity'])
        color_spikes = spikes.opts(color_index=1)
        self._test_colormapping(color_spikes, 1)

    def test_empty_spikes_plot(self):
        spikes = Spikes([], vdims=['Intensity'])
        plot = bokeh_renderer.get_plot(spikes)
        source = plot.handles['source']
        self.assertEqual(len(source.data['x']), 0)
        self.assertEqual(len(source.data['y0']), 0)
        self.assertEqual(len(source.data['y1']), 0)

    def test_batched_spike_plot(self):
        overlay = NdOverlay({
            i: Spikes([i], kdims=['Time']).opts(
                position=0.1*i, spike_length=0.1, show_legend=False)
            for i in range(10)
        })
        plot = bokeh_renderer.get_plot(overlay)
        extents = plot.get_extents(overlay, {})
        self.assertEqual(extents, (0, 0, 9, 1))

    def test_spikes_padding_square(self):
        spikes = Spikes([1, 2, 3]).opts(padding=0.1)
        plot = bokeh_renderer.get_plot(spikes)
        x_range = plot.handles['x_range']
        self.assertEqual(x_range.start, 0.8)
        self.assertEqual(x_range.end, 3.2)

    def test_spikes_padding_square_heights(self):
        spikes = Spikes([(1, 1), (2, 2), (3, 3)], vdims=['Height']).opts(padding=0.1)
        plot = bokeh_renderer.get_plot(spikes)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 0.8)
        self.assertEqual(x_range.end, 3.2)
        self.assertEqual(y_range.start, 0)
        self.assertEqual(y_range.end, 3.2)

    def test_spikes_padding_hard_xrange(self):
        spikes = Spikes([1, 2, 3]).redim.range(x=(0, 3)).opts(padding=0.1)
        plot = bokeh_renderer.get_plot(spikes)
        x_range = plot.handles['x_range']
        self.assertEqual(x_range.start, 0)
        self.assertEqual(x_range.end, 3)

    def test_spikes_padding_soft_xrange(self):
        spikes = Spikes([1, 2, 3]).redim.soft_range(x=(0, 3)).opts(padding=0.1)
        plot = bokeh_renderer.get_plot(spikes)
        x_range = plot.handles['x_range']
        self.assertEqual(x_range.start, 0)
        self.assertEqual(x_range.end, 3)

    def test_spikes_padding_unequal(self):
        spikes = Spikes([1, 2, 3]).opts(padding=(0.05, 0.1))
        plot = bokeh_renderer.get_plot(spikes)
        x_range = plot.handles['x_range']
        self.assertEqual(x_range.start, 0.9)
        self.assertEqual(x_range.end, 3.1)

    def test_spikes_padding_nonsquare(self):
        spikes = Spikes([1, 2, 3]).opts(padding=0.1, width=600)
        plot = bokeh_renderer.get_plot(spikes)
        x_range = plot.handles['x_range']
        self.assertEqual(x_range.start, 0.9)
        self.assertEqual(x_range.end, 3.1)

    def test_spikes_padding_logx(self):
        spikes = Spikes([(1, 1), (2, 2), (3,3)]).opts(padding=0.1, logx=True)
        plot = bokeh_renderer.get_plot(spikes)
        x_range = plot.handles['x_range']
        self.assertEqual(x_range.start, 0.89595845984076228)
        self.assertEqual(x_range.end, 3.3483695221017129)

    def test_spikes_padding_datetime_square(self):
        spikes = Spikes([np.datetime64('2016-04-0%d' % i) for i in range(1, 4)]).opts(
            padding=0.1
        )
        plot = bokeh_renderer.get_plot(spikes)
        x_range = plot.handles['x_range']
        self.assertEqual(x_range.start, np.datetime64('2016-03-31T19:12:00.000000000'))
        self.assertEqual(x_range.end, np.datetime64('2016-04-03T04:48:00.000000000'))

    def test_spikes_padding_datetime_square_heights(self):
        spikes = Spikes([(np.datetime64('2016-04-0%d' % i), i) for i in range(1, 4)], vdims=['Height']).opts(
            padding=0.1
        )
        plot = bokeh_renderer.get_plot(spikes)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, np.datetime64('2016-03-31T19:12:00.000000000'))
        self.assertEqual(x_range.end, np.datetime64('2016-04-03T04:48:00.000000000'))
        self.assertEqual(y_range.start, 0)
        self.assertEqual(y_range.end, 3.2)

    def test_spikes_padding_datetime_nonsquare(self):
        spikes = Spikes([np.datetime64('2016-04-0%d' % i) for i in range(1, 4)]).opts(
            padding=0.1, width=600
        )
        plot = bokeh_renderer.get_plot(spikes)
        x_range = plot.handles['x_range']
        self.assertEqual(x_range.start, np.datetime64('2016-03-31T21:36:00.000000000'))
        self.assertEqual(x_range.end, np.datetime64('2016-04-03T02:24:00.000000000'))

    def test_spikes_datetime_vdim_hover(self):
        points = Spikes([(0, 1, dt.datetime(2017, 1, 1))], vdims=['value', 'date']).opts(tools=['hover'])
        plot = bokeh_renderer.get_plot(points)
        cds = plot.handles['cds']
        self.assertEqual(cds.data['date'].astype('datetime64'), np.array([1483228800000000000]))
        hover = plot.handles['hover']
        self.assertEqual(hover.tooltips, [('x', '@{x}'), ('value', '@{value}'), ('date', '@{date}{%F %T}')])
        self.assertEqual(hover.formatters, {'@{date}': "datetime"})

    def test_spikes_datetime_kdim_hover(self):
        points = Spikes([(dt.datetime(2017, 1, 1), 1)], 'x', 'y').opts(tools=['hover'])
        plot = bokeh_renderer.get_plot(points)
        cds = plot.handles['cds']
        self.assertEqual(cds.data['x'].astype('datetime64'), np.array([1483228800000000000]))
        self.assertEqual(cds.data['y0'], np.array([0]))
        self.assertEqual(cds.data['y1'], np.array([1]))

        hover = plot.handles['hover']
        self.assertEqual(hover.tooltips, [('x', '@{x}{%F %T}'), ('y', '@{y}')])
        self.assertEqual(hover.formatters, {'@{x}': "datetime"})

    def test_spikes_datetime_kdim_hover_spike_length_override(self):
        points = Spikes([(dt.datetime(2017, 1, 1), 1)], 'x', 'y').opts(
            tools=['hover'], spike_length=0.75)
        plot = bokeh_renderer.get_plot(points)
        cds = plot.handles['cds']
        self.assertEqual(cds.data['x'].astype('datetime64'), np.array([1483228800000000000]))
        self.assertEqual(cds.data['y0'], np.array([0]))
        self.assertEqual(cds.data['y1'], np.array([0.75]))
        hover = plot.handles['hover']
        self.assertEqual(hover.tooltips, [('x', '@{x}{%F %T}'), ('y', '@{y}')])
        self.assertEqual(hover.formatters, {'@{x}': "datetime"})

    ###########################
    #    Styling mapping      #
    ###########################

    def test_spikes_color_op(self):
        spikes = Spikes([(0, 0, '#000'), (0, 1, '#F00'), (0, 2, '#0F0')],
                              vdims=['y', 'color']).opts(color='color')
        plot = bokeh_renderer.get_plot(spikes)
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        self.assertEqual(cds.data['color'], np.array(['#000', '#F00', '#0F0']))
        self.assertEqual(property_to_dict(glyph.line_color), {'field': 'color'})

    def test_spikes_linear_color_op(self):
        spikes = Spikes([(0, 0, 0), (0, 1, 1), (0, 2, 2)],
                              vdims=['y', 'color']).opts(color='color')
        plot = bokeh_renderer.get_plot(spikes)
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        cmapper = plot.handles['color_color_mapper']
        self.assertTrue(cmapper, LinearColorMapper)
        self.assertEqual(cmapper.low, 0)
        self.assertEqual(cmapper.high, 2)
        self.assertEqual(cds.data['color'], np.array([0, 1, 2]))
        self.assertEqual(property_to_dict(glyph.line_color), {'field': 'color', 'transform': cmapper})

    def test_spikes_categorical_color_op(self):
        spikes = Spikes([(0, 0, 'A'), (0, 1, 'B'), (0, 2, 'C')],
                              vdims=['y', 'color']).opts(color='color')
        plot = bokeh_renderer.get_plot(spikes)
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        cmapper = plot.handles['color_color_mapper']
        self.assertTrue(cmapper, CategoricalColorMapper)
        self.assertEqual(cmapper.factors, ['A', 'B', 'C'])
        self.assertEqual(cds.data['color'], np.array(['A', 'B', 'C']))
        self.assertEqual(property_to_dict(glyph.line_color), {'field': 'color', 'transform': cmapper})

    def test_spikes_line_color_op(self):
        spikes = Spikes([(0, 0, '#000'), (0, 1, '#F00'), (0, 2, '#0F0')],
                              vdims=['y', 'color']).opts(line_color='color')
        plot = bokeh_renderer.get_plot(spikes)
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        self.assertEqual(cds.data['line_color'], np.array(['#000', '#F00', '#0F0']))
        self.assertEqual(property_to_dict(glyph.line_color), {'field': 'line_color'})

    def test_spikes_alpha_op(self):
        spikes = Spikes([(0, 0, 0), (0, 1, 0.2), (0, 2, 0.7)],
                              vdims=['y', 'alpha']).opts(alpha='alpha')
        plot = bokeh_renderer.get_plot(spikes)
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        self.assertEqual(cds.data['alpha'], np.array([0, 0.2, 0.7]))
        self.assertEqual(property_to_dict(glyph.line_alpha), {'field': 'alpha'})

    def test_spikes_line_alpha_op(self):
        spikes = Spikes([(0, 0, 0), (0, 1, 0.2), (0, 2, 0.7)],
                              vdims=['y', 'alpha']).opts(line_alpha='alpha')
        plot = bokeh_renderer.get_plot(spikes)
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        self.assertEqual(cds.data['line_alpha'], np.array([0, 0.2, 0.7]))
        self.assertEqual(property_to_dict(glyph.line_alpha), {'field': 'line_alpha'})

    def test_spikes_line_width_op(self):
        spikes = Spikes([(0, 0, 1), (0, 1, 4), (0, 2, 8)],
                              vdims=['y', 'line_width']).opts(line_width='line_width')
        plot = bokeh_renderer.get_plot(spikes)
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        self.assertEqual(cds.data['line_width'], np.array([1, 4, 8]))
        self.assertEqual(property_to_dict(glyph.line_width), {'field': 'line_width'})

    def test_op_ndoverlay_value(self):
        colors = ['blue', 'red']
        overlay = NdOverlay({color: Spikes(np.arange(i+2)) for i, color in enumerate(colors)}, 'Color').opts('Spikes', color='Color')
        plot = bokeh_renderer.get_plot(overlay)
        for subplot, color in zip(plot.subplots.values(),  colors):
            self.assertEqual(subplot.handles['glyph'].line_color, color)

    def test_spikes_color_index_color_clash(self):
        spikes = Spikes([(0, 0, 0), (0, 1, 1), (0, 2, 2)],
                    vdims=['y', 'color']).opts(color='color', color_index='color')
        with ParamLogStream() as log:
            bokeh_renderer.get_plot(spikes)
        log_msg = log.stream.read()
        warning = (
            "The `color_index` parameter is deprecated in favor of color style mapping, e.g. "
            "`color=dim('color')` or `line_color=dim('color')`\nCannot declare style mapping "
            "for 'color' option and declare a color_index; ignoring the color_index.\n"
        )
        self.assertEqual(log_msg, warning)
