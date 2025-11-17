import numpy as np
import pandas as pd
import panel as pn
import pytest
from bokeh.models import FactorRange, FixedTicker, HoverTool, Range1d, Span

from holoviews.core import DynamicMap, HoloMap, NdOverlay, Overlay
from holoviews.element import (
    Bars,
    Box,
    Curve,
    ErrorBars,
    HLine,
    Points,
    Scatter,
    Text,
    VLine,
)
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.streams import Pipe, Stream, Tap
from holoviews.util import Dynamic

from ...utils import LoggingComparisonTestCase
from .test_plot import TestBokehPlot, bokeh_renderer


class TestOverlayPlot(LoggingComparisonTestCase, TestBokehPlot):

    def test_overlay_apply_ranges_disabled(self):
        overlay = (Curve(range(10)) * Curve(range(10))).opts('Curve', apply_ranges=False)
        plot = bokeh_renderer.get_plot(overlay)
        self.assertTrue(all(np.isnan(e) for e in plot.get_extents(overlay, {})))

    def test_overlay_update_sources(self):
        hmap = HoloMap({i: (Curve(np.arange(i), label='A') *
                            Curve(np.arange(i)*2, label='B'))
                        for i in range(10, 13)})
        plot = bokeh_renderer.get_plot(hmap)
        plot.update((12,))
        subplot1, subplot2 = plot.subplots.values()
        self.assertEqual(subplot1.handles['source'].data['y'], np.arange(12))
        self.assertEqual(subplot2.handles['source'].data['y'], np.arange(12)*2)

    def test_overlay_framewise_norm(self):
        a = {
            'X': [0, 1, 2],
            'Y': [0, 1, 2],
            'Z': [0, 50, 100]
        }
        b = {
            'X': [3, 4, 5],
            'Y': [0, 10, 20],
            'Z': [50, 50, 150]
        }
        sa = Scatter(a, 'X', ['Y', 'Z']).opts(color='Z', framewise=True)
        sb = Scatter(b, 'X', ['Y', 'Z']).opts(color='Z', framewise=True)
        plot = bokeh_renderer.get_plot(sa * sb)
        sa_plot, sb_plot = plot.subplots.values()
        sa_cmapper = sa_plot.handles['color_color_mapper']
        sb_cmapper = sb_plot.handles['color_color_mapper']
        self.assertEqual(sa_cmapper.low, 0)
        self.assertEqual(sb_cmapper.low, 0)
        self.assertEqual(sa_cmapper.high, 150)
        self.assertEqual(sb_cmapper.high, 150)

    def test_overlay_update_visible(self):
        hmap = HoloMap({i: Curve(np.arange(i), label='A') for i in range(1, 3)})
        hmap2 = HoloMap({i: Curve(np.arange(i), label='B') for i in range(3, 5)})
        plot = bokeh_renderer.get_plot(hmap*hmap2)
        subplot1, subplot2 = plot.subplots.values()
        self.assertTrue(subplot1.handles['glyph_renderer'].visible)
        self.assertFalse(subplot2.handles['glyph_renderer'].visible)
        plot.update((4,))
        self.assertFalse(subplot1.handles['glyph_renderer'].visible)
        self.assertTrue(subplot2.handles['glyph_renderer'].visible)

    def test_hover_tool_instance_renderer_association(self):
        tooltips = [("index", "$index")]
        hover = HoverTool(tooltips=tooltips)
        overlay = Curve(np.random.rand(10,2)).opts(tools=[hover]) * Points(np.random.rand(10,2))
        plot = bokeh_renderer.get_plot(overlay)
        curve_plot = plot.subplots[('Curve', 'I')]
        self.assertEqual(len(curve_plot.handles['hover'].renderers), 1)
        self.assertIn(curve_plot.handles['glyph_renderer'], curve_plot.handles['hover'].renderers)
        self.assertEqual(plot.handles['hover'].tooltips, tooltips)

    # def test_hover_tool_overlay_renderers(self):
    #     overlay = Curve(range(2)).opts(tools=['hover']) * ErrorBars([]).opts(tools=['hover'])
    #     plot = bokeh_renderer.get_plot(overlay)
    #     self.assertEqual(len(plot.handles['hover'].renderers), 1)
    #     self.assertEqual(plot.handles['hover'].tooltips, [('x', '@{x}'), ('y', '@{y}')])

    def test_hover_tool_nested_overlay_renderers(self):
        overlay1 = NdOverlay({0: Curve(range(2)), 1: Curve(range(3))}, kdims=['Test'])
        overlay2 = NdOverlay({0: Curve(range(4)), 1: Curve(range(5))}, kdims=['Test'])
        nested_overlay = (overlay1 * overlay2).opts('Curve', tools=['hover'])
        plot = bokeh_renderer.get_plot(nested_overlay)
        self.assertEqual(len(plot.handles['hover'].renderers), 4)
        self.assertEqual(plot.handles['hover'].tooltips,
                         [('Test', '@{Test}'), ('x', '@{x}'), ('y', '@{y}')])

    def test_overlay_empty_layers(self):
        overlay = Curve(range(10)) * NdOverlay()
        plot = bokeh_renderer.get_plot(overlay)
        self.assertEqual(len(plot.subplots), 1)
        self.log_handler.assertContains('WARNING', 'is empty and will be skipped during plotting')

    def test_overlay_show_frame_disabled(self):
        overlay = (Curve(range(10)) * Curve(range(10))).opts(show_frame=False)
        plot = bokeh_renderer.get_plot(overlay).state
        self.assertEqual(plot.outline_line_alpha, 0)

    def test_overlay_no_xaxis(self):
        overlay = (Curve(range(10)) * Curve(range(10))).opts(xaxis=None)
        plot = bokeh_renderer.get_plot(overlay).state
        self.assertFalse(plot.xaxis[0].visible)

    def test_overlay_no_yaxis(self):
        overlay = (Curve(range(10)) * Curve(range(10))).opts(yaxis=None)
        plot = bokeh_renderer.get_plot(overlay).state
        self.assertFalse(plot.yaxis[0].visible)

    def test_overlay_xlabel_override(self):
        overlay = (Curve(range(10)) * Curve(range(10))).opts(xlabel='custom x-label')
        plot = bokeh_renderer.get_plot(overlay).state
        self.assertEqual(plot.xaxis[0].axis_label, 'custom x-label')

    def test_overlay_ylabel_override(self):
        overlay = (Curve(range(10)) * Curve(range(10))).opts(ylabel='custom y-label')
        plot = bokeh_renderer.get_plot(overlay).state
        self.assertEqual(plot.yaxis[0].axis_label, 'custom y-label')

    def test_overlay_xlabel_override_propagated(self):
        overlay = (Curve(range(10)).opts(xlabel='custom x-label') * Curve(range(10)))
        plot = bokeh_renderer.get_plot(overlay).state
        self.assertEqual(plot.xaxis[0].axis_label, 'custom x-label')

    def test_overlay_ylabel_override_propagated(self):
        overlay = (Curve(range(10)).opts(ylabel='custom y-label') * Curve(range(10)))
        plot = bokeh_renderer.get_plot(overlay).state
        self.assertEqual(plot.yaxis[0].axis_label, 'custom y-label')

    def test_overlay_xrotation(self):
        overlay = (Curve(range(10)) * Curve(range(10))).opts(xrotation=90)
        plot = bokeh_renderer.get_plot(overlay).state
        self.assertEqual(plot.xaxis[0].major_label_orientation, np.pi/2)

    def test_overlay_yrotation(self):
        overlay = (Curve(range(10)) * Curve(range(10))).opts(yrotation=90)
        plot = bokeh_renderer.get_plot(overlay).state
        self.assertEqual(plot.yaxis[0].major_label_orientation, np.pi/2)

    def test_overlay_xticks_list(self):
        overlay = (Curve(range(10)) * Curve(range(10))).opts(xticks=[0, 5, 10])
        plot = bokeh_renderer.get_plot(overlay).state
        self.assertIsInstance(plot.xaxis[0].ticker, FixedTicker)
        self.assertEqual(plot.xaxis[0].ticker.ticks, [0, 5, 10])

    def test_overlay_yticks_list(self):
        overlay = (Curve(range(10)) * Curve(range(10))).opts(yticks=[0, 5, 10])
        plot = bokeh_renderer.get_plot(overlay).state
        self.assertIsInstance(plot.yaxis[0].ticker, FixedTicker)
        self.assertEqual(plot.yaxis[0].ticker.ticks, [0, 5, 10])

    def test_overlay_update_plot_opts(self):
        hmap = HoloMap(
            {0: (Curve([]) * Curve([])).opts(title='A'),
             1: (Curve([]) * Curve([])).opts(title='B')}
        )
        plot = bokeh_renderer.get_plot(hmap)
        self.assertEqual(plot.state.title.text, 'A')
        plot.update((1,))
        self.assertEqual(plot.state.title.text, 'B')

    def test_overlay_update_plot_opts_inherited(self):
        hmap = HoloMap(
            {0: (Curve([]).opts(title='A') * Curve([])),
             1: (Curve([]).opts(title='B') * Curve([]))}
        )
        plot = bokeh_renderer.get_plot(hmap)
        self.assertEqual(plot.state.title.text, 'A')
        plot.update((1,))
        self.assertEqual(plot.state.title.text, 'B')

    def test_points_errorbars_text_ndoverlay_categorical_xaxis(self):
        overlay = NdOverlay({i: Points(([chr(65+i)]*10,np.random.randn(10)))
                             for i in range(5)})
        error = ErrorBars([(el['x'][0], np.mean(el['y']), np.std(el['y']))
                           for el in overlay])
        text = Text('C', 0, 'Test')
        plot = bokeh_renderer.get_plot(overlay*error*text)
        x_range = plot.handles['x_range']
        y_range = plot.handles['y_range']
        self.assertIsInstance(x_range, FactorRange)
        factors = ['A', 'B', 'C', 'D', 'E']
        self.assertEqual(x_range.factors, ['A', 'B', 'C', 'D', 'E'])
        self.assertIsInstance(y_range, Range1d)
        error_plot = plot.subplots[('ErrorBars', 'I')]
        for xs, factor in zip(error_plot.handles['source'].data['base'], factors, strict=None):
            self.assertEqual(factor, xs)

    def test_overlay_categorical_two_level(self):
        bars = Bars([('A', 'a', 1), ('B', 'b', 2), ('A', 'b', 3), ('B', 'a', 4)],
                    kdims=['Upper', 'Lower'])

        plot = bokeh_renderer.get_plot(bars * HLine(2))
        x_range = plot.handles['x_range']
        assert isinstance(x_range, FactorRange)
        assert x_range.factors == [('A', 'a'), ('A', 'b'), ('B', 'a'), ('B', 'b')]
        assert isinstance(plot.state.renderers[-1], Span)

    def test_points_errorbars_text_ndoverlay_categorical_xaxis_invert_axes(self):
        overlay = NdOverlay({i: Points(([chr(65+i)]*10,np.random.randn(10)))
                             for i in range(5)})
        error = ErrorBars([(el['x'][0], np.mean(el['y']), np.std(el['y']))
                           for el in overlay]).opts(invert_axes=True)
        text = Text('C', 0, 'Test')
        plot = bokeh_renderer.get_plot(overlay*error*text)
        x_range = plot.handles['x_range']
        y_range = plot.handles['y_range']
        self.assertIsInstance(x_range, Range1d)
        self.assertIsInstance(y_range, FactorRange)
        self.assertEqual(y_range.factors, ['A', 'B', 'C', 'D', 'E'])

    def test_overlay_empty_element_extent(self):
        overlay = Curve([]).redim.range(x=(-10, 10)) * Points([]).redim.range(y=(-20, 20))
        plot = bokeh_renderer.get_plot(overlay)
        extents = plot.get_extents(overlay, {})
        self.assertEqual(extents, (-10, -20, 10, 20))

    def test_dynamic_subplot_creation(self):
        def cb(X):
            return NdOverlay({i: Curve(np.arange(10)+i) for i in range(X)})
        dmap = DynamicMap(cb, kdims=['X']).redim.range(X=(1, 10))
        plot = bokeh_renderer.get_plot(dmap)
        self.assertEqual(len(plot.subplots), 1)
        plot.update((3,))
        self.assertEqual(len(plot.subplots), 3)
        for i, subplot in enumerate(plot.subplots.values()):
            self.assertEqual(subplot.cyclic_index, i)

    def test_complex_range_example(self):
        errors = [(0.1*i, np.sin(0.1*i), (i+1)/3., (i+1)/5.) for i in np.linspace(0, 100, 11)]
        errors = ErrorBars(errors, vdims=['y', 'yerrneg', 'yerrpos']).redim.range(y=(0, None))
        overlay = Curve(errors) * errors * VLine(4)
        plot = bokeh_renderer.get_plot(overlay)
        x_range = plot.handles['x_range']
        y_range = plot.handles['y_range']
        self.assertEqual(x_range.start, 0)
        self.assertEqual(x_range.end, 10.0)
        self.assertEqual(y_range.start, 0)
        self.assertEqual(y_range.end, 19.655978889110628)

    def test_overlay_muted_renderer(self):
        overlay = Curve((np.arange(5)), label='increase') * Curve((np.arange(5)*-1+5), label='decrease').opts(muted=True)
        plot = bokeh_renderer.get_plot(overlay)
        unmuted, muted = plot.subplots.values()
        self.assertFalse(unmuted.handles['glyph_renderer'].muted)
        self.assertTrue(muted.handles['glyph_renderer'].muted)

    def test_overlay_params_bind_linked_stream(self):
        tap = Tap()
        def test(x):
            return Curve([1, 2, 3]) * VLine(x or 0)
        dmap = DynamicMap(pn.bind(test, x=tap.param.x))
        plot = bokeh_renderer.get_plot(dmap)

        tap.event(x=1)
        _, vline_plot = plot.subplots.values()
        assert vline_plot.handles['glyph'].location == 1

    def test_overlay_params_dict_linked_stream(self):
        tap = Tap()
        def test(x):
            return Curve([1, 2, 3]) * VLine(x or 0)
        dmap = DynamicMap(test, streams={'x': tap.param.x})
        plot = bokeh_renderer.get_plot(dmap)

        tap.event(x=1)
        _, vline_plot = plot.subplots.values()
        assert vline_plot.handles['glyph'].location == 1

    def test_ndoverlay_subcoordinate_y_no_batching(self):
        overlay = NdOverlay({
            i: Curve(np.arange(10)*i).opts(subcoordinate_y=True) for i in range(10)
        }).opts(legend_limit=1)
        plot = bokeh_renderer.get_plot(overlay)

        assert plot.batched == False
        assert len(plot.subplots) == 10

    def test_ndoverlay_subcoordinate_y_ranges(self):
        data = {
            'x': np.arange(10),
            'A': np.arange(10),
            'B': np.arange(10)*2,
            'C': np.arange(10)*3
        }
        overlay = NdOverlay({
            'A': Curve(data, 'x', ('A', 'y')).opts(subcoordinate_y=True),
            'B': Curve(data, 'x', ('B', 'y')).opts(subcoordinate_y=True),
            'C': Curve(data, 'x', ('C', 'y')).opts(subcoordinate_y=True),
        })
        plot = bokeh_renderer.get_plot(overlay)

        assert plot.state.y_range.start == -0.5
        assert plot.state.y_range.end == 2.5
        for sp in plot.subplots.values():
            assert sp.handles['y_range'].start == 0
            assert sp.handles['y_range'].end == 27

    def test_ndoverlay_subcoordinate_y_ranges_update(self):
        def lines(data):
            return NdOverlay({
                'A': Curve(data, 'x', ('A', 'y1')).opts(subcoordinate_y=True, framewise=True),
                'B': Curve(data, 'x', ('B', 'y2')).opts(subcoordinate_y=True, framewise=True),
                'C': Curve(data, 'x', ('C', 'y3')).opts(subcoordinate_y=True, framewise=True),
            })
        data = {
            'x': np.arange(10),
            'A': np.arange(10),
            'B': np.arange(10)*2,
            'C': np.arange(10)*3
        }
        stream = Pipe(data=data)
        dmap = DynamicMap(lines, streams=[stream])
        plot = bokeh_renderer.get_plot(dmap)

        assert plot.state.y_range.start == -0.5
        assert plot.state.y_range.end == 2.5
        for sp in plot.subplots.values():
            y_range = sp.handles['y_range']
            assert y_range.start == 0
            assert y_range.end == data[y_range.name].max()

        new_data = {
            'x': np.arange(10),
            'A': data['A'] + 1,
            'B': data['B'] + 2,
            'C': data['C'] + 3
        }
        stream.event(data=new_data)
        for sp in plot.subplots.values():
            y_range = sp.handles['y_range']
            ydata = new_data[y_range.name]
            assert y_range.start == ydata.min()
            assert y_range.end == ydata.max()

    def test_overlay_subcoordinate_y_ranges(self):
        data = {
            'x': np.arange(10),
            'A': np.arange(10),
            'B': np.arange(10)*2,
            'C': np.arange(10)*3
        }
        overlay = Overlay([
            Curve(data, 'x', ('A', 'y'), label='A').opts(subcoordinate_y=True),
            Curve(data, 'x', ('B', 'y'), label='B').opts(subcoordinate_y=True),
            Curve(data, 'x', ('C', 'y'), label='C').opts(subcoordinate_y=True),
        ])
        plot = bokeh_renderer.get_plot(overlay)

        assert plot.state.y_range.start == -0.5
        assert plot.state.y_range.end == 2.5
        for sp in plot.subplots.values():
            assert sp.handles['y_range'].start == 0
            assert sp.handles['y_range'].end == 27

    def test_overlay_subcoordinate_y_ranges_update(self):
        def lines(data):
            return Overlay([
                Curve(data, 'x', ('A', 'y1'), label='A').opts(subcoordinate_y=True, framewise=True),
                Curve(data, 'x', ('B', 'y2'), label='B').opts(subcoordinate_y=True, framewise=True),
                Curve(data, 'x', ('C', 'y3'), label='C').opts(subcoordinate_y=True, framewise=True),
            ])
        data = {
            'x': np.arange(10),
            'A': np.arange(10),
            'B': np.arange(10)*2,
            'C': np.arange(10)*3
        }
        stream = Pipe(data=data)
        dmap = DynamicMap(lines, streams=[stream])
        plot = bokeh_renderer.get_plot(dmap)

        assert plot.state.y_range.start == -0.5
        assert plot.state.y_range.end == 2.5
        for sp in plot.subplots.values():
            y_range = sp.handles['y_range']
            assert y_range.start == 0
            assert y_range.end == data[y_range.name].max()

        new_data = {
            'x': np.arange(10),
            'A': data['A'] + 1,
            'B': data['B'] + 2,
            'C': data['C'] + 3
        }
        stream.event(data=new_data)
        for sp in plot.subplots.values():
            y_range = sp.handles['y_range']
            ydata = new_data[y_range.name]
            assert y_range.start == ydata.min()
            assert y_range.end == ydata.max()


@pytest.mark.parametrize('order', [("str", "date"), ("date", "str")])
def test_ndoverlay_categorical_y_ranges(order):
    df = pd.DataFrame(
        {
            "str": ["apple", "banana", "cherry", "date", "elderberry"],
            "date": pd.to_datetime(
                ["2023-01-01", "2023-02-14", "2023-03-21", "2023-04-30", "2023-05-15"]
            ),
        }
    )
    overlay = NdOverlay(
        {col: Scatter(df, kdims="index", vdims=col) for col in order}
    )
    plot = bokeh_renderer.get_plot(overlay)
    output = plot.handles["y_range"].factors
    expected = sorted(map(str, df.values.ravel()))
    assert output == expected


class TestLegends(TestBokehPlot):

    def test_overlay_legend(self):
        overlay = Curve(range(10), label='A') * Curve(range(10), label='B')
        plot = bokeh_renderer.get_plot(overlay)
        legend_labels = [l.label['value'] for l in plot.state.legend[0].items]
        self.assertEqual(legend_labels, ['A', 'B'])

    def test_overlay_legend_with_labels(self):
        overlay = (Curve(range(10), label='A') * Curve(range(10), label='B')).opts(
            legend_labels={'A': 'A Curve', 'B': 'B Curve'})
        plot = bokeh_renderer.get_plot(overlay)
        legend_labels = [l.label['value'] for l in plot.state.legend[0].items]
        self.assertEqual(legend_labels, ['A Curve', 'B Curve'])

    def test_holomap_legend_updates(self):
        hmap = HoloMap({i: Curve([1, 2, 3], label=chr(65+i+2)) * Curve([1, 2, 3], label='B')
                        for i in range(3)})
        plot = bokeh_renderer.get_plot(hmap)
        legend_labels = [property_to_dict(item.label) for item in plot.state.legend[0].items]
        self.assertEqual(legend_labels, [{'value': 'C'}, {'value': 'B'}])
        plot.update((1,))
        legend_labels = [property_to_dict(item.label) for item in plot.state.legend[0].items]
        self.assertEqual(legend_labels, [{'value': 'B'}, {'value': 'D'}])
        plot.update((2,))
        legend_labels = [property_to_dict(item.label) for item in plot.state.legend[0].items]
        self.assertEqual(legend_labels, [{'value': 'B'}, {'value': 'E'}])

    def test_holomap_legend_updates_varying_lengths(self):
        hmap = HoloMap({i: Overlay([Curve([1, 2, j], label=chr(65+j)) for j in range(i)]) for i in range(1, 4)})
        plot = bokeh_renderer.get_plot(hmap)
        legend_labels = [property_to_dict(item.label) for item in plot.state.legend[0].items]
        self.assertEqual(legend_labels, [{'value': 'A'}])
        plot.update((2,))
        legend_labels = [property_to_dict(item.label) for item in plot.state.legend[0].items]
        self.assertEqual(legend_labels, [{'value': 'A'}, {'value': 'B'}])
        plot.update((3,))
        legend_labels = [property_to_dict(item.label) for item in plot.state.legend[0].items]
        self.assertEqual(legend_labels, [{'value': 'A'}, {'value': 'B'}, {'value': 'C'}])

    def test_dynamicmap_legend_updates(self):
        hmap = HoloMap({i: Curve([1, 2, 3], label=chr(65+i+2)) * Curve([1, 2, 3], label='B')
                        for i in range(3)})
        dmap = Dynamic(hmap)
        plot = bokeh_renderer.get_plot(dmap)
        legend_labels = [property_to_dict(item.label) for item in plot.state.legend[0].items]
        self.assertEqual(legend_labels, [{'value': 'C'}, {'value': 'B'}])
        plot.update((1,))
        legend_labels = [property_to_dict(item.label) for item in plot.state.legend[0].items]
        self.assertEqual(legend_labels, [{'value': 'B'}, {'value': 'D'}])
        plot.update((2,))
        legend_labels = [property_to_dict(item.label) for item in plot.state.legend[0].items]
        self.assertEqual(legend_labels, [{'value': 'B'}, {'value': 'E'}])

    def test_dynamicmap_legend_updates_add_dynamic_plots(self):
        hmap = HoloMap({i: Overlay([Curve([1, 2, j], label=chr(65+j)) for j in range(i)]) for i in range(1, 4)})
        dmap = Dynamic(hmap)
        plot = bokeh_renderer.get_plot(dmap)
        legend_labels = [property_to_dict(item.label) for item in plot.state.legend[0].items]
        self.assertEqual(legend_labels, [{'value': 'A'}])
        plot.update((2,))
        legend_labels = [property_to_dict(item.label) for item in plot.state.legend[0].items]
        self.assertEqual(legend_labels, [{'value': 'A'}, {'value': 'B'}])
        plot.update((3,))
        legend_labels = [property_to_dict(item.label) for item in plot.state.legend[0].items]
        self.assertEqual(legend_labels, [{'value': 'A'}, {'value': 'B'}, {'value': 'C'}])

    def test_dynamicmap_ndoverlay_shrink_number_of_items(self):
        selected = Stream.define('selected', items=3)()
        def callback(items):
            return NdOverlay({j: Overlay([Curve([1, 2, j])]) for j in range(items)})
        dmap = DynamicMap(callback, streams=[selected])
        plot = bokeh_renderer.get_plot(dmap)
        selected.event(items=2)
        self.assertEqual(len([r for r in plot.state.renderers if r.visible]), 2)

    def test_dynamicmap_variable_length_overlay(self):
        selected = Stream.define('selected', items=[1])()
        def callback(items):
            return Overlay([Box(0, 0, radius*2) for radius in items])
        dmap = DynamicMap(callback, streams=[selected])
        plot = bokeh_renderer.get_plot(dmap)
        assert len(plot.subplots) == 1
        selected.event(items=[1, 2, 4])
        assert len(plot.subplots) == 3
        selected.event(items=[1, 4])
        sp1, sp2, sp3 = plot.subplots.values()
        assert sp1.handles['cds'].data['xs'][0].min() == -1
        assert sp1.handles['glyph_renderer'].visible
        assert sp2.handles['cds'].data['xs'][0].min() == -4
        assert sp2.handles['glyph_renderer'].visible
        assert sp3.handles['cds'].data['xs'][0].min() == -4
        assert not sp3.handles['glyph_renderer'].visible
