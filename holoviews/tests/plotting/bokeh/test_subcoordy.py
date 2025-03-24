import numpy as np
import pytest
from bokeh.models.tools import WheelZoomTool, ZoomInTool, ZoomOutTool

from holoviews.core import NdOverlay, Overlay
from holoviews.element import Curve
from holoviews.element.annotation import VSpan
from holoviews.operation.normalization import subcoordinate_group_ranges
from holoviews.plotting.bokeh.util import BOKEH_GE_3_5_0

from .test_plot import TestBokehPlot, bokeh_renderer


class TestSubcoordinateY(TestBokehPlot):

    # With subcoordinate_y set to True

    def test_bool_base(self):
        overlay = Overlay([Curve(range(10), label=f'Data {i}').opts(subcoordinate_y=True) for i in range(2)])
        plot = bokeh_renderer.get_plot(overlay)
        # subcoordinate_y is propagated to the overlay
        assert plot.subcoordinate_y is True
        # the figure has only one yaxis
        assert len(plot.state.yaxis) == 1
        # the overlay has two subplots
        assert len(plot.subplots) == 2
        assert ('Curve', 'Data_0') in plot.subplots
        assert ('Curve', 'Data_1') in plot.subplots
        # the range per subplots are correctly computed
        sp1 = plot.subplots[('Curve', 'Data_0')]
        assert sp1.handles['glyph_renderer'].coordinates.y_target.start == -0.5
        assert sp1.handles['glyph_renderer'].coordinates.y_target.end == 0.5
        sp2 = plot.subplots[('Curve', 'Data_1')]
        assert sp2.handles['glyph_renderer'].coordinates.y_target.start == 0.5
        assert sp2.handles['glyph_renderer'].coordinates.y_target.end == 1.5
        # y_range is correctly computed
        assert plot.handles['y_range'].start == -0.5
        assert plot.handles['y_range'].end == 1.5
        # extra_y_range is empty
        assert plot.handles['extra_y_ranges'] == {}
        # the ticks show the labels
        assert plot.state.yaxis.ticker.ticks == [0, 1]
        assert plot.state.yaxis.major_label_overrides == {0: 'Data 0', 1: 'Data 1'}

    def test_renderers_reversed(self):
        overlay = Overlay([Curve(range(10), label=f'Data {i}').opts(subcoordinate_y=True) for i in range(2)])
        overlay = VSpan(0, 1, label='back') * overlay * VSpan(2, 3, label='front')
        plot = bokeh_renderer.get_plot(overlay)
        renderers = plot.handles['plot'].renderers
        assert (renderers[0].left, renderers[0].right) == (0, 1)
        # Only the subcoord-y renderers are reversed by default.
        assert renderers[1].name == 'Data 1'
        assert renderers[2].name == 'Data 0'
        assert (renderers[3].left, renderers[3].right) == (2, 3)

    def test_bool_scale(self):
        test_data = [
            (0.5, (-0.25, 0.25), (0.75, 1.25), (-0.25, 1.25)),
            (1, (-0.5, 0.5), (0.5, 1.5), (-0.5, 1.5)),
            (2, (-1, 1), (0, 2), (-1, 2)),
            (5, (-2.5, 2.5), (-1.5, 3.5), (-2.5, 3.5)),
        ]
        for scale, ytarget1, ytarget2, ytarget in test_data:
            overlay = Overlay([
                Curve(range(10), label=f'Data {i}').opts(subcoordinate_y=True, subcoordinate_scale=scale)
                for i in range(2)
            ])
            plot = bokeh_renderer.get_plot(overlay)
            # the range per subplots are correctly computed
            sp1 = plot.subplots[('Curve', 'Data_0')]
            assert sp1.handles['glyph_renderer'].coordinates.y_target.start == ytarget1[0]
            assert sp1.handles['glyph_renderer'].coordinates.y_target.end == ytarget1[1]
            sp2 = plot.subplots[('Curve', 'Data_1')]
            assert sp2.handles['glyph_renderer'].coordinates.y_target.start == ytarget2[0]
            assert sp2.handles['glyph_renderer'].coordinates.y_target.end == ytarget2[1]
            # y_range is correctly computed
            assert plot.handles['y_range'].start == ytarget[0]
            assert plot.handles['y_range'].end == ytarget[1]

    def test_ndoverlay_labels(self):
        overlay = NdOverlay({
            f'Data {i}': Curve(np.arange(10)*i).opts(subcoordinate_y=True)
            for i in range(3)
        }, 'Channel')
        plot = bokeh_renderer.get_plot(overlay)
        assert plot.state.yaxis.ticker.ticks == [0, 1, 2]
        assert plot.state.yaxis.major_label_overrides == {0: 'Data 0', 1: 'Data 1', 2: 'Data 2'}
        for i, sp in enumerate(plot.subplots.values()):
            assert sp.handles['glyph_renderer'].coordinates.y_target.start == (i-0.5)
            assert sp.handles['glyph_renderer'].coordinates.y_target.end == (i+0.5)

    def test_ndoverlay_nd_labels(self):
        overlay = NdOverlay({
            ('A', f'Data {i}'): Curve(np.arange(10)*i).opts(subcoordinate_y=True)
            for i in range(3)
        }, ['Group', 'Channel'])
        plot = bokeh_renderer.get_plot(overlay)
        assert plot.state.yaxis.ticker.ticks == [0, 1, 2]
        assert plot.state.yaxis.major_label_overrides == {0: 'A, Data 0', 1: 'A, Data 1', 2: 'A, Data 2'}
        for i, sp in enumerate(plot.subplots.values()):
            assert sp.handles['glyph_renderer'].coordinates.y_target.start == (i-0.5)
            assert sp.handles['glyph_renderer'].coordinates.y_target.end == (i+0.5)

    def test_no_label(self):
        overlay = Overlay([Curve(range(10)).opts(subcoordinate_y=True) for i in range(2)])
        with pytest.raises(
            ValueError,
            match='Every Element plotted on a subcoordinate_y axis must have a label or be part of an NdOverlay.'
        ):
            bokeh_renderer.get_plot(overlay)

    def test_overlaid_without_label_no_error(self):
        overlay = Overlay([Curve(range(10), label=f'Data {i}').opts(subcoordinate_y=True) for i in range(2)])
        with_span = overlay * VSpan(1, 2)
        bokeh_renderer.get_plot(with_span)

    def test_underlaid_ytick_alignment(self):
        overlay = Overlay([Curve(range(10), label=f'Data {i}').opts(subcoordinate_y=True) for i in range(2)])
        with_span = VSpan(1, 2) * overlay
        plot = bokeh_renderer.get_plot(with_span)
        # the yticks are aligned with their subcoordinate_y axis
        assert plot.state.yaxis.ticker.ticks == [0, 1]

    def test_overlaid_ytick_alignment(self):
        overlay = Overlay([Curve(range(10), label=f'Data {i}').opts(subcoordinate_y=True) for i in range(2)])
        with_span = overlay * VSpan(1, 2)
        plot = bokeh_renderer.get_plot(with_span)
        # the yticks are aligned with their subcoordinate_y axis
        assert plot.state.yaxis.ticker.ticks == [0, 1]

    def test_custom_ylabel(self):
        overlay = Overlay([Curve(range(10), label=f'Data {i}').opts(subcoordinate_y=True) for i in range(2)])
        overlay.opts(ylabel='Y label')
        plot = bokeh_renderer.get_plot(overlay)
        # the figure axis has the label set
        assert plot.state.yaxis.axis_label == 'Y label'
        # the ticks show the labels
        assert plot.state.yaxis.ticker.ticks == [0, 1]
        assert plot.state.yaxis.major_label_overrides == {0: 'Data 0', 1: 'Data 1'}

    def test_legend_label(self):
        overlay = Overlay([Curve(range(10), label=f'Data {i}').opts(subcoordinate_y=True) for i in range(2)])
        plot = bokeh_renderer.get_plot(overlay)
        legend_labels = [l.label['value'] for l in plot.state.legend[0].items]
        # the legend displays the labels
        assert legend_labels == ['Data 0', 'Data 1']

    def test_shared_multi_axes(self):
        overlay1 = Overlay([Curve(np.arange(10), label=f'Data {i}').opts(subcoordinate_y=True) for i in range(2)])
        overlay2 = Overlay([Curve(np.arange(10) + 5, label=f'Data {i}').opts(subcoordinate_y=True) for i in range(2)])

        plot = bokeh_renderer.get_plot(overlay1 + overlay2)

        oplot1 = plot.subplots[(0, 0)].subplots['main']
        oplot2 = plot.subplots[(0, 1)].subplots['main']
        assert (oplot1.handles['y_range'].start, oplot1.handles['y_range'].end) == (-0.5, 1.5)
        assert oplot1.handles['extra_y_ranges'] == {}
        assert (oplot2.handles['y_range'].start, oplot2.handles['y_range'].end) == (-0.5, 1.5)
        assert oplot2.handles['extra_y_ranges'] == {}

    def test_invisible_yaxis(self):
        overlay = Overlay([Curve(range(10), label=f'Data {i}').opts(subcoordinate_y=True) for i in range(2)])
        overlay.opts(yaxis=None)
        plot = bokeh_renderer.get_plot(overlay)
        assert not plot.state.yaxis.visible

    def test_overlay_set_ylim(self):
        ylim = (1, 2.5)
        overlay = Overlay([Curve(range(10), label=f'Data {i}').opts(subcoordinate_y=True) for i in range(2)])
        overlay.opts(ylim=ylim)
        plot = bokeh_renderer.get_plot(overlay)
        y_range = plot.handles['y_range']
        assert y_range.start, y_range.end == ylim

    def test_axis_labels(self):
        overlay = Overlay([Curve(range(10), label=f'Data {i}').opts(subcoordinate_y=True) for i in range(2)])
        plot = bokeh_renderer.get_plot(overlay)

        assert plot.state.xaxis.axis_label == 'x'
        assert plot.state.yaxis.axis_label == 'y'

    def test_only_x_axis_labels(self):
        overlay = Overlay([Curve(range(10), label=f'Data {i}').opts(subcoordinate_y=True) for i in range(2)])
        overlay.opts(labelled=['x'])
        plot = bokeh_renderer.get_plot(overlay)

        assert plot.state.xaxis.axis_label == 'x'
        assert plot.state.yaxis.axis_label == ''

    def test_none_x_axis_labels(self):
        overlay = Overlay([Curve(range(10), vdims=['A'], label=f'Data {i}').opts(subcoordinate_y=True) for i in range(2)])
        plot = bokeh_renderer.get_plot(overlay)

        assert plot.state.xaxis.axis_label == 'x'
        assert plot.state.yaxis.axis_label == 'A'

    # With subcoordinate_y set to a range

    def test_range_base(self):
        overlay = Overlay([
            Curve(range(10), label='Data 0').opts(subcoordinate_y=(0, 0.5)),
            Curve(range(10), label='Data 1').opts(subcoordinate_y=(0.5, 1)),
        ])
        plot = bokeh_renderer.get_plot(overlay)
        # subcoordinate_y is propagated to the overlay, just the first one though :(
        assert plot.subcoordinate_y == (0, 0.5)
        # the figure has only one yaxis
        assert len(plot.state.yaxis) == 1
        # the overlay has two subplots
        assert len(plot.subplots) == 2
        assert ('Curve', 'Data_0') in plot.subplots
        assert ('Curve', 'Data_1') in plot.subplots
        # the range per subplots are correctly computed
        sp1 = plot.subplots[('Curve', 'Data_0')]
        assert sp1.handles['glyph_renderer'].coordinates.y_target.start == 0
        assert sp1.handles['glyph_renderer'].coordinates.y_target.end == 0.5
        sp2 = plot.subplots[('Curve', 'Data_1')]
        assert sp2.handles['glyph_renderer'].coordinates.y_target.start == 0.5
        assert sp2.handles['glyph_renderer'].coordinates.y_target.end == 1
        # y_range is correctly computed
        assert plot.handles['y_range'].start == 0
        assert plot.handles['y_range'].end == 1
        # extra_y_range is empty
        assert plot.handles['extra_y_ranges'] == {}
        # the ticks show the labels
        assert plot.state.yaxis.ticker.ticks == [0.25, 0.75]
        assert plot.state.yaxis.major_label_overrides == {0.25: 'Data 0', 0.75: 'Data 1'}

    def test_plot_standalone(self):
        standalone = Curve(range(10), label='Data 0').opts(subcoordinate_y=True)
        plot = bokeh_renderer.get_plot(standalone)
        assert (plot.state.x_range.start, plot.state.x_range.end) == (0, 9)
        assert (plot.state.y_range.start, plot.state.y_range.end) == (0, 9)

    def test_multi_y_error(self):
        overlay = Overlay([Curve(range(10), label=f'Data {i}').opts(subcoordinate_y=True) for i in range(2)])
        overlay.opts(multi_y=True)
        with pytest.raises(
            ValueError,
            match='multi_y and subcoordinate_y are not supported together'
        ):
            bokeh_renderer.get_plot(overlay)

    def test_same_label_error(self):
        overlay = Overlay([Curve(range(10), label='Same').opts(subcoordinate_y=True) for _ in range(2)])
        with pytest.raises(
            ValueError,
            match='Elements wrapped in a subcoordinate_y overlay must all have a unique label',
        ):
            bokeh_renderer.get_plot(overlay)

    def test_tools_default_wheel_zoom_configured(self):
        overlay = Overlay([Curve(range(10), label=f'Data {i}').opts(subcoordinate_y=True) for i in range(2)])
        plot = bokeh_renderer.get_plot(overlay)
        zoom_subcoordy = plot.handles['zooms_subcoordy']['wheel_zoom']
        assert len(zoom_subcoordy.renderers) == 2
        assert len(set(zoom_subcoordy.renderers)) == 2
        assert zoom_subcoordy.dimensions == 'height'
        assert zoom_subcoordy.level == 1

    def test_tools_string_zoom_in_out_configured(self):
        for zoom in ['zoom_in', 'zoom_out', 'yzoom_in', 'yzoom_out', 'ywheel_zoom']:
            overlay = Overlay([Curve(range(10), label=f'Data {i}').opts(subcoordinate_y=True, tools=[zoom]) for i in range(2)])
            plot = bokeh_renderer.get_plot(overlay)
            zoom_subcoordy = plot.handles['zooms_subcoordy'][zoom]
            assert len(zoom_subcoordy.renderers) == 2
            assert len(set(zoom_subcoordy.renderers)) == 2
            assert zoom_subcoordy.dimensions == 'height'
            assert zoom_subcoordy.level == 1

    def test_tools_string_x_zoom_untouched(self):
        for zoom, zoom_type in [
            ('xzoom_in', ZoomInTool),
            ('xzoom_out', ZoomOutTool),
            ('xwheel_zoom', WheelZoomTool),
        ]:
            overlay = Overlay([Curve(range(10), label=f'Data {i}').opts(subcoordinate_y=True, tools=[zoom]) for i in range(2)])
            plot = bokeh_renderer.get_plot(overlay)
            for tool in plot.state.tools:
                if isinstance(tool, zoom_type) and tool.tags == ['hv_created']:
                    assert tool.level == 0
                    assert tool.dimensions == 'width'
                    break
            else:
                raise AssertionError('Provided zoom not found.')

    def test_tools_instance_zoom_untouched(self):
        for zoom in [WheelZoomTool(), ZoomInTool(), ZoomOutTool()]:
            overlay = Overlay([Curve(range(10), label=f'Data {i}').opts(subcoordinate_y=True, tools=[zoom]) for i in range(2)])
            plot = bokeh_renderer.get_plot(overlay)
            for tool in plot.state.tools:
                if isinstance(tool, type(zoom)) and 'hv_created' not in tool.tags:
                    assert tool.level == 0
                    assert tool.dimensions == 'both'
                    break
            else:
                raise AssertionError('Provided zoom not found.')

    def test_single_group(self):
        # Same as test_bool_base, to check nothing is affected by defining
        # a single group.

        overlay = Overlay([Curve(range(10), label=f'Data {i}', group='Group').opts(subcoordinate_y=True) for i in range(2)])
        plot = bokeh_renderer.get_plot(overlay)
        # subcoordinate_y is propagated to the overlay
        assert plot.subcoordinate_y is True
        # the figure has only one yaxis
        assert len(plot.state.yaxis) == 1
        # the overlay has two subplots
        assert len(plot.subplots) == 2
        assert ('Group', 'Data_0') in plot.subplots
        assert ('Group', 'Data_1') in plot.subplots
        # the range per subplots are correctly computed
        sp1 = plot.subplots[('Group', 'Data_0')]
        assert sp1.handles['glyph_renderer'].coordinates.y_target.start == -0.5
        assert sp1.handles['glyph_renderer'].coordinates.y_target.end == 0.5
        sp2 = plot.subplots[('Group', 'Data_1')]
        assert sp2.handles['glyph_renderer'].coordinates.y_target.start == 0.5
        assert sp2.handles['glyph_renderer'].coordinates.y_target.end == 1.5
        # y_range is correctly computed
        assert plot.handles['y_range'].start == -0.5
        assert plot.handles['y_range'].end == 1.5
        # extra_y_range is empty
        assert plot.handles['extra_y_ranges'] == {}
        # the ticks show the labels
        assert plot.state.yaxis.ticker.ticks == [0, 1]
        assert plot.state.yaxis.major_label_overrides == {0: 'Data 0', 1: 'Data 1'}

    def test_multiple_groups(self):
        overlay = Overlay([
            Curve(range(10), label=f'{group} / {i}', group=group).opts(subcoordinate_y=True)
            for group in ['A', 'B']
            for i in range(2)
        ])
        plot = bokeh_renderer.get_plot(overlay)
        # subcoordinate_y is propagated to the overlay
        assert plot.subcoordinate_y is True
        # the figure has only one yaxis
        assert len(plot.state.yaxis) == 1
        # the overlay has two subplots
        assert len(plot.subplots) == 4
        assert ('A', 'A_over_0') in plot.subplots
        assert ('A', 'A_over_1') in plot.subplots
        assert ('B', 'B_over_0') in plot.subplots
        assert ('B', 'B_over_1') in plot.subplots
        # the range per subplots are correctly computed
        sp1 = plot.subplots[('A', 'A_over_0')]
        assert sp1.handles['glyph_renderer'].coordinates.y_target.start == -0.5
        assert sp1.handles['glyph_renderer'].coordinates.y_target.end == 0.5
        sp2 = plot.subplots[('A', 'A_over_1')]
        assert sp2.handles['glyph_renderer'].coordinates.y_target.start == 0.5
        assert sp2.handles['glyph_renderer'].coordinates.y_target.end == 1.5
        sp3 = plot.subplots[('B', 'B_over_0')]
        assert sp3.handles['glyph_renderer'].coordinates.y_target.start == 1.5
        assert sp3.handles['glyph_renderer'].coordinates.y_target.end == 2.5
        sp4 = plot.subplots[('B', 'B_over_1')]
        assert sp4.handles['glyph_renderer'].coordinates.y_target.start == 2.5
        assert sp4.handles['glyph_renderer'].coordinates.y_target.end == 3.5
        # y_range is correctly computed
        assert plot.handles['y_range'].start == -0.5
        assert plot.handles['y_range'].end == 3.5
        # extra_y_range is empty
        assert plot.handles['extra_y_ranges'] == {}
        # the ticks show the labels
        assert plot.state.yaxis.ticker.ticks == [0, 1, 2, 3]
        assert plot.state.yaxis.major_label_overrides == {
            0: 'A / 0', 1: 'A / 1',
            2: 'B / 0', 3: 'B / 1',
        }

    @pytest.mark.skipif(BOKEH_GE_3_5_0, reason="test Bokeh < 3.5")
    def test_multiple_groups_wheel_zoom_configured(self):
        # Same as test_tools_default_wheel_zoom_configured

        groups = ['A', 'B']
        overlay = Overlay([
            Curve(range(10), label=f'{group} / {i}', group=group).opts(subcoordinate_y=True)
            for group in groups
            for i in range(2)
        ])
        plot = bokeh_renderer.get_plot(overlay)
        zoom_tools = [tool for tool in plot.state.tools if isinstance(tool, WheelZoomTool)]
        assert zoom_tools == plot.handles['zooms_subcoordy']['wheel_zoom']
        assert len(zoom_tools) == len(groups)
        for zoom_tool, group in zip(zoom_tools, reversed(groups), strict=None):
            assert len(zoom_tool.renderers) == 2
            assert len(set(zoom_tool.renderers)) == 2
            assert zoom_tool.dimensions == 'height'
            assert zoom_tool.level == 1
            assert zoom_tool.description == f'Wheel Zoom ({group})'

    @pytest.mark.skipif(not BOKEH_GE_3_5_0, reason="requires Bokeh >= 3.5")
    def test_multiple_groups_wheel_zoom_configured_35(self):
        # Same as test_tools_default_wheel_zoom_configured

        groups = ['A', 'B']
        overlay = Overlay([
            Curve(range(10), label=f'{group} / {i}', group=group).opts(subcoordinate_y=True)
            for group in groups
            for i in range(2)
        ])
        plot = bokeh_renderer.get_plot(overlay)
        zoom_tools = [tool for tool in plot.state.tools if isinstance(tool, WheelZoomTool)]
        assert len(zoom_tools) == 1
        zoom_tool = zoom_tools[0]
        assert zoom_tool == plot.handles['zooms_subcoordy']['wheel_zoom']
        assert len(zoom_tool.renderers) == 4
        assert zoom_tool.dimensions == 'height'
        assert zoom_tool.level == 1
        assert zoom_tool.hit_test is True
        assert zoom_tool.hit_test_mode == 'hline'
        assert len(zoom_tool.hit_test_behavior.groups) == 2

    def test_single_group_overlaid_no_error(self):
        overlay = Overlay([Curve(range(10), label=f'Data {i}', group='Group').opts(subcoordinate_y=True) for i in range(2)])
        with_span = VSpan(1, 2) * overlay * VSpan(3, 4)
        bokeh_renderer.get_plot(with_span)

    def test_multiple_groups_overlaid_no_error(self):
        overlay = Overlay([
            Curve(range(10), label=f'{group} / {i}', group=group).opts(subcoordinate_y=True)
            for group in ['A', 'B']
            for i in range(2)
        ])
        with_span = VSpan(1, 2) * overlay * VSpan(3, 4)
        bokeh_renderer.get_plot(with_span)

    def test_missing_group_error(self):
        curves = []
        for i, group in enumerate(['A', 'B', 'C']):
            for i in range(2):
                label = f'{group}{i}'
                if group == "B":
                    curve = Curve(range(10), label=label, group=group).opts(
                        subcoordinate_y=True
                    )
                else:
                    curve = Curve(range(10), label=label).opts(
                        subcoordinate_y=True
                    )
                curves.append(curve)

        with pytest.raises(
            ValueError,
            match=(
                'The subcoordinate_y overlay contains elements with a defined group, each '
                'subcoordinate_y element in the overlay must have a defined group.'
            )
        ):
            bokeh_renderer.get_plot(Overlay(curves))

    def test_norm_subcoordinate_group_ranges(self):
        x = np.linspace(0, 10 * np.pi, 21)
        curves = []
        j = 0
        for group in ['A', 'B']:
            for i in range(2):
                yvals = j * np.sin(x)
                curves.append(
                    Curve((x + np.pi/2, yvals), label=f'{group}{i}', group=group).opts(subcoordinate_y=True)
                )
                j += 1

        overlay = Overlay(curves)
        noverlay = subcoordinate_group_ranges(overlay)

        expected = [
            (-1.0, 1.0),
            (-1.0, 1.0),
            (-3.0, 3.0),
            (-3.0, 3.0),
        ]
        for i, el in enumerate(noverlay):
            assert el.get_dimension('y').range == expected[i]

        plot = bokeh_renderer.get_plot(noverlay)

        for i, sp in enumerate(plot.subplots.values()):
                y_source = sp.handles['glyph_renderer'].coordinates.y_source
                assert (y_source.start, y_source.end) == expected[i]
