import numpy as np
import pytest

from holoviews.core import Overlay
from holoviews.element import Curve
from holoviews.element.annotation import VSpan

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

    def test_no_label(self):
        overlay = Overlay([Curve(range(10)).opts(subcoordinate_y=True) for i in range(2)])
        with pytest.raises(
            ValueError,
            match='Every element wrapped in a subcoordinate_y overlay must have a label'
        ):
            bokeh_renderer.get_plot(overlay)

    def test_overlaid_without_label_no_error(self):
        overlay = Overlay([Curve(range(10), label=f'Data {i}').opts(subcoordinate_y=True) for i in range(2)])
        with_span = overlay * VSpan(1, 2)
        bokeh_renderer.get_plot(with_span)

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
