from bokeh.models import LinearAxis, LinearScale, LogAxis, LogScale

from holoviews.element import Curve

from ...utils import LoggingComparisonTestCase
from .test_plot import TestBokehPlot, bokeh_renderer


class TestCurveTwinAxes(LoggingComparisonTestCase, TestBokehPlot):

    def test_multi_y_disabled(self):
        overlay = (Curve(range(10)) * Curve(range(10)))
        plot = bokeh_renderer.get_plot(overlay).state
        self.assertEqual(len(plot.yaxis), 1)

    def test_multi_y_enabled_two_curves_one_vdim(self):
        overlay = (Curve(range(10)) * Curve(range(10))).opts(multi_y=True)
        plot = bokeh_renderer.get_plot(overlay).state
        self.assertEqual(len(plot.yaxis), 1)

    def test_multi_y_enabled_two_curves_two_vdim(self):
        overlay = (Curve(range(10), vdims=['A'])
                   * Curve(range(10), vdims=['B'])).opts(multi_y=True)
        plot = bokeh_renderer.get_plot(overlay)
        self.assertEqual(len(plot.state.yaxis), 2)
        y_range = plot.handles['y_range']
        self.assertEqual(y_range.start, 0)
        self.assertEqual(y_range.end, 9)
        extra_y_ranges = plot.handles['extra_y_ranges']
        self.assertEqual(list(extra_y_ranges.keys()), ['B'])
        self.assertEqual(extra_y_ranges['B'].start, 0)
        self.assertEqual(extra_y_ranges['B'].end, 9)

    def test_multi_y_enabled_three_curves_two_vdim(self):
        curve_1A = Curve(range(10), vdims=['A'])
        curve_2B = Curve(range(11), vdims=['B'])
        curve_3A = Curve(range(12), vdims=['A'])
        overlay = (curve_1A * curve_2B * curve_3A ).opts(multi_y=True)
        plot = bokeh_renderer.get_plot(overlay).state
        self.assertEqual(len(plot.yaxis), 2)

    # Testing independent y-lims

    def test_multi_y_lims_left_axis(self):
        overlay = (Curve(range(10), vdims=['A']).opts(ylim=(-10,20))
                   * Curve(range(10), vdims=['B'])).opts(multi_y=True)
        plot = bokeh_renderer.get_plot(overlay)
        y_range = plot.handles['y_range']
        self.assertEqual(y_range.start, -10)
        self.assertEqual(y_range.end, 20)
        extra_y_ranges = plot.handles['extra_y_ranges']
        self.assertEqual(list(extra_y_ranges.keys()), ['B'])
        self.assertEqual(extra_y_ranges['B'].start, 0)
        self.assertEqual(extra_y_ranges['B'].end, 9)

    def test_multi_y_lims_right_axis(self):
        overlay = (Curve(range(10), vdims=['A'])
                   * Curve(range(10), vdims=['B']).opts(ylim=(-10,20))).opts(multi_y=True)
        plot = bokeh_renderer.get_plot(overlay)
        y_range = plot.handles['y_range']
        self.assertEqual(y_range.start, 0)
        self.assertEqual(y_range.end, 9)
        extra_y_ranges = plot.handles['extra_y_ranges']
        self.assertEqual(list(extra_y_ranges.keys()), ['B'])
        self.assertEqual(extra_y_ranges['B'].start, -10)
        self.assertEqual(extra_y_ranges['B'].end, 20)

    def test_multi_y_lims_both_axes(self):
        overlay = (Curve(range(10), vdims=['A']).opts(ylim=(-15,25))
                   * Curve(range(10), vdims=['B']).opts(ylim=(-10,20))).opts(multi_y=True)
        plot = bokeh_renderer.get_plot(overlay)
        y_range = plot.handles['y_range']
        self.assertEqual(y_range.start, -15)
        self.assertEqual(y_range.end, 25)
        extra_y_ranges = plot.handles['extra_y_ranges']
        self.assertEqual(list(extra_y_ranges.keys()), ['B'])
        self.assertEqual(extra_y_ranges['B'].start, -10)
        self.assertEqual(extra_y_ranges['B'].end, 20)

    # Testing independent logy

    def test_multi_log_left_axis(self):
        overlay = (Curve(range(1,9), vdims=['A']).opts(logy=True)
                   * Curve(range(10), vdims=['B'])).opts(multi_y=True)
        plot = bokeh_renderer.get_plot(overlay)
        self.assertEqual(len(plot.state.yaxis), 2)
        self.assertTrue(isinstance(plot.state.yaxis[0], LogAxis))
        self.assertTrue(isinstance(plot.state.yaxis[1], LinearAxis))
        extra_y_ranges = plot.handles['extra_y_scales']
        self.assertTrue(isinstance(extra_y_ranges['B'], LinearScale))

    def test_multi_log_right_axis(self):
        overlay = (Curve(range(1,9), vdims=['A'])
                   * Curve(range(1, 9), vdims=['B']).opts(logy=True)).opts(multi_y=True)
        plot = bokeh_renderer.get_plot(overlay)
        self.assertEqual(len(plot.state.yaxis), 2)
        self.assertTrue(isinstance(plot.state.yaxis[0], LinearAxis))
        self.assertTrue(isinstance(plot.state.yaxis[1], LogAxis))
        extra_y_ranges = plot.handles['extra_y_scales']
        self.assertTrue(isinstance(extra_y_ranges['B'], LogScale))


    def test_multi_log_both_axes(self):
        overlay = (Curve(range(1,9), vdims=['A']).opts(logy=True)
                   * Curve(range(1, 9), vdims=['B']).opts(logy=True)).opts(multi_y=True)
        plot = bokeh_renderer.get_plot(overlay)
        self.assertEqual(len(plot.state.yaxis), 2)
        self.assertTrue(isinstance(plot.state.yaxis[0], LogAxis))
        self.assertTrue(isinstance(plot.state.yaxis[1], LogAxis))
        extra_y_ranges = plot.handles['extra_y_scales']
        self.assertTrue(isinstance(extra_y_ranges['B'], LogScale))

    # BUG! left_axis is not warning (main axis)

    def test_multi_log_right_axis_warn(self):
        overlay = (Curve(range(10), vdims=['A'])
                   * Curve(range(10), vdims=['B']).opts(logy=True)).opts(multi_y=True)
        plot = bokeh_renderer.get_plot(overlay)
        self.assertEqual(len(plot.state.yaxis), 2)
        self.assertTrue(isinstance(plot.state.yaxis[0], LinearAxis))
        self.assertTrue(isinstance(plot.state.yaxis[1], LogAxis))
        extra_y_ranges = plot.handles['extra_y_scales']
        self.assertTrue(isinstance(extra_y_ranges['B'], LogScale))
        #print(self.log_handler)
        substr = "Logarithmic axis range encountered value less than or equal to zero, please supply explicit lower bound to override default of 0.010."
        self.log_handler.assertEndsWith('WARNING', substr)

    # Testing invert_yaxis

    def test_multi_invert_left_axis(self):
        overlay = (Curve(range(10), vdims=['A']).opts(invert_yaxis=True)
                   * Curve(range(10), vdims=['B'])).opts(multi_y=True)
        plot = bokeh_renderer.get_plot(overlay)
        y_range = plot.handles['y_range']
        self.assertEqual(y_range.start, 9)
        self.assertEqual(y_range.end, 0)
        extra_y_ranges = plot.handles['extra_y_ranges']
        self.assertEqual(list(extra_y_ranges.keys()), ['B'])
        self.assertEqual(extra_y_ranges['B'].start, 0)
        self.assertEqual(extra_y_ranges['B'].end, 9)

    def test_multi_invert_right_axis(self):
        overlay = (Curve(range(10), vdims=['A'])
                   * Curve(range(10), vdims=['B']).opts(invert_yaxis=True)).opts(multi_y=True)
        plot = bokeh_renderer.get_plot(overlay)
        y_range = plot.handles['y_range']
        self.assertEqual(y_range.start, 0)
        self.assertEqual(y_range.end, 9)
        extra_y_ranges = plot.handles['extra_y_ranges']
        self.assertEqual(list(extra_y_ranges.keys()), ['B'])
        self.assertEqual(extra_y_ranges['B'].start, 9)
        self.assertEqual(extra_y_ranges['B'].end, 0)


    def test_multi_invert_both_axes(self):
        overlay = (Curve(range(10), vdims=['A']).opts(invert_yaxis=True)
                   * Curve(range(10), vdims=['B']).opts(invert_yaxis=True)).opts(multi_y=True)
        plot = bokeh_renderer.get_plot(overlay)
        y_range = plot.handles['y_range']
        self.assertEqual(y_range.start, 9)
        self.assertEqual(y_range.end, 0)
        extra_y_ranges = plot.handles['extra_y_ranges']
        self.assertEqual(list(extra_y_ranges.keys()), ['B'])
        self.assertEqual(extra_y_ranges['B'].start, 9)
        self.assertEqual(extra_y_ranges['B'].end, 0)


    # Combination test

    def test_inverted_log_ylim_right_axis(self):
        overlay = (Curve(range(10), vdims=['A'])
                   * Curve(range(10), vdims=['B']
                           ).opts(invert_yaxis=True, logy=True, ylim=(2,20))).opts(multi_y=True)
        plot = bokeh_renderer.get_plot(overlay)
        y_range = plot.handles['y_range']
        self.assertEqual(y_range.start, 0)
        self.assertEqual(y_range.end, 9)
        extra_y_ranges = plot.handles['extra_y_ranges']
        self.assertEqual(list(extra_y_ranges.keys()), ['B'])
        print(extra_y_ranges['B'].start, extra_y_ranges['B'].end)
        self.assertEqual(extra_y_ranges['B'].start, 20)
        self.assertEqual(extra_y_ranges['B'].end, 2)
        self.assertTrue(isinstance(plot.handles['extra_y_scales']['B'], LogScale))


    # Test axis sharing in layouts

    def test_shared_multi_axes(self):
        curve1A = Curve([9, 8, 7, 6, 5], vdims='A')
        curve2B = Curve([1, 2, 3], vdims='B')
        overlay1 = (curve1A * curve2B).opts(multi_y=True)

        curve3A = Curve([19, 18, 17, 16, 15], vdims='A')
        curve4B = Curve([11, 12, 13], vdims='B')
        overlay2 = (curve3A * curve4B).opts(multi_y=True)

        plot = bokeh_renderer.get_plot(overlay1 + overlay2)
        plot = plot.subplots[(0, 1)].subplots['main']
        y_range = plot.handles['y_range']
        extra_y_ranges = plot.handles['extra_y_ranges']

        self.assertEqual((y_range.start, y_range.end), (5, 19))
        self.assertEqual((extra_y_ranges['B'].start, extra_y_ranges['B'].end), (1, 13))

    def test_invisible_main_axis(self):
        overlay = (
            Curve(range(10), vdims=['A']).opts(yaxis=None) *
            Curve(range(10), vdims=['B'])
        ).opts(multi_y=True)
        plot = bokeh_renderer.get_plot(overlay)
        assert len(plot.state.yaxis) == 2
        assert not plot.state.yaxis[0].visible
        assert plot.state.yaxis[1].visible

    def test_invisible_extra_axis(self):
        overlay = (
            Curve(range(10), vdims=['A']) *
            Curve(range(10), vdims=['B']).opts(yaxis=None)
        ).opts(multi_y=True)
        plot = bokeh_renderer.get_plot(overlay)
        assert len(plot.state.yaxis) == 2
        assert plot.state.yaxis[0].visible
        assert not plot.state.yaxis[1].visible

    def test_axis_labels(self):
        overlay = (
            Curve(range(10), vdims=['A']) *
            Curve(range(10), vdims=['B'])
        ).opts(multi_y=True)
        plot = bokeh_renderer.get_plot(overlay)

        assert plot.state.xaxis[0].axis_label == 'x'
        assert plot.state.yaxis[0].axis_label == 'A'
        assert plot.state.yaxis[1].axis_label == 'B'

    def test_custom_axis_labels(self):
        overlay = (
            Curve(range(10), vdims=['A']).opts(xlabel='x-custom', ylabel='A-custom') *
            Curve(range(10), vdims=['B']).opts(ylabel='B-custom')
        ).opts(multi_y=True)
        plot = bokeh_renderer.get_plot(overlay)

        assert plot.state.xaxis[0].axis_label == 'x-custom'
        assert plot.state.yaxis[0].axis_label == 'A-custom'
        assert plot.state.yaxis[1].axis_label == 'B-custom'

    def test_only_x_axis_labels(self):
        overlay = (
            Curve(range(10), vdims=['A']) *
            Curve(range(10), vdims=['B'])
        ).opts(multi_y=True, labelled=['x'])
        plot = bokeh_renderer.get_plot(overlay)

        assert plot.state.xaxis[0].axis_label == 'x'
        assert plot.state.yaxis[0].axis_label is None
        assert plot.state.yaxis[1].axis_label is None

    def test_none_x_axis_labels(self):
        overlay = (
            Curve(range(10), vdims=['A']) *
            Curve(range(10), vdims=['B'])
        ).opts(multi_y=True, labelled=['y'])
        plot = bokeh_renderer.get_plot(overlay)

        assert plot.state.xaxis[0].axis_label is None
        assert plot.state.yaxis[0].axis_label == 'A'
        assert plot.state.yaxis[1].axis_label == 'B'

    def test_swapped_position_label(self):
        overlay = (
            Curve(range(10), vdims=['A']).opts(yaxis='right') *
            Curve(range(10), vdims=['B']).opts(yaxis='left')
        ).opts(multi_y=True)
        plot = bokeh_renderer.get_plot(overlay)

        assert plot.state.yaxis[0].axis_label == 'B'
        assert plot.state.yaxis[1].axis_label == 'A'

    def test_swapped_position_custom_y_labels(self):
        overlay = (Curve(range(10), vdims=['A']).opts(yaxis='right', ylabel='A-custom')
                   * Curve(range(10), vdims=['B']).opts(yaxis='left', ylabel='B-custom')
                   ).opts(multi_y=True)
        plot = bokeh_renderer.get_plot(overlay)

        assert plot.state.yaxis[0].axis_label == 'B-custom'
        assert plot.state.yaxis[1].axis_label == 'A-custom'

    def test_position_custom_size_label(self):
        overlay = (
            Curve(range(10), vdims='A').opts(fontsize={'ylabel': '13pt'}) *
            Curve(range(10), vdims='B').opts(fontsize={'ylabel': '15pt'})
        ).opts(multi_y=True)
        plot = bokeh_renderer.get_plot(overlay)
        assert plot.state.yaxis[0].axis_label == 'A'
        assert plot.state.yaxis[0].axis_label_text_font_size == '13pt'
        assert plot.state.yaxis[1].axis_label == 'B'
        assert plot.state.yaxis[1].axis_label_text_font_size == '15pt'

    def test_swapped_position_custom_size_label(self):
        overlay = (
            Curve(range(10), vdims='A').opts(yaxis='right', fontsize={'ylabel': '13pt'}) *
            Curve(range(10), vdims='B').opts(yaxis='left', fontsize={'ylabel': '15pt'})
        ).opts(multi_y=True)
        plot = bokeh_renderer.get_plot(overlay)
        assert plot.state.yaxis[0].axis_label == 'B'
        assert plot.state.yaxis[0].axis_label_text_font_size == '15pt'
        assert plot.state.yaxis[1].axis_label == 'A'
        assert plot.state.yaxis[1].axis_label_text_font_size == '13pt'

    def test_multi_y_on_curve(self):
        # Test for https://github.com/holoviz/holoviews/issues/6322
        overlay = Curve(range(10), vdims='A').opts(multi_y=True)

        # Should not error
        bokeh_renderer.get_plot(overlay)
