from unittest import TestCase, SkipTest

try:
    from unittest.mock import Mock
except:
    from mock import Mock

import uuid

try:
    import plotly.graph_objs as go
except:
    go = None

try:
    from holoviews.plotting.plotly.callbacks import (
        RangeXYCallback, RangeXCallback, RangeYCallback,
        BoundsXYCallback, BoundsXCallback, BoundsYCallback,
        Selection1DCallback
    )
except:
    pass


def mock_plot(trace_uid=None):
    # Build a mock to stand in for a PlotlyPlot subclass
    if trace_uid is None:
        trace_uid = str(uuid.uuid4())

    plot = Mock()
    plot.trace_uid = trace_uid
    return plot


def build_callback_set(callback_cls, trace_uids, num_streams=2):
    """
    Build a collection of plots, callbacks, and streams for a given callback class and
    a list of trace_uids
    """
    plots = []
    streamss = []
    callbacks = []
    for trace_uid in trace_uids:
        plot = mock_plot(trace_uid)
        streams = [Mock() for _ in range(num_streams)]
        callback = callback_cls(plot, streams, None)

        plots.append(plot)
        streamss.append(streams)
        callbacks.append(callback)

    return plots, streamss, callbacks


class TestCallbacks(TestCase):

    def setUp(self):
        if go is None:
            raise SkipTest("Plotly required to test plotly callbacks")
        self.fig_dict = go.Figure({
            'data': [
                {'type': 'scatter',
                 'y': [1, 2, 3],
                 'uid': 'first'},
                {'type': 'bar',
                 'y': [1, 2, 3],
                 'uid': 'second',
                 'xaxis': 'x',
                 'yaxis': 'y'},
                {'type': 'scatter',
                 'y': [1, 2, 3],
                 'uid': 'third',
                 'xaxis': 'x2',
                 'yaxis': 'y2'},
                {'type': 'bar',
                 'y': [1, 2, 3],
                 'uid': 'forth',
                 'xaxis': 'x3',
                 'yaxis': 'y3'},
            ],
            'layout': {
                'title': {'text': 'Figure Title'}}
        }).to_dict()

    def testCallbackClassInstanceTracking(self):
        # Each callback class should track all active instances of its own class in a
        # weak value dictionary. Here we make sure that instances stay separated per
        # class
        plot1 = mock_plot()
        plot2 = mock_plot()
        plot3 = mock_plot()

        # Check RangeXYCallback
        rangexy_cb = RangeXYCallback(plot1, [], None)
        self.assertIn(plot1.trace_uid, RangeXYCallback.instances)
        self.assertIs(rangexy_cb, RangeXYCallback.instances[plot1.trace_uid])

        # Check BoundsXYCallback
        boundsxy_cb = BoundsXYCallback(plot2, [], None)
        self.assertIn(plot2.trace_uid, BoundsXYCallback.instances)
        self.assertIs(boundsxy_cb, BoundsXYCallback.instances[plot2.trace_uid])

        # Check Selection1DCallback
        selection1d_cb = Selection1DCallback(plot3, [], None)
        self.assertIn(plot3.trace_uid, Selection1DCallback.instances)
        self.assertIs(selection1d_cb, Selection1DCallback.instances[plot3.trace_uid])

        # Check that objects don't show up as instances in the wrong class
        self.assertNotIn(plot1.trace_uid, BoundsXYCallback.instances)
        self.assertNotIn(plot1.trace_uid, Selection1DCallback.instances)
        self.assertNotIn(plot2.trace_uid, RangeXYCallback.instances)
        self.assertNotIn(plot2.trace_uid, Selection1DCallback.instances)
        self.assertNotIn(plot3.trace_uid, RangeXYCallback.instances)
        self.assertNotIn(plot3.trace_uid, BoundsXYCallback.instances)

    def testRangeCallbacks(self):

        # Build callbacks
        range_classes = [RangeXYCallback, RangeXCallback, RangeYCallback]

        xyplots, xystreamss, xycallbacks = build_callback_set(
            RangeXYCallback, ['first', 'second', 'third', 'forth', 'other'], 2
        )

        xplots, xstreamss, xcallbacks = build_callback_set(
            RangeXCallback, ['first', 'second', 'third', 'forth', 'other'], 2
        )

        yplots, ystreamss, ycallbacks = build_callback_set(
            RangeYCallback, ['first', 'second', 'third', 'forth', 'other'], 2
        )

        # Sanity check the length of the streams lists
        for xystreams in xystreamss:
            self.assertEqual(len(xystreams), 2)

        # Change viewport on first set of axes
        viewport1 = {'xaxis.range': [1, 4], 'yaxis.range': [-1, 5]}
        for cb_cls in range_classes:
            cb_cls.update_streams_from_property_update(viewport1, self.fig_dict)

        # Check that all streams attached to 'first' and 'second' plots were triggered
        for xystream, xstream, ystream in zip(
                xystreamss[0] + xystreamss[1],
                xstreamss[0] + xstreamss[1],
                ystreamss[0] + ystreamss[1],
        ):
            xystream.event.assert_called_once_with(x_range=(1, 4), y_range=(-1, 5))
            xstream.event.assert_called_once_with(x_range=(1, 4))
            ystream.event.assert_called_once_with(y_range=(-1, 5))

        # And that no other streams were triggered
        for xystream, xstream, ystream in zip(
                xystreamss[2] + xystreamss[3],
                xstreamss[2] + xstreamss[3],
                ystreamss[2] + ystreamss[3],
        ):
            xystream.event.assert_called_with(x_range=None, y_range=None)
            xstream.event.assert_called_with(x_range=None)
            ystream.event.assert_called_with(y_range=None)

        # Change viewport on second set of axes
        viewport2 = {'xaxis2.range': [2, 5], 'yaxis2.range': [0, 6]}
        for cb_cls in range_classes:
            cb_cls.update_streams_from_property_update(viewport2, self.fig_dict)

        # Check that all streams attached to 'third' were triggered
        for xystream, xstream, ystream in zip(
                xystreamss[2], xstreamss[2], ystreamss[2]
        ):
            xystream.event.assert_called_with(x_range=(2, 5), y_range=(0, 6))
            xstream.event.assert_called_with(x_range=(2, 5))
            ystream.event.assert_called_with(y_range=(0, 6))

        # Change viewport on third set of axes
        viewport3 = {'xaxis3.range': [3, 6], 'yaxis3.range': [1, 7]}
        for cb_cls in range_classes:
            cb_cls.update_streams_from_property_update(viewport3, self.fig_dict)

        # Check that all streams attached to 'forth' were triggered
        for xystream, xstream, ystream in zip(
                xystreamss[3], xstreamss[3], ystreamss[3]
        ):
            xystream.event.assert_called_with(x_range=(3, 6), y_range=(1, 7))
            xstream.event.assert_called_with(x_range=(3, 6))
            ystream.event.assert_called_with(y_range=(1, 7))

        # Check that streams attached to a trace not in this plot are not triggered
        for xystream, xstream, ystream in zip(
                xystreamss[4], xstreamss[4], ystreamss[4],
        ):
            xystream.event.assert_not_called()
            xstream.event.assert_not_called()
            ystream.event.assert_not_called()

    def testBoundsCallbacks(self):

        # Build callbacks
        bounds_classes = [BoundsXYCallback, BoundsXCallback, BoundsYCallback]

        xyplots, xystreamss, xycallbacks = build_callback_set(
            BoundsXYCallback, ['first', 'second', 'third', 'forth', 'other'], 2
        )

        xplots, xstreamss, xcallbacks = build_callback_set(
            BoundsXCallback, ['first', 'second', 'third', 'forth', 'other'], 2
        )

        yplots, ystreamss, ycallbacks = build_callback_set(
            BoundsYCallback, ['first', 'second', 'third', 'forth', 'other'], 2
        )

        # box selection on first set of axes
        selected_data1 = {'range': {'x': [1, 4], 'y': [-1, 5]}}
        for cb_cls in bounds_classes:
            cb_cls.update_streams_from_property_update(selected_data1, self.fig_dict)

        # Check that all streams attached to 'first' and 'second' plots were triggered
        for xystream, xstream, ystream in zip(
                xystreamss[0] + xystreamss[1],
                xstreamss[0] + xstreamss[1],
                ystreamss[0] + ystreamss[1],
        ):
            xystream.event.assert_called_once_with(bounds=(1, -1, 4, 5))
            xstream.event.assert_called_once_with(boundsx=(1, 4))
            ystream.event.assert_called_once_with(boundsy=(-1, 5))

        # Check that streams attached to plots in other subplots are called with None
        # to clear their bounds
        for xystream, xstream, ystream in zip(
                xystreamss[2] + xystreamss[3],
                xstreamss[2] + xstreamss[3],
                ystreamss[2] + ystreamss[3],
        ):
            xystream.event.assert_called_once_with(bounds=None)
            xstream.event.assert_called_once_with(boundsx=None)
            ystream.event.assert_called_once_with(boundsy=None)

        # box select on second set of axes
        selected_data2 = {'range': {'x2': [2, 5], 'y2': [0, 6]}}
        for cb_cls in bounds_classes:
            cb_cls.update_streams_from_property_update(selected_data2, self.fig_dict)

        # Check that all streams attached to 'second' were triggered
        for xystream, xstream, ystream in zip(
                xystreamss[2], xstreamss[2], ystreamss[2],
        ):
            xystream.event.assert_called_with(bounds=(2, 0, 5, 6))
            xstream.event.assert_called_with(boundsx=(2, 5))
            ystream.event.assert_called_with(boundsy=(0, 6))

        # box select on third set of axes
        selected_data3 = {'range': {'x3': [3, 6], 'y3': [1, 7]}}
        for cb_cls in bounds_classes:
            cb_cls.update_streams_from_property_update(selected_data3, self.fig_dict)

        # Check that all streams attached to 'third' were triggered
        for xystream, xstream, ystream in zip(
                xystreamss[3], xstreamss[3], ystreamss[3],
        ):
            xystream.event.assert_called_with(bounds=(3, 1, 6, 7))
            xstream.event.assert_called_with(boundsx=(3, 6))
            ystream.event.assert_called_with(boundsy=(1, 7))

        # lasso select on first set of axes should clear all bounds
        selected_data_lasso = {'lassoPoints': {'x': [1, 4, 2], 'y': [-1, 5, 2]}}
        for cb_cls in bounds_classes:
            cb_cls.update_streams_from_property_update(
                selected_data_lasso, self.fig_dict)

        # Check that all streams attached to this figure are called with None
        # to clear their bounds
        for xystream, xstream, ystream in zip(
                xystreamss[0] + xystreamss[1] + xystreamss[2] + xystreamss[3],
                xstreamss[0] + xstreamss[1] + xstreamss[2] + xstreamss[3],
                ystreamss[0] + ystreamss[1] + ystreamss[2] + ystreamss[3],
        ):
            xystream.event.assert_called_with(bounds=None)
            xstream.event.assert_called_with(boundsx=None)
            ystream.event.assert_called_with(boundsy=None)

        # Check that streams attached to plots not in this figure are not called
        for xystream, xstream, ystream in zip(
                xystreamss[4], xstreamss[4], ystreamss[4]
        ):
            xystream.event.assert_not_called()
            xstream.event.assert_not_called()
            ystream.event.assert_not_called()

    def testSelection1DCallback(self):
        plots, streamss, callbacks = build_callback_set(
            Selection1DCallback, ['first', 'second', 'third', 'forth', 'other'], 2
        )

        # Select points from the 'first' plot (first set of axes)
        selected_data1 = {'points': [
            {"pointNumber": 0, "curveNumber": 0},
            {"pointNumber": 2, "curveNumber": 0},
        ]}
        Selection1DCallback.update_streams_from_property_update(
            selected_data1, self.fig_dict)

        # Check that all streams attached to the 'first' plots were triggered
        for stream in streamss[0]:
            stream.event.assert_called_once_with(index=[0, 2])

        # Check that all streams attached to other plots in this figure were triggered
        # with empty selection
        for stream in streamss[1] + streamss[2] + streamss[3]:
            stream.event.assert_called_once_with(index=[])

        # Select points from the 'first' and 'second' plot (first set of axes)
        selected_data1 = {'points': [
            {"pointNumber": 0, "curveNumber": 0},
            {"pointNumber": 1, "curveNumber": 0},
            {"pointNumber": 1, "curveNumber": 1},
            {"pointNumber": 2, "curveNumber": 1},
        ]}
        Selection1DCallback.update_streams_from_property_update(
            selected_data1, self.fig_dict)

        # Check that all streams attached to the 'first' plot were triggered
        for stream in streamss[0]:
            stream.event.assert_called_with(index=[0, 1])

        # Check that all streams attached to the 'second' plot were triggered
        for stream in streamss[1]:
            stream.event.assert_called_with(index=[1, 2])

        # Check that all streams attached to other plots in this figure were triggered
        # with empty selection
        for stream in streamss[2] + streamss[3]:
            stream.event.assert_called_with(index=[])

        # Select points from the 'forth' plot (third set of axes)
        selected_data1 = {'points': [
            {"pointNumber": 0, "curveNumber": 3},
            {"pointNumber": 2, "curveNumber": 3},
        ]}
        Selection1DCallback.update_streams_from_property_update(
            selected_data1, self.fig_dict)

        # Check that all streams attached to the 'forth' plot were triggered
        for stream in streamss[3]:
            stream.event.assert_called_with(index=[0, 2])

        # Check that streams attached to plots not in this figure are not called
        for stream in streamss[4]:
            stream.event.assert_not_called()
