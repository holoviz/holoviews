import uuid

from unittest import TestCase
from unittest.mock import Mock

import plotly.graph_objs as go

from holoviews import Tiles
from holoviews.streams import (
    BoundsXY, BoundsX, BoundsY, RangeXY, RangeX, RangeY, Selection1D
)
from holoviews.plotting.plotly.callbacks import (
    RangeXYCallback, RangeXCallback, RangeYCallback,
    BoundsXYCallback, BoundsXCallback, BoundsYCallback,
    Selection1DCallback
)


def mock_plot(trace_uid=None):
    # Build a mock to stand in for a PlotlyPlot subclass
    if trace_uid is None:
        trace_uid = str(uuid.uuid4())

    plot = Mock()
    plot.trace_uid = trace_uid
    return plot


def build_callback_set(callback_cls, trace_uids, stream_type, num_streams=2):
    """
    Build a collection of plots, callbacks, and streams for a given callback class and
    a list of trace_uids
    """
    plots = []
    streamss = []
    callbacks = []
    eventss = []
    for trace_uid in trace_uids:
        plot = mock_plot(trace_uid)
        streams, event_list = [], []
        for _ in range(num_streams):
            events = []
            stream = stream_type()
            def cb(events=events, **kwargs):
                events.append(kwargs)
            stream.add_subscriber(cb)
            streams.append(stream)
            event_list.append(events)
        callback = callback_cls(plot, streams, None)

        plots.append(plot)
        streamss.append(streams)
        callbacks.append(callback)
        eventss.append(event_list)

    return plots, streamss, callbacks, eventss


class TestCallbacks(TestCase):

    def setUp(self):
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

        self.mapbox_fig_dict = go.Figure({
            'data': [
                {'type': 'scattermapbox', 'uid': 'first', 'subplot': 'mapbox'},
                {'type': 'scattermapbox', 'uid': 'second', 'subplot': 'mapbox2'},
                {'type': 'scattermapbox', 'uid': 'third', 'subplot': 'mapbox3'}
            ],
            'layout': {
                'title': {'text': 'Figure Title'},
            }
        }).to_dict()

        # Precompute a pair of lat/lon, easting/northing, mapbox coord values
        self.lon_range1, self.lat_range1 = (10, 30), (20, 40)
        self.easting_range1, self.northing_range1 = Tiles.lon_lat_to_easting_northing(
            self.lon_range1, self.lat_range1
        )
        self.easting_range1 = tuple(self.easting_range1)
        self.northing_range1 = tuple(self.northing_range1)

        self.mapbox_coords1 = [
            [self.lon_range1[0], self.lat_range1[1]],
            [self.lon_range1[1], self.lat_range1[1]],
            [self.lon_range1[1], self.lat_range1[0]],
            [self.lon_range1[0], self.lat_range1[0]]
        ]

        self.lon_range2, self.lat_range2 = (-50, -30), (-70, -40)
        self.easting_range2, self.northing_range2 = Tiles.lon_lat_to_easting_northing(
            self.lon_range2, self.lat_range2
        )
        self.easting_range2 = tuple(self.easting_range2)
        self.northing_range2 = tuple(self.northing_range2)

        self.mapbox_coords2 = [
            [self.lon_range2[0], self.lat_range2[1]],
            [self.lon_range2[1], self.lat_range2[1]],
            [self.lon_range2[1], self.lat_range2[0]],
            [self.lon_range2[0], self.lat_range2[0]]
        ]

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

    def testRangeXYCallbackEventData(self):
        for viewport in [
            {'xaxis.range': [1, 4], 'yaxis.range': [-1, 5]},
            {'xaxis.range[0]': 1, 'xaxis.range[1]': 4,
             'yaxis.range[0]': -1, 'yaxis.range[1]': 5},
        ]:
            event_data = RangeXYCallback.get_event_data_from_property_update(
                "viewport", viewport, self.fig_dict
            )

            self.assertEqual(event_data, {
                'first': {'x_range': (1, 4), 'y_range': (-1, 5)},
                'second': {'x_range': (1, 4), 'y_range': (-1, 5)},
            })

    def testRangeXCallbackEventData(self):
        for viewport in [
            {'xaxis.range': [1, 4], 'yaxis.range': [-1, 5]},
            {'xaxis.range[0]': 1, 'xaxis.range[1]': 4,
             'yaxis.range[0]': -1, 'yaxis.range[1]': 5},
        ]:
            event_data = RangeXCallback.get_event_data_from_property_update(
                "viewport", viewport, self.fig_dict
            )

            self.assertEqual(event_data, {
                'first': {'x_range': (1, 4)},
                'second': {'x_range': (1, 4)},
            })

    def testRangeYCallbackEventData(self):
        for viewport in [
            {'xaxis.range': [1, 4], 'yaxis.range': [-1, 5]},
            {'xaxis.range[0]': 1, 'xaxis.range[1]': 4,
             'yaxis.range[0]': -1, 'yaxis.range[1]': 5},
        ]:
            event_data = RangeYCallback.get_event_data_from_property_update(
                "viewport", viewport, self.fig_dict
            )

            self.assertEqual(event_data, {
                'first': {'y_range': (-1, 5)},
                'second': {'y_range': (-1, 5)},
            })

    def testMapboxRangeXYCallbackEventData(self):
        relayout_data = {
            'mapbox._derived': {"coordinates": self.mapbox_coords1},
            'mapbox3._derived': {"coordinates": self.mapbox_coords2}
        }

        event_data = RangeXYCallback.get_event_data_from_property_update(
            "relayout_data", relayout_data, self.mapbox_fig_dict
        )

        self.assertEqual(event_data, {
            'first': {'x_range': self.easting_range1, 'y_range': self.northing_range1},
            'third': {'x_range': self.easting_range2, 'y_range': self.northing_range2},
        })

    def testMapboxRangeXCallbackEventData(self):
        relayout_data = {
            'mapbox._derived': {"coordinates": self.mapbox_coords1},
            'mapbox3._derived': {"coordinates": self.mapbox_coords2}
        }

        event_data = RangeXCallback.get_event_data_from_property_update(
            "relayout_data", relayout_data, self.mapbox_fig_dict
        )

        self.assertEqual(event_data, {
            'first': {'x_range': self.easting_range1},
            'third': {'x_range': self.easting_range2},
        })

    def testMapboxRangeYCallbackEventData(self):
        relayout_data = {
            'mapbox._derived': {"coordinates": self.mapbox_coords1},
            'mapbox3._derived': {"coordinates": self.mapbox_coords2}
        }

        event_data = RangeYCallback.get_event_data_from_property_update(
            "relayout_data", relayout_data, self.mapbox_fig_dict
        )

        self.assertEqual(event_data, {
            'first': {'y_range': self.northing_range1},
            'third': {'y_range': self.northing_range2},
        })

    def testRangeCallbacks(self):

        # Build callbacks
        range_classes = [RangeXYCallback, RangeXCallback, RangeYCallback]

        xyplots, xystreamss, xycallbacks, xyevents = build_callback_set(
            RangeXYCallback, ['first', 'second', 'third', 'forth', 'other'],
            RangeXY, 2
        )

        xplots, xstreamss, xcallbacks, xevents = build_callback_set(
            RangeXCallback, ['first', 'second', 'third', 'forth', 'other'],
            RangeX, 2
        )

        yplots, ystreamss, ycallbacks, yevents = build_callback_set(
            RangeYCallback, ['first', 'second', 'third', 'forth', 'other'],
            RangeY, 2
        )

        # Sanity check the length of the streams lists
        for xystreams in xystreamss:
            self.assertEqual(len(xystreams), 2)

        # Change viewport on first set of axes
        viewport1 = {'xaxis.range': [1, 4], 'yaxis.range': [-1, 5]}
        for cb_cls in range_classes:
            cb_cls.update_streams_from_property_update(
                "viewport", viewport1, self.fig_dict
            )

        # Check that all streams attached to 'first' and 'second' plots were triggered
        for xystream, xstream, ystream in zip(
                xystreamss[0] + xystreamss[1],
                xstreamss[0] + xstreamss[1],
                ystreamss[0] + ystreamss[1],
        ):
            assert xystream.x_range == (1, 4)
            assert xystream.y_range == (-1, 5)
            assert xstream.x_range == (1, 4)
            assert ystream.y_range == (-1, 5)

        # And that no other streams were triggered
        for xystream, xstream, ystream in zip(
                xystreamss[2] + xystreamss[3],
                xstreamss[2] + xstreamss[3],
                ystreamss[2] + ystreamss[3],
        ):
            assert xystream.x_range is None
            assert xystream.y_range is None
            assert xstream.x_range is None
            assert ystream.y_range is None

        # Change viewport on second set of axes
        viewport2 = {'xaxis2.range': [2, 5], 'yaxis2.range': [0, 6]}
        for cb_cls in range_classes:
            cb_cls.update_streams_from_property_update(
                "viewport", viewport2, self.fig_dict
            )

        # Check that all streams attached to 'third' were triggered
        for xystream, xstream, ystream in zip(
                xystreamss[2], xstreamss[2], ystreamss[2]
        ):
            assert xystream.x_range == (2, 5)
            assert xystream.y_range == (0, 6)
            assert xstream.x_range == (2, 5)
            assert ystream.y_range == (0, 6)

        # Change viewport on third set of axes
        viewport3 = {'xaxis3.range': [3, 6], 'yaxis3.range': [1, 7]}
        for cb_cls in range_classes:
            cb_cls.update_streams_from_property_update(
                "viewport", viewport3, self.fig_dict
            )

        # Check that all streams attached to 'forth' were triggered
        for xystream, xstream, ystream in zip(
                xystreamss[3], xstreamss[3], ystreamss[3]
        ):
            assert xystream.x_range == (3, 6)
            assert xystream.y_range == (1, 7)
            assert xstream.x_range == (3, 6)
            assert ystream.y_range == (1, 7)

        # Check that streams attached to a trace not in this plot are not triggered
        for xyevent, xevent, yevent in zip(
                xyevents[4], xevents[4], yevents[4]
        ):
            assert len(xyevent) == 0
            assert len(yevent) == 0
            assert len(yevent) == 0

    def testBoundsXYCallbackEventData(self):
        selected_data1 = {'range': {'x': [1, 4], 'y': [-1, 5]}}
        event_data = BoundsXYCallback.get_event_data_from_property_update(
            "selected_data", selected_data1, self.fig_dict
        )

        self.assertEqual(event_data, {
            'first': {'bounds': (1, -1, 4, 5)},
            'second': {'bounds': (1, -1, 4, 5)},
            'third': {'bounds': None},
            'forth': {'bounds': None}
        })

    def testBoundsXCallbackEventData(self):
        selected_data1 = {'range': {'x': [1, 4], 'y': [-1, 5]}}
        event_data = BoundsXCallback.get_event_data_from_property_update(
            "selected_data", selected_data1, self.fig_dict
        )

        self.assertEqual(event_data, {
            'first': {'boundsx': (1, 4)},
            'second': {'boundsx': (1, 4)},
            'third': {'boundsx': None},
            'forth': {'boundsx': None}
        })

    def testBoundsYCallbackEventData(self):
        selected_data1 = {'range': {'x': [1, 4], 'y': [-1, 5]}}
        event_data = BoundsYCallback.get_event_data_from_property_update(
            "selected_data", selected_data1, self.fig_dict
        )

        self.assertEqual(event_data, {
            'first': {'boundsy': (-1, 5)},
            'second': {'boundsy': (-1, 5)},
            'third': {'boundsy': None},
            'forth': {'boundsy': None}
        })

    def testMapboxBoundsXYCallbackEventData(self):
        selected_data = {"range": {'mapbox2': [
            [self.lon_range1[0], self.lat_range1[0]],
            [self.lon_range1[1], self.lat_range1[1]]
        ]}}

        event_data = BoundsXYCallback.get_event_data_from_property_update(
            "selected_data", selected_data, self.mapbox_fig_dict
        )

        self.assertEqual(event_data, {
            'first': {'bounds': None},
            'second': {'bounds': (
                self.easting_range1[0], self.northing_range1[0],
                self.easting_range1[1], self.northing_range1[1]
            )},
            'third': {'bounds': None}
        })

    def testMapboxBoundsXCallbackEventData(self):
        selected_data = {"range": {'mapbox': [
            [self.lon_range1[0], self.lat_range1[0]],
            [self.lon_range1[1], self.lat_range1[1]]
        ]}}

        event_data = BoundsXCallback.get_event_data_from_property_update(
            "selected_data", selected_data, self.mapbox_fig_dict
        )

        self.assertEqual(event_data, {
            'first': {'boundsx': (
                self.easting_range1[0], self.easting_range1[1],
            )},
            'second': {'boundsx': None},
            'third': {'boundsx': None}
        })

    def testMapboxBoundsYCallbackEventData(self):
        selected_data = {"range": {'mapbox3': [
            [self.lon_range1[0], self.lat_range1[0]],
            [self.lon_range1[1], self.lat_range1[1]]
        ]}}

        event_data = BoundsYCallback.get_event_data_from_property_update(
            "selected_data", selected_data, self.mapbox_fig_dict
        )

        self.assertEqual(event_data, {
            'first': {'boundsy': None},
            'second': {'boundsy': None},
            'third': {'boundsy': (
               self.northing_range1[0], self.northing_range1[1]
            )},
        })

    def testBoundsCallbacks(self):

        # Build callbacks
        bounds_classes = [BoundsXYCallback, BoundsXCallback, BoundsYCallback]

        xyplots, xystreamss, xycallbacks, xyevents = build_callback_set(
            BoundsXYCallback, ['first', 'second', 'third', 'forth', 'other'],
            BoundsXY, 2
        )

        xplots, xstreamss, xcallbacks, xevents = build_callback_set(
            BoundsXCallback, ['first', 'second', 'third', 'forth', 'other'],
            BoundsX, 2
        )

        yplots, ystreamss, ycallbacks, yevents = build_callback_set(
            BoundsYCallback, ['first', 'second', 'third', 'forth', 'other'],
            BoundsY, 2
        )

        # box selection on first set of axes
        selected_data1 = {'range': {'x': [1, 4], 'y': [-1, 5]}}
        for cb_cls in bounds_classes:
            cb_cls.update_streams_from_property_update(
                "selected_data", selected_data1, self.fig_dict
            )

        # Check that all streams attached to 'first' and 'second' plots were triggered
        for xystream, xstream, ystream in zip(
                xystreamss[0] + xystreamss[1],
                xstreamss[0] + xstreamss[1],
                ystreamss[0] + ystreamss[1],
        ):
            assert xystream.bounds == (1, -1, 4, 5)
            assert xstream.boundsx == (1, 4)
            assert ystream.boundsy == (-1, 5)

        # Check that streams attached to plots in other subplots are called with None
        # to clear their bounds
        for xystream, xstream, ystream in zip(
                xystreamss[2] + xystreamss[3],
                xstreamss[2] + xstreamss[3],
                ystreamss[2] + ystreamss[3],
        ):
            assert xystream.bounds is None
            assert xstream.boundsx is None
            assert ystream.boundsy is None

        # box select on second set of axes
        selected_data2 = {'range': {'x2': [2, 5], 'y2': [0, 6]}}
        for cb_cls in bounds_classes:
            cb_cls.update_streams_from_property_update(
                "selected_data", selected_data2, self.fig_dict
            )

        # Check that all streams attached to 'second' were triggered
        for xystream, xstream, ystream in zip(
                xystreamss[2], xstreamss[2], ystreamss[2],
        ):
            assert xystream.bounds == (2, 0, 5, 6)
            assert xstream.boundsx == (2, 5)
            assert ystream.boundsy == (0, 6)

        # box select on third set of axes
        selected_data3 = {'range': {'x3': [3, 6], 'y3': [1, 7]}}
        for cb_cls in bounds_classes:
            cb_cls.update_streams_from_property_update(
                "selected_data", selected_data3, self.fig_dict
            )

        # Check that all streams attached to 'third' were triggered
        for xystream, xstream, ystream in zip(
                xystreamss[3], xstreamss[3], ystreamss[3],
        ):
            assert xystream.bounds == (3, 1, 6, 7)
            assert xstream.boundsx == (3, 6)
            assert ystream.boundsy == (1, 7)

        # lasso select on first set of axes should clear all bounds
        selected_data_lasso = {'lassoPoints': {'x': [1, 4, 2], 'y': [-1, 5, 2]}}
        for cb_cls in bounds_classes:
            cb_cls.update_streams_from_property_update(
                "selected_data", selected_data_lasso, self.fig_dict
            )

        # Check that all streams attached to this figure are called with None
        # to clear their bounds
        for xystream, xstream, ystream in zip(
                xystreamss[0] + xystreamss[1] + xystreamss[2] + xystreamss[3],
                xstreamss[0] + xstreamss[1] + xstreamss[2] + xstreamss[3],
                ystreamss[0] + ystreamss[1] + ystreamss[2] + ystreamss[3],
        ):
            assert xystream.bounds is None
            assert xstream.boundsx is None
            assert ystream.boundsy is None

        # Check that streams attached to plots not in this figure are not called
        for xyevent, xevent, yevent in zip(
                xyevents[4], xevents[4], yevents[4]
        ):
            assert xyevent == []
            assert xevent == []
            assert yevent == []

    def testSelection1DCallbackEventData(self):
        selected_data1 = {'points': [
            {"pointNumber": 0, "curveNumber": 0},
            {"pointNumber": 2, "curveNumber": 0},
        ]}

        event_data = Selection1DCallback.get_event_data_from_property_update(
            "selected_data", selected_data1, self.fig_dict
        )

        self.assertEqual(event_data, {
            'first': {'index': [0, 2]},
            'second': {'index': []},
            'third': {'index': []},
            'forth': {'index': []}
        })

    def testMapboxSelection1DCallbackEventData(self):
        selected_data1 = {'points': [
            {"pointNumber": 0, "curveNumber": 1},
            {"pointNumber": 2, "curveNumber": 1},
        ]}

        event_data = Selection1DCallback.get_event_data_from_property_update(
            "selected_data", selected_data1, self.mapbox_fig_dict
        )

        self.assertEqual(event_data, {
            'first': {'index': []},
            'second': {'index': [0, 2]},
            'third': {'index': []},
        })

    def testSelection1DCallback(self):
        plots, streamss, callbacks, sel_events = build_callback_set(
            Selection1DCallback, ['first', 'second', 'third', 'forth', 'other'],
            Selection1D, 2
        )

        # Select points from the 'first' plot (first set of axes)
        selected_data1 = {'points': [
            {"pointNumber": 0, "curveNumber": 0},
            {"pointNumber": 2, "curveNumber": 0},
        ]}
        Selection1DCallback.update_streams_from_property_update(
            "selected_data", selected_data1, self.fig_dict
        )

        # Check that all streams attached to the 'first' plots were triggered
        for stream, events in zip(streamss[0], sel_events[0]):
            assert stream.index == [0, 2]
            assert len(events) == 1

        # Check that all streams attached to other plots in this figure were triggered
        # with empty selection
        for stream in streamss[1] + streamss[2] + streamss[3]:
            assert stream.index == []

        # Select points from the 'first' and 'second' plot (first set of axes)
        selected_data1 = {'points': [
            {"pointNumber": 0, "curveNumber": 0},
            {"pointNumber": 1, "curveNumber": 0},
            {"pointNumber": 1, "curveNumber": 1},
            {"pointNumber": 2, "curveNumber": 1},
        ]}
        Selection1DCallback.update_streams_from_property_update(
            "selected_data", selected_data1, self.fig_dict
        )

        # Check that all streams attached to the 'first' plot were triggered
        for stream in streamss[0]:
            assert stream.index == [0, 1]

        # Check that all streams attached to the 'second' plot were triggered
        for stream in streamss[1]:
            assert stream.index == [1, 2]

        # Check that all streams attached to other plots in this figure were triggered
        # with empty selection
        for stream in streamss[2] + streamss[3]:
            assert stream.index == []

        # Select points from the 'forth' plot (third set of axes)
        selected_data1 = {'points': [
            {"pointNumber": 0, "curveNumber": 3},
            {"pointNumber": 2, "curveNumber": 3},
        ]}
        Selection1DCallback.update_streams_from_property_update(
            "selected_data", selected_data1, self.fig_dict
        )

        # Check that all streams attached to the 'forth' plot were triggered
        for stream, events in zip(streamss[3], sel_events[3]):
            assert stream.index == [0, 2]

        # Check that streams attached to plots not in this figure are not called
        for stream, events in zip(streamss[4], sel_events[4]):
            assert len(events) == 0
