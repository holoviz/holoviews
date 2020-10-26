from weakref import WeakValueDictionary

from param.parameterized import add_metaclass

from ...streams import (
    Stream, Selection1D, RangeXY, RangeX, RangeY, BoundsXY, BoundsX, BoundsY,
    SelectionXY
)

from .util import _trace_to_subplot


class PlotlyCallbackMetaClass(type):
    """
    Metaclass for PlotlyCallback classes.

    We want each callback class to keep track of all of the instances of the class.
    Using a meta class here lets us keep the logic for instance tracking in one place.
    """

    def __init__(cls, name, bases, attrs):
        super(PlotlyCallbackMetaClass, cls).__init__(name, bases, attrs)

        # Create weak-value dictionary to hold instances of the class
        cls.instances = WeakValueDictionary()

    def __call__(cls, *args, **kwargs):
        inst = super(PlotlyCallbackMetaClass, cls).__call__(*args, **kwargs)

        # Store weak reference to the callback instance in the _instances
        # WeakValueDictionary. This will allow instances to be garbage collected and
        # the references will be automatically removed from the colleciton when this
        # happens.
        cls.instances[inst.plot.trace_uid] = inst

        return inst


@add_metaclass(PlotlyCallbackMetaClass)
class PlotlyCallback(object):

    def __init__(self, plot, streams, source, **params):
        self.plot = plot
        self.streams = streams
        self.source = source
        self.last_event = None

    @classmethod
    def update_streams_from_property_update(cls, property_value, fig_dict):
        event_data = cls.get_event_data_from_property_update(property_value, fig_dict)
        streams = []
        for trace_uid, stream_data in event_data.items():
            if trace_uid in cls.instances:
                cb = cls.instances[trace_uid]
                try:
                    unchanged = stream_data == cb.last_event
                except Exception:
                    unchanged = False
                if unchanged:
                    continue
                cb.last_event = stream_data
                for stream in cb.streams:
                    stream.update(**stream_data)
                    streams.append(stream)

        try:
            Stream.trigger(streams)
        except Exception as e:
            raise e

    @classmethod
    def get_event_data_from_property_update(cls, property_value, fig_dict):
        raise NotImplementedError


class Selection1DCallback(PlotlyCallback):
    callback_property = "selected_data"

    @classmethod
    def get_event_data_from_property_update(cls, selected_data, fig_dict):

        traces = fig_dict.get('data', [])

        # build event data and compute which trace UIDs are eligible
        # Look up callback with UID
        # graph reference and update the streams
        point_inds = {}
        if selected_data:
            for point in selected_data['points']:
                point_inds.setdefault(point['curveNumber'], [])
                point_inds[point['curveNumber']].append(point['pointNumber'])

        event_data = {}
        for trace_ind, trace in enumerate(traces):
            trace_uid = trace.get('uid', None)
            new_index = point_inds.get(trace_ind, [])
            event_data[trace_uid] = dict(index=new_index)

        return event_data


class BoundsCallback(PlotlyCallback):
    callback_property = "selected_data"
    boundsx = False
    boundsy = False

    @classmethod
    def get_event_data_from_property_update(cls, selected_data, fig_dict):
        traces = fig_dict.get('data', [])

        if not selected_data or 'range' not in selected_data:
            # No valid box selection
            box = None
        else:
            # Get x and y axis references
            box = selected_data["range"]
            axis_refs = list(box)
            xref = [ref for ref in axis_refs if ref.startswith('x')][0]
            yref = [ref for ref in axis_refs if ref.startswith('y')][0]

        # Process traces
        event_data = {}
        for trace_ind, trace in enumerate(traces):
            trace_type = trace.get('type', 'scatter')
            trace_uid = trace.get('uid', None)

            if _trace_to_subplot.get(trace_type, None) != ['xaxis', 'yaxis']:
                continue

            if (box and trace.get('xaxis', 'x') == xref and
                    trace.get('yaxis', 'y') == yref):

                new_bounds = (box[xref][0], box[yref][0], box[xref][1], box[yref][1])

                if cls.boundsx and cls.boundsy:
                    stream_data = dict(bounds=new_bounds)
                elif cls.boundsx:
                    stream_data = dict(boundsx=(new_bounds[0], new_bounds[2]))
                elif cls.boundsy:
                    stream_data = dict(boundsy=(new_bounds[1], new_bounds[3]))
                else:
                    stream_data = dict()

                event_data[trace_uid] = stream_data
            else:
                if cls.boundsx and cls.boundsy:
                    stream_data = dict(bounds=None)
                elif cls.boundsx:
                    stream_data = dict(boundsx=None)
                elif cls.boundsy:
                    stream_data = dict(boundsy=None)
                else:
                    stream_data = dict()

                event_data[trace_uid] = stream_data

        return event_data


class BoundsXYCallback(BoundsCallback):
    boundsx = True
    boundsy = True


class BoundsXCallback(BoundsCallback):
    boundsx = True


class BoundsYCallback(BoundsCallback):
    boundsy = True


class RangeCallback(PlotlyCallback):
    callback_property = "viewport"
    x_range = False
    y_range = False

    @classmethod
    def get_event_data_from_property_update(cls, viewport, fig_dict):

        traces = fig_dict.get('data', [])

        # Process traces
        event_data = {}
        for trace_ind, trace in enumerate(traces):
            trace_type = trace.get('type', 'scatter')
            trace_uid = trace.get('uid', None)

            if _trace_to_subplot.get(trace_type, None) != ['xaxis', 'yaxis']:
                continue

            xaxis = trace.get('xaxis', 'x').replace('x', 'xaxis')
            yaxis = trace.get('yaxis', 'y').replace('y', 'yaxis')
            xprop = '{xaxis}.range'.format(xaxis=xaxis)
            yprop = '{yaxis}.range'.format(yaxis=yaxis)

            if not viewport:
                x_range = None
                y_range = None
            elif xprop in viewport and yprop in viewport:
                x_range = tuple(viewport[xprop])
                y_range = tuple(viewport[yprop])
            elif xprop + "[0]" in viewport and xprop + "[1]" in viewport and \
                    yprop + "[0]" in viewport and yprop + "[1]" in viewport :
                x_range = (viewport[xprop + "[0]"], viewport[xprop + "[1]"])
                y_range = (viewport[yprop + "[0]"], viewport[yprop + "[1]"])
            else:
                x_range = None
                y_range = None

            stream_data = {}
            if cls.x_range:
                stream_data['x_range'] = x_range

            if cls.y_range:
                stream_data['y_range'] = y_range

            event_data[trace_uid] = stream_data

        return event_data


class RangeXYCallback(RangeCallback):
    x_range = True
    y_range = True


class RangeXCallback(RangeCallback):
    x_range = True


class RangeYCallback(RangeCallback):
    y_range = True


callbacks = Stream._callbacks['plotly']
callbacks[Selection1D] = Selection1DCallback
callbacks[SelectionXY] = BoundsXYCallback
callbacks[BoundsXY] = BoundsXYCallback
callbacks[BoundsX] = BoundsXCallback
callbacks[BoundsY] = BoundsYCallback
callbacks[RangeXY] = RangeXYCallback
callbacks[RangeX] = RangeXCallback
callbacks[RangeY] = RangeYCallback

