from __future__ import absolute_import, division, unicode_literals

from collections import defaultdict

import param
import numpy as np
from bokeh.models import (
    CustomJS, FactorRange, DatetimeAxis, ColumnDataSource, ToolbarBox,
    Range1d, DataRange1d
)
from pyviz_comms import JS_CALLBACK

from ...core import OrderedDict
from ...core.util import dimension_sanitizer, isscalar, dt64_to_dt
from ...streams import (Stream, PointerXY, RangeXY, Selection1D, RangeX,
                        RangeY, PointerX, PointerY, BoundsX, BoundsY,
                        Tap, SingleTap, DoubleTap, MouseEnter, MouseLeave,
                        PlotSize, Draw, BoundsXY, PlotReset, BoxEdit,
                        PointDraw, PolyDraw, PolyEdit, CDSStream, FreehandDraw)
from ..links import Link, RangeToolLink, DataLink
from ..plot import GenericElementPlot, GenericOverlayPlot
from .util import convert_timestamp, bokeh_version


class MessageCallback(object):
    """
    A MessageCallback is an abstract baseclass used to supply Streams
    with events originating from bokeh plot interactions. The baseclass
    defines how messages are handled and the basic specification required
    to define a Callback.
    """

    attributes = {}

    # The plotting handle(s) to attach the JS callback on
    models = []

    # Additional models available to the callback
    extra_models = []

    # Conditions when callback should be skipped
    skip = []

    # Callback will listen to events of the supplied type on the models
    on_events = []

    # List of change events on the models to listen to
    on_changes = []

    _callbacks = {}

    def _process_msg(self, msg):
        """
        Subclassable method to preprocess JSON message in callback
        before passing to stream.
        """
        return msg


    def __init__(self, plot, streams, source, **params):
        self.plot = plot
        self.streams = streams
        if plot.renderer.mode != 'server':
            self.comm = plot.renderer.comm_manager.get_client_comm(on_msg=self.on_msg)
        else:
            self.comm = None
        self.source = source
        self.handle_ids = defaultdict(dict)
        self.reset()


    def cleanup(self):
        self.reset()
        self.handle_ids = None
        self.plot = None
        self.source = None
        self.streams = []
        if self.comm:
            try:
                self.comm.close()
            except:
                pass
        Callback._callbacks = {k: cb for k, cb in Callback._callbacks.items()
                               if cb is not self}


    def reset(self):
        if self.handle_ids:
            handles = self._init_plot_handles()
            for handle_name in self.models:
                handle = handles[handle_name]
                cb_hash = (id(handle), id(type(self)))
                self._callbacks.pop(cb_hash, None)
        self.callbacks = []
        self.plot_handles = {}
        self._queue = []


    def _filter_msg(self, msg, ids):
        """
        Filter event values that do not originate from the plotting
        handles associated with a particular stream using their
        ids to match them.
        """
        filtered_msg = {}
        for k, v in msg.items():
            if isinstance(v, dict) and 'id' in v:
                if v['id'] in ids:
                    filtered_msg[k] = v['value']
            else:
                filtered_msg[k] = v
        return filtered_msg


    def on_msg(self, msg):
        streams = []
        for stream in self.streams:
            handle_ids = self.handle_ids[stream]
            ids = list(handle_ids.values())
            filtered_msg = self._filter_msg(msg, ids)
            processed_msg = self._process_msg(filtered_msg)
            if not processed_msg:
                continue
            stream.update(**processed_msg)
            stream._metadata = {h: {'id': hid, 'events': self.on_events}
                                for h, hid in handle_ids.items()}
            streams.append(stream)

        try:
            Stream.trigger(streams)
        except Exception as e:
            raise e
        finally:
            for stream in streams:
                stream._metadata = {}


    def _init_plot_handles(self):
        """
        Find all requested plotting handles and cache them along
        with the IDs of the models the callbacks will be attached to.
        """
        plots = [self.plot]
        if self.plot.subplots:
            plots += list(self.plot.subplots.values())

        handles = {}
        for plot in plots:
            for k, v in plot.handles.items():
                handles[k] = v
        self.plot_handles = handles

        requested = {}
        for h in self.models+self.extra_models:
            if h in self.plot_handles:
                requested[h] = handles[h]
            elif h in self.extra_models:
                print("Warning %s could not find the %s model. "
                      "The corresponding stream may not work."
                      % (type(self).__name__, h))
        self.handle_ids.update(self._get_stream_handle_ids(requested))

        return requested


    def _get_stream_handle_ids(self, handles):
        """
        Gather the ids of the plotting handles attached to this callback
        This allows checking that a stream is not given the state
        of a plotting handle it wasn't attached to
        """
        stream_handle_ids = defaultdict(dict)
        for stream in self.streams:
            for h in self.models:
                if h in handles:
                    handle_id = handles[h].ref['id']
                    stream_handle_ids[stream][h] = handle_id
        return stream_handle_ids



class CustomJSCallback(MessageCallback):
    """
    The CustomJSCallback attaches CustomJS callbacks to a bokeh plot,
    which looks up the requested attributes and sends back a message
    to Python using a Comms instance.
    """

    js_callback = JS_CALLBACK

    code = ""

    # Timeout if a comm message is swallowed
    timeout = 20000

    # Timeout before the first event is processed
    debounce = 20

    @classmethod
    def attributes_js(cls, attributes):
        """
        Generates JS code to look up attributes on JS objects from
        an attributes specification dictionary. If the specification
        references a plotting particular plotting handle it will also
        generate JS code to get the ID of the object.

        Simple example (when referencing cb_data or cb_obj):

        Input  : {'x': 'cb_data.geometry.x'}

        Output : data['x'] = cb_data['geometry']['x']

        Example referencing plot handle:

        Input  : {'x0': 'x_range.attributes.start'}

        Output : if ((x_range !== undefined)) {
                    data['x0'] = {id: x_range['id'], value: x_range['attributes']['start']}
                 }
        """
        assign_template = '{assign}{{id: {obj_name}["id"], value: {obj_name}{attr_getters}}};\n'
        conditional_template = 'if (({obj_name} != undefined)) {{ {assign} }}'
        code = ''
        for key, attr_path in sorted(attributes.items()):
            data_assign = 'data["{key}"] = '.format(key=key)
            attrs = attr_path.split('.')
            obj_name = attrs[0]
            attr_getters = ''.join(['["{attr}"]'.format(attr=attr)
                                    for attr in attrs[1:]])
            if obj_name not in ['cb_obj', 'cb_data']:
                assign_str = assign_template.format(
                    assign=data_assign, obj_name=obj_name, attr_getters=attr_getters
                )
                code += conditional_template.format(
                    obj_name=obj_name, assign=assign_str
                )
            else:
                assign_str = ''.join([data_assign, obj_name, attr_getters, ';\n'])
                code += assign_str
        return code


    def get_customjs(self, references, plot_id=None):
        """
        Creates a CustomJS callback that will send the requested
        attributes back to python.
        """
        # Generate callback JS code to get all the requested data
        if plot_id is None:
            plot_id = self.plot.id or 'PLACEHOLDER_PLOT_ID'
        self_callback = self.js_callback.format(comm_id=self.comm.id,
                                                timeout=self.timeout,
                                                debounce=self.debounce,
                                                plot_id=plot_id)

        attributes = self.attributes_js(self.attributes)
        conditions = ["%s" % cond for cond in self.skip]
        conditional = ''
        if conditions:
            conditional = 'if (%s) { return };\n' % (' || '.join(conditions))
        data = "var data = {};\n"
        code = conditional + data + attributes + self.code + self_callback
        return CustomJS(args=references, code=code)


    def set_customjs_callback(self, js_callback, handle):
        """
        Generates a CustomJS callback by generating the required JS
        code and gathering all plotting handles and installs it on
        the requested callback handle.
        """
        if self.on_events:
            for event in self.on_events:
                handle.js_on_event(event, js_callback)
        elif self.on_changes:
            for change in self.on_changes:
                handle.js_on_change(change, js_callback)
        elif hasattr(handle, 'callback'):
            handle.callback = js_callback



class ServerCallback(MessageCallback):
    """
    Implements methods to set up bokeh server callbacks. A ServerCallback
    resolves the requested attributes on the Python end and then hands
    the msg off to the general on_msg handler, which will update the
    Stream(s) attached to the callback.
    """

    def __init__(self, plot, streams, source, **params):
        super(ServerCallback, self).__init__(plot, streams, source, **params)
        self._active = False


    @classmethod
    def resolve_attr_spec(cls, spec, cb_obj, model=None):
        """
        Resolves a Callback attribute specification looking the
        corresponding attribute up on the cb_obj, which should be a
        bokeh model. If not model is supplied cb_obj is assumed to
        be the same as the model.
        """
        if not cb_obj:
            raise Exception('Bokeh plot attribute %s could not be found' % spec)
        if model is None:
            model = cb_obj
        spec = spec.split('.')
        resolved = cb_obj
        for p in spec[1:]:
            if p == 'attributes':
                continue
            if isinstance(resolved, dict):
                resolved = resolved.get(p)
            else:
                resolved = getattr(resolved, p, None)
        return {'id': model.ref['id'], 'value': resolved}


    def on_change(self, attr, old, new):
        """
        Process change events adding timeout to process multiple concerted
        value change at once rather than firing off multiple plot updates.
        """
        self._queue.append((attr, old, new))
        if not self._active and self.plot.document:
            self.plot.document.add_timeout_callback(self.process_on_change, 50)
            self._active = True


    def on_event(self, event):
        """
        Process bokeh UIEvents adding timeout to process multiple concerted
        value change at once rather than firing off multiple plot updates.
        """
        self._queue.append((event))
        if not self._active and self.plot.document:
            self.plot.document.add_timeout_callback(self.process_on_event, 50)
            self._active = True


    def process_on_event(self):
        """
        Trigger callback change event and triggering corresponding streams.
        """
        if not self._queue:
            self._active = False
            return
        # Get unique event types in the queue
        events = list(OrderedDict([(event.event_name, event)
                                   for event in self._queue]).values())
        self._queue = []

        # Process event types
        for event in events:
            msg = {}
            for attr, path in self.attributes.items():
                model_obj = self.plot_handles.get(self.models[0])
                msg[attr] = self.resolve_attr_spec(path, event, model_obj)
            self.on_msg(msg)
        self.plot.document.add_timeout_callback(self.process_on_event, 50)


    def process_on_change(self):
        if not self._queue:
            self._active = False
            return
        self._queue = []

        msg = {}
        for attr, path in self.attributes.items():
            attr_path = path.split('.')
            if attr_path[0] == 'cb_obj':
                obj_handle = self.models[0]
                path = '.'.join(self.models[:1]+attr_path[1:])
            else:
                obj_handle = attr_path[0]
            cb_obj = self.plot_handles.get(obj_handle)
            msg[attr] = self.resolve_attr_spec(path, cb_obj)

        self.on_msg(msg)
        self.plot.document.add_timeout_callback(self.process_on_change, 50)


    def set_server_callback(self, handle):
        """
        Set up on_change events for bokeh server interactions.
        """
        if self.on_events:
            for event in self.on_events:
                handle.on_event(event, self.on_event)
        elif self.on_changes:
            for change in self.on_changes:
                handle.on_change(change, self.on_change)



class Callback(CustomJSCallback, ServerCallback):
    """
    Provides a baseclass to define callbacks, which return data from
    bokeh model callbacks, events and attribute changes. The callback
    then makes this data available to any streams attached to it.

    The definition of a callback consists of a number of components:

    * models      : Defines which bokeh models the callback will be
                    attached on referencing the model by its key in
                    the plots handles, e.g. this could be the x_range,
                    y_range, plot, a plotting tool or any other
                    bokeh mode.

    * extra_models: Any additional models available in handles which
                    should be made available in the namespace of the
                    objects, e.g. to make a tool available to skip
                    checks.

    * attributes  : The attributes define which attributes to send
                    back to Python. They are defined as a dictionary
                    mapping between the name under which the variable
                    is made available to Python and the specification
                    of the attribute. The specification should start
                    with the variable name that is to be accessed and
                    the location of the attribute separated by
                    periods.  All models defined by the models and
                    extra_models attributes can be addressed in this
                    way, e.g. to get the start of the x_range as 'x'
                    you can supply {'x': 'x_range.attributes.start'}.
                    Additionally certain handles additionally make the
                    cb_data and cb_obj variables available containing
                    additional information about the event.

    * skip        : Conditions when the Callback should be skipped
                    specified as a list of valid JS expressions, which
                    can reference models requested by the callback,
                    e.g. ['pan.attributes.active'] would skip the
                    callback if the pan tool is active.

    * code        : Defines any additional JS code to be executed,
                    which can modify the data object that is sent to
                    the backend.

    * on_events   : If the Callback should listen to bokeh events this
                    should declare the types of event as a list (optional)

    * on_changes  : If the Callback should listen to model attribute
                    changes on the defined ``models`` (optional)

    If either on_events or on_changes are declared the Callback will
    be registered using the on_event or on_change machinery, otherwise
    it will be treated as a regular callback on the model.  The
    callback can also define a _process_msg method, which can modify
    the data sent by the callback before it is passed to the streams.
    """

    def initialize(self, plot_id=None):
        handles = self._init_plot_handles()
        for handle_name in self.models:
            if handle_name not in handles:
                warn_args = (handle_name, type(self.plot).__name__,
                             type(self).__name__)
                print('%s handle not found on %s, cannot '
                      'attach %s callback' % warn_args)
                continue
            handle = handles[handle_name]

            # Hash the plot handle with Callback type allowing multiple
            # callbacks on one handle to be merged
            cb_hash = (id(handle), id(type(self)))
            if cb_hash in self._callbacks:
                # Merge callbacks if another callback has already been attached
                cb = self._callbacks[cb_hash]
                cb.streams = list(set(cb.streams+self.streams))
                for k, v in self.handle_ids.items():
                    cb.handle_ids[k].update(v)
                continue

            if self.plot.renderer.mode == 'server':
                self.set_server_callback(handle)
            else:
                js_callback = self.get_customjs(handles, plot_id=plot_id)
                self.set_customjs_callback(js_callback, handle)
                self.callbacks.append(js_callback)
            self._callbacks[cb_hash] = self



class PointerXYCallback(Callback):
    """
    Returns the mouse x/y-position on mousemove event.
    """

    attributes = {'x': 'cb_obj.x', 'y': 'cb_obj.y'}
    models = ['plot']
    extra_models= ['x_range', 'y_range']

    on_events = ['mousemove']

    # Clip x and y values to available axis range
    code = """
    if (x_range.type.endsWith('Range1d')) {
      xstart = x_range.start;
      xend = x_range.end;
      if (xstart > xend) {
        [xstart, xend] = [xend, xstart]
      }
      if (cb_obj.x < xstart) {
        data['x'] = xstart;
      } else if (cb_obj.x > xend) {
        data['x'] = xend;
      }
    }
    if (y_range.type.endsWith('Range1d')) {
      ystart = y_range.start;
      yend = y_range.end;
      if (ystart > yend) {
        [ystart, yend] = [yend, ystart]
      }
      if (cb_obj.y < ystart) {
        data['y'] = ystart;
      } else if (cb_obj.y > yend) {
        data['y'] = yend;
      }
    }
    """

    def _process_out_of_bounds(self, value, start, end):
        "Clips out of bounds values"
        if isinstance(value, np.datetime64):
            v = dt64_to_dt(value)
            if isinstance(start, (int, float)):
                start = convert_timestamp(start)
            if isinstance(end, (int, float)):
                end = convert_timestamp(end)
            s, e = start, end
            if isinstance(s, np.datetime64):
                s = dt64_to_dt(s)
            if isinstance(e, np.datetime64):
                e = dt64_to_dt(e)
        else:
            v, s, e = value, start, end

        if v < s:
            value = start
        elif v > e:
            value = end

        return value

    def _process_msg(self, msg):
        x_range = self.plot.handles.get('x_range')
        y_range = self.plot.handles.get('y_range')
        xaxis = self.plot.handles.get('xaxis')
        yaxis = self.plot.handles.get('yaxis')

        if 'x' in msg and isinstance(xaxis, DatetimeAxis):
            msg['x'] = convert_timestamp(msg['x'])
        if 'y' in msg and isinstance(yaxis, DatetimeAxis):
            msg['y'] = convert_timestamp(msg['y'])

        server_mode = self.plot.renderer.mode == 'server'
        if isinstance(x_range, FactorRange) and isinstance(msg.get('x'), (int, float)):
            msg['x'] = x_range.factors[int(msg['x'])]
        elif 'x' in msg and isinstance(x_range, (Range1d, DataRange1d)) and server_mode:
            xstart, xend = x_range.start, x_range.end
            if xstart > xend:
                xstart, xend = xend, xstart
            x = self._process_out_of_bounds(msg['x'], xstart, xend)
            if x is None:
                msg = {}
            else:
                msg['x'] = x

        if isinstance(y_range, FactorRange) and isinstance(msg.get('y'), (int, float)):
            msg['y'] = y_range.factors[int(msg['y'])]
        elif 'y' in msg and isinstance(y_range, (Range1d, DataRange1d)) and server_mode:
            ystart, yend = y_range.start, y_range.end
            if ystart > yend:
                ystart, yend = yend, ystart
            y = self._process_out_of_bounds(msg['y'], ystart, yend)
            if y is None:
                msg = {}
            else:
                msg['y'] = y

        return msg


class PointerXCallback(PointerXYCallback):
    """
    Returns the mouse x-position on mousemove event.
    """

    attributes = {'x': 'cb_obj.x'}
    extra_models= ['x_range']
    code = """
    if (x_range.type.endsWith('Range1d')) {
      xstart = x_range.start;
      xend = x_range.end;
      if (xstart > xend) {
        [xstart, xend] = [xend, xstart]
      }
      if (cb_obj.x < xstart) {
        data['x'] = xstart;
      } else if (cb_obj.x > xend) {
        data['x'] = xend;
      }
    }
    """

class PointerYCallback(PointerXYCallback):
    """
    Returns the mouse x/y-position on mousemove event.
    """

    attributes = {'y': 'cb_obj.y'}
    extra_models= ['y_range']
    code = """
    if (y_range.type.endsWith('Range1d')) {
      ystart = y_range.start;
      yend = y_range.end;
      if (ystart > yend) {
        [ystart, yend] = [yend, ystart]
      }
      if (cb_obj.y < ystart) {
        data['y'] = ystart;
      } else if (cb_obj.y > yend) {
        data['y'] = yend;
      }
    }
    """

class DrawCallback(PointerXYCallback):
    on_events = ['pan', 'panstart', 'panend']
    models = ['plot']
    extra_models=['pan', 'box_zoom', 'x_range', 'y_range']
    skip = ['pan && pan.attributes.active', 'box_zoom && box_zoom.attributes.active']
    attributes = {'x': 'cb_obj.x', 'y': 'cb_obj.y', 'event': 'cb_obj.event_name'}

    def __init__(self, *args, **kwargs):
        self.stroke_count = 0
        super(DrawCallback, self).__init__(*args, **kwargs)

    def _process_msg(self, msg):
        event = msg.pop('event')
        if event == 'panend':
            self.stroke_count += 1
        return dict(msg, stroke_count=self.stroke_count)


class TapCallback(PointerXYCallback):
    """
    Returns the mouse x/y-position on tap event.

    Note: As of bokeh 0.12.5, there is no way to distinguish the
    individual tap events within a doubletap event.
    """

    # Skip if tap is outside axis range
    code = """
    if (x_range.type.endsWith('Range1d')) {
      xstart = x_range.start;
      xend = x_range.end;
      if (xstart > xend) {
        [xstart, xend] = [xend, xstart]
      }
      if ((cb_obj.x < xstart) || (cb_obj.x > xend)) {
        return
      }
    }
    if (y_range.type.endsWith('Range1d')) {
      ystart = y_range.start;
      yend = y_range.end;
      if (ystart > yend) {
        [ystart, yend] = [yend, ystart]
      }
      if ((cb_obj.y < ystart) || (cb_obj.y > yend)) {
        return
      }
    }
    """

    on_events = ['tap', 'doubletap']

    def _process_out_of_bounds(self, value, start, end):
        "Sets out of bounds values to None"
        if isinstance(value, np.datetime64):
            v = dt64_to_dt(value)
            if isinstance(start, (int, float)):
                start = convert_timestamp(start)
            if isinstance(end, (int, float)):
                end = convert_timestamp(end)
            s, e = start, end
            if isinstance(s, np.datetime64):
                s = dt64_to_dt(s)
            if isinstance(e, np.datetime64):
                e = dt64_to_dt(e)
        else:
            v, s, e = value, start, end

        if v < s or v > e:
            value = None
        return value


class SingleTapCallback(TapCallback):
    """
    Returns the mouse x/y-position on tap event.
    """

    on_events = ['tap']


class DoubleTapCallback(TapCallback):
    """
    Returns the mouse x/y-position on doubletap event.
    """

    on_events = ['doubletap']


class MouseEnterCallback(PointerXYCallback):
    """
    Returns the mouse x/y-position on mouseenter event, i.e. when
    mouse enters the plot canvas.
    """

    on_events = ['mouseenter']


class MouseLeaveCallback(PointerXYCallback):
    """
    Returns the mouse x/y-position on mouseleave event, i.e. when
    mouse leaves the plot canvas.
    """

    on_events = ['mouseleave']


class RangeXYCallback(Callback):
    """
    Returns the x/y-axis ranges of a plot.
    """

    attributes = {'x0': 'x_range.attributes.start',
                  'x1': 'x_range.attributes.end',
                  'y0': 'y_range.attributes.start',
                  'y1': 'y_range.attributes.end'}
    models = ['x_range', 'y_range']
    on_changes = ['start', 'end']

    def _process_msg(self, msg):
        data = {}
        if 'x0' in msg and 'x1' in msg:
            x0, x1 = msg['x0'], msg['x1']
            if isinstance(self.plot.handles.get('xaxis'), DatetimeAxis):
                x0 = convert_timestamp(x0)
                x1 = convert_timestamp(x1)
            if self.plot.invert_xaxis:
                x0, x1 = x1, x0
            data['x_range'] = (x0, x1)
        if 'y0' in msg and 'y1' in msg:
            y0, y1 = msg['y0'], msg['y1']
            if isinstance(self.plot.handles.get('yaxis'), DatetimeAxis):
                y0 = convert_timestamp(y0)
                y1 = convert_timestamp(y1)
            if self.plot.invert_yaxis:
                y0, y1 = y1, y0
            data['y_range'] = (y0, y1)
        return data


class RangeXCallback(RangeXYCallback):
    """
    Returns the x-axis range of a plot.
    """

    attributes = {'x0': 'x_range.attributes.start',
                  'x1': 'x_range.attributes.end'}
    models = ['x_range']


class RangeYCallback(RangeXYCallback):
    """
    Returns the y-axis range of a plot.
    """

    attributes = {'y0': 'y_range.attributes.start',
                  'y1': 'y_range.attributes.end'}
    models = ['y_range']



class PlotSizeCallback(Callback):
    """
    Returns the actual width and height of a plot once the layout
    solver has executed.
    """

    models = ['plot']
    attributes = {'width': 'cb_obj.inner_width',
                  'height': 'cb_obj.inner_height'}
    on_changes = ['inner_width', 'inner_height']

    def _process_msg(self, msg):
        if msg.get('width') and msg.get('height'):
            return msg
        else:
            return {}


class BoundsCallback(Callback):
    """
    Returns the bounds of a box_select tool.
    """
    attributes = {'x0': 'cb_obj.geometry.x0',
                  'x1': 'cb_obj.geometry.x1',
                  'y0': 'cb_obj.geometry.y0',
                  'y1': 'cb_obj.geometry.y1'}
    models = ['plot']
    extra_models = ['box_select']
    on_events = ['selectiongeometry']
    skip = ["cb_obj.geometry.type != 'rect'"]

    def _process_msg(self, msg):
        if all(c in msg for c in ['x0', 'y0', 'x1', 'y1']):
            if isinstance(self.plot.handles.get('xaxis'), DatetimeAxis):
                msg['x0'] = convert_timestamp(msg['x0'])
                msg['x1'] = convert_timestamp(msg['x1'])
            if isinstance(self.plot.handles.get('yaxis'), DatetimeAxis):
                msg['y0'] = convert_timestamp(msg['y0'])
                msg['y1'] = convert_timestamp(msg['y1'])
            return {'bounds': (msg['x0'], msg['y0'], msg['x1'], msg['y1'])}
        else:
            return {}


class BoundsXCallback(Callback):
    """
    Returns the bounds of a xbox_select tool.
    """

    attributes = {'x0': 'cb_obj.geometry.x0', 'x1': 'cb_obj.geometry.x1'}
    models = ['plot']
    extra_models = ['xbox_select']
    on_events = ['selectiongeometry']
    skip = ["cb_obj.geometry.type != 'rect'"]

    def _process_msg(self, msg):
        if all(c in msg for c in ['x0', 'x1']):
            if isinstance(self.plot.handles.get('xaxis'), DatetimeAxis):
                msg['x0'] = convert_timestamp(msg['x0'])
                msg['x1'] = convert_timestamp(msg['x1'])
            return {'boundsx': (msg['x0'], msg['x1'])}
        else:
            return {}


class BoundsYCallback(Callback):
    """
    Returns the bounds of a ybox_select tool.
    """

    attributes = {'y0': 'cb_obj.geometry.y0', 'y1': 'cb_obj.geometry.y1'}
    models = ['plot']
    extra_models = ['ybox_select']
    on_events = ['selectiongeometry']
    skip = ["cb_obj.geometry.type != 'rect'"]

    def _process_msg(self, msg):
        if all(c in msg for c in ['y0', 'y1']):
            if isinstance(self.plot.handles.get('yaxis'), DatetimeAxis):
                msg['y0'] = convert_timestamp(msg['y0'])
                msg['y1'] = convert_timestamp(msg['y1'])
            return {'boundsy': (msg['y0'], msg['y1'])}
        else:
            return {}


class Selection1DCallback(Callback):
    """
    Returns the current selection on a ColumnDataSource.
    """

    attributes = {'index': 'cb_obj.indices'}
    models = ['selected']
    on_changes = ['indices']

    def _process_msg(self, msg):
        if 'index' in msg:
            return {'index': [int(v) for v in msg['index']]}
        else:
            return {}


class ResetCallback(Callback):
    """
    Signals the Reset stream if an event has been triggered.
    """

    models = ['plot']
    on_events = ['reset']

    def _process_msg(self, msg):
        return {'resetting': True}


class CDSCallback(Callback):
    """
    A Stream callback that syncs the data on a bokeh ColumnDataSource
    model with Python.
    """

    attributes = {'data': 'source.data'}
    models = ['source']
    on_changes = ['data']

    def initialize(self, plot_id=None):
        super(CDSCallback, self).initialize(plot_id)
        plot = self.plot
        data = self._process_msg({'data': plot.handles['source'].data})['data']
        for stream in self.streams:
            stream.update(data=data)

    def _process_msg(self, msg):
        if 'data' not in msg:
            return {}
        msg['data'] = dict(msg['data'])
        for col, values in msg['data'].items():
            if isinstance(values, dict):
                items = sorted([(int(k), v) for k, v in values.items()])
                values = [v for k, v in items]
            elif isinstance(values, list) and values and isinstance(values[0], dict):
                new_values = []
                for vals in values:
                    if isinstance(vals, dict):
                        vals = sorted([(int(k), v) for k, v in vals.items()])
                        vals = [v for k, v in vals]
                    new_values.append(vals)
                values = new_values
            msg['data'][col] = values
        return msg


class PointDrawCallback(CDSCallback):

    def initialize(self, plot_id=None):
        try:
            from bokeh.models import PointDrawTool
        except Exception:
            param.main.param.warning('PointDraw requires bokeh >= 0.12.14')
            return

        stream = self.streams[0]
        renderers = [self.plot.handles['glyph_renderer']]
        kwargs = {}
        if stream.num_objects:
            if bokeh_version >= '1.0.0':
                kwargs['num_objects'] = stream.num_objects
            else:
                param.main.param.warning(
                    'Specifying num_objects to PointDraw stream requires '
                    'a bokeh version >= 1.0.0.')
        point_tool = PointDrawTool(drag=all(s.drag for s in self.streams),
                                   empty_value=stream.empty_value,
                                   renderers=renderers, **kwargs)
        self.plot.state.tools.append(point_tool)
        source = self.plot.handles['source']

        # Add any value dimensions not already in the CDS data
        # ensuring the element can be reconstituted in entirety
        element = self.plot.current_frame
        for d in element.vdims:
            dim = dimension_sanitizer(d.name)
            if dim not in source.data:
                source.data[dim] = element.dimension_values(d)
        super(PointDrawCallback, self).initialize(plot_id)


class PolyDrawCallback(CDSCallback):

    def initialize(self, plot_id=None):
        try:
            from bokeh.models import PolyDrawTool
        except:
            param.main.param.warning('PolyDraw requires bokeh >= 0.12.14')
            return
        plot = self.plot
        stream = self.streams[0]
        kwargs = {}
        if stream.num_objects:
            if bokeh_version >= '1.0.0':
                kwargs['num_objects'] = stream.num_objects
            else:
                param.main.param.warning(
                    'Specifying num_objects to PointDraw stream requires '
                    'a bokeh version >=1.0.0.')
        if stream.show_vertices:
            if bokeh_version >= '1.0.0':
                vertex_style = dict({'size': 10}, **stream.vertex_style)
                r1 = plot.state.scatter([], [], **vertex_style)
                kwargs['vertex_renderer'] = r1
            else:
                param.main.param.warning(
                    'Enabling vertices on the PointDraw stream requires '
                    'a bokeh version >=1.0.0.')
        poly_tool = PolyDrawTool(drag=all(s.drag for s in self.streams),
                                 empty_value=stream.empty_value,
                                 renderers=[plot.handles['glyph_renderer']],
                                 **kwargs)
        plot.state.tools.append(poly_tool)
        self._update_cds_vdims()
        super(PolyDrawCallback, self).initialize(plot_id)

    def _update_cds_vdims(self):
        # Add any value dimensions not already in the CDS data
        # ensuring the element can be reconstituted in entirety
        element = self.plot.current_frame
        cds = self.plot.handles['cds']
        for d in element.vdims:
            scalar = element.interface.isscalar(element, d)
            dim = dimension_sanitizer(d.name)
            if dim not in cds.data:
                if scalar:
                    cds.data[dim] = element.dimension_values(d, not scalar)
                else:
                    cds.data[dim] = [arr[:, 0] for arr in element.split(datatype='array', dimensions=[dim])]


class FreehandDrawCallback(PolyDrawCallback):

    def initialize(self, plot_id=None):
        try:
            from bokeh.models import FreehandDrawTool
        except:
            param.main.param.warning('FreehandDraw requires bokeh >= 0.13.0')
            return
        plot = self.plot
        stream = self.streams[0]
        poly_tool = FreehandDrawTool(
            empty_value=stream.empty_value,
            num_objects=stream.num_objects,
            renderers=[plot.handles['glyph_renderer']],
        )
        plot.state.tools.append(poly_tool)
        self._update_cds_vdims()
        CDSCallback.initialize(self, plot_id)


class BoxEditCallback(CDSCallback):

    attributes = {'data': 'rect_source.data'}
    models = ['rect_source']

    def initialize(self, plot_id=None):
        try:
            from bokeh.models import BoxEditTool
        except:
            param.main.param.warning('BoxEdit requires bokeh >= 0.12.14')
            return

        plot = self.plot
        data = plot.handles['cds'].data
        element = self.plot.current_frame
        stream = self.streams[0]
        kwargs = {}
        if stream.num_objects:
            if bokeh_version >= '1.0.0':
                kwargs['num_objects'] = stream.num_objects
            else:
                param.main.param.warning(
                    'Specifying num_objects to BoxEdit stream requires '
                    'a bokeh version >=1.0.0.')
        xs, ys, widths, heights = [], [], [], []
        for x, y in zip(data['xs'], data['ys']):
            x0, x1 = (np.nanmin(x), np.nanmax(x))
            y0, y1 = (np.nanmin(y), np.nanmax(y))
            xs.append((x0+x1)/2.)
            ys.append((y0+y1)/2.)
            widths.append(x1-x0)
            heights.append(y1-y0)
        data = {'x': xs, 'y': ys, 'width': widths, 'height': heights}
        data.update({vd.name: element.dimension_values(vd, expanded=False) for vd in element.vdims})
        rect_source = ColumnDataSource(data=data)
        style = self.plot.style[self.plot.cyclic_index]
        style.pop('cmap', None)
        r1 = plot.state.rect('x', 'y', 'width', 'height', source=rect_source, **style)
        plot.handles['rect_source'] = rect_source
        box_tool = BoxEditTool(renderers=[r1], **kwargs)
        plot.state.tools.append(box_tool)
        if plot.handles['glyph_renderer'] in self.plot.state.renderers:
            self.plot.state.renderers.remove(plot.handles['glyph_renderer'])
        super(CDSCallback, self).initialize()
        data = self._process_msg({'data': data})['data']
        for stream in self.streams:
            stream.update(data=data)


    def _process_msg(self, msg):
        data = super(BoxEditCallback, self)._process_msg(msg)
        if 'data' not in data:
            return {}
        data = data['data']
        x0s, x1s, y0s, y1s = [], [], [], []
        for x, y, w, h in zip(data['x'], data['y'], data['width'], data['height']):
            x0s.append(x-w/2.)
            x1s.append(x+w/2.)
            y0s.append(y-h/2.)
            y1s.append(y+h/2.)
        return {'data': {'x0': x0s, 'x1': x1s, 'y0': y0s, 'y1': y1s}}


class PolyEditCallback(PolyDrawCallback):

    def initialize(self, plot_id=None):
        try:
            from bokeh.models import PolyEditTool
        except:
            param.main.param.warning('PolyEdit requires bokeh >= 0.12.14')
            return
        plot = self.plot
        vertex_tool = None
        if all(s.shared for s in self.streams):
            tools = [tool for tool in plot.state.tools if isinstance(tool, PolyEditTool)]
            vertex_tool = tools[0] if tools else None
        if vertex_tool is None:
            vertex_style = dict({'size': 10}, **self.streams[0].vertex_style)
            r1 = plot.state.scatter([], [], **vertex_style)
            vertex_tool = PolyEditTool(vertex_renderer=r1)
            plot.state.tools.append(vertex_tool)
        vertex_tool.renderers.append(plot.handles['glyph_renderer'])
        self._update_cds_vdims()
        CDSCallback.initialize(self, plot_id)



callbacks = Stream._callbacks['bokeh']

callbacks[PointerXY]   = PointerXYCallback
callbacks[PointerX]    = PointerXCallback
callbacks[PointerY]    = PointerYCallback
callbacks[Tap]         = TapCallback
callbacks[SingleTap]   = SingleTapCallback
callbacks[DoubleTap]   = DoubleTapCallback
callbacks[MouseEnter]  = MouseEnterCallback
callbacks[MouseLeave]  = MouseLeaveCallback
callbacks[RangeXY]     = RangeXYCallback
callbacks[RangeX]      = RangeXCallback
callbacks[RangeY]      = RangeYCallback
callbacks[BoundsXY]    = BoundsCallback
callbacks[BoundsX]     = BoundsXCallback
callbacks[BoundsY]     = BoundsYCallback
callbacks[Selection1D] = Selection1DCallback
callbacks[PlotSize]    = PlotSizeCallback
callbacks[Draw]        = DrawCallback
callbacks[PlotReset]   = ResetCallback
callbacks[CDSStream]   = CDSCallback
callbacks[BoxEdit]     = BoxEditCallback
callbacks[PointDraw]   = PointDrawCallback
callbacks[FreehandDraw]   = FreehandDrawCallback
callbacks[PolyDraw]    = PolyDrawCallback
callbacks[PolyEdit]    = PolyEditCallback



class LinkCallback(param.Parameterized):

    source_model = None
    target_model = None
    source_handles = []
    target_handles = []

    on_source_events = []
    on_source_changes = []

    on_target_events = []
    on_target_changes = []

    source_code = None
    target_code = None

    def __init__(self, root_model, link, source_plot, target_plot=None):
        self.root_model = root_model
        self.link = link
        self.source_plot = source_plot
        self.target_plot = target_plot
        self.validate()

        references = {k: v for k, v in link.get_param_values()
                      if k not in ('source', 'target', 'name')}

        for sh in self.source_handles+[self.source_model]:
            key = '_'.join(['source', sh])
            references[key] = source_plot.handles[sh]

        for p, value in link.get_param_values():
            if p in ('name', 'source', 'target'):
                continue
            references[p] = value

        if target_plot is not None:
            for sh in self.target_handles+[self.target_model]:
                key = '_'.join(['target', sh])
                references[key] = target_plot.handles[sh]

        if self.source_model in source_plot.handles:
            src_model = source_plot.handles[self.source_model]
            src_cb = CustomJS(args=references, code=self.source_code)
            for ch in self.on_source_changes:
                src_model.js_on_change(ch, src_cb)
            for ev in self.on_source_events:
                src_model.js_on_event(ev, src_cb)

        if target_plot is not None and self.target_model in target_plot.handles and self.target_code:
            tgt_model = target_plot.handles[self.target_model]
            tgt_cb = CustomJS(args=references, code=self.target_code)
            for ch in self.on_target_changes:
                tgt_model.js_on_change(ch, tgt_cb)
            for ev in self.on_target_events:
                tgt_model.js_on_event(ev, tgt_cb)

    @classmethod
    def find_links(cls, root_plot):
        """
        Traverses the supplied plot and searches for any Links on
        the plotted objects.
        """
        plot_fn = lambda x: isinstance(x, GenericElementPlot) and not isinstance(x, GenericOverlayPlot)
        plots = root_plot.traverse(lambda x: x, [plot_fn])
        potentials = [cls.find_link(plot) for plot in plots]
        source_links = [p for p in potentials if p is not None]
        found = []
        for plot, links in source_links:
            for link in links:
                if not link._requires_target:
                    # If link has no target don't look further
                    found.append((link, plot, None))
                    continue
                potentials = [cls.find_link(p, link) for p in plots]
                tgt_links = [p for p in potentials if p is not None]
                if tgt_links:
                    found.append((link, plot, tgt_links[0][0]))
        return found

    @classmethod
    def find_link(cls, plot, link=None):
        """
        Searches a GenericElementPlot for a Link.
        """
        registry = Link.registry.items()
        for source in plot.link_sources:
            if link is None:
                links = [
                    l for src, links in registry for l in links
                    if src is source or (src._plot_id is not None and
                                         src._plot_id == source._plot_id)]
                if links:
                    return (plot, links)
            else:
                if ((link.target is source) or
                    (link.target is not None and
                     link.target._plot_id is not None and
                     link.target._plot_id == source._plot_id)):
                    return (plot, [link])

    def validate(self):
        """
        Should be subclassed to check if the source and target plots
        are compatible to perform the linking.
        """


class RangeToolLinkCallback(LinkCallback):
    """
    Attaches a RangeTool to the source plot and links it to the
    specified axes on the target plot
    """

    def __init__(self, root_model, link, source_plot, target_plot):
        try:
            from bokeh.models.tools import RangeTool
        except:
            raise Exception('RangeToolLink requires bokeh >= 0.13')
        toolbars = list(root_model.select({'type': ToolbarBox}))
        axes = {}
        if 'x' in link.axes:
            axes['x_range'] = target_plot.handles['x_range']
        if 'y' in link.axes:
            axes['y_range'] = target_plot.handles['y_range']
        tool = RangeTool(**axes)
        source_plot.state.add_tools(tool)
        if toolbars:
            toolbar = toolbars[0].toolbar
            toolbar.tools.append(tool)


class DataLinkCallback(LinkCallback):
    """
    Merges the source and target ColumnDataSource
    """

    def __init__(self, root_model, link, source_plot, target_plot):
        src_cds = source_plot.handles['source']
        tgt_cds = target_plot.handles['source']
        if src_cds is tgt_cds:
            return

        src_len = [len(v) for v in src_cds.data.values()]
        tgt_len = [len(v) for v in tgt_cds.data.values()]
        if src_len[0] != tgt_len[0]:
            raise Exception('DataLink source data length must match target '
                            'data length, found source length of %d and '
                            'target length of %d.' % (src_len[0], tgt_len[0]))

        # Ensure the data sources are compatible (i.e. overlapping columns are equal)
        for k, v in tgt_cds.data.items():
            if k not in src_cds.data:
                continue
            v = np.asarray(v)
            col = np.asarray(src_cds.data[k])
            if not ((isscalar(v) and v == col) or
                    (v.dtype.kind not in 'iufc' and (v==col).all()) or
                    np.allclose(v, np.asarray(src_cds.data[k]))):
                raise ValueError('DataLink can only be applied if overlapping '
                                 'dimension values are equal, %s column on source '
                                 'does not match target' % k)

        src_cds.data.update(tgt_cds.data)
        renderer = target_plot.handles.get('glyph_renderer')
        if renderer is None:
            pass
        elif 'data_source' in renderer.properties():
            renderer.update(data_source=src_cds)
        else:
            renderer.update(source=src_cds)
        if hasattr(renderer, 'view'):
            renderer.view.update(source=src_cds)
        target_plot.handles['source'] = src_cds
        for callback in target_plot.callbacks:
            callback.initialize(plot_id=root_model.ref['id'])


callbacks = Link._callbacks['bokeh']

callbacks[RangeToolLink] = RangeToolLinkCallback
callbacks[DataLink] = DataLinkCallback
