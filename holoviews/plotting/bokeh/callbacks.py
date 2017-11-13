from collections import defaultdict

from bokeh.models import CustomJS, FactorRange, DatetimeAxis

from ...core import OrderedDict
from ...streams import (Stream, PointerXY, RangeXY, Selection1D, RangeX,
                        RangeY, PointerX, PointerY, BoundsX, BoundsY,
                        Tap, SingleTap, DoubleTap, MouseEnter, MouseLeave,
                        PlotSize, Draw, BoundsXY, PlotReset)
from ...streams import PositionX, PositionY, PositionXY, Bounds # Deprecated: remove in 2.0
from ..comms import JupyterCommJS, Comm
from .util import convert_timestamp


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
            try:
                self.comm = self._comm_type(plot, on_msg=self.on_msg)
            except AttributeError:
                self.comm = Comm(plot)
        self.source = source
        self.handle_ids = defaultdict(dict)
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
            if not processed_msg and not stream.transient:
                continue
            stream.update(**processed_msg)
            stream._metadata = {h: {'id': hid, 'events': self.on_events}
                                for h, hid in handle_ids.items()}
            streams.append(stream)
        Stream.trigger(streams)
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
                      "The corresponding stream may not work.")
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

    js_callback = """
        function unique_events(events) {{
            // Processes the event queue ignoring duplicate events
            // of the same type
            var unique = [];
            var unique_events = [];
            for (var i=0; i<events.length; i++) {{
                [event, data] = events[i];
                if (unique_events.indexOf(event)===-1) {{
                    unique.unshift(data);
                    unique_events.push(event);
                }}
            }}
            return unique;
        }}

        function process_events(comm_state) {{
            // Iterates over event queue and sends events via Comm
            var events = unique_events(comm_state.event_buffer);
            for (var i=0; i<events.length; i++) {{
                var data = events[i];
                var comm = HoloViewsWidget.comms[data["comm_id"]];
                comm.send(data);
            }}
            comm_state.event_buffer = [];
        }}

        function on_msg(msg){{
          // Receives acknowledgement from Python, processing event
          // and unblocking Comm if event queue empty
          msg = JSON.parse(msg.content.data);
          var comm_id = msg["comm_id"]
          var comm_state = HoloViewsWidget.comm_state[comm_id];
          if (comm_state.event_buffer.length) {{
            process_events(comm_state);
            comm_state.blocked = true;
            comm_state.time = Date.now()+{debounce};
          }} else {{
            comm_state.blocked = false;
          }}
          comm_state.event_buffer = [];
          if ((msg.msg_type == "Ready") && msg.content) {{
            console.log("Python callback returned following output:", msg.content);
          }} else if (msg.msg_type == "Error") {{
            console.log("Python failed with the following traceback:", msg['traceback'])
          }}
        }}

        // Initialize Comm
        if ((window.Jupyter !== undefined) && (Jupyter.notebook.kernel != null)) {{
          var comm_manager = Jupyter.notebook.kernel.comm_manager;
          var comm = HoloViewsWidget.comms["{comm_id}"];
          if (comm == null) {{
            comm = comm_manager.new_comm("{comm_id}", {{}}, {{}}, {{}});
            comm.on_msg(on_msg);
            comm_manager["{comm_id}"] = comm;
            HoloViewsWidget.comms["{comm_id}"] = comm;
          }}
        }} else {{
          return
        }}

        // Initialize event queue and timeouts for Comm
        var comm_state = HoloViewsWidget.comm_state["{comm_id}"];
        if (comm_state === undefined) {{
            comm_state = {{event_buffer: [], blocked: false, time: Date.now()}}
            HoloViewsWidget.comm_state["{comm_id}"] = comm_state
        }}

        // Add current event to queue and process queue if not blocked
        event_name = cb_obj.event_name
        data['comm_id'] = "{comm_id}";
        timeout = comm_state.time + {timeout};
        if ((window.Jupyter == null) | (Jupyter.notebook.kernel == null)) {{
        }} else if ((comm_state.blocked && (Date.now() < timeout))) {{
            comm_state.event_buffer.unshift([event_name, data]);
        }} else {{
            comm_state.event_buffer.unshift([event_name, data]);
            setTimeout(function() {{ process_events(comm_state); }}, {debounce});
            comm_state.blocked = true;
            comm_state.time = Date.now()+{debounce};
        }}
    """

    code = ""

    # Timeout if a comm message is swallowed
    timeout = 20000

    # Timeout before the first event is processed
    debounce = 20

    _comm_type = JupyterCommJS

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


    def get_customjs(self, references):
        """
        Creates a CustomJS callback that will send the requested
        attributes back to python.
        """
        # Generate callback JS code to get all the requested data
        self_callback = self.js_callback.format(comm_id=self.comm.id,
                                                timeout=self.timeout,
                                                debounce=self.debounce)

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
        if not self._active:
            self.plot.document.add_timeout_callback(self.process_on_change, 50)
            self._active = True


    def on_event(self, event):
        """
        Process bokeh UIEvents adding timeout to process multiple concerted
        value change at once rather than firing off multiple plot updates.
        """
        self._queue.append((event))
        if not self._active:
            self.plot.document.add_timeout_callback(self.process_on_event, 50)
            self._active = True

    def process_on_event(self):
        """
        Trigger callback change event and triggering corresponding streams.
        """
        if not self._queue:
            self._active = False
            return
        self._queue = []
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

    def initialize(self):
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
                js_callback = self.get_customjs(handles)
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
             if (cb_obj.x < x_range.start) {
               data['x'] = x_range.start }
             else if (cb_obj.x > x_range.end) {
               data['x'] = x_range.end }}
           if (y_range.type.endsWith('Range1d')) {
             if (cb_obj.y < y_range.start) {
               data['y'] = y_range.start }
             else if (cb_obj.y > y_range.end) {
               data['y'] = y_range.end }}
           """

    def _process_msg(self, msg):
        x_range = self.plot.handles.get('x_range')
        y_range = self.plot.handles.get('y_range')
        if isinstance(x_range, FactorRange) and isinstance(msg.get('x'), (int, float)):
            msg['x'] = x_range.factors[int(msg['x'])]
        if isinstance(y_range, FactorRange) and isinstance(msg.get('y'), (int, float)):
            msg['y'] = y_range.factors[int(msg['y'])]

        xaxis = self.plot.handles.get('xaxis')
        yaxis = self.plot.handles.get('yaxis')
        if 'x' in msg and isinstance(xaxis, DatetimeAxis):
            msg['x'] = convert_timestamp(msg['x'])
        if 'y' in msg and isinstance(yaxis, DatetimeAxis):
            msg['y'] = convert_timestamp(msg['y'])
        return msg


class PointerXCallback(PointerXYCallback):
    """
    Returns the mouse x-position on mousemove event.
    """

    attributes = {'x': 'cb_obj.x'}
    extra_models= ['x_range']
    code = """
           if (x_range.type.endsWith('Range1d')) {
             if (cb_obj.x < x_range.start) {
               data['x'] = x_range.start }
             else if (cb_obj.x > x_range.end) {
               data['x'] = x_range.end }}
           """

class PointerYCallback(PointerXYCallback):
    """
    Returns the mouse x/y-position on mousemove event.
    """

    attributes = {'y': 'cb_obj.y'}
    extra_models= ['y_range']
    code = """
           if (y_range.type.endsWith('Range1d')) {
             if (cb_obj.y < y_range.start) {
               data['y'] = y_range.start }
             else if (cb_obj.y > y_range.end) {
               data['y'] = y_range.end }}
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

    on_events = ['tap', 'doubletap']


class SingleTapCallback(PointerXYCallback):
    """
    Returns the mouse x/y-position on tap event.
    """

    on_events = ['tap']


class DoubleTapCallback(PointerXYCallback):
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

    attributes = {'index': 'cb_obj.selected.1d.indices'}
    models = ['source']
    on_changes = ['selected']

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
        return {'reset': True}


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
callbacks[Bounds]      = BoundsCallback
callbacks[BoundsXY]    = BoundsCallback
callbacks[BoundsX]     = BoundsXCallback
callbacks[BoundsY]     = BoundsYCallback
callbacks[Selection1D] = Selection1DCallback
callbacks[PlotSize]    = PlotSizeCallback
callbacks[Draw]        = DrawCallback
callbacks[PlotReset]   = ResetCallback

# Aliases for deprecated streams
callbacks[PositionXY]  = PointerXYCallback
callbacks[PositionX]   = PointerXCallback
callbacks[PositionY]   = PointerYCallback
