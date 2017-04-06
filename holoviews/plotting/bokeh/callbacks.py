from collections import defaultdict

import param
import numpy as np
from bokeh.models import CustomJS

from ...streams import (Stream, PositionXY, RangeXY, Selection1D, RangeX,
                        RangeY, PositionX, PositionY, Bounds, Tap,
                        DoubleTap, MouseEnter, MouseLeave, PlotSize)
from ..comms import JupyterCommJS
from .util import bokeh_version


def attributes_js(attributes, handles):
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
    code = ''
    for key, attr_path in attributes.items():
        data_assign = "data['{key}'] = ".format(key=key)
        attrs = attr_path.split('.')
        obj_name = attrs[0]
        attr_getters = ''.join(["['{attr}']".format(attr=attr)
                                for attr in attrs[1:]])
        if obj_name not in ['cb_obj', 'cb_data']:
            assign_str = '{assign}{{id: {obj_name}["id"], value: {obj_name}{attr_getters}}};\n'.format(
                assign=data_assign, obj_name=obj_name, attr_getters=attr_getters
            )
            code += 'if (({obj_name} != undefined)) {{ {assign} }}'.format(
                obj_name=obj_name, id=handles[obj_name].ref['id'], assign=assign_str
                )
        else:
            assign_str = ''.join([data_assign, obj_name, attr_getters, ';\n'])
            code += assign_str
    return code


class Callback(object):
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

    * events      : If the Callback should listen to bokeh events this
                    should declare the types of event as a list (optional)

    * change      : If the Callback should listen to model attribute
                    changes on the defined ``models`` (optional)

    If either the event or change attributes are declared the Callback
    will be registered using the on_event or on_change machinery,
    otherwise it will be treated as a regular callback on the model.
    The callback can also define a _process_msg method, which can
    modify the data sent by the callback before it is passed to the
    streams.
    """

    code = ""

    attributes = {}

    js_callback = """
        function unique_events(events) {{
            // Processes the event queue ignoring duplicate events
            // of the same type
            var unique = [];
            var unique_events = [];
            for (var i=0; i<events.length; i++) {{
                [event, data] = events[i];
                if (!unique_events.includes(event)) {{
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
        if ((window.Jupyter !== undefined) && (Jupyter.notebook.kernel !== undefined)) {{
          var comm_manager = Jupyter.notebook.kernel.comm_manager;
          var comms = HoloViewsWidget.comms["{comm_id}"];
          if (comms && ("{comm_id}" in comms)) {{
            comm = comms["{comm_id}"];
          }} else {{
            comm = comm_manager.new_comm("{comm_id}", {{}}, {{}}, {{}});
            comm.on_msg(on_msg);
            comm_manager["{comm_id}"] = comm;
            HoloViewsWidget.comms["{comm_id}"] = comm;
          }}
          comm_manager["{comm_id}"] = comm;
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
        if ((window.Jupyter == undefined) | (Jupyter.notebook.kernel == undefined)) {{
        }} else if ((comm_state.blocked && (Date.now() < timeout))) {{
            comm_state.event_buffer.unshift([event_name, data]);
        }} else {{
            comm_state.event_buffer.unshift([event_name, data]);
            setTimeout(function() {{ process_events(comm_state); }}, {debounce});
            comm_state.blocked = true;
            comm_state.time = Date.now()+{debounce};
        }}
    """

    # The plotting handle(s) to attach the JS callback on
    models = []

    # Additional models available to the callback
    extra_models = []

    # Conditions when callback should be skipped
    skip = []

    # Callback will listen to events of the supplied type on the models
    events = []

    # List of attributes on the models to listen to
    change = []

    _comm_type = JupyterCommJS

    # Timeout if a comm message is swallowed
    timeout = 20000

    # Timeout before the first event is processed
    debounce = 20

    _callbacks = {}

    def __init__(self, plot, streams, source, **params):
        self.plot = plot
        self.streams = streams
        if plot.renderer.mode != 'server':
            self.comm = self._comm_type(plot, on_msg=self.on_msg)
        self.source = source
        self.handle_ids = defaultdict(dict)
        self.callbacks = []
        self.plot_handles = {}
        self._event_queue = []


    def initialize(self):
        plots = [self.plot]
        if self.plot.subplots:
            plots += list(self.plot.subplots.values())

        self.plot_handles = self._get_plot_handles(plots)
        requested = {}
        for h in self.models+self.extra_models:
            if h in self.plot_handles:
                requested[h] = self.plot_handles[h]
            elif h in self.extra_models:
                print("Warning %s could not find the %s model. "
                      "The corresponding stream may not work.")
        self.handle_ids.update(self._get_stream_handle_ids(requested))

        found = []
        for plot in plots:
            for handle_name in self.models:
                if handle_name not in self.plot_handles:
                    warn_args = (handle_name, type(self.plot).__name__,
                                 type(self).__name__)
                    print('%s handle not found on %s, cannot '
                          'attach %s callback' % warn_args)
                    continue
                handle = self.plot_handles[handle_name]

                # Hash the plot handle with Callback type allowing multiple
                # callbacks on one handle to be merged
                cb_hash = (id(handle), id(type(self)))
                if cb_hash in self._callbacks:
                    # Merge callbacks if another callback has already been attached
                    cb = self._callbacks[cb_hash]
                    cb.streams += self.streams
                    for k, v in self.handle_ids.items():
                        cb.handle_ids[k].update(v)
                    continue

                if self.plot.renderer.mode == 'server':
                    self.set_onchange(handle)
                else:
                    js_callback = self.get_customjs(requested)
                    self.set_customjs(js_callback, handle)
                    self.callbacks.append(js_callback)
                self._callbacks[cb_hash] = self


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
        for stream in self.streams:
            handle_ids = self.handle_ids[stream]
            ids = list(handle_ids.values())
            filtered_msg = self._filter_msg(msg, ids)
            processed_msg = self._process_msg(filtered_msg)
            if not processed_msg:
                continue
            stream.update(trigger=False, **processed_msg)
            stream._metadata = {h: {'id': hid, 'events': self.events}
                                for h, hid in handle_ids.items()}
        Stream.trigger(self.streams)
        for stream in self.streams:
            stream._metadata = {}


    def _process_msg(self, msg):
        """
        Subclassable method to preprocess JSON message in callback
        before passing to stream.
        """
        return msg


    def _get_plot_handles(self, plots):
        """
        Iterate over plots and find all unique plotting handles.
        """
        handles = {}
        for plot in plots:
            for k, v in plot.handles.items():
                handles[k] = v
        return handles


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


    def on_change(self, attr, old, new):
        """
        Process change events adding timeout to process multiple concerted
        value change at once rather than firing off multiple plot updates.
        """
        self._event_queue.append((attr, old, new))
        if self.trigger not in self.plot.document._session_callbacks:
            self.plot.document.add_timeout_callback(self.trigger, 50)


    def trigger(self):
        """
        Trigger callback change event and triggering corresponding streams.
        """
        if not self._event_queue:
            return
        self._event_queue = []

        values = {}
        for attr, path in self.attributes.items():
            attr_path = path.split('.')
            if attr_path[0] == 'cb_obj':
                attr_path = self.models[0]
            obj = self.plot_handles.get(attr_path[0])
            attr_val = obj
            if not obj:
                raise Exception('Bokeh plot attribute %s could not be found' % path)
            for p in attr_path[1:]:
                if p == 'attributes':
                    continue
                if isinstance(attr_val, dict):
                    attr_val = attr_val.get(p)
                else:
                    attr_val = getattr(attr_val, p, None)
            values[attr] = {'id': obj.ref['id'], 'value': attr_val}
        self.on_msg(values)
        self.plot.document.add_timeout_callback(self.trigger, 50)


    def set_onchange(self, handle):
        """
        Set up on_change events for bokeh server interactions.
        """
        if self.events and bokeh_version >= '0.12.5':
            for event in self.events:
                handle.on_event(event, self.on_change)
        elif self.change:
            for change in self.change:
                handle.on_change(change, self.on_change)


    def get_customjs(self, references):
        """
        Creates a CustomJS callback that will send the requested
        attributes back to python.
        """
        # Generate callback JS code to get all the requested data
        self_callback = self.js_callback.format(comm_id=self.comm.id,
                                                timeout=self.timeout,
                                                debounce=self.debounce)

        attributes = attributes_js(self.attributes, references)
        conditions = ["%s" % cond for cond in self.skip]
        conditional = ''
        if conditions:
            conditional = 'if (%s) { return };\n' % (' || '.join(conditions))
        data = "var data = {};\n"
        code = conditional + data + attributes + self.code + self_callback
        return CustomJS(args=references, code=code)


    def set_customjs(self, js_callback, handle):
        """
        Generates a CustomJS callback by generating the required JS
        code and gathering all plotting handles and installs it on
        the requested callback handle.
        """
        if self.events and bokeh_version >= '0.12.5':
            for event in self.events:
                handle.js_on_event(event, js_callback)
        elif self.change:
            for change in self.change:
                handle.js_on_change(change, js_callback)
        elif hasattr(handle, 'callback'):
            handle.callback = js_callback




class PositionXYCallback(Callback):
    """
    Returns the mouse x/y-position on mousemove event.
    """

    attributes = {'x': 'cb_obj.x', 'y': 'cb_obj.y'}
    models = ['plot']
    events = ['mousemove']


class PositionXCallback(PositionXYCallback):
    """
    Returns the mouse x-position on mousemove event.
    """

    attributes = {'x': 'cb_obj.x'}


class PositionYCallback(PositionXYCallback):
    """
    Returns the mouse x/y-position on mousemove event.
    """

    attributes = {'y': 'cb_data.y'}


class TapCallback(PositionXYCallback):
    """
    Returns the mouse x/y-position on tap event.
    """

    events = ['tap']


class DoubleTapCallback(PositionXYCallback):
    """
    Returns the mouse x/y-position on doubletap event.
    """

    events = ['doubletap']


class MouseEnterCallback(PositionXYCallback):
    """
    Returns the mouse x/y-position on mouseenter event, i.e. when
    mouse enters the plot canvas.
    """

    events = ['mouseenter']


class MouseLeaveCallback(PositionXYCallback):
    """
    Returns the mouse x/y-position on mouseleave event, i.e. when
    mouse leaves the plot canvas.
    """

    events = ['mouseleave']


class RangeXYCallback(Callback):
    """
    Returns the x/y-axis ranges of a plot.
    """

    attributes = {'x0': 'x_range.attributes.start',
                  'x1': 'x_range.attributes.end',
                  'y0': 'y_range.attributes.start',
                  'y1': 'y_range.attributes.end'}
    models = ['x_range', 'y_range']
    change = ['start', 'end']

    def _process_msg(self, msg):
        data = {}
        if 'x0' in msg and 'x1' in msg:
            data['x_range'] = (msg['x0'], msg['x1'])
        if 'y0' in msg and 'y1' in msg:
            data['y_range'] = (msg['y0'], msg['y1'])
        return data


class RangeXCallback(RangeXYCallback):
    """
    Returns the x-axis range of a plot.
    """

    attributes = {'x0': 'x_range.attributes.start',
                  'x1': 'x_range.attributes.end'}
    models = ['x_range']

    def _process_msg(self, msg):
        if 'x0' in msg and 'x1' in msg:
            return {'x_range': (msg['x0'], msg['x1'])}
        else:
            return {}


class RangeYCallback(RangeXYCallback):
    """
    Returns the y-axis range of a plot.
    """

    attributes = {'y0': 'y_range.attributes.start',
                  'y1': 'y_range.attributes.end'}
    models = ['y_range']

    def _process_msg(self, msg):
        if 'y0' in msg and 'y1' in msg:
            return {'y_range': (msg['y0'], msg['y1'])}
        else:
            return {}


class PlotSizeCallback(Callback):
    """
    Returns the actual width and height of a plot once the layout
    solver has executed.
    """

    models = ['plot']
    attributes = {'width': 'cb_obj.inner_width',
                  'height': 'cb_obj.inner_height'}
    change = ['inner_width', 'inner_height']


class BoundsCallback(Callback):
    """
    Returns the bounds of a box_select tool.
    """

    attributes = {'x0': 'cb_data.geometry.x0',
                  'x1': 'cb_data.geometry.x1',
                  'y0': 'cb_data.geometry.y0',
                  'y1': 'cb_data.geometry.y1'}
    models = ['box_select']

    def _process_msg(self, msg):
        if all(c in msg for c in ['x0', 'y0', 'x1', 'y1']):
            return {'bounds': (msg['x0'], msg['y0'], msg['x1'], msg['y1'])}
        else:
            return {}


class Selection1DCallback(Callback):
    """
    Returns the current selection on a ColumnDataSource.
    """

    attributes = {'index': 'cb_obj.selected.1d.indices'}
    models = ['source']
    change = ['selected']

    def _process_msg(self, msg):
        if 'index' in msg:
            return {'index': [int(v) for v in msg['index']]}
        else:
            return {}


callbacks = Stream._callbacks['bokeh']

callbacks[PositionXY]  = PositionXYCallback
callbacks[PositionX]   = PositionXCallback
callbacks[PositionY]   = PositionYCallback
callbacks[Tap]         = TapCallback
callbacks[DoubleTap]   = DoubleTapCallback
callbacks[MouseEnter]  = MouseEnterCallback
callbacks[MouseLeave]  = MouseLeaveCallback
callbacks[RangeXY]     = RangeXYCallback
callbacks[RangeX]      = RangeXCallback
callbacks[RangeY]      = RangeYCallback
callbacks[Bounds]      = BoundsCallback
callbacks[Selection1D] = Selection1DCallback
callbacks[PlotSize]    = PlotSizeCallback
