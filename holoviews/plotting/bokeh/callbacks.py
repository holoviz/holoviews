from collections import defaultdict

import param
import numpy as np
from bokeh.models import CustomJS

from ...streams import (Stream, PositionXY, RangeXY, Selection1D, RangeX,
                        RangeY, PositionX, PositionY, Bounds)
from ..comms import JupyterCommJS


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
    bokeh models such as the plot ranges or various tools. The callback
    then makes this data available to any streams attached to it.

    The definition of a callback consists of a number of components:

    * handles    :  The handles define which plotting handles the
                    callback will be attached on, e.g. this could be
                    the x_range, y_range, a plotting tool or any other
                    bokeh object that allows callbacks.

    * attributes :  The attributes define which attributes to send
                    back to Python. They are defined as a dictionary
                    mapping between the name under which the variable
                    is made available to Python and the specification
                    of the attribute. The specification should start
                    with the variable name that is to be accessed and
                    the location of the attribute separated by periods.
                    All plotting handles such as tools, the x_range,
                    y_range and (data)source can be addressed in this
                    way, e.g. to get the start of the x_range as 'x'
                    you can supply {'x': 'x_range.attributes.start'}.
                    Additionally certain handles additionally make the
                    cb_data and cb_obj variables available containing
                    additional information about the event.

    * code        : Defines any additional JS code to be executed,
                    which can modify the data object that is sent to
                    the backend.

    The callback can also define a _process_msg method, which can
    modify the data sent by the callback before it is passed to the
    streams.
    """

    code = ""

    attributes = {}

    js_callback = """
        function on_msg(msg){{
          msg = JSON.parse(msg.content.data);
          var comm_id = msg["comm_id"]
          var comm = HoloViewsWidget.comms[comm_id];
          var comm_state = HoloViewsWidget.comm_state[comm_id];
          if (comm_state.event) {{
            comm.send(comm_state.event);
            comm_state.blocked = true;
            comm_state.timeout = Date.now()+{debounce};
          }} else {{
            comm_state.blocked = false;
          }}
          comm_state.event = undefined;
          if ((msg.msg_type == "Ready") && msg.content) {{
            console.log("Python callback returned following output:", msg.content);
          }} else if (msg.msg_type == "Error") {{
            console.log("Python failed with the following traceback:", msg['traceback'])
          }}
        }}

        data['comm_id'] = "{comm_id}";
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

        var comm_state = HoloViewsWidget.comm_state["{comm_id}"];
        if (comm_state === undefined) {{
            comm_state = {{event: undefined, blocked: false, timeout: Date.now()}}
            HoloViewsWidget.comm_state["{comm_id}"] = comm_state
        }}

        function trigger() {{
            if (comm_state.event != undefined) {{
               var comm_id = comm_state.event["comm_id"]
               var comm = HoloViewsWidget.comms[comm_id];
               comm.send(comm_state.event);
            }}
            comm_state.event = undefined;
        }}

        timeout = comm_state.timeout + {timeout};
        if ((window.Jupyter == undefined) | (Jupyter.notebook.kernel == undefined)) {{
        }} else if ((comm_state.blocked && (Date.now() < timeout))) {{
            comm_state.event = data;
        }} else {{
            comm_state.event = data;
            setTimeout(trigger, {debounce});
            comm_state.blocked = true;
            comm_state.timeout = Date.now()+{debounce};
        }}
    """

    # The plotting handle(s) to attach the JS callback on
    handles = []

    _comm_type = JupyterCommJS

    # Timeout if a comm message is swallowed
    timeout = 20000

    # Timeout before the first event is processed
    debounce = 20

    _callbacks = {}

    event = False

    def __init__(self, plot, streams, source, **params):
        self.plot = plot
        self.streams = streams
        self.comm = self._comm_type(plot, on_msg=self.on_msg)
        self.source = source
        self.handle_ids = defaultdict(list)


    def initialize(self):
        plots = [self.plot]
        if self.plot.subplots:
            plots += list(self.plot.subplots.values())

        handles = self._get_plot_handles(plots)
        self.handle_ids.update(self._get_stream_handle_ids(handles))

        for plot in plots:
            for handle_name in self.handles:
                if handle_name not in handles:
                    warn_args = (handle_name, type(self.plot).__name__,
                                 type(self).__name__)
                    self.warning('%s handle not found on %s, cannot'
                                 'attach %s callback' % warn_args)
                    continue
                handle = handles[handle_name]
                self.set_customjs(handle, handles)


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
            ids = self.handle_ids[stream]
            filtered_msg = self._filter_msg(msg, ids)
            processed_msg = self._process_msg(filtered_msg)
            if not processed_msg:
                continue
            stream.update(trigger=False, **processed_msg)
        Stream.trigger(self.streams)


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
                if k not in handles and k in self.handles:
                    handles[k] = v
        return handles


    def _get_stream_handle_ids(self, handles):
        """
        Gather the ids of the plotting handles attached to this callback
        This allows checking that a stream is not given the state
        of a plotting handle it wasn't attached to
        """
        stream_handle_ids = defaultdict(list)
        for stream in self.streams:
            for h in self.handles:
                if h in handles:
                    handle_id = handles[h].ref['id']
                    stream_handle_ids[stream].append(handle_id)
        return stream_handle_ids


    def set_customjs(self, handle, references):
        """
        Generates a CustomJS callback by generating the required JS
        code and gathering all plotting handles and installs it on
        the requested callback handle.
        """

        # Generate callback JS code to get all the requested data
        self_callback = self.js_callback.format(comm_id=self.comm.id,
                                                timeout=self.timeout,
                                                debounce=self.debounce)

        attributes = attributes_js(self.attributes, references)
        code = 'var data = {};\n' + attributes + self.code + self_callback

        if self.event:
            js_callback = CustomJS(args=references, code=code)
            handle.js_on_event(self.event, js_callback)
            return

        # Merge callbacks if another callback has already been attached
        # otherwise set it
        if id(handle.callback) in self._callbacks:
            cb = self._callbacks[id(handle.callback)]
            if isinstance(cb, type(self)):
                cb.streams += self.streams
                for k, v in self.handle_ids.items():
                    cb.handle_ids[k] += v
            else:
                handle.callback.code += code
        else:
            js_callback = CustomJS(args=references, code=code)
            self._callbacks[id(js_callback)] = self
            handle.callback = js_callback



class PositionXYCallback(Callback):

    attributes = {'x': 'cb_data.geometry.x', 'y': 'cb_data.geometry.y'}

    handles = ['hover']


class PositionXCallback(Callback):

    attributes = {'x': 'cb_data.geometry.x'}

    handles = ['hover']


class PositionYCallback(Callback):

    attributes = {'y': 'cb_data.geometry.y'}

    handles = ['hover']


class RangeXYCallback(Callback):

    attributes = {'x0': 'x_range.attributes.start',
                  'x1': 'x_range.attributes.end',
                  'y0': 'y_range.attributes.start',
                  'y1': 'y_range.attributes.end'}

    handles = ['x_range', 'y_range']

    def _process_msg(self, msg):
        data = {}
        if 'x0' in msg and 'x1' in msg:
            data['x_range'] = (msg['x0'], msg['x1'])
        if 'y0' in msg and 'y1' in msg:
            data['y_range'] = (msg['y0'], msg['y1'])
        return data


class RangeXCallback(Callback):

    attributes = {'x0': 'x_range.attributes.start',
                  'x1': 'x_range.attributes.end'}

    handles = ['x_range']

    def _process_msg(self, msg):
        if 'x0' in msg and 'x1' in msg:
            return {'x_range': (msg['x0'], msg['x1'])}
        else:
            return {}


class RangeYCallback(Callback):

    attributes = {'y0': 'y_range.attributes.start',
                  'y1': 'y_range.attributes.end'}

    handles = ['y_range']

    def _process_msg(self, msg):
        if 'y0' in msg and 'y1' in msg:
            return {'y_range': (msg['y0'], msg['y1'])}
        else:
            return {}


class BoundsCallback(Callback):

    attributes = {'x0': 'cb_data.geometry.x0',
                  'x1': 'cb_data.geometry.x1',
                  'y0': 'cb_data.geometry.y0',
                  'y1': 'cb_data.geometry.y1'}

    handles = ['box_select']

    def _process_msg(self, msg):
        if all(c in msg for c in ['x0', 'y0', 'x1', 'y1']):
            return {'bounds': (msg['x0'], msg['y0'], msg['x1'], msg['y1'])}
        else:
            return {}


class Selection1DCallback(Callback):

    attributes = {'index': 'source.selected.1d.indices'}

    handles = ['source']

    def _process_msg(self, msg):
        if 'index' in msg:
            return {'index': [int(v) for v in msg['index']]}
        else:
            return {}


callbacks = Stream._callbacks['bokeh']

callbacks[PositionXY] = PositionXYCallback
callbacks[PositionX] = PositionXCallback
callbacks[PositionY] = PositionYCallback
callbacks[RangeXY] = RangeXYCallback
callbacks[RangeX] = RangeXCallback
callbacks[RangeY] = RangeYCallback
callbacks[Bounds] = BoundsCallback
callbacks[Selection1D] = Selection1DCallback
