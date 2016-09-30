import json
from collections import defaultdict

import param
import numpy as np
from bokeh.models import CustomJS

from ...streams import (Stream, PositionXY, RangeXY, Selection1D, RangeX,
                        RangeY, PositionX, PositionY, Bounds)
from ..comms import JupyterCommJS


def attributes_js(attributes):
    """
    Generates JS code to look up attributes on JS objects from
    an attributes specification dictionary.

    Example:

    Input  : {'x': 'cb_data.geometry.x'}

    Output : data['x'] = cb_data['geometry']['x']
    """
    code = ''
    for key, attr_path in attributes.items():
        data_assign = "data['{key}'] = ".format(key=key)
        attrs = attr_path.split('.')
        obj_name = attrs[0]
        attr_getters = ''.join(["['{attr}']".format(attr=attr)
                                for attr in attrs[1:]])
        code += ''.join([data_assign, obj_name, attr_getters, ';\n'])
    return code


class Callback(object):
    """
    Provides a baseclass to define callbacks, which return data from
    bokeh models such as the plot ranges or various tools. The callback
    then makes this data available to any streams attached to it.

    The defintion of a callback consists of a number of components:

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
        var argstring = JSON.stringify(data);
        if ((window.Jupyter !== undefined) && (Jupyter.notebook.kernel !== undefined)) {{
          var comm_manager = Jupyter.notebook.kernel.comm_manager;
          var comms = HoloViewsWidget.comms["{comms_target}"];
          if (comms && ("{comms_target}" in comms)) {{
            comm = comms["{comms_target}"];
          }} else {{
            comm = comm_manager.new_comm("{comms_target}", {{}}, {{}}, {{}});

          comm_manager["{comms_target}"] = comm;
            HoloViewsWidget.comms["{comms_target}"] = comm;
          }}
          comm_manager["{comms_target}"] = comm;
          comm.send(argstring)
        }}
    """

    # The plotting handle(s) to attach the JS callback on
    handles = []
    msgs=[]

    def __init__(self, plot, streams, source, **params):
        self.plot = plot
        self.streams = streams
        self.comm = JupyterCommJS(plot, on_msg=self.on_msg)
        self.source = source


    def initialize(self):
        plots = [self.plot]
        if self.plot.subplots:
            plots += list(self.plot.subplots.values())

        found = []
        for plot in plots:
            for handle in self.handles:
                if handle not in plot.handles or handle in found:
                    continue
                self.set_customjs(plot.handles[handle])
                found.append(handle)
        if len(found) != len(self.handles):
            self.warning('Plotting handle for JS callback not found')


    def on_msg(self, msg):
        msg = json.loads(msg)
        msg = self._process_msg(msg)
        if any(v is None for v in msg.values()):
            return
        for stream in self.streams:
            stream.update(**msg)


    def _process_msg(self, msg):
        return msg


    def set_customjs(self, handle):
        """
        Generates a CustomJS callback by generating the required JS
        code and gathering all plotting handles and installs it on
        the requested callback handle.
        """

        # Generate callback JS code to get all the requested data
        self_callback = self.js_callback.format(comms_target=self.comm.target)
        attributes = attributes_js(self.attributes)
        code = 'var data = {};\n' + attributes + self.code + self_callback

        handles = dict(self.plot.handles)
        plots = [self.plot] + (self.plot.subplots.values()[::-1] if self.plot.subplots else [])
        for plot in plots:
            handles.update(plot.handles)
        # Set callback
        handle.callback = CustomJS(args=handles, code=code)



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
        return {'x_range': (msg['x0'], msg['x1']),
                'y_range': (msg['y0'], msg['y1'])}


class RangeXCallback(Callback):

    attributes = {'x0': 'x_range.attributes.start',
                  'x1': 'x_range.attributes.end'}

    handles = ['x_range']

    def _process_msg(self, msg):
        return {'x_range': (msg['x0'], msg['x1'])}


class RangeYCallback(Callback):

    attributes = {'y0': 'y_range.attributes.start',
                  'y1': 'y_range.attributes.end'}

    handles = ['y_range']

    def _process_msg(self, msg):
        return {'y_range': (msg['y0'], msg['y1'])}


class BoundsCallback(Callback):

    attributes = {'x0': 'cb_data.geometry.x0',
                  'x1': 'cb_data.geometry.x1',
                  'y0': 'cb_data.geometry.y0',
                  'y1': 'cb_data.geometry.y1'}

    handles = ['box_select']

    def _process_msg(self, msg):
        return {'bounds': (msg['x0'], msg['y0'], msg['x1'], msg['y1'])}


class Selection1DCallback(Callback):

    attributes = {'index': 'source.selected.1d.indices'}

    handles = ['source']


callbacks = Stream._callbacks['bokeh']

callbacks[PositionXY] = PositionXYCallback
callbacks[PositionX] = PositionXCallback
callbacks[PositionY] = PositionYCallback
callbacks[RangeXY] = RangeXYCallback
callbacks[RangeX] = RangeXCallback
callbacks[RangeY] = RangeYCallback
callbacks[Bounds] = BoundsCallback
callbacks[Selection1D] = Selection1DCallback
