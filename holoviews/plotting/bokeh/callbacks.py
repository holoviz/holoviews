import json
from collections import defaultdict

import param
import numpy as np
from bokeh.models import CustomJS

from ...streams import (Stream, PositionXY, RangeXY, BoxSelect, RangeX,
                        RangeY, PositionX, PositionY)
from ..comms import JupyterCommJS


def get_attributes(attributes):
    code = ''
    for key, attr_path in attributes.items():
        data_assign = "data['{key}'] = ".format(key=key)
        attrs = attr_path.split('.')
        obj_name = attrs[0]
        attr_getters = ''.join(["['{attr}']".format(attr=attr)
                                for attr in attrs[1:]])
        code += ''.join([data_assign, obj_name, attr_getters, ';\n'])
    return code


class Callback(param.Parameterized):

    code = param.String(default="", doc="""
        Custom javascript code executed on the callback. The code
        has access to the plot, source and cb_obj and may modify
        the data javascript object sent back to Python.""")

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

    def __init__(self, plot, streams, **params):
        self.plot = plot
        self.streams = streams
        self.comm = JupyterCommJS(plot, on_msg=self.on_msg)


    def initialize(self):
        for handle in self.handles:
            if handle not in self.plot.handles:
                self.warning('Plotting handle for JS callback not found')
                continue
            self.set_customjs(self.plot.handles[handle])


    def on_msg(self, msg):
        msg = json.loads(msg)
        msg = self._process_msg(msg)
        if any(v is None for v in msg.values()):
            return
        for stream in self.streams:
            stream.update(**msg)


    def _process_msg(self, msg):
        return msg


    def set_customjs(self, cb_obj):
        # Generate callback JS code to get all the requested data
        self_callback = self.js_callback.format(comms_target=self.comm.target)
        attributes = get_attributes(self.attributes)
        code = 'var data = {};\n' + attributes + self.code + self_callback

        # Set cb_obj
        cb_obj.callback = CustomJS(args=self.plot.handles, code=code)



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


class BoxCallback(Callback):

    attributes = {'x0': 'cb_data.geometry.x0',
                  'x1': 'cb_data.geometry.x1',
                  'y0': 'cb_data.geometry.y0',
                  'y1': 'cb_data.geometry.y1'}

    handles = ['box_select']

    def _process_msg(self, msg):
        return {'bounds': (msg['x0'], msg['y0'], msg['x1'], msg['y1'])}


callbacks = Stream._callbacks['bokeh']

callbacks[PositionXY] = PositionXYCallback
callbacks[PositionX] = PositionXCallback
callbacks[PositionY] = PositionYCallback
callbacks[RangeXY] = RangeXYCallback
callbacks[RangeX] = RangeXCallback
callbacks[RangeY] = RangeYCallback
callbacks[BoxSelect] = BoxCallback
