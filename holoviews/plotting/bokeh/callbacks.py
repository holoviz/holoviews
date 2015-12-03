from collections import defaultdict

import numpy as np
from bokeh.models import CustomJS, TapTool, ColumnDataSource

try:
    from bokeh.models import PlotObject
    from bokeh.protocol import serialize_json
    old_bokeh = True
except ImportError:
    from bokeh.models import Component as PlotObject
    from bokeh._json_encoder import serialize_json
    old_bokeh = False

import param

from .plot import BokehPlot


class Callback(param.ParameterizedFunction):
    """
    Callback functions provide an easy way to interactively modify
    the plot based on changes to the bokeh axis ranges, data sources,
    tools or widgets.

    The data sent to the Python callback from javascript is defined
    by the plot attributes and callback_obj attributes and optionally
    custom javascript code.

    The user should define any plot_attributes and cb_attributes
    he wants access to, which will be supplied in the form of a
    dictionary to the call method. The call method can then apply
    any processing depending on the callback data and return the
    modified bokeh plot objects.
    """

    callback_obj = param.ClassSelector(class_=(PlotObject,), doc="""
        Bokeh PlotObject the callback is applied to.""")

    cb_attributes = param.List(default=[], doc="""
        Callback attributes returned to the Python callback.""")

    code = param.String(default="", doc="""
        Custom javascript code executed on the callback. The code
        has access to the plot, source and cb_obj and may modify
        the data javascript object sent back to Python.""")

    current_data = param.Dict(default={})

    initialize_cb = param.Boolean(default=True, doc="""
        Whether the callback should be initialized when it's first
        added to a plot""")

    plot_attributes = param.Dict(default={}, doc="""
        Plot attributes returned to the Python callback.""")

    plots = param.List(default=[], doc="""
        The HoloViews plot object the callback applies to.""")

    streams = param.List(default=[], doc="""
        List of streams attached to this callback.""")

    reinitialize = param.Boolean(default=False, doc="""
        Whether the Callback should be reinitialized per plot instance""")

    skip_unchanged = param.Boolean(default=False, doc="""
        Avoid running the callback if the callback data is unchanged.
        Useful for avoiding infinite loops.""")

    JS_callback = """
        function callback(msg){
          if (msg.msg_type == "execute_result") {
            var data = JSON.parse(msg.content.data['text/plain'].slice(1, -1));
            $.each(data, function(id, value) {
              var ds = Bokeh.Collections(value.type).get(id);
                if (ds != undefined) {
                  ds.set(value.data);
                }
            });
          } else {
            console.log("Python callback returned unexpected message:", msg)
          }
        }
        callbacks = {iopub: {output: callback}};
        var data = {};
    """

    IPython_callback = """
        if (!(_.isEmpty(data))) {{
           var argstring = JSON.stringify(data);
           argstring = argstring.replace('true', 'True').replace('false','False');
           var kernel = IPython.notebook.kernel;
           var cmd = "Callbacks.callbacks[{callback_id}].update(" + argstring + ")";
           var pyimport = "from holoviews.plotting.bokeh import Callbacks;";
           kernel.execute(pyimport + cmd, callbacks, {{silent : false}});
        }}
    """

    def initialize(self, data):
        """
        Initialize is called when the callback is added to a new plot
        and the initialize option is enabled. May avoid repeat
        initialization by setting initialize_cb parameter to False
        inside this method.
        """


    def __call__(self, data):
        """
        The call method can modify any bokeh plot object
        depending on the supplied data dictionary. It
        should return the modified plot objects as a list.
        """
        return []


    def update(self, data, chained=False):
        """
        The update method is called by the javascript callback
        with the supplied data and will return the json serialized
        string representation of the changes to the Bokeh plot.
        When chained=True it will return a list of the plot objects
        to be updated, allowing chaining of callback operations.
        """
        if self.skip_unchanged and self.current_data == data:
            return [] if chained else "{}"
        self.current_data = data

        objects = self(data)

        for stream in self.streams:
            objects += stream.update(data, True)

        if chained:
            return objects
        else:
            return self.serialize(objects)


    def serialize(self, objects):
        """
        Serializes any Bokeh plot objects passed to it as a list.
        """
        data = {}
        for obj in objects:
            if old_bokeh:
                json = obj.vm_serialize(changed_only=True)
            else:
                json = obj.to_json(False)
            data[obj.ref['id']] = {'type': obj.ref['type'], 'data': json}
        return serialize_json(data)



class DownsampleImage(Callback):
    """
    Downsamples any Image plot to the specified
    max_width and max_height by slicing the
    Image to the specified x_range and y_range
    and then finding step values matching the
    constraints.
    """

    max_width = param.Integer(default=250, doc="""
        Maximum plot width in pixels after slicing and downsampling.""")

    max_height = param.Integer(default=250, doc="""
        Maximum plot height in pixels after slicing and downsampling.""")

    plot_attributes = param.Dict(default={'x_range': ['start', 'end'],
                                          'y_range': ['start', 'end']})

    def __call__(self, data):
        xstart, xend = data['x_range']
        ystart, yend = data['y_range']

        ranges = self.plots[0].current_ranges
        element = self.plots[0].current_frame

        # Slice Element to match selected ranges
        xdim, ydim = element.dimensions('key', True)
        sliced = element.select(**{xdim: (xstart, xend),
                                   ydim: (ystart, yend)})

        # Get dimensions of sliced element
        shape = sliced.data.shape
        max_shape = (self.max_height, self.max_width)

        #Find minimum downsampling to fit requirement
        steps = []
        for s, max_s in zip(shape, max_shape):
            step = 1
            while s/step > max_s: step += 1
            steps.append(step)
        resampled = sliced.clone(sliced.data[::steps[0], ::steps[1]])

        # Update data source
        new_data = self.plots[0].get_data(resampled, ranges)[0]
        source = self.plots[0].handles['source']
        source.data.update(new_data)
        return [source]



class DownsampleColumns(Callback):
    """
    Downsamples any column based Element by randomizing
    the rows and updating the ColumnDataSource with
    up to max_samples.
    """

    max_samples = param.Integer(default=800)

    plot_attributes = param.Dict(default={'x_range': ['start', 'end'],
                                          'y_range': ['start', 'end']})

    def initialize(self, data):
        return self(data)

    def __call__(self, data):
        xstart, xend = data['x_range']
        ystart, yend = data['y_range']

        plot = self.plots[0]
        element = plot.current_frame
        ranges  = plot.current_ranges

        # Slice element to current ranges
        xdim, ydim = element.dimensions(label=True)[0:2]
        sliced = element.select(**{xdim: (xstart, xend),
                                   ydim: (ystart, yend)})

        # Avoid randomizing if possible (expensive)
        if len(sliced) > self.max_samples:
            # Randomize element samples and slice to region
            # Randomization consistent to avoid "flicker".
            np.random.seed(42)
            inds = np.random.choice(len(element), len(element), False)
            data = element.data[inds, :]
            randomized = element.clone(data)
            sliced = randomized.select(**{xdim: (xstart, xend),
                                          ydim: (ystart, yend)})

        sliced = sliced.clone(sliced.data[:self.max_samples, :])

        # Update data source
        new_data = plot.get_data(sliced, ranges)[0]
        source = plot.handles['source']
        source.data.update(new_data)
        return [source]


class Callbacks(param.Parameterized):
    """
    Callbacks allows defining a number of callbacks to be applied
    to a plot. Callbacks should
    """

    selection = param.ClassSelector(class_=(CustomJS, Callback, list), doc="""
        Callback that gets triggered when user applies a selection to a
        data source.""")

    ranges = param.ClassSelector(class_=(CustomJS, Callback, list), doc="""
        Callback applied to plot x_range and y_range, data will
        supply 'x_range' and 'y_range' lists of the form [low, high].""")

    x_range = param.ClassSelector(class_=(CustomJS, Callback, list), doc="""
        Callback applied to plot x_range, data will supply
        'x_range' as a list of the form [low, high].""")

    y_range = param.ClassSelector(class_=(CustomJS, Callback, list), doc="""
        Callback applied to plot x_range, data will supply
        'y_range' as a list of the form [low, high].""")

    tap = param.ClassSelector(class_=(CustomJS, Callback, list), doc="""
        Callback that gets triggered when user clicks on a glyph.""")

    callbacks = {}

    plot_callbacks = defaultdict(list)

    def initialize_callback(self, cb_obj, plot, pycallback):
        """
        Initialize the callback with the appropriate data
        and javascript, execute once and return bokeh CustomJS
        object to be installed on the appropriate plot object.
        """
        if pycallback.reinitialize:
            pycallback = pycallback.instance()
        pycallback.callback_obj = cb_obj
        pycallback.plots.append(plot)

        # Register the callback to allow calling it from JS
        cb_id = id(pycallback)
        self.callbacks[cb_id] = pycallback
        self.plot_callbacks[id(cb_obj)].append(pycallback)

        # Generate callback JS code to get all the requested data
        self_callback = Callback.IPython_callback.format(callback_id=cb_id)
        data, code = {}, ''
        for k, v in pycallback.plot_attributes.items():
            format_kwargs = dict(key=repr(k), attrs=repr(v))
            if v is None:
                code += "data[{key}] = plot.get({key});\n".format(**format_kwargs)
                data[k] = plot.state.vm_props().get(k)
            else:
                code += "data[{key}] = {attrs}.map(function(attr) {{" \
                        "  return plot.get({key}).get(attr)" \
                        "}})\n".format(**format_kwargs)
                data[k] = [plot.state.vm_props().get(k).vm_props().get(attr)
                           for attr in v]
        if pycallback.cb_attributes:
            code += "data['cb_obj'] = {attrs}.map(function(attr) {{"\
                    "  return cb_obj.get(attr)}});\n".format(attrs=repr(pycallback.cb_attributes))
            data['cb_obj'] = [pycallback.callback_obj.vm_props().get(attr)
                              for attr in pycallback.cb_attributes]
        code = Callback.JS_callback + code + pycallback.code + self_callback

        # Generate CustomJS object
        customjs = CustomJS(args=plot.handles, code=code)

        # Get initial callback data and call to initialize
        if pycallback.initialize_cb:
            pycallback.initialize(data)

        return customjs, pycallback


    def _chain_callbacks(self, plot, cb_obj, callbacks):
        """
        Initializes new callbacks and chains them to
        existing callbacks, allowing multiple callbacks
        on the same plot object.
        """
        other_callbacks = self.plot_callbacks[id(cb_obj)]
        chain_callback = other_callbacks[-1] if other_callbacks else None
        if not isinstance(callbacks, list): callbacks = [callbacks]
        for callback in callbacks:
            if isinstance(callback, Callback):
                jscb, pycb = self.initialize_callback(cb_obj, plot, callback)
                if chain_callback and pycb is not chain_callback:
                    chain_callback.streams.append(pycb)
                    chain_callback = pycb
                else:
                    cb_obj.callback = jscb
                    chain_callback = pycb
            else:
                cb_obj.callback = callback


    def __call__(self, plot):
        """
        Initialize callbacks, chaining them as necessary
        and setting them on the appropriate plot object.
        """
        # Initialize range callbacks
        xrange_cb = self.ranges if self.ranges else self.x_range
        yrange_cb = self.ranges if self.ranges else self.y_range
        if xrange_cb:
            self._chain_callbacks(plot, plot.state.x_range, xrange_cb)
        if yrange_cb:
            self._chain_callbacks(plot, plot.state.y_range, yrange_cb)

        if self.tap:
            for tool in plot.state.select(type=TapTool):
                self._chain_callbacks(plot, tool, self.tap)

        if self.selection:
            for tool in plot.state.select(type=(ColumnDataSource)):
                self._chain_callbacks(plot, tool, self.selection)
