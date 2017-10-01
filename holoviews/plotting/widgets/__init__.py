from __future__ import unicode_literals

import os, uuid, json, math

import param
import numpy as np

from ...core import OrderedDict, NdMapping
from ...core.options import Store
from ...core.util import (dimension_sanitizer, bytes_to_unicode,
                          unique_array, unicode, isnumeric,
                          wrap_tuple_streams, drop_streams)
from ...core.traversal import hierarchical

def escape_vals(vals, escape_numerics=True):
    """
    Escapes a list of values to a string, converting to
    unicode for safety.
    """
    # Ints formatted as floats to disambiguate with counter mode
    ints, floats = "%.1f", "%.10f"

    escaped = []
    for v in vals:
        if not isnumeric(v):
            v = "'"+unicode(bytes_to_unicode(v))+"'"
        elif isinstance(v, (np.datetime64, np.timedelta64)):
            v = "'"+str(v)+"'"
        else:
            if v % 1 == 0:
                v = ints % v
            else:
                v = (floats % v)[:-1]
            if escape_numerics:
                v = "'"+v+"'"
        escaped.append(v)
    return escaped

def escape_tuple(vals):
    return "(" + ", ".join(vals) + (",)" if len(vals) == 1 else ")")

def escape_list(vals):
    return "[" + ", ".join(vals) + "]"

def escape_dict(vals):
    vals = [': '.join([k, escape_list(v)]) for k, v in
            zip(escape_vals(vals.keys()), vals.values())]
    return "{" + ", ".join(vals) + "}"


subdirs = [p[0] for p in os.walk(os.path.join(os.path.split(__file__)[0], '..'))]

class NdWidget(param.Parameterized):
    """
    NdWidget is an abstract base class implementing a method to find
    the dimensions and keys of any ViewableElement, GridSpace or
    UniformNdMapping type.  In the process it creates a mock_obj to
    hold the dimensions and keys.
    """

    display_options = param.Dict(default={}, doc="""
        The display options used to generate individual frames""")

    embed = param.Boolean(default=True, doc="""
        Whether to embed all plots in the Javascript, generating
        a static widget not dependent on the IPython server.""")

    #######################
    # JSON export options #
    #######################

    export_json = param.Boolean(default=False, doc="""Whether to export
         plots as json files, which can be dynamically loaded through
         a callback from the slider.""")

    json_save_path = param.String(default='./json_figures', doc="""
         If export_json is enabled the widget will save the json
         data to this path. If None data will be accessible via the
         json_data attribute.""")

    json_load_path = param.String(default=None, doc="""
         If export_json is enabled the widget JS code will load the data
         from this path, if None defaults to json_save_path. For loading
         the data from within the notebook the path must be relative,
         when exporting the notebook the path can be set to another
         location like a webserver where the json files can be uploaded to.""")

    ##############################
    # Javascript include options #
    ##############################

    CDN = param.Dict(default={'underscore': 'https://cdnjs.cloudflare.com/ajax/libs/underscore.js/1.8.3/underscore-min.js',
                              'jQueryUI':   'https://code.jquery.com/ui/1.10.4/jquery-ui.min.js'})

    css = param.String(default=None, doc="""
        Defines the local CSS file to be loaded for this widget.""")

    basejs = param.String(default='widgets.js', doc="""
        JS file containing javascript baseclasses for the widget.""")

    extensionjs = param.String(default=None, doc="""
        Optional javascript extension file for a particular backend.""")

    widgets = {}
    counter = 0

    def __init__(self, plot, renderer=None, **params):
        super(NdWidget, self).__init__(**params)
        self.id = plot.comm.id if plot.comm else uuid.uuid4().hex
        self.plot = plot
        streams = []
        for stream in plot.streams:
            if any(k in plot.dimensions for k in stream.contents):
                streams.append(stream)
        self.dimensions, self.keys = drop_streams(streams,
                                                  plot.dimensions,
                                                  plot.keys)

        self.json_data = {}
        if self.plot.dynamic: self.embed = False
        if renderer is None:
            backend = Store.current_backend
            self.renderer = Store.renderers[backend]
        else:
            self.renderer = renderer
        # Create mock NdMapping to hold the common dimensions and keys
        self.mock_obj = NdMapping([(k, None) for k in self.keys],
                                  kdims=list(self.dimensions), sort=False)

        NdWidget.widgets[self.id] = self

        # Set up jinja2 templating
        import jinja2
        templateLoader = jinja2.FileSystemLoader(subdirs)
        self.jinjaEnv = jinja2.Environment(loader=templateLoader)


    def __call__(self):
        return self.render_html(self._get_data())


    def _get_data(self):
        delay = int(1000./self.display_options.get('fps', 5))
        CDN = {k: v[:-3] for k, v in self.CDN.items()}
        template = self.jinjaEnv.get_template(self.base_template)
        name = type(self).__name__
        cached = str(self.embed).lower()
        load_json = str(self.export_json).lower()
        mode = str(self.renderer.mode)
        json_path = (self.json_save_path if self.json_load_path is None
                     else self.json_load_path)
        if json_path and json_path[-1] != '/':
            json_path = json_path + '/'
        dynamic = json.dumps(self.plot.dynamic) if self.plot.dynamic else 'false'
        return dict(CDN=CDN, frames=self.get_frames(), delay=delay,
                    cached=cached, load_json=load_json, mode=mode, id=self.id,
                    Nframes=len(self.plot), widget_name=name, json_path=json_path,
                    widget_template=template, dynamic=dynamic)


    def render_html(self, data):
        template = self.jinjaEnv.get_template(self.template)
        return template.render(**data)


    def get_frames(self):
        if self.embed:
            frames = OrderedDict([(idx, self._plot_figure(idx))
                                  for idx in range(len(self.plot))])
        else:
            frames = {}
        return self.encode_frames(frames)


    def encode_frames(self, frames):
        if isinstance(frames, dict):
            frames = dict(frames)
        return json.dumps(frames)

    def save_json(self, frames):
        """
        Saves frames data into a json file at the
        specified json_path, named with the widget uuid.
        """
        if self.json_save_path is None: return
        path = os.path.join(self.json_save_path, '%s.json' % self.id)
        if not os.path.isdir(self.json_save_path):
            os.mkdir(self.json_save_path)
        with open(path, 'w') as f:
            json.dump(frames, f)
        self.json_data = frames

    def _plot_figure(self, idx):
        with self.renderer.state():
            self.plot.update(idx)
            css = self.display_options.get('css', {})
            figure_format = self.display_options.get('figure_format',
                                                     self.renderer.fig)
            return self.renderer.html(self.plot, figure_format, css=css,
                                      comm=False)


    def update(self, key):
        if not self.plot.dimensions:
            self.plot.refresh()
        else:
            self.plot.update(key)
            self.plot.push()
        return 'Complete'



class ScrubberWidget(NdWidget):
    """
    ScrubberWidget generates a basic animation widget with a slider
    and various play/rewind/stepping options. It has been adapted
    from Jake Vanderplas' JSAnimation library, which was released
    under BSD license.

    Optionally the individual plots can be exported to json, which can
    be dynamically loaded by serving the data the data for each frame
    on a simple server.
    """

    base_template = param.String('jsscrubber.jinja', doc="""
        The jinja2 template used to generate the html output.""")

    template = param.String('jsscrubber.jinja', doc="""
        The jinja2 template used to generate the html output.""")



class SelectionWidget(NdWidget):
    """
    Javascript based widget to select and view ViewableElement objects
    contained in an NdMapping. For each dimension in the NdMapping a
    slider or dropdown selection widget is created and can be used to
    select the html output associated with the selected
    ViewableElement type. The widget maybe set to embed all frames in
    the supplied object into the rendered html or to dynamically
    update the widget with a live IPython kernel.

    The widget supports all current HoloViews figure backends
    including png, svg and nbagg output. To select nbagg output,
    the SelectionWidget must not be set to embed.

    Just like the ScrubberWidget the data can be optionally saved
    to json and dynamically loaded from a server.
    """

    base_template = param.String('jsslider.jinja', doc="""
        The jinja2 template used to generate the html output.""")

    css = param.String(default='jsslider.css', doc="""
        Defines the local CSS file to be loaded for this widget.""")

    template = param.String('jsslider.jinja', doc="""
        The jinja2 template used to generate the html output.""")

    ##############################
    # Javascript include options #
    ##############################

    throttle = {True: 0, False: 100}

    def get_widgets(self):
        # Generate widget data
        step = 1
        widgets, dimensions, init_dim_vals = [], [], []
        hierarchy = hierarchical(list(self.mock_obj.data.keys()))
        for idx, dim in enumerate(self.mock_obj.kdims):
            next_dim = ''
            next_vals = {}

            # Hide widget if it has 1-to-1 mapping to next widget
            visible = True
            if self.plot.dynamic:
                if dim.values:
                    if all(isnumeric(v) for v in dim.values):
                        # Widgets currently detect dynamic mode by type
                        # this value representation is now redundant
                        # and should be removed in a refactor
                        dim_vals = {i: i for i, v in enumerate(dim.values)}
                        widget_type = 'slider'
                        value_labels = escape_list(escape_vals([dim.pprint_value(v)
                                                                for v in dim.values]))
                    else:
                        dim_vals = list(range(len(dim.values)))
                        value_labels = escape_list(escape_vals([dim.pprint_value(v)
                                                                for v in dim.values]))
                        widget_type = 'dropdown'
                    init_dim_vals.append(dim_vals[0])
                else:
                    widget_type = 'slider'
                    value_labels = []
                    dim_vals = [dim.soft_range[0] if dim.soft_range[0] else dim.range[0],
                                dim.soft_range[1] if dim.soft_range[1] else dim.range[1]]
                    dim_range = dim_vals[1] - dim_vals[0]
                    int_type = isinstance(dim.type, type) and issubclass(dim.type, int)
                    if isinstance(dim_range, int) or int_type:
                        step = 1
                    elif dim.step is not None:
                        step = dim.step
                    else:
                        step = 10**(round(math.log10(dim_range))-3)
                    init_dim_vals.append(dim_vals[0])
                    dim_vals = escape_list(escape_vals(dim_vals))
            else:
                if next_vals:
                    dim_vals = next_vals[init_dim_vals[idx-1]]
                else:
                    dim_vals = (dim.values if dim.values else
                                list(unique_array(self.mock_obj.dimension_values(dim.name))))
                    visible = visible and len(dim_vals) > 1

                if idx < self.mock_obj.ndims-1:
                    next_vals = hierarchy[idx]
                    next_dim = bytes_to_unicode(self.mock_obj.kdims[idx+1])
                else:
                    next_vals = {}

                value_labels = escape_list(escape_vals([dim.pprint_value(v)
                                                        for v in dim_vals]))

                if isinstance(dim_vals[0], np.datetime64):
                    dim_vals = [str(v) for v in dim_vals]
                    widget_type = 'slider'
                elif isnumeric(dim_vals[0]):
                    dim_vals = [round(v, 10) for v in dim_vals]
                    if next_vals:
                        next_vals = {round(k, 10): [round(v, 10) if isnumeric(v) else v
                                                    for v in vals]
                                     for k, vals in next_vals.items()}
                    widget_type = 'slider'
                else:
                    next_vals = dict(next_vals)
                    widget_type = 'dropdown'
                init_dim_vals.append(dim_vals[0])
                dim_vals = escape_list(escape_vals(dim_vals))
                next_vals = escape_dict({k: escape_vals(v) for k, v in next_vals.items()})

            visibility = '' if visible else 'display: none'
            dim_str = dim.pprint_label
            escaped_dim = dimension_sanitizer(dim_str)
            widget_data = dict(dim=escaped_dim, dim_label=dim_str,
                               dim_idx=idx, vals=dim_vals, type=widget_type,
                               visibility=visibility, step=step, next_dim=next_dim,
                               next_vals=next_vals, labels=value_labels)

            widgets.append(widget_data)
            dimensions.append(escaped_dim)
        init_dim_vals = escape_list(escape_vals(init_dim_vals, not self.plot.dynamic))
        return widgets, dimensions, init_dim_vals


    def get_key_data(self):
        # Generate key data
        key_data = OrderedDict()
        for i, k in enumerate(self.mock_obj.data.keys()):
            key = escape_tuple(escape_vals(k))
            key_data[key] = i
        return json.dumps(key_data)


    def _get_data(self):
        data = super(SelectionWidget, self)._get_data()
        widgets, dimensions, init_dim_vals = self.get_widgets()
        key_data = {} if self.plot.dynamic else self.get_key_data()
        notfound_msg = "<h2 style='vertical-align: middle>No frame at selected dimension value.<h2>"
        throttle = self.throttle[self.embed]
        return dict(data, Nframes=len(self.mock_obj),
                    Nwidget=self.mock_obj.ndims,
                    dimensions=dimensions, key_data=key_data,
                    widgets=widgets, init_dim_vals=init_dim_vals,
                    throttle=throttle, notFound=notfound_msg)


    def update(self, key):
        if self.plot.dynamic:
            key = tuple(dim.values[k] if dim.values else k
                        for dim, k in zip(self.mock_obj.kdims, tuple(key)))
            key = [key[self.dimensions.index(kdim)] if kdim in self.dimensions else None
                   for kdim in self.plot.dimensions]
            key = wrap_tuple_streams(tuple(key), self.plot.dimensions,
                                     self.plot.streams)
        self.plot.update(key)
        self.plot.push()
        return 'Complete'
