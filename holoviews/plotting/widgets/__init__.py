from __future__ import unicode_literals

import os, uuid, json, math

import param
import numpy as np

from ...core import OrderedDict, NdMapping
from ...core.options import Store
from ...core.ndmapping import item_check
from ...core.util import (
    dimension_sanitizer, bytes_to_unicode, unique_array, unicode,
    isnumeric, cross_index, wrap_tuple_streams, drop_streams
)
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
        if isinstance(v, np.timedelta64):
            v = "'"+str(v)+"'"
        elif isinstance(v, np.datetime64):
            v = "'"+str(v.astype('datetime64[ns]'))+"'"
        elif not isnumeric(v):
            v = "'"+unicode(bytes_to_unicode(v))+"'"
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

    export_json = param.Boolean(default=False, doc="""
         Whether to export plots as JSON files, which can be
         dynamically loaded through a callback from the slider.""")

    json_save_path = param.String(default='./json_figures', doc="""
         If export_json is enabled the widget will save the JSON
         data to this path. If None data will be accessible via the
         json_data attribute.""")

    json_load_path = param.String(default=None, doc="""
         If export_json is enabled the widget JS code will load the data
         from this path, if None defaults to json_save_path. For loading
         the data from within the notebook the path must be relative,
         when exporting the notebook the path can be set to another
         location like a webserver where the JSON files can be uploaded to.""")

    ##############################
    # Javascript include options #
    ##############################

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
        self.plot_id = plot.id
        streams = []
        for stream in plot.streams:
            if any(k in plot.dimensions for k in stream.contents):
                streams.append(stream)

        keys = plot.keys[:1] if self.plot.dynamic else plot.keys
        self.dimensions, self.keys = drop_streams(streams,
                                                  plot.dimensions,
                                                  keys)
        defaults = [kd.default for kd in self.dimensions]
        self.init_key = tuple(v if d is None else d for v, d in
                              zip(self.keys[0], defaults))

        self.json_data = {}
        if self.plot.dynamic: self.embed = False
        if renderer is None:
            backend = Store.current_backend
            self.renderer = Store.renderers[backend]
        else:
            self.renderer = renderer

        # Create mock NdMapping to hold the common dimensions and keys
        sorted_dims = []
        for dim in self.dimensions:
            if dim.values and all(isnumeric(v) for v in dim.values):
                dim = dim.clone(values=sorted(dim.values))
            sorted_dims.append(dim)

        if self.plot.dynamic:
            self.length = np.product([len(d.values) for d in sorted_dims if d.values])
        else:
            self.length = len(self.plot)

        with item_check(False):
            self.mock_obj = NdMapping([(k, None) for k in self.keys],
                                      kdims=sorted_dims, sort=False)

        NdWidget.widgets[self.id] = self

        # Set up jinja2 templating
        import jinja2
        templateLoader = jinja2.FileSystemLoader(subdirs)
        self.jinjaEnv = jinja2.Environment(loader=templateLoader)
        if not self.embed:
            comm_manager = self.renderer.comm_manager
            self.comm = comm_manager.get_client_comm(id=self.id+'_client',
                                                     on_msg=self._process_update)


    def cleanup(self):
        self.plot.cleanup()
        del NdWidget.widgets[self.id]


    def _process_update(self, msg):
        if 'content' not in msg:
            raise ValueError('Received widget comm message has no content.')
        self.update(msg['content'])


    def __call__(self, as_script=False):
        data = self._get_data()
        html = self.render_html(data)
        js = self.render_js(data)
        if as_script:
            return js, html
        js = '<script type="text/javascript">%s</script>' % js
        html = '\n'.join([html, js])
        return html


    def _get_data(self):
        delay = int(1000./self.display_options.get('fps', 5))
        CDN = {}
        for name, resources in self.plot.renderer.core_dependencies.items():
            if 'js' in resources:
                CDN[name] = resources['js'][0]
        for name, resources in self.plot.renderer.extra_dependencies.items():
            if 'js' in resources:
                CDN[name] = resources['js'][0]
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
                    Nframes=self.length, widget_name=name, json_path=json_path,
                    dynamic=dynamic, plot_id=self.plot_id)


    def render_html(self, data):
        template = self.jinjaEnv.get_template(self.html_template)
        return template.render(**data)


    def render_js(self, data):
        template = self.jinjaEnv.get_template(self.js_template)
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
            return self.renderer.html(self.plot, figure_format, css=css)


    def update(self, key):
        pass




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

    html_template = param.String('htmlscrubber.jinja', doc="""
        The jinja2 template used to generate the html output.""")

    js_template = param.String('jsscrubber.jinja', doc="""
        The jinja2 template used to generate the html output.""")

    def update(self, key):
        if not self.plot.dimensions:
            self.plot.refresh()
        else:
            if self.plot.dynamic:
                key = cross_index([d.values for d in self.mock_obj.kdims], key)
            self.plot.update(key)
            self.plot.push()


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
    including png and svg output..

    Just like the ScrubberWidget the data can be optionally saved
    to json and dynamically loaded from a server.
    """

    css = param.String(default='jsslider.css', doc="""
        Defines the local CSS file to be loaded for this widget.""")

    html_template = param.String('htmlslider.jinja', doc="""
        The jinja2 template used to generate the html output.""")

    js_template = param.String('jsslider.jinja', doc="""
        The jinja2 template used to generate the html output.""")

    ##############################
    # Javascript include options #
    ##############################

    throttle = {True: 0, False: 100}

    def get_widgets(self):
        # Generate widget data
        widgets, dimensions, init_dim_vals = [], [], []
        if self.plot.dynamic:
            hierarchy = None
        else:
            hierarchy = hierarchical(list(self.mock_obj.data.keys()))
        for idx, dim in enumerate(self.mock_obj.kdims):
            # Hide widget if it has 1-to-1 mapping to next widget
            if self.plot.dynamic:
                widget_data = self._get_dynamic_widget(idx, dim)
            else:
                widget_data = self._get_static_widget(idx, dim, self.mock_obj, hierarchy,
                                                      init_dim_vals)
            init_dim_vals.append(widget_data['init_val'])
            visibility = '' if widget_data.get('visible', True) else 'display: none'
            dim_str = dim.pprint_label
            escaped_dim = dimension_sanitizer(dim_str)
            widget_data = dict(widget_data, dim=escaped_dim, dim_label=dim_str,
                               dim_idx=idx, visibility=visibility)
            widgets.append(widget_data)
            dimensions.append(escaped_dim)
        init_dim_vals = escape_list(escape_vals(init_dim_vals, not self.plot.dynamic))
        return widgets, dimensions, init_dim_vals


    @classmethod
    def _get_static_widget(cls, idx, dim, mock_obj, hierarchy, init_dim_vals):
        next_dim = ''
        next_vals = {}
        visible = True
        if next_vals:
            dim_vals = next_vals[init_dim_vals[idx-1]]
        else:
            dim_vals = (list(dim.values) if dim.values else
                        list(unique_array(mock_obj.dimension_values(dim.name))))
            visible = visible and len(dim_vals) > 1

        if idx < mock_obj.ndims-1:
            next_vals = hierarchy[idx]
            next_dim = bytes_to_unicode(mock_obj.kdims[idx+1])
        else:
            next_vals = {}

        value_labels = escape_list(escape_vals([dim.pprint_value(v)
                                                for v in dim_vals]))

        if isinstance(dim_vals[0], np.datetime64):
            dim_vals = sorted([str(v.astype('datetime64[ns]')) for v in dim_vals])
            widget_type = 'slider'
        elif isnumeric(dim_vals[0]):
            dim_vals = sorted([round(v, 10) for v in dim_vals])
            if next_vals:
                next_vals = {round(k, 10): [round(v, 10) if isnumeric(v) else v
                                            for v in vals]
                             for k, vals in next_vals.items()}
            widget_type = 'slider'
        else:
            next_vals = dict(next_vals)
            widget_type = 'dropdown'

        if dim.default is None:
            default = 0
            init_val = dim_vals[0];
        elif dim.default not in dim_vals:
            raise ValueError("%s dimension default %r is not in dimension values: %s"
                             % (dim, dim.default, dim.values))
        else:
            default = repr(dim_vals.index(dim.default))
            init_val = dim.default

        dim_vals = escape_list(escape_vals(dim_vals))
        next_vals = escape_dict({k: escape_vals(v) for k, v in next_vals.items()})
        return {'type': widget_type, 'vals': dim_vals, 'labels': value_labels,
                'step': 1, 'default': default, 'next_vals': next_vals,
                'next_dim': next_dim or None, 'init_val': init_val, 'visible': visible}


    @classmethod
    def _get_dynamic_widget(cls, idx, dim):
        step = 1
        if dim.values:
            if all(isnumeric(v) for v in dim.values):
                # Widgets currently detect dynamic mode by type
                # this value representation is now redundant
                # and should be removed in a refactor
                values = dim.values
                dim_vals = {i: i for i, v in enumerate(values)}
                widget_type = 'slider'
            else:
                values = list(dim.values)
                dim_vals = list(range(len(values)))
                widget_type = 'dropdown'

            value_labels = escape_list(escape_vals([dim.pprint_value(v)
                                                    for v in values]))

            if dim.default is None:
                default = dim_vals[0]
            elif widget_type == 'slider':
                default = values.index(dim.default)
            else:
                default = repr(values.index(dim.default))
            init_val = default
        else:
            widget_type = 'slider'
            value_labels = []
            dim_vals = [dim.soft_range[0] if dim.soft_range[0] else dim.range[0],
                        dim.soft_range[1] if dim.soft_range[1] else dim.range[1]]
            dim_range = dim_vals[1] - dim_vals[0]
            int_type = isinstance(dim.type, type) and issubclass(dim.type, int)
            if dim.step is not None:
                step = dim.step
            elif isinstance(dim_range, int) or int_type:
                step = 1
            else:
                step = 10**(round(math.log10(dim_range))-3)

            if dim.default is None:
                default = dim_vals[0]
            else:
                default = dim.default
            init_val = default
            dim_vals = escape_list(escape_vals(sorted(dim_vals)))
        return {'type': widget_type, 'vals': dim_vals, 'labels': value_labels,
                'step': step, 'default': default, 'next_vals': {},
                'next_dim': None, 'init_val': init_val}


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
