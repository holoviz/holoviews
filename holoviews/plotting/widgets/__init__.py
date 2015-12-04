import os, uuid, json, math

import param

import numpy as np
from ...core import OrderedDict, NdMapping
from ...core.options import Store
from ...core.util import (dimension_sanitizer, safe_unicode, basestring,
                          unique_iterator)
from ...core.traversal import hierarchical

def isnumeric(val):
    if isinstance(val, basestring):
        return False
    try:
        float(val)
        return True
    except:
        return False

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
         from this relative path, if None defaults to json_save_path.""")

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
        self.id = uuid.uuid4().hex
        self.plot = plot
        self.dimensions = plot.dimensions
        self.keys = plot.keys

        self.json_data = {}
        if self.plot.dynamic: self.embed = False
        if renderer is None:
            backend = Store.current_backend
            self.renderer = Store.renderers[backend]
        else:
            self.renderer = renderer
        # Create mock NdMapping to hold the common dimensions and keys
        self.mock_obj = NdMapping([(k, None) for k in self.keys],
                                  kdims=self.dimensions)

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
        mode = repr(self.renderer.mode)
        json_path = (self.json_save_path if self.json_load_path is None
                     else self.json_load_path)
        dynamic = repr(self.plot.dynamic) if self.plot.dynamic else 'false'
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
            frames = {0: self._plot_figure(0)}
        return self.encode_frames(frames)


    def encode_frames(self, frames):
        if isinstance(frames, dict):
            frames = {idx: frame for idx, frame in frames.items()}
        return frames

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
        return self._plot_figure(key)



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
    including png, svg, mpld3 and nbagg output. To select nbagg
    output, the SelectionWidget must not be set to embed.

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
        widgets = []
        dimensions = []
        init_dim_vals = []
        hierarchy = hierarchical(list(self.mock_obj.data.keys()))
        next_vals = {}
        for idx, dim in enumerate(self.mock_obj.kdims):
            step = 1
            next_dim = ''
            if self.plot.dynamic:
                if dim.values:
                    if all(isnumeric(v) for v in dim.values):
                        dim_vals = {i: v for i, v in enumerate(dim.values)}
                        widget_type = 'slider'
                    else:
                        dim_vals = dim.values
                        widget_type = 'dropdown'
                else:
                    dim_vals = list(dim.range)
                    int_type = isinstance(dim.type, type) and issubclass(dim.type, int)
                    widget_type = 'slider'
                    dim_range = dim_vals[1] - dim_vals[0]
                    if not isinstance(dim_range, int) or int_type:
                        step = 10**(round(math.log10(dim_range))-3)
                init_dim_vals.append(dim_vals[0])
            else:
                if next_vals:
                    dim_vals = next_vals[init_dim_vals[idx-1]]
                else:
                    dim_vals = (dim.values if dim.values else
                                list(unique_iterator(self.mock_obj.dimension_values(dim.name))))
                if idx < self.mock_obj.ndims-1:
                    next_vals = hierarchy[idx]
                    next_dim = safe_unicode(self.mock_obj.kdims[idx+1])
                else:
                    next_vals = {}
                if isnumeric(dim_vals[0]):
                    dim_vals = [round(v, 10) for v in dim_vals]
                    if next_vals:
                        next_vals = {round(k, 10): [round(v, 10) if isnumeric(v) else v for v in vals]
                                     for k, vals in next_vals.items()}
                    widget_type = 'slider'
                else:
                    next_vals = dict(next_vals)
                    widget_type = 'dropdown'
                init_dim_vals.append(dim_vals[0])
                dim_vals = repr([v for v in dim_vals if v is not None])
            dim_str = safe_unicode(dim.name)
            visibility = 'visibility: visible' if len(dim_vals) > 1 else 'visibility: hidden; height: 0;'
            widget_data = dict(dim=dimension_sanitizer(dim_str), dim_label=dim_str,
                               dim_idx=idx, vals=dim_vals, type=widget_type,
                               visibility=visibility, step=step, next_dim=next_dim,
                               next_vals=next_vals)
            widgets.append(widget_data)
            dimensions.append(dim_str)
        return widgets, dimensions, init_dim_vals


    def get_key_data(self):
        # Generate key data
        key_data = OrderedDict()
        for i, k in enumerate(self.mock_obj.data.keys()):
            key = [("%.1f" % v if v % 1 == 0 else "%.10f" % v)
                   if isnumeric(v) else v for v in k]
            key = str(tuple(key))
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
        if self.plot.dynamic: key = tuple(key)
        return self._plot_figure(key)
