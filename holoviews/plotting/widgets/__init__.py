import os, uuid, json

import param

from ...core import OrderedDict, NdMapping
from ...core.util import sanitize_identifier, safe_unicode, basestring

def isnumeric(val):
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

    json_path = param.String(default='./json_figures', doc="""
         If export_json is True the json files will be written to this
         directory.""")

    server_url = param.String(default='', doc="""If export_json is
         True the slider widget will expect to be served the plot data
         from this URL. Data should be served from:
         server_url/fig_{id}/{frame}.""")

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
        if renderer is None:
            self.renderer = plot.renderer.instance(dpi=self.display_options.get('dpi', 72))
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
        return dict(CDN=CDN, frames=self.get_frames(), delay=delay,
                    server=self.server_url, cached=cached,
                    load_json=load_json, mode=mode, id=self.id,
                    Nframes=len(self.plot), widget_name=name,
                    widget_template=template)


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
        if self.export_json:
            if not os.path.isdir(self.json_path):
                os.mkdir(self.json_path)
            with open(self.json_path+'/fig_%s.json' % self.id, 'wb') as f:
                json.dump(frames, f)
            frames = {}
        return frames


    def _plot_figure(self, idx):
        with self.renderer.state():
            self.plot.update(idx)
            css = self.display_options.get('css', {})
            figure_format = self.display_options.get('figure_format',
                                                     self.renderer.fig)
            return self.renderer.html(self.plot, figure_format, css=css)


    def update(self, n):
        return self._plot_figure(n)



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
        for idx, dim in enumerate(self.mock_obj.kdims):
            dim_vals = dim.values if dim.values else sorted(set(self.mock_obj.dimension_values(dim.name)))
            dim_vals = [v for v in dim_vals if v is not None]
            val = dim_vals[0]
            if not isinstance(val, basestring) and isnumeric(val):
                dim_vals = [round(v, 10) for v in dim_vals]
                widget_type = 'slider'
            else:
                widget_type = 'dropdown'
            init_dim_vals.append(val)
            dim_str = safe_unicode(dim.name)
            visibility = 'visibility: visible' if len(dim_vals) > 1 else 'visibility: hidden; height: 0;'
            widgets.append(dict(dim=sanitize_identifier(dim_str), dim_label=dim_str, dim_idx=idx, vals=repr(dim_vals),
                                type=widget_type, visibility=visibility))
            dimensions.append(dim_str)
        return widgets, dimensions, init_dim_vals


    def get_key_data(self):
        # Generate key data
        key_data = OrderedDict()
        for i, k in enumerate(self.mock_obj.data.keys()):
            key = [("%.1f" % v if v % 1 == 0 else "%.10f" % v)
                   if not isinstance(v, basestring) and isnumeric(v) else v
                   for v in k]
            key_data[str(tuple(key))] = i
        return json.dumps(key_data)


    def _get_data(self):
        data = super(SelectionWidget, self)._get_data()
        widgets, dimensions, init_dim_vals = self.get_widgets()
        key_data = self.get_key_data()
        notfound_msg = "<h2 style='vertical-align: middle>No frame at selected dimension value.<h2>"
        throttle = self.throttle[self.embed]
        return dict(data, Nframes=len(self.mock_obj),
                    Nwidget=self.mock_obj.ndims,
                    dimensions=dimensions, key_data=key_data,
                    widgets=widgets, init_dim_vals=init_dim_vals,
                    throttle=throttle, notFound=notfound_msg)

