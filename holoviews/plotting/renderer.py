"""
Public API for all plotting renderers supported by HoloViews,
regardless of plotting package or backend.
"""
from __future__ import unicode_literals

from io import BytesIO
import os, base64
from contextlib import contextmanager

import param
from ..core.io import Exporter
from ..core.options import Store, StoreOptions, SkipRendering, Compositor
from ..core.util import find_file, unicode, unbound_dimensions, basestring
from .. import Layout, HoloMap, AdjointLayout, DynamicMap
from .widgets import NdWidget, ScrubberWidget, SelectionWidget

from . import Plot
from pyviz_comms import CommManager, JupyterCommManager, embed_js
from .util import displayable, collate, initialize_dynamic

from param.parameterized import bothmethod

# Tags used when visual output is to be embedded in HTML
IMAGE_TAG = "<img src='{src}' style='max-width:100%; margin: auto; display: block; {css}'/>"
VIDEO_TAG = """
<video controls style='max-width:100%; margin: auto; display: block; {css}'>
<source src='{src}' type='{mime_type}'>
Your browser does not support the video tag.
</video>"""
PDF_TAG = "<iframe src='{src}' style='width:100%; margin: auto; display: block; {css}'></iframe>"
HTML_TAG = "{src}"

HTML_TAGS = {
    'base64': 'data:{mime_type};base64,{b64}', # Use to embed data
    'svg':  IMAGE_TAG,
    'png':  IMAGE_TAG,
    'gif':  IMAGE_TAG,
    'webm': VIDEO_TAG,
    'mp4':  VIDEO_TAG,
    'pdf':  PDF_TAG,
    'html': HTML_TAG
}

MIME_TYPES = {
    'svg':  'image/svg+xml',
    'png':  'image/png',
    'gif':  'image/gif',
    'webm': 'video/webm',
    'mp4':  'video/mp4',
    'pdf':  'application/pdf',
    'html': 'text/html',
    'json': 'text/json',
    'js':   'application/javascript',
    'jlab-hv-exec': 'application/vnd.holoviews_exec.v0+json',
    'jlab-hv-load': 'application/vnd.holoviews_load.v0+json',
    'server': None
}

static_template = """
<html>
  <head>
    {css}
    {js}
  </head>
  <body>
    {html}
  </body>
</html>
"""

class Renderer(Exporter):
    """
    The job of a Renderer is to turn the plotting state held within
    Plot classes into concrete, visual output in the form of the PNG,
    SVG, MP4 or WebM formats (among others). Note that a Renderer is a
    type of Exporter and must therefore follow the Exporter interface.

    The Renderer needs to be able to use the .state property of the
    appropriate Plot classes associated with that renderer in order to
    generate output. The process of 'drawing' is execute by the Plots
    and the Renderer turns the final plotting state into output.
    """

    backend = param.String(doc="""
        The full, lowercase name of the rendering backend or third
        part plotting package used e.g 'matplotlib' or 'cairo'.""")

    dpi=param.Integer(None, doc="""
        The render resolution in dpi (dots per inch)""")

    fig = param.ObjectSelector(default='auto', objects=['auto'], doc="""
        Output render format for static figures. If None, no figure
        rendering will occur. """)

    fps=param.Number(20, doc="""
        Rendered fps (frames per second) for animated formats.""")

    holomap = param.ObjectSelector(default='auto',
                                   objects=['scrubber','widgets', None, 'auto'], doc="""
        Output render multi-frame (typically animated) format. If
        None, no multi-frame rendering will occur.""")

    mode = param.ObjectSelector(default='default', objects=['default'], doc="""
         The available rendering modes. As a minimum, the 'default'
         mode must be supported.""")

    size=param.Integer(100, doc="""
        The rendered size as a percentage size""")

    widget_mode = param.ObjectSelector(default='embed', objects=['embed', 'live'], doc="""
        The widget mode determining whether frames are embedded or generated
        'live' when interacting with the widget.""")

    css = param.Dict(default={},
                     doc="Dictionary of CSS attributes and values to apply to HTML output")

    info_fn = param.Callable(None, allow_None=True, constant=True,  doc="""
        Renderers do not support the saving of object info metadata""")

    key_fn = param.Callable(None, allow_None=True, constant=True,  doc="""
        Renderers do not support the saving of object key metadata""")

    post_render_hooks = param.Dict(default={'svg':[], 'png':[]}, doc="""
       Optional dictionary of hooks that are applied to the rendered
       data (according to the output format) before it is returned.

       Each hook is passed the rendered data and the object that is
       being rendered. These hooks allow post-processing of rendered
       data before output is saved to file or displayed.""")

    # Defines the valid output formats for each mode.
    mode_formats = {'fig': {'default': [None, 'auto']},
                    'holomap': {'default': [None, 'auto']}}

    # The comm_manager handles the creation and registering of client,
    # and server side comms
    comm_manager = CommManager

    # JS code which handles comm messages and updates the plot
    comm_msg_handler = None

    # Define appropriate widget classes
    widgets = {'scrubber': ScrubberWidget, 'widgets': SelectionWidget}

    core_dependencies = {'jQueryUI': {'js': ['https://code.jquery.com/ui/1.10.4/jquery-ui.min.js'],
                                      'css': ['https://code.jquery.com/ui/1.10.4/themes/smoothness/jquery-ui.css']}}

    extra_dependencies = {'jQuery': {'js': ['https://code.jquery.com/jquery-2.1.4.min.js']},
                          'underscore': {'js': ['https://cdnjs.cloudflare.com/ajax/libs/underscore.js/1.8.3/underscore-min.js']},
                          'require': {'js': ['https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.20/require.min.js']},
                          'bootstrap': {'css': ['https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/css/bootstrap.min.css']}}

    # Any additional JS and CSS dependencies required by a specific backend
    backend_dependencies = {}

    # Whether in a notebook context, set when running Renderer.load_nb
    notebook_context = False

    # Plot registry
    _plots = {}

    def __init__(self, **params):
        self.last_plot = None
        super(Renderer, self).__init__(**params)


    @bothmethod
    def get_plot(self_or_cls, obj, renderer=None, **kwargs):
        """
        Given a HoloViews Viewable return a corresponding plot instance.
        """
        if isinstance(obj, DynamicMap) and obj.unbounded:
            dims = ', '.join('%r' % dim for dim in obj.unbounded)
            msg = ('DynamicMap cannot be displayed without explicit indexing '
                   'as {dims} dimension(s) are unbounded. '
                   '\nSet dimensions bounds with the DynamicMap redim.range '
                   'or redim.values methods.')
            raise SkipRendering(msg.format(dims=dims))

        # Initialize DynamicMaps with first data item
        initialize_dynamic(obj)

        if not isinstance(obj, Plot):
            if not displayable(obj):
                obj = collate(obj)
                initialize_dynamic(obj)
            obj = Compositor.map(obj, mode='data', backend=self_or_cls.backend)

        if not renderer:
            renderer = self_or_cls
            if not isinstance(self_or_cls, Renderer):
                renderer = self_or_cls.instance()
        if not isinstance(obj, Plot):
            obj = Layout.from_values(obj) if isinstance(obj, AdjointLayout) else obj
            plot_opts = dict(self_or_cls.plot_options(obj, self_or_cls.size),
                             **kwargs)
            plot = self_or_cls.plotting_class(obj)(obj, renderer=renderer,
                                                   **plot_opts)
            defaults = [kd.default for kd in plot.dimensions]
            init_key = tuple(v if d is None else d for v, d in
                             zip(plot.keys[0], defaults))
            plot.update(init_key)
        else:
            plot = obj
        return plot


    def _validate(self, obj, fmt, **kwargs):
        """
        Helper method to be used in the __call__ method to get a
        suitable plot or widget object and the appropriate format.
        """
        if isinstance(obj, tuple(self.widgets.values())):
            return obj, 'html'
        plot = self.get_plot(obj, renderer=self, **kwargs)

        fig_formats = self.mode_formats['fig'][self.mode]
        holomap_formats = self.mode_formats['holomap'][self.mode]

        if fmt in ['auto', None]:
            if (((len(plot) == 1 and not plot.dynamic)
                or (len(plot) > 1 and self.holomap is None) or
                (plot.dynamic and len(plot.keys[0]) == 0)) or
                not unbound_dimensions(plot.streams, plot.dimensions, no_duplicates=False)):
                fmt = fig_formats[0] if self.fig=='auto' else self.fig
            else:
                fmt = holomap_formats[0] if self.holomap=='auto' else self.holomap

        if fmt in self.widgets:
            plot = self.get_widget(plot, fmt, display_options={'fps': self.fps})
            fmt = 'html'

        all_formats = set(fig_formats + holomap_formats)
        if fmt not in all_formats:
            raise Exception("Format %r not supported by mode %r. Allowed formats: %r"
                            % (fmt, self.mode, fig_formats + holomap_formats))
        self.last_plot = plot
        return plot, fmt


    def __call__(self, obj, fmt=None):
        """
        Render the supplied HoloViews component or plot instance using
        the appropriate backend. The output is not a file format but a
        suitable, in-memory byte stream together with any suitable
        metadata.
        """
        plot, fmt =  self._validate(obj, fmt)
        if plot is None: return
        # [Backend specific code goes here to generate data]
        data = None

        # Example of how post_render_hooks are applied
        data = self._apply_post_render_hooks(data, obj, fmt)
        # Example of the return format where the first value is the rendered data.
        return data, {'file-ext':fmt, 'mime_type':MIME_TYPES[fmt]}


    def _apply_post_render_hooks(self, data, obj, fmt):
        """
        Apply the post-render hooks to the data.
        """
        hooks = self.post_render_hooks.get(fmt,[])
        for hook in hooks:
            try:
                data = hook(data, obj)
            except Exception as e:
                self.warning("The post_render_hook %r could not be applied:\n\n %s"
                             % (hook, e))
        return data


    def html(self, obj, fmt=None, css=None, **kwargs):
        """
        Renders plot or data structure and wraps the output in HTML.
        The comm argument defines whether the HTML output includes
        code to initialize a Comm, if the plot supplies one.
        """
        plot, fmt =  self._validate(obj, fmt)
        figdata, _ = self(plot, fmt, **kwargs)
        if css is None: css = self.css

        if fmt in ['html', 'json']:
            return figdata
        else:
            if fmt == 'svg':
                figdata = figdata.encode("utf-8")
            elif fmt == 'pdf' and 'height' not in css:
                _, h = self.get_size(plot)
                css['height'] = '%dpx' % (h*self.dpi*1.15)

        if isinstance(css, dict):
            css = '; '.join("%s: %s" % (k, v) for k, v in css.items())
        else:
            raise ValueError("CSS must be supplied as Python dictionary")

        b64 = base64.b64encode(figdata).decode("utf-8")
        (mime_type, tag) = MIME_TYPES[fmt], HTML_TAGS[fmt]
        src = HTML_TAGS['base64'].format(mime_type=mime_type, b64=b64)
        html = tag.format(src=src, mime_type=mime_type, css=css)
        return html


    def components(self, obj, fmt=None, comm=True, **kwargs):
        """
        Returns data and metadata dictionaries containing HTML and JS
        components to include render in app, notebook, or standalone
        document. Depending on the backend the fmt defines the format
        embedded in the HTML, e.g. png or svg. If comm is enabled the
        JS code will set up a Websocket comm channel using the
        currently defined CommManager.
        """
        if isinstance(obj, (Plot, NdWidget)):
            plot = obj
        else:
            plot, fmt = self._validate(obj, fmt)

        widget_id = None
        data, metadata = {}, {}
        if isinstance(plot, NdWidget):
            js, html = plot(as_script=True)
            plot_id = plot.plot_id
            widget_id = plot.id
        else:
            html, js = self._figure_data(plot, fmt, as_script=True, **kwargs)
            plot_id = plot.id
            if comm and plot.comm is not None and self.comm_msg_handler:
                msg_handler = self.comm_msg_handler.format(plot_id=plot_id)
                html = plot.comm.html_template.format(init_frame=html,
                                                      plot_id=plot_id)
                comm_js = plot.comm.js_template.format(msg_handler=msg_handler,
                                                       comm_id=plot.comm.id,
                                                       plot_id=plot_id)
                js = '\n'.join([js, comm_js])
            html = "<div id='%s' style='display: table; margin: 0 auto;'>%s</div>" % (plot_id, html)
        if not os.environ.get('HV_DOC_HTML', False) and js is not None:
            js = embed_js.format(widget_id=widget_id, plot_id=plot_id, html=html) + js

        data['text/html'] = html
        if js:
            data[MIME_TYPES['js']] = js
            data[MIME_TYPES['jlab-hv-exec']] = ''
            metadata['id'] = plot_id
            self._plots[plot_id] = plot
        return (data, {MIME_TYPES['jlab-hv-exec']: metadata})


    def static_html(self, obj, fmt=None, template=None):
        """
        Generates a static HTML with the rendered object in the
        supplied format. Allows supplying a template formatting string
        with fields to interpolate 'js', 'css' and the main 'html'.
        """
        js_html, css_html = self.html_assets()
        if template is None: template = static_template
        html = self.html(obj, fmt)
        return template.format(js=js_html, css=css_html, html=html)


    @bothmethod
    def get_widget(self_or_cls, plot, widget_type, **kwargs):
        if not isinstance(plot, Plot):
            plot = self_or_cls.get_plot(plot)
        dynamic = plot.dynamic
        # Whether dimensions define discrete space
        discrete = all(d.values for d in plot.dimensions)
        if widget_type == 'auto':
            isuniform = plot.uniform
            if not isuniform:
                widget_type = 'scrubber'
            else:
                widget_type = 'widgets'
        elif dynamic and not discrete:
            widget_type = 'widgets'

        if widget_type in [None, 'auto']:
            holomap_formats = self_or_cls.mode_formats['holomap'][self_or_cls.mode]
            widget_type = holomap_formats[0] if self_or_cls.holomap=='auto' else self_or_cls.holomap

        widget_cls = self_or_cls.widgets[widget_type]
        renderer = self_or_cls
        if not isinstance(self_or_cls, Renderer):
            renderer = self_or_cls.instance()
        embed = self_or_cls.widget_mode == 'embed'
        return widget_cls(plot, renderer=renderer, embed=embed, **kwargs)


    @bothmethod
    def export_widgets(self_or_cls, obj, filename, fmt=None, template=None,
                       json=False, json_path='', **kwargs):
        """
        Render and export object as a widget to a static HTML
        file. Allows supplying a custom template formatting string
        with fields to interpolate 'js', 'css' and the main 'html'
        containing the widget. Also provides options to export widget
        data to a json file in the supplied json_path (defaults to
        current path).
        """
        if fmt not in list(self_or_cls.widgets.keys())+['auto', None]:
            raise ValueError("Renderer.export_widget may only export "
                             "registered widget types.")

        if not isinstance(obj, NdWidget):
            if not isinstance(filename, BytesIO):
                filedir = os.path.dirname(filename)
                current_path = os.getcwd()
                html_path = os.path.abspath(filedir)
                rel_path = os.path.relpath(html_path, current_path)
                save_path = os.path.join(rel_path, json_path)
            else:
                save_path = json_path
            kwargs['json_save_path'] = save_path
            kwargs['json_load_path'] = json_path
            widget = self_or_cls.get_widget(obj, fmt, **kwargs)
        else:
            widget = obj

        html = self_or_cls.static_html(widget, fmt, template)
        encoded = self_or_cls.encode((html, {'mime_type': 'text/html'}))
        if isinstance(filename, BytesIO):
            filename.write(encoded)
            filename.seek(0)
        else:
            with open(filename, 'wb') as f:
                f.write(encoded)


    @classmethod
    def plotting_class(cls, obj):
        """
        Given an object or Element class, return the suitable plotting
        class needed to render it with the current renderer.
        """
        if isinstance(obj, AdjointLayout) or obj is AdjointLayout:
            obj  = Layout
        if isinstance(obj, type):
            element_type = obj
        else:
            element_type = obj.type if isinstance(obj, HoloMap) else type(obj)
        try:
            plotclass = Store.registry[cls.backend][element_type]
        except KeyError:
            raise SkipRendering("No plotting class for {0} "
                                "found".format(element_type.__name__))
        return plotclass


    @classmethod
    def html_assets(cls, core=True, extras=True, backends=None, script=False):
        """
        Returns JS and CSS and for embedding of widgets.
        """
        if backends is None:
            backends = [cls.backend] if cls.backend else []

        # Get all the widgets and find the set of required js widget files
        widgets = [wdgt for r in [Renderer]+Renderer.__subclasses__()
                   for wdgt in r.widgets.values()]
        css = list({wdgt.css for wdgt in widgets})
        basejs = list({wdgt.basejs for wdgt in widgets})
        extensionjs = list({wdgt.extensionjs for wdgt in widgets})

        # Join all the js widget code into one string
        path = os.path.dirname(os.path.abspath(__file__))

        def open_and_read(path, f):
            with open(find_file(path, f), 'r') as f:
                txt = f.read()
            return txt

        widgetjs = '\n'.join(open_and_read(path, f)
                             for f in basejs + extensionjs if f is not None)
        widgetcss = '\n'.join(open_and_read(path, f)
                              for f in css if f is not None)

        dependencies = {}
        if core:
            dependencies.update(cls.core_dependencies)
        if extras:
            dependencies.update(cls.extra_dependencies)
        for backend in backends:
            dependencies[backend] = Store.renderers[backend].backend_dependencies

        js_html, css_html = '', ''
        for _, dep in sorted(dependencies.items(), key=lambda x: x[0]):
            js_data = dep.get('js', [])
            if isinstance(js_data, tuple):
                for js in js_data:
                    if script:
                        js_html += js
                    else:
                        js_html += '\n<script type="text/javascript">%s</script>' % js
            elif not script:
                for js in js_data:
                    js_html += '\n<script src="%s" type="text/javascript"></script>' % js
            css_data = dep.get('css', [])
            if isinstance(js_data, tuple):
                for css in css_data:
                    css_html += '\n<style>%s</style>' % css
            else:
                for css in css_data:
                    css_html += '\n<link rel="stylesheet" href="%s">' % css
        if script:
            js_html += widgetjs
        else:
            js_html += '\n<script type="text/javascript">%s</script>' % widgetjs
        css_html += '\n<style>%s</style>' % widgetcss

        comm_js = cls.comm_manager.js_manager
        if script:
            js_html += comm_js
        else:
            js_html += '\n<script type="text/javascript">%s</script>' % comm_js

        return unicode(js_html), unicode(css_html)


    @classmethod
    def plot_options(cls, obj, percent_size):
        """
        Given an object and a percentage size (as supplied by the
        %output magic) return all the appropriate plot options that
        would be used to instantiate a plot class for that element.

        Default plot sizes at the plotting class level should be taken
        into account.
        """
        raise NotImplementedError


    @bothmethod
    def save(self_or_cls, obj, basename, fmt='auto', key={}, info={}, options=None, **kwargs):
        """
        Save a HoloViews object to file, either using an explicitly
        supplied format or to the appropriate default.
        """
        if info or key:
            raise Exception('Renderer does not support saving metadata to file.')

        if isinstance(obj, (Plot, NdWidget)):
            plot = obj
        else:
            with StoreOptions.options(obj, options, **kwargs):
                plot = self_or_cls.get_plot(obj)

        if (fmt in list(self_or_cls.widgets.keys())+['auto']) and len(plot) > 1:
            with StoreOptions.options(obj, options, **kwargs):
                if isinstance(basename, basestring):
                    basename = basename+'.html'
                self_or_cls.export_widgets(plot, basename, fmt)
            return

        rendered = self_or_cls(plot, fmt)
        if rendered is None: return
        (data, info) = rendered
        encoded = self_or_cls.encode(rendered)
        prefix = self_or_cls._save_prefix(info['file-ext'])
        if prefix:
            encoded = prefix + encoded
        if isinstance(basename, BytesIO):
            basename.write(encoded)
            basename.seek(0)
        else:
            filename ='%s.%s' % (basename, info['file-ext'])
            with open(filename, 'wb') as f:
                f.write(encoded)

    @bothmethod
    def _save_prefix(self_or_cls, ext):
        "Hook to prefix content for instance JS when saving HTML"
        return


    @bothmethod
    def get_size(self_or_cls, plot):
        """
        Return the display size associated with a plot before
        rendering to any particular format. Used to generate
        appropriate HTML display.

        Returns a tuple of (width, height) in pixels.
        """
        raise NotImplementedError

    @classmethod
    @contextmanager
    def state(cls):
        """
        Context manager to handle global state for a backend,
        allowing Plot classes to temporarily override that state.
        """
        yield


    @classmethod
    def validate(cls, options):
        """
        Validate an options dictionary for the renderer.
        """
        return options


    @classmethod
    def load_nb(cls, inline=True):
        """
        Loads any resources required for display of plots
        in the Jupyter notebook
        """
        with param.logging_level('ERROR'):
            cls.notebook_context = True
            cls.comm_manager = JupyterCommManager


    @classmethod
    def _delete_plot(cls, plot_id):
        """
        Deletes registered plots and calls Plot.cleanup
        """
        plot = cls._plots.get(plot_id)
        if plot is None:
            return
        plot.cleanup()
        del cls._plots[plot_id]
