"""
Public API for all plotting renderers supported by HoloViews,
regardless of plotting package or backend.
"""
from __future__ import unicode_literals, absolute_import

import base64
from io import BytesIO
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
from contextlib import contextmanager

import param

from panel import config
from panel.io.notebook import load_notebook, render_model, render_mimebundle
from panel.io.state import state
from panel.pane import HoloViews as HoloViewsPane
from panel.widgets.player import PlayerBase
from panel.viewable import Viewable

from ..core.io import Exporter
from ..core.options import Store, StoreOptions, SkipRendering, Compositor
from ..core.util import unbound_dimensions
from .. import Layout, HoloMap, AdjointLayout, DynamicMap

from . import Plot
from pyviz_comms import CommManager, JupyterCommManager
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

    dpi = param.Integer(None, doc="""
        The render resolution in dpi (dots per inch)""")

    fig = param.ObjectSelector(default='auto', objects=['auto'], doc="""
        Output render format for static figures. If None, no figure
        rendering will occur. """)

    fps = param.Number(20, doc="""
        Rendered fps (frames per second) for animated formats.""")

    holomap = param.ObjectSelector(default='auto',
                                   objects=['scrubber','widgets', None, 'auto'], doc="""
        Output render multi-frame (typically animated) format. If
        None, no multi-frame rendering will occur.""")

    mode = param.ObjectSelector(default='default',
                                objects=['default', 'server'], doc="""
        Whether to render the object in regular or server mode. In server
        mode a bokeh Document will be returned which can be served as a
        bokeh server app. By default renders all output is rendered to HTML.""")

    size = param.Integer(100, doc="""
        The rendered size as a percentage size""")

    widget_location = param.ObjectSelector(default=None, allow_None=True, objects=[
        'left', 'bottom', 'right', 'top', 'top_left', 'top_right',
        'bottom_left', 'bottom_right', 'left_top', 'left_bottom',
        'right_top', 'right_bottom'], doc="""
        The position of the widgets relative to the plot.""")

    widget_mode = param.ObjectSelector(default='embed', objects=['embed', 'live'], doc="""
        The widget mode determining whether frames are embedded or generated
        'live' when interacting with the widget.""")

    css = param.Dict(default={}, doc="""
        Dictionary of CSS attributes and values to apply to HTML output.""")

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
    mode_formats = {'fig': [None, 'auto'],
                    'holomap': [None, 'auto']}

    # The comm_manager handles the creation and registering of client,
    # and server side comms
    comm_manager = CommManager

    # Define appropriate widget classes
    widgets = ['scrubber', 'widgets']

    # Whether in a notebook context, set when running Renderer.load_nb
    notebook_context = False

    # Plot registry
    _plots = {}

    # Whether to render plots with Panel
    _render_with_panel = False

    def __init__(self, **params):
        self.last_plot = None
        super(Renderer, self).__init__(**params)


    def __call__(self, obj, fmt='auto', **kwargs):
        plot, fmt = self._validate(obj, fmt)
        info = {'file-ext': fmt, 'mime_type': MIME_TYPES[fmt]}

        if plot is None:
            return None, info
        elif self.mode == 'server':
            return self.server_doc(plot, doc=kwargs.get('doc')), info
        elif isinstance(plot, Viewable):
            return self.static_html(plot), info
        else:
            data = self._figure_data(plot, fmt, **kwargs)
            data = self._apply_post_render_hooks(data, obj, fmt)
            return data, info


    @bothmethod
    def get_plot(self_or_cls, obj, doc=None, renderer=None, comm=None, **kwargs):
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

        if not renderer:
            renderer = self_or_cls
            if not isinstance(self_or_cls, Renderer):
                renderer = self_or_cls.instance()

        if not isinstance(obj, Plot):
            if not displayable(obj):
                obj = collate(obj)
                initialize_dynamic(obj)
            obj = Compositor.map(obj, mode='data', backend=self_or_cls.backend)
            obj = Layout(obj) if isinstance(obj, AdjointLayout) else obj
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

        if isinstance(self_or_cls, Renderer):
            self_or_cls.last_plot = plot

        if comm:
            plot.comm = comm

        if comm or self_or_cls.mode == 'server':
            from bokeh.document import Document
            from bokeh.io import curdoc
            if doc is None:
                doc = Document() if self_or_cls.notebook_context else curdoc()
            plot.document = doc
        return plot


    @bothmethod
    def get_plot_state(self_or_cls, obj, renderer=None, **kwargs):
        """
        Given a HoloViews Viewable return a corresponding plot state.
        """
        if not isinstance(obj, Plot):
            obj = self_or_cls.get_plot(obj, renderer, **kwargs)
        return obj.state


    def _validate(self, obj, fmt, **kwargs):
        """
        Helper method to be used in the __call__ method to get a
        suitable plot or widget object and the appropriate format.
        """
        if isinstance(obj, Viewable):
            return obj, 'html'

        fig_formats = self.mode_formats['fig']
        holomap_formats = self.mode_formats['holomap']

        holomaps = obj.traverse(lambda x: x, [HoloMap])
        dynamic = any(isinstance(m, DynamicMap) for m in holomaps)

        if fmt in ['auto', None]:
            if any(len(o) > 1 or (isinstance(o, DynamicMap) and unbound_dimensions(o.streams, o.kdims))
                   for o in holomaps):
                fmt = holomap_formats[0] if self.holomap in ['auto', None] else self.holomap
            else:
                fmt = fig_formats[0] if self.fig == 'auto' else self.fig

        if fmt in self.widgets:
            plot = self.get_widget(obj, fmt)
            fmt = 'html'
        elif dynamic or (self._render_with_panel and fmt == 'html'):
            plot, fmt = HoloViewsPane(obj, center=True, backend=self.backend, renderer=self), fmt
        else:
            plot = self.get_plot(obj, renderer=self, **kwargs)

        all_formats = set(fig_formats + holomap_formats)
        if fmt not in all_formats:
            raise Exception("Format %r not supported by mode %r. Allowed formats: %r"
                            % (fmt, self.mode, fig_formats + holomap_formats))
        self.last_plot = plot
        return plot, fmt


    def _apply_post_render_hooks(self, data, obj, fmt):
        """
        Apply the post-render hooks to the data.
        """
        hooks = self.post_render_hooks.get(fmt,[])
        for hook in hooks:
            try:
                data = hook(data, obj)
            except Exception as e:
                self.param.warning("The post_render_hook %r could not "
                                   "be applied:\n\n %s" % (hook, e))
        return data


    def html(self, obj, fmt=None, css=None, resources='CDN', **kwargs):
        """
        Renders plot or data structure and wraps the output in HTML.
        The comm argument defines whether the HTML output includes
        code to initialize a Comm, if the plot supplies one.
        """
        plot, fmt =  self._validate(obj, fmt)
        figdata, _ = self(plot, fmt, **kwargs)
        if css is None: css = self.css

        if isinstance(plot, Viewable):
            from bokeh.document import Document
            from bokeh.embed import file_html
            from bokeh.resources import CDN, INLINE
            doc = Document()
            plot._render_model(doc)
            if resources == 'cdn':
                resources = CDN
            elif resources == 'inline':
                resources = INLINE
            return file_html(doc, resources)
        elif fmt in ['html', 'json']:
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
        document.
        """
        if isinstance(obj, Plot):
            plot = obj
        else:
            plot, fmt = self._validate(obj, fmt)

        data, metadata = {}, {}
        if isinstance(plot, Viewable):
            from bokeh.document import Document
            dynamic = bool(plot.object.traverse(lambda x: x, [DynamicMap]))
            embed = (not (dynamic or self.widget_mode == 'live') or config.embed)
            comm = self.comm_manager.get_server_comm() if comm else None
            doc = Document()
            with config.set(embed=embed):
                model = plot.layout._render_model(doc, comm)
            return render_model(model, comm) if embed else render_mimebundle(model, doc, comm)
        else:
            html = self._figure_data(plot, fmt, as_script=True, **kwargs)
        data['text/html'] = html

        return (data, {MIME_TYPES['jlab-hv-exec']: metadata})


    def static_html(self, obj, fmt=None, template=None):
        """
        Generates a static HTML with the rendered object in the
        supplied format. Allows supplying a template formatting string
        with fields to interpolate 'js', 'css' and the main 'html'.
        """
        html_bytes = StringIO()
        self.save(obj, html_bytes, fmt)
        html_bytes.seek(0)
        return html_bytes.read()


    @bothmethod
    def get_widget(self_or_cls, plot, widget_type, **kwargs):
        if widget_type == 'scrubber':
            widget_location = self_or_cls.widget_location or 'bottom'
        else:
            widget_type = 'individual'
            widget_location = self_or_cls.widget_location or 'right'

        layout = HoloViewsPane(plot, widget_type=widget_type, center=True,
                               widget_location=widget_location, renderer=self_or_cls)
        interval = int((1./self_or_cls.fps) * 1000)
        for player in layout.layout.select(PlayerBase):
            player.interval = interval
        return layout


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
        if fmt not in self_or_cls.widgets+['auto', None]:
            raise ValueError("Renderer.export_widget may only export "
                             "registered widget types.")
        self_or_cls.get_widget(obj, fmt).save(filename)


    @bothmethod
    def _widget_kwargs(self_or_cls):
        if self_or_cls.holomap in ('auto', 'widgets'):
            widget_type = 'individual'
            loc = self_or_cls.widget_location or 'right'
        else:
            widget_type = 'scrubber'
            loc = self_or_cls.widget_location or 'bottom'
        return {'widget_location': loc, 'widget_type': widget_type, 'center': True}


    @bothmethod
    def app(self_or_cls, plot, show=False, new_window=False, websocket_origin=None, port=0):
        """
        Creates a bokeh app from a HoloViews object or plot. By
        default simply attaches the plot to bokeh's curdoc and returns
        the Document, if show option is supplied creates an
        Application instance and displays it either in a browser
        window or inline if notebook extension has been loaded.  Using
        the new_window option the app may be displayed in a new
        browser tab once the notebook extension has been loaded.  A
        websocket origin is required when launching from an existing
        tornado server (such as the notebook) and it is not on the
        default port ('localhost:8888').
        """
        if isinstance(plot, HoloViewsPane):
            pane = plot
        else:
            pane = HoloViewsPane(plot, backend=self_or_cls.backend, renderer=self_or_cls,
                                 **self_or_cls._widget_kwargs())
        if new_window:
            return pane._get_server(port, websocket_origin, show=show)
        else:
            kwargs = {'notebook_url': websocket_origin} if websocket_origin else {}
            return pane.app(port=port, **kwargs)


    @bothmethod
    def server_doc(self_or_cls, obj, doc=None):
        """
        Get a bokeh Document with the plot attached. May supply
        an existing doc, otherwise bokeh.io.curdoc() is used to
        attach the plot to the global document instance.
        """
        if not isinstance(obj, HoloViewsPane):
            obj = HoloViewsPane(obj, renderer=self_or_cls, backend=self_or_cls.backend,
                                **self_or_cls._widget_kwargs())
        return obj.layout.server_doc(doc)


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
        Deprecated: No longer needed
        """
        param.main.warning("Renderer.html_assets is deprecated as all "
                           "JS and CSS dependencies are now handled by "
                           "Panel.")

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
    def save(self_or_cls, obj, basename, fmt='auto', key={}, info={},
             options=None, resources='inline', **kwargs):
        """
        Save a HoloViews object to file, either using an explicitly
        supplied format or to the appropriate default.
        """
        if info or key:
            raise Exception('Renderer does not support saving metadata to file.')

        with StoreOptions.options(obj, options, **kwargs):
            plot, fmt = self_or_cls._validate(obj, fmt)

        if isinstance(plot, Viewable):
            from bokeh.resources import CDN, INLINE, Resources
            if isinstance(resources, Resources):
                pass
            elif resources.lower() == 'cdn':
                resources = CDN
            elif resources.lower() == 'inline':
                resources = INLINE
            plot.layout.save(basename, embed=True, resources=resources)
            return

        rendered = self_or_cls(plot, fmt)
        if rendered is None: return
        (data, info) = rendered
        encoded = self_or_cls.encode(rendered)
        prefix = self_or_cls._save_prefix(info['file-ext'])
        if prefix:
            encoded = prefix + encoded
        if isinstance(basename, (BytesIO, StringIO)):
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
        load_notebook(inline)
        with param.logging_level('ERROR'):
            try:
                ip = get_ipython() # noqa
            except:
                ip = None
            if not ip or not hasattr(ip, 'kernel'):
                return
            cls.notebook_context = True
            cls.comm_manager = JupyterCommManager
            state._comm_manager = JupyterCommManager


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
