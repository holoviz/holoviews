from io import BytesIO
import logging

import param
from param.parameterized import bothmethod

import bokeh.core
from bokeh.application.handlers import FunctionHandler
from bokeh.application import Application
from bokeh.document import Document
from bokeh.io import curdoc, show as bkshow
from bokeh.models import Model
from bokeh.resources import CDN, INLINE
from bokeh.server.server import Server

from ...core import Store, HoloMap
from ..comms import JupyterComm, Comm
from ..plot import Plot, GenericElementPlot
from ..renderer import Renderer, MIME_TYPES
from .widgets import BokehScrubberWidget, BokehSelectionWidget, BokehServerWidgets
from .util import attach_periodic, compute_plot_size

from bokeh.io.notebook import (load_notebook, publish_display_data,
                               JS_MIME_TYPE, LOAD_MIME_TYPE, EXEC_MIME_TYPE)
from bokeh.protocol import Protocol
from bokeh.embed.notebook import encode_utf8, notebook_content

NOTEBOOK_DIV = """
{plot_div}
<script type="text/javascript">
  {plot_script}
</script>
"""


class BokehRenderer(Renderer):

    backend = param.String(default='bokeh', doc="The backend name.")

    fig = param.ObjectSelector(default='auto', objects=['html', 'json', 'auto', 'png'], doc="""
        Output render format for static figures. If None, no figure
        rendering will occur. """)

    holomap = param.ObjectSelector(default='auto',
                                   objects=['widgets', 'scrubber', 'server',
                                            None, 'auto'], doc="""
        Output render multi-frame (typically animated) format. If
        None, no multi-frame rendering will occur.""")

    mode = param.ObjectSelector(default='default',
                                objects=['default', 'server'], doc="""
        Whether to render the object in regular or server mode. In server
        mode a bokeh Document will be returned which can be served as a
        bokeh server app. By default renders all output is rendered to HTML.""")

    # Defines the valid output formats for each mode.
    mode_formats = {'fig': {'default': ['html', 'json', 'auto', 'png'],
                            'server': ['html', 'json', 'auto']},
                    'holomap': {'default': ['widgets', 'scrubber', 'auto', None],
                                'server': ['server', 'auto', None]}}

    webgl = param.Boolean(default=False, doc="""Whether to render plots with WebGL
        if bokeh version >=0.10""")

    widgets = {'scrubber': BokehScrubberWidget,
               'widgets': BokehSelectionWidget,
               'server': BokehServerWidgets}

    backend_dependencies = {'js': CDN.js_files if CDN.js_files else tuple(INLINE.js_raw),
                            'css': CDN.css_files if CDN.css_files else tuple(INLINE.css_raw)}

    comms = {'default': (JupyterComm, None),
             'server': (Comm, None)}

    _loaded = False

    def __call__(self, obj, fmt=None, doc=None):
        """
        Render the supplied HoloViews component using the appropriate
        backend. The output is not a file format but a suitable,
        in-memory byte stream together with any suitable metadata.
        """
        plot, fmt =  self._validate(obj, fmt)
        info = {'file-ext': fmt, 'mime_type': MIME_TYPES[fmt]}

        if self.mode == 'server':
            return self.server_doc(plot, doc), info
        elif isinstance(plot, tuple(self.widgets.values())):
            return plot(), info
        elif fmt == 'png':
            from bokeh.io.export import get_screenshot_as_png
            img = get_screenshot_as_png(plot.state, None)
            imgByteArr = BytesIO()
            img.save(imgByteArr, format='PNG')
            return imgByteArr.getvalue(), info
        elif fmt == 'html':
            html = self._figure_data(plot, doc=doc)
            html = "<div style='display: table; margin: 0 auto;'>%s</div>" % html
            return self._apply_post_render_hooks(html, obj, fmt), info
        elif fmt == 'json':
            return self.diff(plot), info

    @bothmethod
    def _save_prefix(self_or_cls, ext):
        "Hook to prefix content for instance JS when saving HTML"
        if ext == 'html':
            return '\n'.join(self_or_cls.html_assets()).encode('utf8')
        return

    @bothmethod
    def get_plot(self_or_cls, obj, doc=None, renderer=None):
        """
        Given a HoloViews Viewable return a corresponding plot instance.
        Allows supplying a document attach the plot to, useful when
        combining the bokeh model with another plot.
        """
        plot = super(BokehRenderer, self_or_cls).get_plot(obj, renderer)
        if self_or_cls.mode == 'server' and doc is None:
            doc = curdoc()
        if doc is not None:
            plot.document = doc
        return plot


    @bothmethod
    def get_widget(self_or_cls, plot, widget_type, doc=None, **kwargs):
        if not isinstance(plot, Plot):
            plot = self_or_cls.get_plot(plot, doc)
        if self_or_cls.mode == 'server':
            return BokehServerWidgets(plot, renderer=self_or_cls.instance(), **kwargs)
        else:
            return super(BokehRenderer, self_or_cls).get_widget(plot, widget_type, **kwargs)


    @bothmethod
    def app(self_or_cls, plot, show=False, new_window=False, websocket_origin=None):
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
        renderer = self_or_cls.instance(mode='server')
        # If show=False and not in notebook context return document
        if not show and not self_or_cls.notebook_context:
            doc, _ = renderer(plot)
            return doc

        def modify_doc(doc):
            renderer(plot, doc=doc)
        handler = FunctionHandler(modify_doc)
        app = Application(handler)

        if not show:
            # If not showing and in notebook context return app
            return app
        elif self_or_cls.notebook_context and not new_window:
            # If in notebook, show=True and no new window requested
            # display app inline
            opts = dict(notebook_url=websocket_origin) if websocket_origin else {}
            return bkshow(app, **opts)

        # If app shown outside notebook or new_window requested
        # start server and open in new browser tab
        from tornado.ioloop import IOLoop
        loop = IOLoop.current()
        opts = dict(allow_websocket_origin=[websocket_origin]) if websocket_origin else {}
        opts['io_loop'] = loop
        server = Server({'/': app}, port=0, **opts)
        def show_callback():
            server.show('/')
        server.io_loop.add_callback(show_callback)
        server.start()
        try:
            loop.start()
        except RuntimeError:
            pass
        return server


    @bothmethod
    def server_doc(self_or_cls, obj, doc=None):
        """
        Get a bokeh Document with the plot attached. May supply
        an existing doc, otherwise bokeh.io.curdoc() is used to
        attach the plot to the global document instance.
        """
        if doc is None:
            doc = curdoc()
        if not isinstance(obj, (Plot, BokehServerWidgets)):
            renderer = self_or_cls.instance(mode='server')
            plot, _ =  renderer._validate(obj, 'auto')
        else:
            plot = obj
        root = plot.state
        if isinstance(plot, BokehServerWidgets):
            plot = plot.plot
        plot.document = doc
        plot.traverse(lambda x: attach_periodic(x), [GenericElementPlot])
        doc.add_root(root)
        return doc


    def _figure_data(self, plot, fmt='html', doc=None, **kwargs):
        model = plot.state
        doc = Document() if doc is None else doc
        for m in model.references():
            m._document = None
        doc.add_root(model)
        comm_id = plot.comm.id if plot.comm else None
        # Bokeh raises warnings about duplicate tools and empty subplots
        # but at the holoviews level these are not issues
        logger = logging.getLogger(bokeh.core.validation.check.__file__)
        logger.disabled = True
        try:
            js, div, _ = notebook_content(model, comm_id)
            html = NOTEBOOK_DIV.format(plot_script=js, plot_div=div)
            div = encode_utf8(html)
            doc.hold()
        except:
            logger.disabled = False
            raise
        logger.disabled = False
        plot.document = doc
        return div


    def diff(self, plot, binary=True):
        """
        Returns a json diff required to update an existing plot with
        the latest plot data.
        """
        events = list(plot.document._held_events)
        if not events:
            return None
        msg = Protocol("1.0").create("PATCH-DOC", events, use_buffers=binary)
        plot.document._held_events = []
        return msg


    @classmethod
    def plot_options(cls, obj, percent_size):
        """
        Given a holoviews object and a percentage size, apply heuristics
        to compute a suitable figure size. For instance, scaling layouts
        and grids linearly can result in unwieldy figure sizes when there
        are a large number of elements. As ad hoc heuristics are used,
        this functionality is kept separate from the plotting classes
        themselves.

        Used by the IPython Notebook display hooks and the save
        utility. Note that this can be overridden explicitly per object
        using the fig_size and size plot options.
        """
        obj = obj.last if isinstance(obj, HoloMap) else obj
        plot = Store.registry[cls.backend].get(type(obj), None)
        if not hasattr(plot, 'width') or not hasattr(plot, 'height'):
            from .plot import BokehPlot
            plot = BokehPlot
        options = plot.lookup_options(obj, 'plot').options
        width = options.get('width', plot.width)
        height = options.get('height', plot.height)
        return dict(options, **{'width':int(width), 'height': int(height)})


    @bothmethod
    def get_size(self_or_cls, plot):
        """
        Return the display size associated with a plot before
        rendering to any particular format. Used to generate
        appropriate HTML display.

        Returns a tuple of (width, height) in pixels.
        """
        if isinstance(plot, Plot):
            plot = plot.state
        elif not isinstance(plot, Model):
            raise ValueError('Can only compute sizes for HoloViews '
                             'and bokeh plot objects.')
        return compute_plot_size(plot)


    @classmethod
    def load_nb(cls, inline=True):
        """
        Loads the bokeh notebook resources.
        """
        from bokeh.io.notebook import curstate
        load_notebook(hide_banner=True, resources=INLINE if inline else CDN)
        curstate().output_notebook()
