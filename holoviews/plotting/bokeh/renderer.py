from io import BytesIO
import base64
import logging
import signal

import param
from param.parameterized import bothmethod

import bokeh
from bokeh.application.handlers import FunctionHandler
from bokeh.application import Application
from bokeh.document import Document
from bokeh.io import curdoc, show as bkshow
from bokeh.models import Model
from bokeh.resources import CDN, INLINE
from bokeh.server.server import Server

from ...core import Store, HoloMap
from ..plot import Plot, GenericElementPlot
from ..renderer import Renderer, MIME_TYPES, HTML_TAGS
from .widgets import BokehScrubberWidget, BokehSelectionWidget, BokehServerWidgets
from .util import attach_periodic, compute_plot_size, bokeh_version

from bokeh.io.notebook import load_notebook
from bokeh.protocol import Protocol
from bokeh.embed.notebook import encode_utf8, notebook_content
from bokeh.themes.theme import Theme

NOTEBOOK_DIV = """
{plot_div}
<script type="text/javascript">
  {plot_script}
</script>
"""

# Following JS block becomes body of the message handler callback
bokeh_msg_handler = """
var plot_id = "{plot_id}";
if (plot_id in HoloViews.plot_index) {{
  var plot = HoloViews.plot_index[plot_id];
}} else {{
  var plot = Bokeh.index[plot_id];
}}

if (plot_id in HoloViews.receivers) {{
  var receiver = HoloViews.receivers[plot_id];
}} else if (Bokeh.protocol === undefined) {{
  return;
}} else {{
  var receiver = new Bokeh.protocol.Receiver();
  HoloViews.receivers[plot_id] = receiver;
}}

if (buffers.length > 0) {{
  receiver.consume(buffers[0].buffer)
}} else {{
  receiver.consume(msg)
}}

const comm_msg = receiver.message;
if (comm_msg != null) {{
  plot.model.document.apply_json_patch(comm_msg.content, comm_msg.buffers)
}}
"""

default_theme = Theme(json={
    'attrs': {
        'Title': {'text_color': 'black', 'text_font_size': '12pt'}
    }
})


class BokehRenderer(Renderer):

    theme = param.ClassSelector(default=default_theme, class_=(Theme, str),
                                allow_None=True, doc="""
       The applicable Bokeh Theme object (if any).""")

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

    webgl = param.Boolean(default=False, doc="""
        Whether to render plots with WebGL if available""")

    widgets = {'scrubber': BokehScrubberWidget,
               'widgets': BokehSelectionWidget,
               'server': BokehServerWidgets}

    backend_dependencies = {'js': CDN.js_files if CDN.js_files else tuple(INLINE.js_raw),
                            'css': CDN.css_files if CDN.css_files else tuple(INLINE.css_raw)}

    _loaded = False

    # Define the handler for updating bokeh plots
    comm_msg_handler = bokeh_msg_handler if bokeh_version > '0.12.14' else None

    def __call__(self, obj, fmt=None, doc=None):
        """
        Render the supplied HoloViews component using the appropriate
        backend. The output is not a file format but a suitable,
        in-memory byte stream together with any suitable metadata.
        """
        plot, fmt =  self._validate(obj, fmt, doc=doc)
        info = {'file-ext': fmt, 'mime_type': MIME_TYPES[fmt]}

        if self.mode == 'server':
            return self.server_doc(plot, doc), info
        elif isinstance(plot, tuple(self.widgets.values())):
            return plot(), info
        elif fmt == 'png':
            png = self._figure_data(plot, fmt=fmt, doc=doc)
            return png, info
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
    def get_plot(self_or_cls, obj, doc=None, renderer=None, **kwargs):
        """
        Given a HoloViews Viewable return a corresponding plot instance.
        Allows supplying a document attach the plot to, useful when
        combining the bokeh model with another plot.
        """
        if doc is None:
            doc = Document() if self_or_cls.notebook_context else curdoc()

        if self_or_cls.notebook_context:
            curdoc().theme = self_or_cls.theme
        doc.theme = self_or_cls.theme
        plot = super(BokehRenderer, self_or_cls).get_plot(obj, renderer, **kwargs)
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
        if not isinstance(self_or_cls, BokehRenderer) or self_or_cls.mode != 'server':
            renderer = self_or_cls.instance(mode='server')
        else:
            renderer = self_or_cls

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
            if isinstance(websocket_origin, list):
                if len(websocket_origin) > 1:
                    raise ValueError('In the notebook only a single websocket origin '
                                     'may be defined, which must match the URL of the '
                                     'notebook server.')
                websocket_origin = websocket_origin[0]
            opts = dict(notebook_url=websocket_origin) if websocket_origin else {}
            return bkshow(app, **opts)

        # If app shown outside notebook or new_window requested
        # start server and open in new browser tab
        from tornado.ioloop import IOLoop
        loop = IOLoop.current()
        if websocket_origin and not isinstance(websocket_origin, list):
            websocket_origin = [websocket_origin]
        opts = dict(allow_websocket_origin=websocket_origin) if websocket_origin else {}
        opts['io_loop'] = loop
        server = Server({'/': app}, port=port, **opts)
        def show_callback():
            server.show('/')
        server.io_loop.add_callback(show_callback)
        server.start()

        def sig_exit(*args, **kwargs):
            loop.add_callback_from_signal(do_stop)

        def do_stop(*args, **kwargs):
            loop.stop()

        signal.signal(signal.SIGINT, sig_exit)
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
        if not isinstance(obj, (Plot, BokehServerWidgets)):
            if not isinstance(self_or_cls, BokehRenderer) or self_or_cls.mode != 'server':
                renderer = self_or_cls.instance(mode='server')
            else:
                renderer = self_or_cls
            plot, _ =  renderer._validate(obj, 'auto')
        else:
            plot = obj

        root = plot.state
        if isinstance(plot, BokehServerWidgets):
            plot = plot.plot

        if doc is None:
            doc = plot.document
        else:
            plot.document = doc

        plot.traverse(lambda x: attach_periodic(x), [GenericElementPlot])
        doc.add_root(root)
        return doc


    def components(self, obj, fmt=None, comm=True, **kwargs):
        # Bokeh has to handle comms directly in <0.12.15
        comm = False if bokeh_version < '0.12.15' else comm
        return super(BokehRenderer, self).components(obj,fmt, comm, **kwargs)


    def _figure_data(self, plot, fmt='html', doc=None, as_script=False, **kwargs):
        """
        Given a plot instance, an output format and an optional bokeh
        document, return the corresponding data. If as_script is True,
        the content will be split in an HTML and a JS component.
        """
        model = plot.state
        if doc is None:
            doc = plot.document
        else:
            plot.document = doc

        for m in model.references():
            m._document = None

        doc.theme = self.theme
        doc.add_root(model)

        comm_id = plot.comm.id if plot.comm else None
        # Bokeh raises warnings about duplicate tools and empty subplots
        # but at the holoviews level these are not issues
        logger = logging.getLogger(bokeh.core.validation.check.__file__)
        logger.disabled = True

        if fmt == 'png':
            from bokeh.io.export import get_screenshot_as_png
            img = get_screenshot_as_png(plot.state, None)
            imgByteArr = BytesIO()
            img.save(imgByteArr, format='PNG')
            data = imgByteArr.getvalue()
            if as_script:
                b64 = base64.b64encode(data).decode("utf-8")
                (mime_type, tag) = MIME_TYPES[fmt], HTML_TAGS[fmt]
                src = HTML_TAGS['base64'].format(mime_type=mime_type, b64=b64)
                div = tag.format(src=src, mime_type=mime_type, css='')
                js = ''
        else:
            try:
                js, div, _ = notebook_content(model, comm_id)
                html = NOTEBOOK_DIV.format(plot_script=js, plot_div=div)
                data = encode_utf8(html)
                doc.hold()
            except:
                logger.disabled = False
                raise
            logger.disabled = False

        plot.document = doc
        if as_script:
            return div, js
        return data


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
        LOAD_MIME_TYPE = bokeh.io.notebook.LOAD_MIME_TYPE
        bokeh.io.notebook.LOAD_MIME_TYPE = MIME_TYPES['jlab-hv-load']
        load_notebook(hide_banner=True, resources=INLINE if inline else CDN)
        bokeh.io.notebook.LOAD_MIME_TYPE = LOAD_MIME_TYPE
        bokeh.io.notebook.curstate().output_notebook()
