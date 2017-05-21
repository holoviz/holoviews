import numpy as np
import param
from param.parameterized import bothmethod


from bokeh.application.handlers import FunctionHandler
from bokeh.application import Application
from bokeh.charts import Chart
from bokeh.document import Document
from bokeh.embed import notebook_div, autoload_server
from bokeh.io import load_notebook, curdoc, show as bkshow
from bokeh.models import (Row, Column, Plot, Model, ToolbarBox,
                          WidgetBox, Div, DataTable, Tabs)
from bokeh.plotting import Figure
from bokeh.resources import CDN, INLINE
from bokeh.server.server import Server

from ...core import Store, HoloMap
from ..comms import JupyterComm, Comm
from ..plot import GenericElementPlot
from ..renderer import Renderer, MIME_TYPES
from .widgets import BokehScrubberWidget, BokehSelectionWidget, BokehServerWidgets
from .util import compute_static_patch, serialize_json, attach_periodic, bokeh_version



class BokehRenderer(Renderer):

    backend = param.String(default='bokeh', doc="The backend name.")

    fig = param.ObjectSelector(default='auto', objects=['html', 'json', 'auto'], doc="""
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
    mode_formats = {'fig': {'default': ['html', 'json', 'auto'],
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
        elif fmt == 'html':
            html = self.figure_data(plot, doc=doc)
            html = "<div style='display: table; margin: 0 auto;'>%s</div>" % html
            return self._apply_post_render_hooks(html, obj, fmt), info
        elif fmt == 'json':
            return self.diff(plot), info


    @bothmethod
    def get_plot(self_or_cls, obj, doc=None, renderer=None):
        """
        Given a HoloViews Viewable return a corresponding plot instance.
        Allows supplying a document attach the plot to, useful when
        combining the bokeh model with another plot.
        """
        plot = super(BokehRenderer, self_or_cls).get_plot(obj, renderer)
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
    def app(self_or_cls, plot, show=False, new_window=False):
        """
        Creates a bokeh app from a HoloViews object or plot. By
        default simply uses attaches plot to bokeh's curdoc and
        returns the Document, if show option is supplied creates
        an Application instance and displays it either in a browser
        window or inline if notebook extension has been loaded.
        Using the new_window option the app may be displayed in a
        new browser tab once the notebook extension has been loaded.
        """
        renderer = self_or_cls.instance(mode='server')
        # If show=False and not in noteboook context return document
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
            return bkshow(app)

        # If app shown outside notebook or new_window requested
        # start server and open in new browser tab
        from tornado.ioloop import IOLoop
        loop = IOLoop.current()
        server = Server({'/': app}, port=0, loop=loop)
        def show_callback():
            server.show('/')
        server.io_loop.add_callback(show_callback)
        server.start()
        try:
            loop.start()
        except RuntimeError:
            pass
        return server


    def server_doc(self, plot, doc=None):
        """
        Get server document.
        """
        if doc is None:
            doc = curdoc()
        root = plot.state
        if isinstance(plot, BokehServerWidgets):
            plot = plot.plot
        plot.document = doc
        plot.traverse(lambda x: attach_periodic(plot),
                      [GenericElementPlot])
        doc.add_root(root)
        return doc


    def figure_data(self, plot, fmt='html', doc=None, **kwargs):
        model = plot.state
        doc = Document() if doc is None else doc
        for m in model.references():
            m._document = None
        doc.add_root(model)
        comm_id = plot.comm.id if plot.comm else None
        div = notebook_div(model, comm_id)
        plot.document = doc
        return div


    def diff(self, plot, serialize=True):
        """
        Returns a json diff required to update an existing plot with
        the latest plot data.
        """
        plotobjects = [h for handles in plot.traverse(lambda x: x.current_handles, [lambda x: x._updated])
                       for h in handles]
        plot.traverse(lambda x: setattr(x, '_updated', False))
        patch = compute_static_patch(plot.document, plotobjects)
        processed = self._apply_post_render_hooks(patch, plot, 'json')
        return serialize_json(processed) if serialize else processed


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
        factor = percent_size / 100.0
        obj = obj.last if isinstance(obj, HoloMap) else obj
        plot = Store.registry[cls.backend].get(type(obj), None)
        if not hasattr(plot, 'width') or not hasattr(plot, 'height'):
            from .plot import BokehPlot
            plot = BokehPlot
        options = plot.lookup_options(obj, 'plot').options
        width = options.get('width', plot.width) * factor
        height = options.get('height', plot.height) * factor
        return dict(options, **{'width':int(width), 'height': int(height)})


    @bothmethod
    def get_size(self_or_cls, plot):
        """
        Return the display size associated with a plot before
        rendering to any particular format. Used to generate
        appropriate HTML display.

        Returns a tuple of (width, height) in pixels.
        """
        if not isinstance(plot, Model):
            plot = plot.state
        if isinstance(plot, Div):
            # Cannot compute size for Div
            return 0, 0
        elif isinstance(plot, (Row, Column, ToolbarBox, WidgetBox, Tabs)):
            if not plot.children: return 0, 0
            if isinstance(plot, Row) or (isinstance(plot, ToolbarBox) and plot.toolbar_location not in ['right', 'left']):
                w_agg, h_agg = (np.sum, np.max)
            elif isinstance(plot, Tabs):
                w_agg, h_agg = (np.max, np.max)
            else:
                w_agg, h_agg = (np.max, np.sum)
            widths, heights = zip(*[self_or_cls.get_size(child) for child in plot.children])
            width, height = w_agg(widths), h_agg(heights)
        elif isinstance(plot, (Chart, Figure)):
            width, height = plot.plot_width, plot.plot_height
        elif isinstance(plot, (Plot, DataTable)):
            width, height = plot.width, plot.height
        return width, height

    @classmethod
    def load_nb(cls, inline=True):
        """
        Loads the bokeh notebook resources.
        """
        kwargs = {'notebook_type': 'jupyter'} if bokeh_version > '0.12.5' else {}
        load_notebook(hide_banner=True, resources=INLINE if inline else CDN, **kwargs)
