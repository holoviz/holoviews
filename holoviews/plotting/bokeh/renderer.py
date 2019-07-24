from __future__ import absolute_import, division, unicode_literals

import base64
import logging
from io import BytesIO

import param
import bokeh

from pyviz_comms import bokeh_msg_handler
from param.parameterized import bothmethod
from bokeh.core.validation.warnings import EMPTY_LAYOUT, MISSING_RENDERERS
from bokeh.document import Document
from bokeh.embed.notebook import encode_utf8, notebook_content
from bokeh.io import curdoc
from bokeh.io.notebook import load_notebook
from bokeh.models import Model
from bokeh.protocol import Protocol
from bokeh.resources import CDN, INLINE
from bokeh.themes.theme import Theme
from panel.pane import HoloViews, Viewable

from ...core import Store, HoloMap
from ..plot import Plot
from ..renderer import Renderer, MIME_TYPES, HTML_TAGS
from .util import compute_plot_size, silence_warnings


NOTEBOOK_DIV = """
{plot_div}
<script type="text/javascript">
  {plot_script}
</script>
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

    backend_dependencies = {'js': CDN.js_files if CDN.js_files else tuple(INLINE.js_raw),
                            'css': CDN.css_files if CDN.css_files else tuple(INLINE.css_raw)}

    _loaded = False

    # Define the handler for updating bokeh plots
    comm_msg_handler = bokeh_msg_handler

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
        elif isinstance(plot, Viewable):
            return plot, info
        elif fmt == 'png':
            png = self._figure_data(plot, fmt=fmt, doc=doc)
            return png, info
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
        pane = HoloViews(plot)
        if new_window:
            return pane._get_server(port, websocket_origin, show=show)
        else:
            kwargs = {'notebook_url': websocket_origin} if websocket_origin else {} 
            return pane.app(port, **kwargs)

    @bothmethod
    def server_doc(self_or_cls, obj, doc=None):
        """
        Get a bokeh Document with the plot attached. May supply
        an existing doc, otherwise bokeh.io.curdoc() is used to
        attach the plot to the global document instance.
        """
        return HoloViews(obj).server_doc(doc)


    def components(self, obj, fmt=None, comm=True, **kwargs):
        return super(BokehRenderer, self).components(obj, fmt, comm, **kwargs)


    def _figure_data(self, plot, fmt, doc=None, as_script=False, **kwargs):
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
        else:
            raise ValueError('Unsupported format: {fmt}'.format(fmt=fmt))

        plot.document = doc
        if as_script:
            return div
        else:
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
        if width is not None:
            options['width'] = int(width)
        if height is not None:
            options['height'] = int(height)
        return dict(options)


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
