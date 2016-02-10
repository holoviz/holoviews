import uuid
from ...core import Store, HoloMap
from ..renderer import Renderer, MIME_TYPES
from .widgets import BokehScrubberWidget, BokehSelectionWidget
from .util import models_to_json

import param
from param.parameterized import bothmethod

from bokeh.embed import notebook_div
from bokeh.io import load_notebook
from bokeh.resources import CDN, INLINE

try:
    from bokeh.protocol import serialize_json
    bokeh_lt_011 = True
except ImportError:
    from bokeh.core.json_encoder import serialize_json
    from bokeh.model import _ModelInDocument as add_to_document
    bokeh_lt_011 = False


class BokehRenderer(Renderer):

    backend = param.String(default='bokeh', doc="The backend name.")

    fig = param.ObjectSelector(default='auto', objects=['html', 'json', 'auto'], doc="""
        Output render format for static figures. If None, no figure
        rendering will occur. """)

    # Defines the valid output formats for each mode.
    mode_formats = {'fig': {'default': ['html', 'json', 'auto']},
                    'holomap': {'default': ['widgets', 'scrubber', 'auto', None]}}

    webgl = param.Boolean(default=True, doc="""Whether to render plots with WebGL
        if bokeh version >=0.10""")

    widgets = {'scrubber': BokehScrubberWidget,
               'widgets': BokehSelectionWidget}

    backend_dependencies = {'js': CDN.js_files, 'css': CDN.css_files}

    _loaded = False

    def __call__(self, obj, fmt=None):
        """
        Render the supplied HoloViews component using the appropriate
        backend. The output is not a file format but a suitable,
        in-memory byte stream together with any suitable metadata.
        """
        plot, fmt =  self._validate(obj, fmt)
        info = {'file-ext': fmt, 'mime_type': MIME_TYPES[fmt]}

        if isinstance(plot, tuple(self.widgets.values())):
            return plot(), info
        elif fmt == 'html':
            html = self.figure_data(plot)
            html = '<center>%s</center>' % html
            return self._apply_post_render_hooks(html, obj, fmt), info
        elif fmt == 'json':
            plotobjects = [h for handles in plot.traverse(lambda x: x.current_handles)
                           for h in handles]
            data = dict(data=[])
            if not bokeh_lt_011:
                data['root'] = plot.state._id
            data['data'] = models_to_json(plotobjects)
            return self._apply_post_render_hooks(serialize_json(data), obj, fmt), info


    def figure_data(self, plot, fmt='html', **kwargs):
        if not bokeh_lt_011:
            doc_handler = add_to_document(plot.state)
            with doc_handler:
                doc = doc_handler._doc
                comms_target = str(uuid.uuid4())
                doc.last_comms_target = comms_target
                div = notebook_div(plot.state, comms_target)
            plot.document = doc
            return div
        else:
            return notebook_div(plot.state)


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
        options = Store.lookup_options(cls.backend, obj, 'plot').options
        if not hasattr(plot, 'width') or not hasattr(plot, 'height'):
            from .plot import BokehPlot
            plot = BokehPlot
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
        return (plot.state.height, plot.state.height)

    @classmethod
    def load_nb(cls, inline=True):
        """
        Loads the bokeh notebook resources.
        """
        load_notebook(hide_banner=True, resources=INLINE if inline else CDN)
