from ...core import Store, HoloMap, OrderedDict
from ..renderer import Renderer, MIME_TYPES
from .widgets import BokehWidget, BokehScrubberWidget, BokehSelectionWidget

import param
from param.parameterized import bothmethod

from bokeh.embed import notebook_div
from bokeh.io import load_notebook, Document
from bokeh.models import DataSource
from bokeh.plotting import Figure
from bokeh.resources import CDN

try:
    from bokeh.protocol import serialize_json
    old_bokeh = True
except ImportError:
    from bokeh._json_encoder import serialize_json
    old_bokeh = False

class BokehRenderer(Renderer):

    backend = param.String(default='bokeh', doc="The backend name.")

    fig = param.ObjectSelector(default='auto', objects=['html', 'json', 'auto'], doc="""
        Output render format for static figures. If None, no figure
        rendering will occur. """)

    # Defines the valid output formats for each mode.
    mode_formats = {'fig': {'default': ['html', 'json', 'auto']},
                    'holomap': {'default': ['widgets', 'scrubber', 'auto', None]}}

    widgets = {'scrubber': BokehScrubberWidget,
               'widgets': BokehSelectionWidget}

    js_dependencies = Renderer.js_dependencies + CDN.js_files

    css_dependencies = Renderer.css_dependencies + CDN.css_files

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
            return html, info
        elif fmt == 'json':
            plotobjects = [h for handles in plot.traverse(lambda x: x.current_handles)
                           for h in handles]
            data = OrderedDict()
            if not old_bokeh:
                data['root'] = plot.state._id
            for plotobj in plotobjects:
                if old_bokeh:
                    json = plotobj.vm_serialize(changed_only=True)
                else:
                    json = plotobj.to_json(False)
                data[plotobj.ref['id']] = {'type': plotobj.ref['type'],
                                           'data': json}
            return serialize_json(data), info


    def figure_data(self, plot, fmt='html', **kwargs):
        if not old_bokeh:
            doc = Document()
            doc.add_root(plot.state)
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
    def load_nb(cls):
        """
        Loads the bokeh notebook resources.
        """
        load_notebook(hide_banner=True)
