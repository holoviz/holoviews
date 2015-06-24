from ...core import Store, HoloMap
from ..renderer import Renderer, MIME_TYPES
from .widgets import ScrubberWidget, SelectionWidget

from param.parameterized import bothmethod

from bokeh.embed import file_html, notebook_div

class BokehRenderer(Renderer):

    backend = 'bokeh'

    session = None

    # Defines the valid output formats for each mode.
    mode_formats = {'fig': {'default': ['html']}, 'holomap': {'default': [None]}}
    widgets = {'scrubber': ScrubberWidget,
               'selection': SelectionWidget}


    def __init__(self, **params):
        super(BokehRenderer, self).__init__(**params)


    def __call__(self, obj, fmt=None):
        """
        Render the supplied HoloViews component using the appropriate
        backend. The output is not a file format but a suitable,
        in-memory byte stream together with any suitable metadata.
        """
        # Example of the return format where the first value is the rendered data.
        html = self.figure_data(obj)
        html = '<center>%s</center>' % html
        return html, {'file-ext':fmt, 'mime_type':MIME_TYPES[fmt]}


    def figure_data(self, plot, fmt='html', **kwargs):
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
        from .plot import BokehPlot
        factor = percent_size / 100.0
        obj = obj.last if isinstance(obj, HoloMap) else obj
        options = Store.lookup_options(cls.backend, obj, 'plot').options
        width = options.get('width', BokehPlot.width) * factor
        height = options.get('height', BokehPlot.height) * factor
        return dict(options, **{'width':int(width), 'height': int(height)})


    @bothmethod
    def save(self_or_cls, obj, basename, fmt=None, key={}, info={}, options=None, **kwargs):
        """
        Given an object, a basename for the output file, a file format
        and some options, save the element in a suitable format to disk.
        """
        raise NotImplementedError

    @bothmethod
    def get_size(self_or_cls, plot):
        """
        Return the display size associated with a plot before
        rendering to any particular format. Used to generate
        appropriate HTML display.

        Returns a tuple of (width, height) in pixels.
        """
        return (plot.state.height, plot.state.height)
