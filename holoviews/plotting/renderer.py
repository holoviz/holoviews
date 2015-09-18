"""
Public API for all plotting renderers supported by HoloViews,
regardless of plotting package or backend.
"""

import os
import base64
from contextlib import contextmanager

import param
from ..core.io import Exporter
from ..core.util import find_file
from .. import Store, Layout, HoloMap, AdjointLayout
from .widgets import ScrubberWidget, SelectionWidget

from . import Plot

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
    'html':  None,
    'json':  None
}


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

    mode = param.ObjectSelector(default='default', objects=['default'], doc="""
         The available rendering modes. As a minimum, the 'default'
         mode must be supported.""")

    fig = param.ObjectSelector(default='auto', doc="""
        Output render format for static figures. If None, no figure
        rendering will occur. """)

    holomap = param.ObjectSelector(default='auto', doc="""
        Output render multi-frame (typically animated) format. If
        None, no multi-frame rendering will occur.""")

    size=param.Integer(100, doc="""
        The rendered size as a percentage size""")

    fps=param.Integer(20, doc="""
        Rendered fps (frames per second) for animated formats.""")

    dpi=param.Integer(None, allow_None=True, doc="""
        The render resolution in dpi (dots per inch)""")

    info_fn = param.Callable(None, allow_None=True, constant=True,  doc="""
        Renderers do not support the saving of object info metadata""")

    key_fn = param.Callable(None, allow_None=True, constant=True,  doc="""
        Renderers do not support the saving of object key metadata""")

    # Defines the valid output formats for each mode.
    mode_formats = {'fig': {'default': [None]}, 'holomap': {'default': [None]}}

    # Define appropriate widget classes
    widgets = {'scrubber': ScrubberWidget, 'selection': SelectionWidget}

    def __init__(self, **params):
        super(Renderer, self).__init__(**params)


    def _validate(self, obj, fmt):
        """
        Helper method to be used in the __call__ method to get a
        suitable plot object and the appropriate format.
        """
        fig_formats = self.mode_formats['fig'][self.mode]
        holomap_formats = self.mode_formats['holomap'][self.mode]

        if not isinstance(obj, Plot):
            obj = Layout.from_values(obj) if isinstance(obj, AdjointLayout) else obj
            plot = self.plotting_class(obj)(obj, **self.plot_options(obj, self.size))
            plot.update(0)
        elif fmt is None:
            raise Exception("Format must be specified when supplying a plot instance")
        else:
            plot = obj

        if fmt is None: return (None,None)
        elif fmt =='auto' and len(plot) == 1:
            fmt = fig_formats[0] if self.fig=='auto' else self.fig
        elif fmt ==  'auto':
            fmt = holomap_formats[0] if self.holomap=='auto' else self.holomap

        all_formats = set(fig_formats + holomap_formats)
        if fmt not in all_formats:
            raise Exception("Format %r not supported by mode %r. Allowed formats: %r"
                            % (fmt, self.mode, fig_formats + holomap_formats))
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
        # [Backend specific code goes here]

        # Example of the return format where the first value is the rendered data.
        return None, {'file-ext':fmt, 'mime_type':MIME_TYPES[fmt]}


    def html(self, obj, fmt=None, css={}):
        """
        Renders plot or data structure and wraps the output in HTML.
        """
        plot, fmt =  self._validate(obj, fmt)
        figdata, _ = self(plot, fmt)

        if fmt in ['html', 'json']:
            return figdata
        else:
            if fmt == 'svg':
                figdata = figdata.encode("utf-8")
            elif fmt == 'pdf' and 'height' not in css:
                w,h = self.get_size(plot)
                css['height'] = '%dpx' % (h*self.dpi*1.15)

        if isinstance(css, dict):
            css = '; '.join("%s: %s" % (k, v) for k, v in css.items())
        else:
            raise ValueError("CSS must be supplied as Python dictionary")

        b64 = base64.b64encode(figdata).decode("utf-8")
        (mime_type, tag) = MIME_TYPES[fmt], HTML_TAGS[fmt]
        src = HTML_TAGS['base64'].format(mime_type=mime_type, b64=b64)
        return tag.format(src=src, mime_type=mime_type, css=css)


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
            raise Exception("No corresponding plot type found for %r" % type(obj))
        return plotclass


    @classmethod
    def embed_assets(cls):
        """
        Returns JS and CSS and for embedding of widgets.
        """
        # Get all the widgets and find the set of required js widget files
        widgets = [wdgt for cls in Renderer.__subclasses__()
                   for wdgt in cls.widgets.values()]
        css = list({wdgt.css for wdgt in widgets})
        basejs = list({wdgt.basejs for wdgt in widgets})
        extensionjs = list({wdgt.extensionjs for wdgt in widgets})

        # Join all the js widget code into one string
        path = os.path.dirname(os.path.abspath(__file__))
        widgetjs = '\n'.join(open(find_file(path, f), 'r').read()
                             for f in basejs + extensionjs
                             if f is not None )
        widgetcss = '\n'.join(open(find_file(path, f), 'r').read()
                              for f in css if f is not None)
        return widgetjs, widgetcss


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
        raise NotImplementedError

    @classmethod
    @contextmanager
    def state(cls):
        """
        Context manager to handle global state for a backend,
        allowing Plot classes to temporarily override that state.
        """
        yield

