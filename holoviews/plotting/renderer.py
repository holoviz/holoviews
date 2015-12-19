"""
Public API for all plotting renderers supported by HoloViews,
regardless of plotting package or backend.
"""

from io import BytesIO
import os, json, base64
from contextlib import contextmanager

import param
from ..core.io import Exporter
from ..core.options import Store, StoreOptions
from ..core.util import find_file
from .. import Layout, HoloMap, AdjointLayout
from .widgets import NdWidget, ScrubberWidget, SelectionWidget

from .. import DynamicMap
from . import Plot
from .util import displayable, collate

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

    dpi=param.Integer(None, allow_None=True, doc="""
        The render resolution in dpi (dots per inch)""")

    fig = param.ObjectSelector(default='auto', objects=['auto'], doc="""
        Output render format for static figures. If None, no figure
        rendering will occur. """)

    fps=param.Integer(20, doc="""
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

    # Defines the valid output formats for each mode.
    mode_formats = {'fig': {'default': [None, 'auto']},
                    'holomap': {'default': [None, 'auto']}}

    # Define appropriate widget classes
    widgets = {'scrubber': ScrubberWidget, 'widgets': SelectionWidget}

    js_dependencies = ['https://code.jquery.com/jquery-2.1.4.min.js',
                       'https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.20/require.min.js']

    css_dependencies = ['https://maxcdn.bootstrapcdn.com/bootstrap/3.3.5/css/bootstrap.min.css']

    def __init__(self, **params):
        super(Renderer, self).__init__(**params)


    @bothmethod
    def get_plot(self_or_cls, obj):
        """
        Given a HoloViews Viewable return a corresponding plot instance.
        """
        if not isinstance(obj, Plot) and not displayable(obj):
            obj = collate(obj)

        # Initialize DynamicMaps with first data item
        dmaps = obj.traverse(lambda x: x, specs=[DynamicMap])
        for dmap in dmaps:
            if dmap.sampled:
                # Skip initialization until plotting code
                continue
            if dmap.call_mode == 'key':
                dmap[dmap._initial_key()]
            else:
                next(dmap)

        if not isinstance(obj, Plot):
            obj = Layout.from_values(obj) if isinstance(obj, AdjointLayout) else obj
            plot_opts = self_or_cls.plot_options(obj, self_or_cls.size)
            plot = self_or_cls.plotting_class(obj)(obj, **plot_opts)
            plot.update(0)
        else:
            plot = obj
        return plot


    def _validate(self, obj, fmt):
        """
        Helper method to be used in the __call__ method to get a
        suitable plot or widget object and the appropriate format.
        """
        if isinstance(obj, tuple(self.widgets.values())):
            return obj, 'html'
        plot = self.get_plot(obj)

        fig_formats = self.mode_formats['fig'][self.mode]
        holomap_formats = self.mode_formats['holomap'][self.mode]

        if fmt in ['auto', None] and len(plot) == 1 and not plot.dynamic:
            fmt = fig_formats[0] if self.fig=='auto' else self.fig
        elif fmt is None:
            fmt = holomap_formats[0] if self.holomap=='auto' else self.holomap

        if fmt in self.widgets:
            plot = self.get_widget(plot, fmt)
            fmt = 'html'

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


    def html(self, obj, fmt=None, css=None):
        """
        Renders plot or data structure and wraps the output in HTML.
        """
        plot, fmt =  self._validate(obj, fmt)
        figdata, _ = self(plot, fmt)
        if css is None: css = self.css

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


    def static_html(self, obj, fmt=None, template=None):
        """
        Generates a static HTML with the rendered object in the
        supplied format. Allows supplying a template formatting string
        with fields to interpolate 'js', 'css' and the main 'html'.
        """
        css_html, js_html = '', ''
        js, css = self.embed_assets()
        for url in self.js_dependencies:
            js_html += '<script src="%s" type="text/javascript"></script>' % url
        js_html += '<script type="text/javascript">%s</script>' % js

        for url in self.css_dependencies:
            css_html += '<link rel="stylesheet" href="%s">' % url
        css_html += '<style>%s</style>' % css

        if template is None: template = static_template

        html = self.html(obj, fmt)
        return template.format(js=js_html, css=css_html, html=html)


    @bothmethod
    def get_widget(self_or_cls, plot, widget_type, **kwargs):
        if not isinstance(plot, Plot):
            plot = self_or_cls.get_plot(plot)
        dynamic = plot.dynamic
        if widget_type == 'auto':
            isuniform = plot.uniform
            if not isuniform:
                widget_type = 'scrubber'
            else:
                widget_type = 'widgets'
        elif dynamic == 'open': widget_type = 'scrubber'
        elif dynamic == 'closed': widget_type = 'widgets'
        elif widget_type == 'widgets' and dynamic == 'open':
            raise ValueError('Selection widgets not supported in dynamic open mode')
        elif widget_type == 'scrubber' and dynamic == 'closed':
            raise ValueError('Scrubber widget not supported in dynamic closed mode')

        if widget_type in [None, 'auto']:
            holomap_formats = self_or_cls.mode_formats['holomap'][self_or_cls.mode]
            widget_type = holomap_formats[0] if self_or_cls.holomap=='auto' else self_or_cls.holomap

        widget_cls = self_or_cls.widgets[widget_type]
        return widget_cls(plot, renderer=self_or_cls,
                          embed=self_or_cls.widget_mode == 'embed', **kwargs)


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
        if isinstance(filename, BytesIO):
            filename.write(html)
            filename.seek(0)
        else:
            with open(filename, 'w') as f:
                f.write(html)


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
        widgets = [wdgt for r in Renderer.__subclasses__()
                   for wdgt in r.widgets.values()]
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
    def save(self_or_cls, obj, basename, fmt='auto', key={}, info={}, options=None, **kwargs):
        """
        Save a HoloViews object to file, either using an explicitly
        supplied format or to the appropriate default.
        """
        if info or key:
            raise Exception('MPLRenderer does not support saving metadata to file.')

        with StoreOptions.options(obj, options, **kwargs):
            plot = self_or_cls.get_plot(obj)

        if (fmt in list(self_or_cls.widgets.keys())+['auto']) and len(plot) > 1:
            with StoreOptions.options(obj, options, **kwargs):
                self_or_cls.export_widgets(plot, basename+'.html', fmt)
            return

        with StoreOptions.options(obj, options, **kwargs):
            rendered = self_or_cls(plot, fmt)
        if rendered is None: return
        (data, info) = rendered
        if isinstance(basename, BytesIO):
            basename.write(data)
            basename.seek(0)
        else:
            encoded = self_or_cls.encode(rendered)
            filename ='%s.%s' % (basename, info['file-ext'])
            with open(filename, 'wb') as f:
                f.write(encoded)


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
    def load_nb(cls):
        """
        Loads any resources required for display of plots
        in the Jupyter notebook
        """
