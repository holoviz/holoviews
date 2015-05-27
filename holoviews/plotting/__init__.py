import os
import warnings
from io import BytesIO
from tempfile import NamedTemporaryFile

# Python3 compatibility
try: basestring = basestring
except: basestring = str

from matplotlib import animation
from matplotlib import ticker
from matplotlib import rc_params_from_file

from param.parameterized import bothmethod

from ..core.options import Cycle, Palette, Options, StoreOptions
from ..core import Dimension, Layout, NdLayout, Overlay
from ..core.io import Exporter
from .annotation import * # pyflakes:ignore (API import)
from .chart import * # pyflakes:ignore (API import)
from .chart3d import * # pyflakes:ignore (API import)
from .path import * # pyflakes:ignore (API import)
from .plot import * # pyflakes:ignore (API import)
from .raster import * # pyflakes:ignore (API import)
from .tabular import * # pyflakes:ignore (API import)
from . import pandas # pyflakes:ignore (API import)
from . import seaborn # pyflakes:ignore (API import)

# Tags used when matplotlib output is to be embedded in HTML
IMAGE_TAG = "<img src='{src}' style='max-width:100%; margin: auto; display: block; {css}'/>"
VIDEO_TAG = """
<video controls style='max-width:100%; margin: auto; display: block; {css}'>
<source src='{src}' type='{mime_type}'>
Your browser does not support the video tag.
</video>"""
PDF_TAG = "<iframe src='{src}' style='width:100%; margin: auto; display: block; {css}'></iframe>"


HTML_TAGS = {
    'base64': 'data:{mime_type};base64,{b64}', # Use to embed data
    'svg':  IMAGE_TAG,
    'png':  IMAGE_TAG,
    'gif':  IMAGE_TAG,
    'webm': VIDEO_TAG,
    'mp4':  VIDEO_TAG,
    'pdf':  PDF_TAG
}

MIME_TYPES = {
    'svg':  'image/svg+xml',
    'png':  'image/png',
    'gif':  'image/gif',
    'webm': 'video/webm',
    'mp4':  'video/mp4',
    'pdf':  'application/pdf'
}

# <format name> : (animation writer, format,  anim_kwargs, extra_args)
ANIMATION_OPTS = {
    'webm': ('ffmpeg', 'webm', {},
             ['-vcodec', 'libvpx', '-b', '1000k']),
    'mp4': ('ffmpeg', 'mp4', {'codec': 'libx264'},
             ['-pix_fmt', 'yuv420p']),
    'gif': ('imagemagick', 'gif', {'fps': 10}, []),
    'scrubber': ('html', None, {'fps': 5}, None)
}


def opts(el, percent_size):
    "Returns the plot options with supplied size (if not overridden)"
    obj = el.last if isinstance(el, HoloMap) else el
    return dict(dict(fig_inches=get_plot_size(obj, percent_size)),
                **Store.lookup_options(obj, 'plot').options)


def get_plot_size(obj, percent_size):
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
    plot_type = obj.type if isinstance(obj, HoloMap) else type(obj)
    options = Store.lookup_options(obj, 'plot').options
    fig_inches = options.get('fig_inches', Plot.fig_inches)
    if isinstance(fig_inches, (list, tuple)):
        return (fig_inches[0] * factor,
                fig_inches[1] * factor)
    else:
        return Plot.fig_inches * factor 


class MPLPlotRenderer(Exporter):
    """
    Exporter used to render data from matplotlib, either to a stream
    or directly to file.

    The __call__ method renders an HoloViews component to raw data of
    a specified matplotlib format.  The save method is the
    corresponding method for saving a HoloViews objects to disk.

    The save_fig and save_anim methods are used to save matplotlib
    figure and animation objects. These match the two primary return
    types of plotting class implemented with matplotlib.
    """

    fig = param.ObjectSelector(default='svg',
                               objects=['png', 'svg', 'pdf', None], doc="""
        Output render format for static figures. If None, no figure
        rendering will occur. """)

    holomap = param.ObjectSelector(default='gif',
                                   objects=['webm','mp4', 'gif', None], doc="""
        Output render multi-frame (typically animated) format. If
        None, no multi-frame rendering will occur.""")

    size=param.Integer(100, doc="""
        The rendered size as a percentage size""")

    fps=param.Integer(20, doc="""
        Rendered fps (frames per second) for animated formats.""")

    dpi=param.Integer(None, allow_None=True, doc="""
        The render resolution in dpi (dots per inch)""")

    info_fn = param.Callable(None, allow_None=True, constant=True,  doc="""
        MPLPlotRenderer does not support the saving of object info metadata""")

    key_fn = param.Callable(None, allow_None=True, constant=True,  doc="""
        MPLPlotRenderer does not support the saving of object key metadata""")

    # Error messages generated when testing potentially supported formats
    HOLOMAP_FORMAT_ERROR_MESSAGES = {}

    def __call__(self, obj, fmt=None):
        """
        Render the supplied HoloViews component using matplotlib.
        """
        if isinstance(obj, AdjointLayout):
            obj = Layout.from_values(obj)

        element_type = obj.type if isinstance(obj, HoloMap) else type(obj)
        try:
            plotclass = Store.registry[element_type]
        except KeyError:
            raise Exception("No corresponding plot type found for %r" % type(obj))

        plot = plotclass(obj, **opts(obj,  self.size))

        if fmt is None:
            fmt = self.holomap if len(plot) > 1 else self.fig
            if fmt is None: return

        if len(plot) > 1:
            (writer, _, anim_kwargs, extra_args) = ANIMATION_OPTS[fmt]
            anim = plot.anim(fps=self.fps)
            if extra_args != []:
                anim_kwargs = dict(anim_kwargs, extra_args=extra_args)

            data = self.anim_data(anim, fmt, writer, **anim_kwargs)
        else:
            data = self.figure_data(plot(), fmt, **({'dpi':self.dpi} if self.dpi else {}))

        return data, {'file-ext':fmt,
                      'mime_type':MIME_TYPES[fmt]}

    @bothmethod
    def supported_holomap_formats(self_or_cls, optional_formats):
        "Optional formats that are actually supported by this renderer"
        supported = []
        with param.logging_level('CRITICAL'):
            self_or_cls.HOLOMAP_FORMAT_ERROR_MESSAGES = {}
        for fmt in optional_formats:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    fig = plt.figure()
                    anim = animation.FuncAnimation(fig, lambda x: x, frames=[0,1])
                    (writer, fmt, anim_kwargs, extra_args) = ANIMATION_OPTS[fmt]
                    if extra_args != []:
                        anim_kwargs = dict(anim_kwargs, extra_args=extra_args)
                        renderer = self_or_cls.instance(dpi=72)
                        renderer.anim_data(anim, fmt, writer, **anim_kwargs)
                    plt.close(fig)
                    supported.append(fmt)
                except Exception as e:
                    self_or_cls.HOLOMAP_FORMAT_ERROR_MESSAGES[fmt] = str(e)
        return supported


    @bothmethod
    def save(self_or_cls, obj, basename, fmt=None, key={}, info={}, options=None, **kwargs):
        """
        Save a HoloViews object to file, either using an explicitly
        supplied format or to the appropriate default.
        """
        if info or key:
            raise Exception('MPLPlotRenderer does not support saving metadata to file.')

        with StoreOptions.options(obj, options, **kwargs):
            rendered = self_or_cls(obj, fmt)
        if rendered is None: return
        (data, info) = rendered
        filename ='%s.%s' % (basename, info['file-ext'])
        with open(filename, 'wb') as f:
            f.write(self_or_cls.encode(rendered))

    def anim_data(self, anim, fmt, writer, **anim_kwargs):
        """
        Render a matplotlib animation object and return the corresponding data.
        """
        anim_kwargs = dict(anim_kwargs, **({'dpi':self.dpi} if self.dpi is not None else {}))
        anim_kwargs = dict({'fps':self.fps} if fmt =='gif' else {}, **anim_kwargs)
        if not hasattr(anim, '_encoded_video'):
            with NamedTemporaryFile(suffix='.%s' % fmt) as f:
                anim.save(f.name, writer=writer,
                          **dict(anim_kwargs, **({'dpi':self.dpi} if self.dpi else {})))
                video = open(f.name, "rb").read()
        return video


    def figure_data(self, fig, fmt='png', bbox_inches='tight', **kwargs):
        """
        Render matplotlib figure object and return the corresponding data.

        Similar to IPython.core.pylabtools.print_figure but without
        any IPython dependency.
        """
        kw = dict(
            format=fmt,
            facecolor=fig.get_facecolor(),
            edgecolor=fig.get_edgecolor(),
            dpi=self.dpi,
            bbox_inches=bbox_inches,
        )
        kw.update(kwargs)

        bytes_io = BytesIO()
        fig.canvas.print_figure(bytes_io, **kw)
        data = bytes_io.getvalue()
        if fmt == 'svg':
            data = data.decode('utf-8')
        return data


Store.renderer = MPLPlotRenderer

def set_style(key):
    """
    Select a style by name, e.g. set_style('default'). To revert to the
    previous style use the key 'unset' or False.
    """
    if key is None:
        return
    elif not key or key in ['unset', 'backup']:
        if 'backup' in styles:
            plt.rcParams.update(styles['backup'])
        else:
            raise Exception('No style backed up to restore')
    elif key not in styles:
        raise KeyError('%r not in available styles.')
    else:
        path = os.path.join(os.path.dirname(__file__), styles[key])
        new_style = rc_params_from_file(path)
        styles['backup'] = dict(plt.rcParams)

        plt.rcParams.update(new_style)

styles = {'default': './default.mplstyle'}
set_style('default')

# Upgrade Dimension formatters to matplotlib
wrapped_formatters = {k: fn if isinstance(fn, ticker.Formatter) else ticker.FuncFormatter(fn)
                      for k, fn in Dimension.type_formatters.items()}
Dimension.type_formatters.update(wrapped_formatters)

# Define matplotlib based style cycles and Palettes
Cycle.default_cycles.update({'default_colors': plt.rcParams['axes.color_cycle']})
Palette.colormaps.update({cm: plt.get_cmap(cm) for cm in plt.cm.datad})

style_aliases = {'edgecolor': ['ec', 'ecolor'], 'facecolor': ['fc'],
                 'linewidth': ['lw'], 'edgecolors': ['ec', 'edgecolor'],
                 'linestyle': ['ls'], 'size': ['s'], 'color': ['c'],
                 'markeredgecolor': ['mec'], 'markeredgewidth': ['mew'],
                 'markerfacecolor': ['mfc'], 'markersize': ['ms']}

def default_options(options):
    # Charts
    options.Curve = Options('style', color=Cycle(), linewidth=2)
    options.Scatter = Options('style', color=Cycle(), marker='o')
    options.ErrorBars = Options('style', ecolor='k')
    options.Bars = Options('style', ec='k', color=Cycle())
    options.Histogram = Options('style', ec='k', fc=Cycle())
    options.Points = Options('style', color=Cycle(), marker='o')
    options.Scatter3D = Options('style', color=Cycle(), marker='o')
    # Rasters
    options.Image = Options('style', cmap='hot', interpolation='nearest')
    options.Raster = Options('style', cmap='hot', interpolation='nearest')
    options.HeatMap = Options('style', cmap='RdYlBu_r', interpolation='nearest')
    options.HeatMap = Options('plot', show_values=True, xticks=20, yticks=20)
    options.RGB = Options('style', interpolation='nearest')
    # Composites
    options.Layout = Options('plot', sublabel_format='{Alpha}')
    options.GridSpace = Options('style', **{'font.size': 10, 'axes.labelsize': 'small',
                                                  'axes.titlesize': 'small'})
    # Annotations
    options.VLine = Options('style', color=Cycle())
    options.HLine = Options('style', color=Cycle())
    options.Spline = Options('style', linewidth=2, ec='r')
    options.Text = Options('style', fontsize=13)
    options.Arrow = Options('style', color='k', linewidth=2, fontsize=13)
    # Paths
    options.Contours = Options('style', color=Cycle())
    options.Path = Options('style', color=Cycle())
    options.Box = Options('style', color=Cycle())
    options.Bounds = Options('style', color=Cycle())
    options.Ellipse = Options('style', color=Cycle())
    # Interface
    options.TimeSeries = Options('style', color=Cycle())

# Register the default options
Store.option_setters.append(default_options)

# Register default Element options
Store.register_plots(style_aliases=style_aliases)


# Defining the most common style options for HoloViews
GrayNearest = Options(key='style', cmap='gray', interpolation='nearest')

def public(obj):
    if not isinstance(obj, type): return False
    baseclasses = [Plot, Cycle]
    return any([issubclass(obj, bc) for bc in baseclasses])


_public = ["MPLPlotRenderer", "GrayNearest"] + list(set([_k for _k, _v in locals().items() if public(_v)]))
__all__ = _public
