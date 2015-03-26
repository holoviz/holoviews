import os
from io import BytesIO
from tempfile import NamedTemporaryFile

# Python3 compatibility
try: basestring = basestring
except: basestring = str

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc_params_from_file

import param
from param.parameterized import bothmethod

from ..core.options import Cycle, Palette, Options, Store, StoreOptions
from ..core import Dimension, Layout, NdLayout, GridSpace, HoloMap
from ..core.io import Exporter
from .annotation import * # pyflakes:ignore (API import)
from .chart import * # pyflakes:ignore (API import)
from .chart3d import * # pyflakes:ignore (API import)
from .plot import * # pyflakes:ignore (API import)
from .raster import * # pyflakes:ignore (API import)
from .tabular import * # pyflakes:ignore (API import)
from . import pandas # pyflakes:ignore (API import)
from . import seaborn # pyflakes:ignore (API import)


# Tags used when matplotlib output is to be embedded in HTML
IMAGE_TAG = "<center><img src='{src}' style='max-width:100%'/><center/>"
VIDEO_TAG = """
<center><video controls style='max-width:100%'>
<source src='{src}' type='{mime_type}'>
Your browser does not support the video tag.
</video><center/>"""

HTML_TAGS = {
    'base64': 'data:{mime_type};base64,{b64}', # Use to embed data
    'svg':  ('image/svg+xml', IMAGE_TAG),
    'png':  ('image/png',     IMAGE_TAG),
    'gif':  ('image/gif',     IMAGE_TAG),
    'webm': ('video/webm',   VIDEO_TAG),
    'mp4':  ('video/mp4',    VIDEO_TAG)
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


def opts(el, size):
    "Returns the plot options with supplied size (if not overridden)"
    obj = el.last if isinstance(el, HoloMap) else el
    return dict(figure_inches=size, **Store.lookup_options(obj, 'plot').options)


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
    using the figure_size and size plot options.
    """
    def rescale_figure(percent_size):
        factor = percent_size / 100.0
        return (Plot.figure_inches[0] * factor,
                Plot.figure_inches[1] * factor)

    if isinstance(obj, (Layout, NdLayout)):
        return (obj.shape[1]*rescale_figure(percent_size)[1],
                obj.shape[0]*rescale_figure(percent_size)[0])
    elif isinstance(obj, GridSpace):
        max_dim = max(obj.shape)
        # Reduce plot size as GridSpace gets larger
        shape_factor = 1. / max_dim
        # Expand small grids to a sensible viewing size
        expand_factor = 1 + (max_dim - 1) * 0.1
        scale_factor = expand_factor * shape_factor
        return (scale_factor * obj.shape[0] * rescale_figure(percent_size)[0],
                scale_factor * obj.shape[1] * rescale_figure(percent_size)[1])
    else:
        return rescale_figure(percent_size)



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
                               objects=['png', 'svg', None], doc="""
        Output render format for static figures. If None, no figure
        rendering will occur. """)

    holomap = param.ObjectSelector(default='gif',
                                   objects=['webm','h264', 'gif', None], doc="""
        Output render multi-frame (typically animated) format. If
        None, no multi-frame rendering will occur.""")

    size=param.Integer(100, doc="""
        The rendered size as a percentage size""")

    fps=param.Integer(20, doc="""
        Rendered fps (frames per second) for animated formats.""")

    dpi=param.Integer(None, allow_None=True, doc="""
        The render resolution in dpi (dots per inch)""")


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

        plot = plotclass(obj, **opts(obj,  get_plot_size(obj, self.size)))

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
                      'mime_type':HTML_TAGS[fmt][0]}


    @bothmethod
    def save(self_or_cls, obj, basename, fmt=None, options=None, **kwargs):
        """
        Save a HoloViews object to file, either using an explicitly
        supplied format or to the appropriate default.
        """
        if 'metadata' in kwargs or self_or_cls.metadata_fn:
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
        anim_kwargs = dict(anim_kwargs, **({'fps':self.fps} if fmt =='gif' else {}))
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
        from matplotlib import rcParams
        kw = dict(
            format=fmt,
            facecolor=fig.get_facecolor(),
            edgecolor=fig.get_edgecolor(),
            dpi=rcParams['savefig.dpi'],
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

# Define matplotlib based style cycles and Palettes
Cycle.default_cycles.update({'default_colors': plt.rcParams['axes.color_cycle']})
Palette.colormaps.update({cm: plt.get_cmap(cm) for cm in plt.cm.datad})

# Register default Element options
Store.register_plots()

# Charts
Store.options.Curve = Options('style', color=Cycle(key='default_colors'), linewidth=2)
Store.options.Scatter = Options('style', color=Cycle(key='default_colors'), marker='o')
Store.options.Bars = Options('style', ec='k', color=Cycle(key='default_colors'))
Store.options.Histogram = Options('style', ec='k', fc=Cycle(key='default_colors'))
Store.options.Points = Options('style', color=Cycle(key='default_colors'), marker='o')
Store.options.Scatter3D = Options('style', color=Cycle(key='default_colors'), marker='o')
# Rasters
Store.options.Image = Options('style', cmap='hot', interpolation='nearest')
Store.options.Raster = Options('style', cmap='hot', interpolation='nearest')
Store.options.HeatMap = Options('style', cmap='RdYlBu_r', interpolation='nearest')
Store.options.HeatMap = Options('plot', show_values=True, xticks=20, yticks=20)
Store.options.RGBA = Options('style', interpolation='nearest')
Store.options.RGB = Options('style', interpolation='nearest')
# Composites
Store.options.Layout = Options('plot', sublabel_format='{Alpha}')
Store.options.GridSpace = Options('style', **{'font.size': 10, 'axes.labelsize': 'small',
                                              'axes.titlesize': 'small'})
# Annotations
Store.options.VLine = Options('style', color=Cycle(key='default_colors'))
Store.options.HLine = Options('style', color=Cycle(key='default_colors'))
Store.options.Spline = Options('style', lw=2)
Store.options.Text = Options('style', fontsize=13)
Store.options.Arrow = Options('style', color='k', lw=2, fontsize=13)
# Paths
Store.options.Contours = Options('style', color=Cycle(key='default_colors'))
Store.options.Path = Options('style', color=Cycle(key='default_colors'))
Store.options.Box = Options('style', color=Cycle(key='default_colors'))
Store.options.Bounds = Options('style', color=Cycle(key='default_colors'))
Store.options.Ellipse = Options('style', color=Cycle(key='default_colors'))
# Interface
Store.options.TimeSeries = Options('style', color=Cycle(key='default_colors'))

# Defining the most common style options for HoloViews
GrayNearest = Options(key='style', cmap='gray', interpolation='nearest')

def public(obj):
    if not isinstance(obj, type): return False
    baseclasses = [Plot, Cycle]
    return any([issubclass(obj, bc) for bc in baseclasses])


_public = ["MPLPlotRenderer", "GrayNearest"] + list(set([_k for _k, _v in locals().items() if public(_v)]))
__all__ = _public
