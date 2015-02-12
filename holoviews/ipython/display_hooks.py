"""
Definition and registration of display hooks for the IPython Notebook.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

try:    from matplotlib import animation
except: animation = None

from IPython.core.pylabtools import print_figure

from tempfile import NamedTemporaryFile
from functools import wraps
import sys, traceback, base64

try:
    import mpld3
    from ..plotting import hooks
except:
    mpld3 = None

import param

from ..core.options import Store
from ..core import Element, ViewableElement, HoloMap, AdjointLayout, NdLayout,\
    NdOverlay, AxisLayout, LayoutTree, Overlay
from ..core.traversal import unique_dimkeys
from ..element import Raster
from ..plotting import LayoutPlot, GridPlot, MatrixGridPlot, Plot
from .magics import ViewMagic, OptsMagic
from .widgets import IPySelectionWidget, SelectionWidget, ScrubberWidget

# To assist with debugging of display hooks
ENABLE_TRACEBACKS=True

#==================#
# Helper functions #
#==================#

def opts(el, size):
    "Returns the plot options with supplied size (if not overridden)"
    return dict(figure_size=size, **Store.lookup_options(el, 'plot').options)


def get_plot_size(size):
    factor = size / 100.0
    return (Plot.figure_size[0] * factor,
            Plot.figure_size[1] * factor)


def animate(anim, dpi, writer, mime_type, anim_kwargs, extra_args, tag):
    if extra_args != []:
        anim_kwargs = dict(anim_kwargs, extra_args=extra_args)

    if not hasattr(anim, '_encoded_video'):
        with NamedTemporaryFile(suffix='.%s' % mime_type) as f:
            anim.save(f.name, writer=writer, dpi=dpi, **anim_kwargs)
            video = open(f.name, "rb").read()
        anim._encoded_video = base64.b64encode(video).decode("utf-8")
    return tag.format(b64=anim._encoded_video,
                      mime_type=mime_type)


def HTML_video(plot):
    dpi = ViewMagic.options['dpi']
    anim = plot.anim(fps=ViewMagic.options['fps'])
    writers = animation.writers.avail
    current_format = ViewMagic.options['holomap']
    for fmt in [current_format] + list(ViewMagic.ANIMATION_OPTS.keys()):
        if ViewMagic.ANIMATION_OPTS[fmt][0] in writers:
            try:
                return animate(anim, dpi, *ViewMagic.ANIMATION_OPTS[fmt])
            except: pass
    msg = "<b>Could not generate %s animation</b>" % current_format
    if sys.version_info[0] == 3 and mpl.__version__[:-2] in ['1.2', '1.3']:
        msg = "<b>Python 3 Matplotlib animation support broken &lt;= 1.3</b>"
    raise Exception(msg)


def first_frame(plot):
    "Only display the first frame of an animated plot"
    return display_figure(plot[0])

def middle_frame(plot):
    "Only display the (approximately) middle frame of an animated plot"
    middle_frame = int(len(plot) / 2)
    return display_figure(plot[middle_frame])

def last_frame(plot):
    "Only display the last frame of an animated plot"
    return display_figure(plot[len(plot)])


def sanitized_repr(obj):
    "Sanitize text output for HTML display"
    return repr(obj).replace('\n', '<br>').replace(' ', '&nbsp;')

def max_frame_warning(max_frames):
    sys.stderr.write("Skipping matplotlib display to avoid "
                     "lengthy animation render times\n"
                     "[Total item frames exceeds max_frames on ViewMagic (%d)]"
                     % max_frames)

def process_cell_magics(obj):
    "Hook into %%opts and %%channels magics to process displayed element"
    invalid_options = OptsMagic.process_view(obj)
    if invalid_options: return invalid_options


def render(plot):
    try:
        return render_anim(plot)
    except Exception as e:
        return str(e)+'<br/>'+display_figure(plot())


def display_widgets(plot):
    "Display widgets applicable to the specified view"
    widget_mode = ViewMagic.options['widgets']
    widget_format = ViewMagic.options['holomap']
    assert widget_mode is not None, "Mistaken call to display_widgets method"

    isuniform = plot.uniform
    if not isuniform and widget_format == 'widgets':
        param.Parameterized.warning("%s is not uniform, falling back to scrubber widget."
                                    % type(plot).__name__)
        widget_format == 'scrubber'

    if widget_format == 'auto':
        dims = plot.dimensions
        widget_format = 'scrubber' if len(dims) == 1 or not isuniform else 'widgets'

    if widget_format == 'scrubber':
        return ScrubberWidget(plot)()
    if widget_mode == 'embed':
        return SelectionWidget(plot)()
    elif widget_mode == 'cached':
        return IPySelectionWidget(plot, cached=True)()
    else:
        return IPySelectionWidget(plot, cached=False)()


def display_figure(fig, message=None, max_width='100%'):
    "Display widgets applicable to the specified view"
    figure_format = ViewMagic.options['fig']
    dpi = ViewMagic.options['dpi']
    backend = ViewMagic.options['backend']

    if backend == 'd3' and mpld3:
        fig.dpi = dpi
        mpld3.plugins.connect(fig, mpld3.plugins.MousePosition(fontsize=14))
        html = "<center>" + mpld3.fig_to_html(fig) + "<center/>"
    else:
        figdata = print_figure(fig, figure_format, dpi=dpi)
        if figure_format=='svg':
            mime_type = 'svg+xml'
            figdata = figdata.encode("utf-8")
        else:
            mime_type = 'png'
        prefix = 'data:image/%s;base64,' % mime_type
        b64 = prefix + base64.b64encode(figdata).decode("utf-8")
        html = "<center><img src='%s' style='max-width:%s'/><center/>" % (b64, max_width)
    plt.close(fig)
    return html if (message is None) else '<b>%s</b></br>%s' % (message, html)


#===============#
# Display hooks #
#===============#


def display_hook(fn):
    @wraps(fn)
    def wrapped(view, **kwargs):
        # If pretty printing is off, return None (will fallback to repr)
        ip = get_ipython()
        if not ip.display_formatter.formatters['text/plain'].pprint:
            return None
        try:
            widget_mode = ViewMagic.options['widgets']
            map_format  = ViewMagic.options['holomap']
            # If widget_mode is None, widgets are not being used
            widget_mode = (widget_mode if map_format in ViewMagic.inbuilt_formats else None)
            return fn(view,
                      size=ViewMagic.options['size'],
                      dpi=ViewMagic.options['dpi'],
                      max_frames=ViewMagic.options['max_frames'],
                      max_branches = ViewMagic.options['max_branches'],
                      map_format = map_format,
                      widget_mode = widget_mode,
                      **kwargs)
        except:
            if ENABLE_TRACEBACKS:
                traceback.print_exc()
    return wrapped


@display_hook
def animation_display(anim, map_format, dpi=72, **kwargs):
    return animate(anim, dpi, *ViewMagic.ANIMATION_OPTS[map_format])


@display_hook
def view_display(view, size, **kwargs):
    if not isinstance(view, ViewableElement): return None
    magic_info = process_cell_magics(view)
    if magic_info: return magic_info
    fig = Store.defaults[view.__class__](view,
                                         **opts(view, get_plot_size(size)))()
    return display_figure(fig)


@display_hook
def map_display(vmap, size, map_format, max_frames, widget_mode, **kwargs):
    if not isinstance(vmap, HoloMap): return None
    magic_info = process_cell_magics(vmap)
    if magic_info: return magic_info
    mapplot = Store.defaults[vmap.type](vmap,
                                        **opts(vmap.last, get_plot_size(size)))
    if len(mapplot) == 0:
        return sanitized_repr(vmap)
    elif len(mapplot) > max_frames:
        max_frame_warning(max_frames)
        return sanitized_repr(vmap)
    elif len(mapplot) == 1:
        fig = mapplot()
        return display_figure(fig)
    elif widget_mode is not None:
        return display_widgets(mapplot)
    else:
        return render(mapplot)


@display_hook
def layout_display(layout, size, map_format, max_frames, max_branches, widget_mode, **kwargs):
    if isinstance(layout, AdjointLayout): layout = LayoutTree.from_values(layout)
    if not isinstance(layout, (LayoutTree, NdLayout)): return None
    nframes = len(unique_dimkeys(layout)[1])

    shape = layout.shape
    magic_info = process_cell_magics(layout)
    if magic_info: return magic_info
    grid_size = (shape[1]*get_plot_size(size)[1],
                 shape[0]*get_plot_size(size)[0])

    layoutplot = LayoutPlot(layout, **opts(layout, grid_size))
    if isinstance(layout, LayoutTree):
        if layout._display == 'auto':
            branches = len(set([path[0] for path in list(layout.data.keys())]))
            if branches > max_branches:
                return '<tt>'+ sanitized_repr(layout) + '</tt>'
            elif len(layout.data) * nframes > max_frames:
                max_frame_warning(max_frames)
                return '<tt>'+ sanitized_repr(layout) + '</tt>'

    if nframes == 1:
        fig = layoutplot()
        return display_figure(fig)
    elif widget_mode is not None:
        return display_widgets(layoutplot)
    else:
        return render(layoutplot)


@display_hook
def grid_display(grid, size, map_format, max_frames, max_branches, widget_mode, **kwargs):
    if not isinstance(grid, AxisLayout): return None
    max_dim = max(grid.shape)
    # Reduce plot size as AxisLayout gets larger
    shape_factor = 1. / max_dim
    # Expand small grids to a sensible viewing size
    expand_factor = 1 + (max_dim - 1) * 0.1
    scale_factor = expand_factor * shape_factor
    grid_size = (scale_factor * grid.shape[0] * get_plot_size(size)[0],
                 scale_factor * grid.shape[1] * get_plot_size(size)[1])

    magic_info = process_cell_magics(grid)
    if magic_info: return magic_info

    raster_fn = lambda x: True if isinstance(x, Raster) else False
    all_raster = all(grid.traverse(raster_fn, [Element]))
    if all_raster:
        plot_type = MatrixGridPlot
    else:
        plot_type = GridPlot
    gridplot = plot_type(grid, **opts(grid, grid_size))

    if len(gridplot) > max_frames:
        max_frame_warning(max_frames)
        return sanitized_repr(grid)
    elif len(gridplot) == 1:
        fig = gridplot()
        return display_figure(fig)
    if widget_mode is not None:
        return display_widgets(gridplot)
    else:
        return render(gridplot)


# HTML_video output by default, but may be set to first_frame,
# middle_frame or last_frame (e.g. for testing purposes)
render_anim = HTML_video

def set_display_hooks(ip):
    html_formatter = ip.display_formatter.formatters['text/html']
    html_formatter.for_type_by_name('matplotlib.animation', 'FuncAnimation', animation_display)
    html_formatter.for_type(LayoutTree, layout_display)
    html_formatter.for_type(ViewableElement, view_display)
    html_formatter.for_type(Overlay, view_display)
    html_formatter.for_type(NdOverlay, view_display)
    html_formatter.for_type(HoloMap, map_display)
    html_formatter.for_type(AdjointLayout, layout_display)
    html_formatter.for_type(NdLayout, layout_display)
    html_formatter.for_type(AxisLayout, grid_display)
