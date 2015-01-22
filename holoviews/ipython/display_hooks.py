
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
except:
    mpld3 = None

from ..core import DataElement, Element, HoloMap, AdjointLayout, GridLayout,\
 AxisLayout, ViewTree, Overlay
from ..element import Raster
from ..plotting import LayoutPlot, GridPlot, MatrixGridPlot, Plot
from . import magics
from .magics import ViewMagic, ChannelMagic, OptsMagic
from .widgets import IPySelectionWidget, SelectionWidget, ScrubberWidget

# To assist with debugging of display hooks
ENABLE_TRACEBACKS=True

#==================#
# Helper functions #
#==================#


def get_plot_size():
    factor = ViewMagic.PERCENTAGE_SIZE / 100.0
    return (Plot.size[0] * factor,
            Plot.size[1] * factor)


def animate(anim, writer, mime_type, anim_kwargs, extra_args, tag):
    if extra_args != []:
        anim_kwargs = dict(anim_kwargs, extra_args=extra_args)

    if not hasattr(anim, '_encoded_video'):
        with NamedTemporaryFile(suffix='.%s' % mime_type) as f:
            anim.save(f.name, writer=writer, **anim_kwargs)
            video = open(f.name, "rb").read()
        anim._encoded_video = base64.b64encode(video).decode("utf-8")
    return tag.format(b64=anim._encoded_video,
                      mime_type=mime_type)


def HTML_video(plot):
    anim = plot.anim(fps=ViewMagic.FPS)
    writers = animation.writers.avail
    for fmt in [ViewMagic.VIDEO_FORMAT] + list(magics.ANIMATION_OPTS.keys()):
        if magics.ANIMATION_OPTS[fmt][0] in writers:
            try:
                return animate(anim, *magics.ANIMATION_OPTS[fmt])
            except: pass
    msg = "<b>Could not generate %s animation</b>" % ViewMagic.VIDEO_FORMAT
    if sys.version_info[0] == 3 and mpl.__version__[:-2] in ['1.2', '1.3']:
        msg = "<b>Python 3 Matplotlib animation support broken &lt;= 1.3</b>"
    raise Exception(msg)


def first_frame(plot):
    "Only display the first frame of an animated plot"
    return figure_display(plot[0])

def middle_frame(plot):
    "Only display the (approximately) middle frame of an animated plot"
    middle_frame = int(len(plot) / 2)
    return figure_display(plot[middle_frame])

def last_frame(plot):
    "Only display the last frame of an animated plot"
    return figure_display(plot[len(plot)])


def figure_display(fig, size=None, message=None, max_width='100%'):
    if size is not None:
        inches = size / float(fig.dpi)
        fig.set_size_inches(inches, inches)

    if ViewMagic.FIGURE_FORMAT.lower() == 'mpld3' and mpld3:
        mpld3.plugins.connect(fig, mpld3.plugins.MousePosition(fontsize=14))
        html = "<center>" + mpld3.fig_to_html(fig) + "<center/>"
    else:
        figdata = print_figure(fig, ViewMagic.FIGURE_FORMAT)
        if ViewMagic.FIGURE_FORMAT.lower()=='svg':
            mime_type = 'svg+xml'
            figdata = figdata.encode("utf-8")
        else:
            mime_type = 'png'
        prefix = 'data:image/%s;base64,' % mime_type
        b64 = prefix + base64.b64encode(figdata).decode("utf-8")
        if size is not None:
            html = "<center><img height='%d' width='%d' style='max-width:%s' " \
                   "src='%s'/><center/>" % (size, size, b64, max_width)
        else:
            html = "<center><img src='%s' style='max-width:%s'/><center/>" % (b64, max_width)
    plt.close(fig)
    return html if (message is None) else '<b>%s</b></br>%s' % (message, html)



#===============#
# Display hooks #
#===============#

def sanitized_repr(obj):
    "Sanitize text output for HTML display"
    return repr(obj).replace('\n', '<br>').replace(' ', '&nbsp;')

def max_frame_warning(max_frames):
    sys.stderr.write("Skipping matplotlib display to avoid "
                     "lengthy animation render times\n"
                     "[Total item frames exceeds ViewMagic.MAX_FRAMES (%d)]"
                     % max_frames)

def process_view_magics(obj):
    "Hook into %%opts and %%channels magics to process displayed element"
    invalid_styles = OptsMagic.set_view_options(obj)
    if invalid_styles: return invalid_styles
    invalid_channels = ChannelMagic.set_channels(obj)
    if invalid_channels: return invalid_channels

def display_hook(fn):
    @wraps(fn)
    def wrapped(view, **kwargs):
        try:
            return fn(view, **kwargs)
        except:
            if ENABLE_TRACEBACKS:
                traceback.print_exc()
    return wrapped

def render(plot):
    try:
        return render_anim(plot)
    except Exception as e:
        return str(e)+'<br/>'+figure_display(plot())

@display_hook
def animation_display(anim):
    return animate(anim, *magics.ANIMATION_OPTS[ViewMagic.VIDEO_FORMAT])

def widget_display(view):
    if ViewMagic.VIDEO_FORMAT == 'scrubber':
        return ScrubberWidget(view)()
    mode = ViewMagic.VIDEO_FORMAT[1]
    if mode == 'embedded':
        return SelectionWidget(view)()
    elif mode == 'cached':
        return IPySelectionWidget(view, cached=True)()
    else:
        return IPySelectionWidget(view, cached=False)()

@display_hook
def map_display(vmap, size=256):
    if not isinstance(vmap, HoloMap): return None
    magic_info = process_view_magics(vmap)
    if magic_info: return magic_info
    opts = dict(Element.options.plotting(vmap).opts, size=get_plot_size())
    mapplot = Plot.defaults[vmap.type](vmap, **opts)
    if len(mapplot) == 0:
        return sanitized_repr(vmap)
    elif len(mapplot) > ViewMagic.MAX_FRAMES:
        max_frame_warning(ViewMagic.MAX_FRAMES)
        return sanitized_repr(vmap)
    elif len(mapplot) == 1:
        fig = mapplot()
        return figure_display(fig)
    elif isinstance(ViewMagic.VIDEO_FORMAT, tuple) or\
                    ViewMagic.VIDEO_FORMAT == 'scrubber':
        return widget_display(vmap)

    return render(mapplot)


@display_hook
def layout_display(layout, size=256):
    if isinstance(layout, AdjointLayout): layout = GridLayout([layout])
    if not isinstance(layout, (ViewTree, GridLayout)): return None
    shape = layout.shape
    magic_info = process_view_magics(layout)
    if magic_info: return magic_info
    grid_size = (shape[1]*get_plot_size()[1],
                 shape[0]*get_plot_size()[0])

    opts = dict(Element.options.plotting(layout).opts, size=grid_size)
    layoutplot = LayoutPlot(layout, **opts)

    if isinstance(layout, ViewTree):
        if layout._display == 'auto':
            branches = len(set([path[0] for path in layout.data.keys()]))
            if branches > ViewMagic.MAX_BRANCHES:
                return '<tt>'+ sanitized_repr(layout) + '</tt>'
            elif len(layout.data) * len(layoutplot) > ViewMagic.MAX_FRAMES:
                max_frame_warning(ViewMagic.MAX_FRAMES)
                return '<tt>'+ sanitized_repr(layout) + '</tt>'

    if len(layoutplot) == 1:
        fig = layoutplot()
        return figure_display(fig)
    elif isinstance(ViewMagic.VIDEO_FORMAT, tuple) or\
                    ViewMagic.VIDEO_FORMAT == 'scrubber':
        return widget_display(layoutplot)

    return render(layoutplot)

@display_hook
def grid_display(grid, size=256):
    if not isinstance(grid, AxisLayout): return None

    max_dim = max(grid.shape)
    # Reduce plot size as AxisLayout gets larger
    shape_factor = 1. / max_dim
    # Expand small grids to a sensible viewing size
    expand_factor = 1 + (max_dim - 1) * 0.1
    scale_factor = expand_factor * shape_factor
    grid_size = (scale_factor * grid.shape[0] * get_plot_size()[0],
                 scale_factor * grid.shape[1] * get_plot_size()[1])

    magic_info = process_view_magics(grid)
    if magic_info: return magic_info
    layer_types = grid.layer_types
    if len(layer_types) == 1 and issubclass(layer_types[0], Raster):
        plot_type = MatrixGridPlot
    else:
        plot_type = GridPlot
    gridplot = plot_type(grid, size=grid_size)
    if len(gridplot) > ViewMagic.MAX_FRAMES:
        max_frame_warning(ViewMagic.MAX_FRAMES)
        return sanitized_repr(grid)
    if len(gridplot) == 1:
        fig = gridplot()
        return figure_display(fig)
    elif isinstance(ViewMagic.VIDEO_FORMAT, tuple) or\
                    ViewMagic.VIDEO_FORMAT == 'scrubber':
        return widget_display(grid)

    return render(gridplot)

@display_hook
def view_display(view, size=256):
    if not isinstance(view, DataElement): return None
    magic_info = process_view_magics(view)
    if magic_info: return magic_info
    opts = dict(Element.options.plotting(view).opts, size=get_plot_size())
    fig = Plot.defaults[view.__class__](view, **opts)()
    return figure_display(fig)

render_anim = HTML_video

def set_display_hooks(ip):
    html_formatter = ip.display_formatter.formatters['text/html']
    html_formatter.for_type_by_name('matplotlib.animation', 'FuncAnimation', animation_display)
    html_formatter.for_type(Element, view_display)
    html_formatter.for_type(HoloMap, map_display)
    html_formatter.for_type(AdjointLayout, layout_display)
    html_formatter.for_type(GridLayout, layout_display)
    html_formatter.for_type(AxisLayout, grid_display)
    html_formatter.for_type(ViewTree, layout_display)
    html_formatter.for_type(Overlay, view_display)
