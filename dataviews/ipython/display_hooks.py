
import matplotlib as mpl
import matplotlib.pyplot as plt

try:    from matplotlib import animation
except: animation = None

from IPython.core.pylabtools import print_figure

from tempfile import NamedTemporaryFile
from functools import wraps
import sys, traceback, base64


from ..dataviews import Stack, View
from ..views import Annotation, Layout
from ..sheetviews import GridLayout, CoordinateGrid
from ..plots import Plot, GridLayoutPlot


from . import magics
from .magics import ViewMagic, ChannelMagic, OptsMagic
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


def figure_display(fig, size=None, message=None):
    if size is not None:
        inches = size / float(fig.dpi)
        fig.set_size_inches(inches, inches)

    mime_type = 'svg+xml' if ViewMagic.FIGURE_FORMAT.lower()=='svg' else 'png'
    prefix = 'data:image/%s;base64,' % mime_type
    b64 = prefix + base64.b64encode(print_figure(fig, ViewMagic.FIGURE_FORMAT)).decode("utf-8")
    if size is not None:
        html = "<center><img height='%d' width='%d' src='%s'/><center/>" % (size, size, b64)
    else:
        html = "<center><img src='%s' /><center/>" % b64
    plt.close(fig)
    return html if (message is None) else '<b>%s</b></br>%s' % (message, html)



#===============#
# Display hooks #
#===============#


def process_view_magics(obj):
    "Hook into %%opts and %%channels magics to process display view"
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

@display_hook
def stack_display(stack, size=256):
    if not isinstance(stack, Stack): return None
    magic_info = process_view_magics(stack)
    if magic_info: return magic_info
    opts = dict(View.options.plotting(stack).opts, size=get_plot_size())
    stackplot = Plot.defaults[stack.type](stack, **opts)
    if len(stackplot) == 0:
        return repr(stack)
    elif len(stackplot) == 1:
        fig = stackplot()
        return figure_display(fig)

    return render(stackplot)

@display_hook
def layout_display(grid, size=256):
    grid = GridLayout([grid]) if isinstance(grid, Layout) else grid
    if not isinstance(grid, (GridLayout)): return None
    magic_info = process_view_magics(grid)
    if magic_info: return magic_info
    grid_size = (grid.shape[1]*get_plot_size()[1],
                 grid.shape[0]*get_plot_size()[0])

    opts = dict(size=grid_size)
    gridplot = GridLayoutPlot(grid, **opts)
    if len(gridplot)==1:
        fig =  gridplot()
        return figure_display(fig)

    return render(gridplot)

@display_hook
def projection_display(grid, size=256):
    if not isinstance(grid, CoordinateGrid): return None
    size_factor = 0.17
    grid_size = (size_factor*grid.shape[1]*get_plot_size()[1],
                 size_factor*grid.shape[0]*get_plot_size()[0])
    magic_info = process_view_magics(grid)
    if magic_info: return magic_info
    opts = dict(View.options.plotting(list(grid.values())[-1]).opts, size=grid_size)
    gridplot = Plot.defaults[grid.__class__](grid, **opts)
    if len(gridplot)==1:
        fig =  gridplot()
        return figure_display(fig)

    return render(gridplot)

@display_hook
def view_display(view, size=256):
    if not isinstance(view, View): return None
    if isinstance(view, Annotation): return None
    magic_info = process_view_magics(view)
    if magic_info: return magic_info
    opts = dict(View.options.plotting(view).opts, size=get_plot_size())
    fig = Plot.defaults[view.__class__](view, **opts)()
    return figure_display(fig)

render_anim = HTML_video

def set_display_hooks(ip):
    html_formatter = ip.display_formatter.formatters['text/html']
    html_formatter.for_type_by_name('matplotlib.animation', 'FuncAnimation', animation_display)
    html_formatter.for_type(View, view_display)
    html_formatter.for_type(Stack, stack_display)
    html_formatter.for_type(Layout, layout_display)
    html_formatter.for_type(GridLayout, layout_display)
    html_formatter.for_type(CoordinateGrid, projection_display)
