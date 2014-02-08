import matplotlib.pyplot as plt
try:    from matplotlib import animation
except: animation = None

from IPython.core.pylabtools import print_figure

from tempfile import NamedTemporaryFile

from dataviews import DataStack, DataLayer
from plots import Plot, GridLayoutPlot, viewmap
from sheetviews import SheetStack, SheetLayer, GridLayout, CoordinateGrid
from views import Stack, View


GIF_TAG = "<img src='data:image/gif;base64,{b64}'/>"

VIDEO_TAG = """<video controls>
 <source src="data:video/{mime_type};base64,{b64}" type="video/{mime_type}">
 Your browser does not support the video tag.
</video>"""

# 'format name':(animation writer, mime_type,  anim_kwargs, extra_args, tag)
ANIMATION_OPTS = {
    'webm':('ffmpeg', 'webm',  {},
            ['-vcodec', 'libvpx', '-b', '1000k'],
            VIDEO_TAG),
    'h264':('ffmpeg', 'mp4', {'codec':'libx264'},
            ['-pix_fmt', 'yuv420p'],
            VIDEO_TAG),
    'gif':('imagemagick', 'gif', {'fps':10}, [],
           GIF_TAG)
}

def select_format(format_priority):
    for fmt in format_priority:
        try:
            anim = animation.FuncAnimation(plt.figure(),
                                           lambda x: x, frames=[0,1])
            animate(anim, *ANIMATION_OPTS[fmt])
            return fmt
        except: pass
    return format_priority[-1]


def opts(obj, additional_opts=[]):
    default_options = ['size']
    options = default_options + additional_opts
    return dict((k, obj.metadata.get(k)) for k in options if (k in obj.metadata))


def anim_opts(obj, additional_opts=[]):
    default_options = ['fps']
    options = default_options + additional_opts
    return dict((k, obj.metadata.get(k)) for k in options if (k in obj.metadata))


def animate(anim, writer, mime_type, anim_kwargs, extra_args, tag):
    if extra_args != []:
        anim_kwargs = dict(anim_kwargs, extra_args=extra_args)

    if not hasattr(anim, '_encoded_video'):
        with NamedTemporaryFile(suffix='.%s' % mime_type) as f:
            anim.save(f.name, writer=writer, **anim_kwargs)
            video = open(f.name, "rb").read()
        anim._encoded_video = video.encode("base64")
    return tag.format(b64=anim._encoded_video,
                      mime_type=mime_type)


def HTML_video(plot, view):
    anim_kwargs =  dict((k, view.metadata[k]) for k in ['fps']
                        if (k in view.metadata))
    video_format = view.metadata.get('video_format', VIDEO_FORMAT)
    if video_format not in ANIMATION_OPTS.keys():
        raise Exception("Unrecognized video format: %s" % video_format)
    anim = plot.anim(**anim_kwargs)

    writers = animation.writers.avail
    for fmt in [video_format] + ANIMATION_OPTS.keys():
        if ANIMATION_OPTS[fmt][0] in writers:
            try:
                return animate(anim, *ANIMATION_OPTS[fmt])
            except: pass
    return "<b>Could not generate %s animation</b>" % video_format


def figure_display(fig, size=None, format='svg', message=None):
    if size is not None:
        inches = size / float(fig.dpi)
        fig.set_size_inches(inches, inches)
    prefix = 'data:image/png;base64,'
    b64 = prefix + print_figure(fig, 'png').encode("base64")
    if size is not None:
        html = "<img height='%d' width='%d' src='%s' />" % (size, size, b64)
    else:
        html = "<img src='%s' />" % b64
    plt.close(fig)
    return html if (message is None) else '<b>%s</b></br>%s' % (message, html)


def figure_fallback(plotobj):
        message = ('Cannot import matplotlib.animation' if animation is None
                   else 'Failed to generate matplotlib animation')
        fig =  plotobj()
        return figure_display(fig, message=message)

#===============#
# Display hooks #
#===============#

def animation_display(anim):
    return animate(anim, *ANIMATION_OPTS[VIDEO_FORMAT])


def stack_display(stack, size=256, format='svg'):
    if not isinstance(stack, Stack): return None
    stackplot = viewmap[stack.type](stack, **opts(stack))
    if len(stack) == 1:
        fig = stackplot()
        return figure_display(fig)

    try:    return HTML_video(stackplot, stack)
    except: return figure_fallback(stackplot)


def layout_display(grid, size=256, format='svg'):
    if not isinstance(grid, GridLayout): return None
    grid_size = grid.shape[1]*Plot.size[1], grid.shape[0]*Plot.size[0]
    gridplot = GridLayoutPlot(grid, **dict(opts(grid), size=grid_size))
    if len(grid)==1:
        fig =  gridplot()
        return figure_display(fig)

    try:     return HTML_video(gridplot, grid)
    except:  return figure_fallback(gridplot)


def projection_display(grid, size=256, format='svg'):
    if not isinstance(grid, CoordinateGrid): return None
    size_factor = 0.17
    grid_size = (size_factor*grid.shape[1]*Plot.size[1],
                 size_factor*grid.shape[0]*Plot.size[0])
    gridplot = viewmap[grid.__class__](grid, **dict(opts(grid), size=grid_size))
    if len(grid)==1:
        fig =  gridplot()
        return figure_display(fig)

    try:     return HTML_video(gridplot, grid)
    except:  return figure_fallback(gridplot)


def view_display(view, size=256, format='svg'):
    if not isinstance(view, View): return None
    fig = viewmap[view.__class__](view, **opts(view))()
    return figure_display(fig)


def update_matplotlib_rc():
    """
    Default changes to the matplotlib rc used by IPython Notebook.
    """
    import matplotlib
    rc= {'figure.figsize': (6.0,4.0),
         'figure.facecolor': 'white',
         'figure.edgecolor': 'white',
         'font.size': 10,
         'savefig.dpi': 72,
         'figure.subplot.bottom' : .125
         }
    matplotlib.rcParams.update(rc)


message = """Welcome to the Imagen IPython extension! (http://ioam.github.io/imagen/)"""

_loaded = False
VIDEO_FORMAT = select_format(['webm','h264','gif'])

def load_ipython_extension(ip, verbose=True):

    if verbose: print message

    global _loaded
    if not _loaded:
        _loaded = True
        html_formatter = ip.display_formatter.formatters['text/html']
        html_formatter.for_type_by_name('matplotlib.animation', 'FuncAnimation', animation_display)
        html_formatter.for_type(SheetLayer, view_display)
        html_formatter.for_type(DataLayer, view_display)
        html_formatter.for_type(SheetStack, stack_display)
        html_formatter.for_type(DataStack, stack_display)
        html_formatter.for_type(GridLayout, layout_display)
        html_formatter.for_type(CoordinateGrid, projection_display)

        update_matplotlib_rc()
