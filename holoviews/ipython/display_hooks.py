"""
Definition and registration of display hooks for the IPython Notebook.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

try:    from matplotlib import animation
except: animation = None

from functools import wraps
import sys, traceback, base64

try:
    import mpld3
except:
    mpld3 = None

import param

from ..core.options import Store
from ..core import Element, ViewableElement, HoloMap, AdjointLayout, NdLayout,\
    NdOverlay, GridSpace, Layout, Overlay
from ..core.traversal import unique_dimkeys, bijective
from ..element import Raster
from ..plotting import LayoutPlot, GridPlot, RasterGridPlot
from ..plotting import ANIMATION_OPTS, HTML_TAGS, opts, get_plot_size
from .magics import OutputMagic, OptsMagic
from .widgets import IPySelectionWidget, SelectionWidget, ScrubberWidget

from .archive import notebook_archive

OutputMagic.ANIMATION_OPTS = ANIMATION_OPTS

# To assist with debugging of display hooks
ENABLE_TRACEBACKS=True


#==================#
# Helper functions #
#==================#


def animate(anim, dpi, writer, fmt, anim_kwargs, extra_args):
    if extra_args != []:
        anim_kwargs = dict(anim_kwargs, extra_args=extra_args)

    renderer = Store.renderer.instance(dpi=dpi)
    data = renderer.anim_data(anim, fmt, writer, **anim_kwargs)
    b64data = base64.b64encode(data).decode("utf-8")
    (mime_type, tag) = HTML_TAGS[fmt]
    src = HTML_TAGS['base64'].format(mime_type=mime_type, b64=b64data)
    return  tag.format(src=src, mime_type=mime_type)


def HTML_video(plot):
    if OutputMagic.options['holomap'] == 'repr': return None
    dpi = OutputMagic.options['dpi']
    anim = plot.anim(fps=OutputMagic.options['fps'])
    writers = animation.writers.avail
    current_format = OutputMagic.options['holomap']
    for fmt in [current_format] + list(OutputMagic.ANIMATION_OPTS.keys()):
        if OutputMagic.ANIMATION_OPTS[fmt][0] in writers:
            try:
                return animate(anim, dpi, *OutputMagic.ANIMATION_OPTS[fmt])
            except: pass
    msg = "<b>Could not generate %s animation</b>" % current_format
    if sys.version_info[0] == 3 and mpl.__version__[:-2] in ['1.2', '1.3']:
        msg = "<b>Python 3 matplotlib animation support broken &lt;= 1.3</b>"
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


def sanitize_HTML(obj):
    "Sanitize text output for HTML display"
    return repr(obj).replace('\n', '<br>').replace(' ', '&nbsp;')

def max_frame_warning(max_frames):
    sys.stderr.write("Skipping matplotlib display to avoid "
                     "lengthy animation render times\n"
                     "[Total item frames exceeds max_frames on OutputMagic (%d)]"
                     % max_frames)

def process_object(obj):
    "Hook to process the object currently being displayed."
    invalid_options = OptsMagic.process_element(obj)
    if invalid_options: return invalid_options
    OutputMagic.info(obj)


def render(plot):
    try:
        return render_anim(plot)
    except Exception as e:
        return str(e)+'<br/>'+display_figure(plot())


def display_widgets(plot):
    "Display widgets applicable to the specified element"
    if OutputMagic.options['holomap'] == 'repr': return None
    if OutputMagic.options['fig'] == 'repr':
        return  "<center><b>Figure format must not be 'repr' when using widgets.</b></center>"
    widget_mode = OutputMagic.options['widgets']
    widget_format = OutputMagic.options['holomap']
    assert widget_mode is not None, "Mistaken call to display_widgets method"


    isuniform = plot.uniform
    islinear = bijective(plot.keys)
    if not isuniform and widget_format == 'widgets':
        param.Parameterized.warning("%s is not uniform, falling back to scrubber widget."
                                    % type(plot).__name__)
        widget_format == 'scrubber'

    if widget_format == 'auto':
        widget_format = 'scrubber' if islinear or not isuniform else 'widgets'

    if widget_format == 'scrubber':
        return ScrubberWidget(plot)()
    if widget_mode == 'embed':
        return SelectionWidget(plot)()
    elif widget_mode == 'cached':
        return IPySelectionWidget(plot, cached=True)()
    else:
        return IPySelectionWidget(plot, cached=False)()


def display_figure(fig, message=None, max_width='100%'):
    "Display widgets applicable to the specified element"
    if OutputMagic.options['fig'] == 'repr': return None

    figure_format = OutputMagic.options['fig']
    dpi = OutputMagic.options['dpi']
    backend = OutputMagic.options['backend']

    if backend == 'd3' and mpld3:
        fig.dpi = dpi
        mpld3.plugins.connect(fig, mpld3.plugins.MousePosition(fontsize=14))
        html = "<center>" + mpld3.fig_to_html(fig) + "<center/>"
    else:
        renderer = Store.renderer.instance(dpi=dpi)
        figdata = renderer.figure_data(fig, figure_format)
        if figure_format=='svg':
            figdata = figdata.encode("utf-8")
        b64 = base64.b64encode(figdata).decode("utf-8")
        (mime_type, tag) = HTML_TAGS[figure_format]
        src = HTML_TAGS['base64'].format(mime_type=mime_type, b64=b64)
        html = tag.format(src=src)
    plt.close(fig)
    return html if (message is None) else '<b>%s</b></br>%s' % (message, html)


#===============#
# Display hooks #
#===============#


def display_hook(fn):
    @wraps(fn)
    def wrapped(element, **kwargs):
        # If pretty printing is off, return None (will fallback to repr)
        ip = get_ipython()  #  # pyflakes:ignore (in IPython namespace)
        if not ip.display_formatter.formatters['text/plain'].pprint:
            return None
        try:
            widget_mode = OutputMagic.options['widgets']
            map_format  = OutputMagic.options['holomap']
            # If widget_mode is None, widgets are not being used
            widget_mode = (widget_mode if map_format in OutputMagic.inbuilt_formats else None)
            html = fn(element,
                      size=OutputMagic.options['size'],
                      dpi=OutputMagic.options['dpi'],
                      max_frames=OutputMagic.options['max_frames'],
                      max_branches = OutputMagic.options['max_branches'],
                      map_format = map_format,
                      widget_mode = widget_mode,
                      **kwargs)
            notebook_archive.add(element, html=html)
            keys = ['fig', 'holomap', 'size', 'fps', 'dpi']
            filename = OutputMagic.options['filename']
            if filename:
                options = {k:OutputMagic.options[k] for k in keys}
                if options['holomap']  in OutputMagic.inbuilt_formats:
                    options['holomap'] = None
                Store.renderer.instance(**options).save(element, filename)

            return html
        except:
            if ENABLE_TRACEBACKS:
                traceback.print_exc()
    return wrapped


@display_hook
def animation_display(anim, map_format, dpi=72, **kwargs):
    return animate(anim, dpi, *OutputMagic.ANIMATION_OPTS[map_format])


@display_hook
def element_display(element, size, **kwargs):
    if not isinstance(element, ViewableElement): return None
    if type(element) == Element:                 return None
    info = process_object(element)
    if info: return info
    if element.__class__ not in Store.registry: return None
    fig = Store.registry[element.__class__](element,
                                         **opts(element, get_plot_size(element, size)))()
    return display_figure(fig)


@display_hook
def map_display(vmap, size, map_format, max_frames, widget_mode, **kwargs):
    if not isinstance(vmap, HoloMap): return None
    info = process_object(vmap)
    if info: return info
    if vmap.type not in Store.registry:  return None
    mapplot = Store.registry[vmap.type](vmap,
                                        **opts(vmap, get_plot_size(vmap,size)))
    if len(mapplot) == 0:
        return sanitize_HTML(vmap)
    elif len(mapplot) > max_frames:
        max_frame_warning(max_frames)
        return sanitize_HTML(vmap)
    elif len(mapplot) == 1:
        fig = mapplot()
        return display_figure(fig)
    elif widget_mode is not None:
        return display_widgets(mapplot)
    else:
        return render(mapplot)


@display_hook
def layout_display(layout, size, map_format, max_frames, max_branches, widget_mode, **kwargs):
    if isinstance(layout, AdjointLayout): layout = Layout.from_values(layout)
    if not isinstance(layout, (Layout, NdLayout)): return None
    nframes = len(unique_dimkeys(layout)[1])

    info = process_object(layout)
    if info: return info
    layoutplot = LayoutPlot(layout, **opts(layout, get_plot_size(layout, size)))
    if isinstance(layout, Layout):
        if layout._display == 'auto':
            branches = len(set([path[0] for path in list(layout.data.keys())]))
            if branches > max_branches:
                return '<tt>'+ sanitize_HTML(layout) + '</tt>'
            elif len(layout.data) * nframes > max_frames:
                max_frame_warning(max_frames)
                return '<tt>'+ sanitize_HTML(layout) + '</tt>'

    if nframes == 1:
        fig = layoutplot()
        return display_figure(fig)
    elif widget_mode is not None:
        return display_widgets(layoutplot)
    else:
        return render(layoutplot)


@display_hook
def grid_display(grid, size, map_format, max_frames, max_branches, widget_mode, **kwargs):
    if not isinstance(grid, GridSpace): return None
    info = process_object(grid)
    if info: return info

    raster_fn = lambda x: True if isinstance(x, Raster) else False
    all_raster = all(grid.traverse(raster_fn, [Element]))
    if all_raster:
        plot_type = RasterGridPlot
    else:
        plot_type = GridPlot
    gridplot = plot_type(grid, **opts(grid, get_plot_size(grid, size)))

    if len(gridplot) > max_frames:
        max_frame_warning(max_frames)
        return sanitize_HTML(grid)
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
    html_formatter.for_type(Layout, layout_display)
    html_formatter.for_type(ViewableElement, element_display)
    html_formatter.for_type(Overlay, element_display)
    html_formatter.for_type(NdOverlay, element_display)
    html_formatter.for_type(HoloMap, map_display)
    html_formatter.for_type(AdjointLayout, layout_display)
    html_formatter.for_type(NdLayout, layout_display)
    html_formatter.for_type(GridSpace, grid_display)
