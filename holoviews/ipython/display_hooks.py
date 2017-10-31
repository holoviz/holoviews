"""
Definition and registration of display hooks for the IPython Notebook.
"""
from functools import wraps
from contextlib import contextmanager

import sys, traceback

import IPython
from IPython import get_ipython

import holoviews
from holoviews.plotting import Plot
from ..core.options import (Store, StoreOptions, SkipRendering,
                            AbbreviatedException)
from ..core import (ViewableElement, UniformNdMapping,
                    HoloMap, AdjointLayout, NdLayout, GridSpace, Layout,
                    CompositeOverlay, DynamicMap)
from ..core.traversal import unique_dimkeys
from ..core.io import FileArchive
from ..util.settings import OutputSettings
from .magics import OptsMagic, OutputMagic

# To assist with debugging of display hooks
FULL_TRACEBACK = None
ABBREVIATE_TRACEBACKS = True

#==================#
# Helper functions #
#==================#


def max_frame_warning(max_frames):
    sys.stderr.write("Skipping regular visual display to avoid "
                     "lengthy animation render times\n"
                     "[Total item frames exceeds max_frames on OutputSettings (%d)]"
                     % max_frames)

def process_object(obj):
    "Hook to process the object currently being displayed."
    invalid_options = OptsMagic.process_element(obj)
    if invalid_options: return invalid_options
    OutputMagic.info(obj)


def render(obj, **kwargs):
    info = process_object(obj)
    if info:
        IPython.display.display(IPython.display.HTML(info))
        return


    if render_anim is not None:
        return render_anim(obj)

    backend = Store.current_backend
    renderer = Store.renderers[backend]

    # Drop back to png if pdf selected, notebook PDF rendering is buggy
    if renderer.fig == 'pdf':
        renderer = renderer.instance(fig='png')
    return renderer.html(obj, **kwargs)


def single_frame_plot(obj):
    """
    Returns plot, renderer and format for single frame export.
    """
    obj = Layout.from_values(obj) if isinstance(obj, AdjointLayout) else obj

    backend = Store.current_backend
    renderer = Store.renderers[backend]

    plot_cls = renderer.plotting_class(obj)
    plot = plot_cls(obj, **renderer.plot_options(obj, renderer.size))
    fmt = renderer.params('fig').objects[0] if renderer.fig == 'auto' else renderer.fig
    return plot, renderer, fmt


def first_frame(obj):
    "Only display the first frame of an animated plot"
    plot, renderer, fmt = single_frame_plot(obj)
    plot.update(0)
    return renderer.html(plot, fmt)

def middle_frame(obj):
    "Only display the (approximately) middle frame of an animated plot"
    plot, renderer, fmt = single_frame_plot(obj)
    middle_frame = int(len(plot) / 2)
    plot.update(middle_frame)
    return renderer.html(plot, fmt)

def last_frame(obj):
    "Only display the last frame of an animated plot"
    plot, renderer, fmt = single_frame_plot(obj)
    plot.update(len(plot))
    return renderer.html(plot, fmt)

#===============#
# Display hooks #
#===============#

def dynamic_optstate(element, state=None):
    # Temporary fix to avoid issues with DynamicMap traversal
    DynamicMap._deep_indexable = False
    optstate = StoreOptions.state(element,state=state)
    DynamicMap._deep_indexable = True
    return optstate

@contextmanager
def option_state(element):
    optstate = dynamic_optstate(element)
    raised_exception = False
    try:
        yield
    except Exception:
        raised_exception = True
        raise
    finally:
        if raised_exception:
            dynamic_optstate(element, state=optstate)

def display_hook(fn):
    @wraps(fn)
    def wrapped(element):
        global FULL_TRACEBACK
        if Store.current_backend is None:
            return

        try:
            html = fn(element,
                      max_frames=OutputSettings.options['max_frames'])

            # Only want to add to the archive for one display hook...
            disabled_suffixes = ['png_display', 'svg_display']
            if not any(fn.__name__.endswith(suffix) for suffix in disabled_suffixes):
                if type(holoviews.archive) is not FileArchive:
                    holoviews.archive.add(element, html=html)
            filename = OutputSettings.options['filename']
            if filename:
                Store.renderers[Store.current_backend].save(element, filename)

            return html
        except SkipRendering as e:
            if e.warn:
                sys.stderr.write(str(e))
            return None
        except AbbreviatedException as e:

            FULL_TRACEBACK = '\n'.join(traceback.format_exception(e.etype,
                                                                  e.value,
                                                                  e.traceback))
            info = dict(name=e.etype.__name__,
                        message=str(e.value).replace('\n','<br>'))
            msg =  '<i> [Call holoviews.ipython.show_traceback() for details]</i>'
            return "<b>{name}</b>{msg}<br>{message}".format(msg=msg, **info)

        except Exception:
            raise
    return wrapped


@display_hook
def element_display(element, max_frames):
    info = process_object(element)
    if info:
        IPython.display.display(IPython.display.HTML(info))
        return

    backend = Store.current_backend
    if type(element) not in Store.registry[backend]:
        return None

    # Drop back to png if pdf selected, notebook PDF rendering is buggy
    renderer = Store.renderers[backend]
    if renderer.fig == 'pdf':
        renderer = renderer.instance(fig='png')
    return renderer.html(element, fmt=renderer.fig)


@display_hook
def map_display(vmap, max_frames):
    if not isinstance(vmap, (HoloMap, DynamicMap)): return None
    if  isinstance(vmap, DynamicMap) and vmap.unbounded:
        dims = ', '.join('%r' % dim for dim in  vmap.unbounded)
        msg = ('DynamicMap cannot be displayed without explicit indexing '
               'as {dims} dimension(s) are unbounded. '
               '\nSet dimensions bounds with the DynamicMap redim.range '
               'or redim.values methods.')
        sys.stderr.write(msg.format(dims=dims))
        return None

    if len(vmap) == 0 and not isinstance(vmap, DynamicMap):
        return None

    elif len(vmap) > max_frames:
        max_frame_warning(max_frames)
        return None

    return render(vmap)


@display_hook
def layout_display(layout, max_frames):
    if isinstance(layout, AdjointLayout): layout = Layout.from_values(layout)
    if not isinstance(layout, (Layout, NdLayout)): return None

    nframes = len(unique_dimkeys(layout)[1])
    if nframes > max_frames:
        max_frame_warning(max_frames)
        return None

    return render(layout)


@display_hook
def grid_display(grid, max_frames):
    if not isinstance(grid, GridSpace): return None

    nframes = len(unique_dimkeys(grid)[1])
    if nframes > max_frames:
        max_frame_warning(max_frames)
        return None

    return render(grid)


def display(obj, raw=False, **kwargs):
    """
    Renders any HoloViews object to HTML and displays it
    using the IPython display function. If raw is enabled
    the raw HTML is returned instead of displaying it directly.
    """
    if isinstance(obj, GridSpace):
        with option_state(obj):
            html = grid_display(obj)
    elif isinstance(obj, (CompositeOverlay, ViewableElement)):
        with option_state(obj):
            html = element_display(obj)
    elif isinstance(obj, (Layout, NdLayout, AdjointLayout)):
        with option_state(obj):
            html = layout_display(obj)
    elif isinstance(obj, (HoloMap, DynamicMap)):
        with option_state(obj):
            html = map_display(obj)
    else:
        return repr(obj) if raw else IPython.display.display(obj, **kwargs)
    return html if raw else IPython.display.display(IPython.display.HTML(html))


def pprint_display(obj):
    if 'html' not in Store.display_formats:
        return None

    # If pretty printing is off, return None (fallback to next display format)
    ip = get_ipython()  #  # noqa (in IPython namespace)
    if not ip.display_formatter.formatters['text/plain'].pprint:
        return None
    return display(obj, raw=True)


@display_hook
def element_png_display(element, max_frames):
    """
    Used to render elements to PNG if requested in the display formats.
    """
    if 'png' not in Store.display_formats:
        return None
    info = process_object(element)
    if info:
        IPython.display.display(IPython.display.HTML(info))
        return


    backend = Store.current_backend
    if type(element) not in Store.registry[backend]:
        return None
    renderer = Store.renderers[backend]
    # Current renderer does not support PNG
    if 'png' not in renderer.params('fig').objects:
        return None

    data, info = renderer(element, fmt='png')
    return data


@display_hook
def element_svg_display(element, max_frames):
    """
    Used to render elements to SVG if requested in the display formats.
    """
    if 'svg' not in Store.display_formats:
        return None
    info = process_object(element)
    if info:
        IPython.display.display(IPython.display.HTML(info))
        return


    backend = Store.current_backend
    if type(element) not in Store.registry[backend]:
        return None
    renderer = Store.renderers[backend]
    # Current renderer does not support SVG
    if 'svg' not in renderer.params('fig').objects:
        return None
    data, info = renderer(element, fmt='svg')
    return data


# display_video output by default, but may be set to first_frame,
# middle_frame or last_frame (e.g. for testing purposes)
render_anim = None

def plot_display(plot):
    return plot.renderer.html(plot)


def set_display_hooks(ip):
    html_formatter = ip.display_formatter.formatters['text/html']
    html_formatter.for_type(ViewableElement, pprint_display)
    html_formatter.for_type(UniformNdMapping, pprint_display)
    html_formatter.for_type(AdjointLayout, pprint_display)
    html_formatter.for_type(Layout, pprint_display)
    # Give plot instances rich display
    html_formatter.for_type(Plot, plot_display)

    # Note: Disable additional hooks from calling archive
    #       (see disabled_suffixes variable in the display decorator)
    png_formatter = ip.display_formatter.formatters['image/png']
    png_formatter.for_type(ViewableElement, element_png_display)

    svg_formatter = ip.display_formatter.formatters['image/svg+xml']
    svg_formatter.for_type(ViewableElement, element_svg_display)
