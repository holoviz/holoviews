"""
Definition and registration of display hooks for the IPython Notebook.
"""
from __future__ import unicode_literals, division, absolute_import, print_function

from functools import wraps
import sys, traceback, inspect, io

import IPython
from IPython.core.ultratb import AutoFormattedTB
import param

from ..core.options import Store, StoreOptions, BackendError
from ..core import (LabelledData, Element, ViewableElement, UniformNdMapping,
                    HoloMap, AdjointLayout, NdLayout, GridSpace, Layout,
                    CompositeOverlay, DynamicMap)
from ..core.traversal import unique_dimkeys, bijective
from .archive import notebook_archive
from .magics import OutputMagic, OptsMagic

# To assist with debugging of display hooks
FULL_TRACEBACK = None
ABBREVIATE_TRACEBACKS = True

#==================#
# Helper functions #
#==================#

def sanitize_HTML(obj):
    "Sanitize text output for HTML display"
    return repr(obj).replace('\n', '<br>').replace(' ', '&nbsp;')

def max_frame_warning(max_frames):
    sys.stderr.write("Skipping regular visual display to avoid "
                     "lengthy animation render times\n"
                     "[Total item frames exceeds max_frames on OutputMagic (%d)]"
                     % max_frames)

def process_object(obj):
    "Hook to process the object currently being displayed."
    invalid_options = OptsMagic.process_element(obj)
    if invalid_options: return invalid_options
    OutputMagic.info(obj)


def render(obj, **kwargs):
    info = process_object(obj)
    if info: return info

    if render_anim is not None:
        return render_anim(obj)

    backend = Store.current_backend
    return Store.renderers[backend].html(obj, **kwargs)


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


def display_hook(fn):
    @wraps(fn)
    def wrapped(element):
        global FULL_TRACEBACK
        optstate = StoreOptions.state(element)
        try:
            html = fn(element,
                      max_frames=OutputMagic.options['max_frames'],
                      max_branches = OutputMagic.options['max_branches'])
            notebook_archive.add(element, html=html)
            filename = OutputMagic.options['filename']
            if filename:
                Store.renderers[Store.current_backend].save(element, filename)

            return html
        except Exception as e:
            StoreOptions.state(element, state=optstate)
            frame = inspect.trace()[-1]
            mod = inspect.getmodule(frame[0])
            module = (mod.__name__ if mod else frame[1]).split('.')[0]
            backends = Store.renderers.keys()
            abbreviate =  isinstance(e, BackendError) or module in backends
            if ABBREVIATE_TRACEBACKS and abbreviate:
                AutoTB = AutoFormattedTB(mode = 'Verbose',color_scheme='Linux')
                buff = io.StringIO()
                AutoTB(out=buff)
                buff.seek(0)
                FULL_TRACEBACK = buff.read()
                info = dict(name=type(e).__name__,
                            message=str(e).replace('\n','<br>'))
                msg ='<i> [Call ipython.show_traceback() for details]</i>'
                return "<b>{name}</b>{msg}<br>{message}".format(msg=msg, **info)
            else:
                traceback.print_exc()
    return wrapped


@display_hook
def element_display(element, max_frames, max_branches):
    info = process_object(element)
    if info: return info

    backend = Store.current_backend
    if type(element) not in Store.registry[backend]:
        return None
    renderer = Store.renderers[backend]
    return renderer.html(element, fmt=renderer.fig)


@display_hook
def map_display(vmap, max_frames, max_branches):
    if not isinstance(vmap, (HoloMap, DynamicMap)): return None
    if len(vmap) == 0 and (not isinstance(vmap, DynamicMap) or vmap.sampled):
        return sanitize_HTML(vmap)
    elif len(vmap) > max_frames:
        max_frame_warning(max_frames)
        return sanitize_HTML(vmap)

    return render(vmap)


@display_hook
def layout_display(layout, max_frames, max_branches):
    if isinstance(layout, AdjointLayout): layout = Layout.from_values(layout)
    if not isinstance(layout, (Layout, NdLayout)): return None

    nframes = len(unique_dimkeys(layout)[1])
    if isinstance(layout, Layout):
        if layout._display == 'auto':
            branches = len(set([path[0] for path in list(layout.data.keys())]))
            if branches > max_branches:
                return '<tt>'+ sanitize_HTML(layout) + '</tt>'
            elif len(layout.data) * nframes > max_frames:
                max_frame_warning(max_frames)
                return '<tt>'+ sanitize_HTML(layout) + '</tt>'

    return render(layout)


@display_hook
def grid_display(grid, max_frames, max_branches):
    if not isinstance(grid, GridSpace): return None

    nframes = len(unique_dimkeys(grid)[1])
    if nframes > max_frames:
        max_frame_warning(max_frames)
        return sanitize_HTML(grid)

    return render(grid)


def display(obj, raw=False, **kwargs):
    """
    Renders any HoloViews object to HTML and displays it
    using the IPython display function. If raw is enabled
    the raw HTML is returned instead of displaying it directly.
    """
    if isinstance(obj, GridSpace):
        html = grid_display(obj)
    elif isinstance(obj, (CompositeOverlay, ViewableElement)):
        html = element_display(obj)
    elif isinstance(obj, (Layout, NdLayout, AdjointLayout)):
        html = layout_display(obj)
    elif isinstance(obj, (HoloMap, DynamicMap)):
        html = map_display(obj)
    else:
        return repr(obj) if raw else IPython.display.display(obj, **kwargs)
    return html if raw else IPython.display.display(IPython.display.HTML(html))


def pprint_display(obj):
    if 'html' not in Store.display_formats:
        return None

    # If pretty printing is off, return None (fallback to next display format)
    ip = get_ipython()  #  # pyflakes:ignore (in IPython namespace)
    if not ip.display_formatter.formatters['text/plain'].pprint:
        return None
    return display(obj, raw=True)


@display_hook
def element_png_display(element, max_frames, max_branches):
    """
    Used to render elements to PNG if requested in the display formats.
    """
    if 'png' not in Store.display_formats:
        return None
    info = process_object(element)
    if info: return info

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
def element_svg_display(element, max_frames, max_branches):
    """
    Used to render elements to SVG if requested in the display formats.
    """
    if 'svg' not in Store.display_formats:
        return None
    info = process_object(element)
    if info: return info

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

def set_display_hooks(ip):
    html_formatter = ip.display_formatter.formatters['text/html']
    html_formatter.for_type(ViewableElement, pprint_display)
    html_formatter.for_type(UniformNdMapping, pprint_display)
    html_formatter.for_type(AdjointLayout, pprint_display)
    html_formatter.for_type(Layout, pprint_display)

    png_formatter = ip.display_formatter.formatters['image/png']
    png_formatter.for_type(ViewableElement, element_png_display)

    svg_formatter = ip.display_formatter.formatters['image/svg+xml']
    svg_formatter.for_type(ViewableElement, element_svg_display)
