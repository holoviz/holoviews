"""
Definition and registration of display hooks for the IPython Notebook.
"""
from functools import wraps
import sys, traceback

import IPython
import param

from ..core.options import Store, StoreOptions
from ..core import Element, ViewableElement, UniformNdMapping, HoloMap, AdjointLayout, NdLayout,\
    GridSpace, Layout, displayable, undisplayable_info, CompositeOverlay
from ..core.traversal import unique_dimkeys, bijective
from .magics import OutputMagic, OptsMagic

from .archive import notebook_archive
# To assist with debugging of display hooks
ABBREVIATE_TRACEBACKS=True

#==================#
# Helper functions #
#==================#

def first_frame(plot, renderer, **kwargs):
    "Only display the first frame of an animated plot"
    plot.update(0)
    return display_frame(plot, renderer, **kwargs)

def middle_frame(plot, renderer, **kwargs):
    "Only display the (approximately) middle frame of an animated plot"
    middle_frame = int(len(plot) / 2)
    plot.update(middle_frame)
    return display_frame(plot, renderer, **kwargs)

def last_frame(plot, renderer, **kwargs):
    "Only display the last frame of an animated plot"
    plot.update(len(plot))
    return display_frame(plot, renderer, **kwargs)

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


#==================================================#
# HTML/Javascript generation given a Plot instance #
#==================================================#


def display_video(plot, renderer, holomap_format, dpi, fps, css, **kwargs):
    """
    Allows the animation render policy to be changed, e.g to show only
    the middle frame using middle_frame for testing notebooks.

    See render_anim variable below (default is display_video)
    """
    if OutputMagic.options['holomap'] == 'repr': return None
    try:
        if render_anim is not None:
            return render_anim(plot, renderer, dpi=dpi, css=css, **kwargs)
        return renderer.html(plot, holomap_format, css)
    except Exception as e:
        plot.update(0)
        return str(e)+'<br/>'+display_frame(plot, renderer,  dpi=dpi, css=css, **kwargs)


def display_widgets(plot, renderer, holomap_format, widget_mode, **kwargs):
    "Display widgets applicable to the specified element"
    isuniform = plot.uniform
    islinear = bijective(plot.keys)
    if not isuniform and holomap_format == 'widgets':
        param.Parameterized.warning("%s is not uniform, falling back to scrubber widget."
                                    % type(plot).__name__)
        holomap_format = 'scrubber'

    if holomap_format == 'auto':
        holomap_format = 'scrubber' if islinear or not isuniform else 'widgets'

    widget = 'scrubber' if holomap_format == 'scrubber' else 'selection'
    widget_cls = plot.renderer.widgets[widget]

    return widget_cls(plot, renderer=renderer, embed=(widget_mode == 'embed'),
                      display_options=kwargs)()



def display_frame(plot, renderer, figure_format, backend, dpi, css, message, **kwargs):
    """
    Display specified element as a figure. Note the plot instance
    needs to be initialized appropriately first.
    """
    html = renderer.html(plot, figure_format, css)
    return html if (message is None) else '<b>%s</b></br>%s' % (message, html)


def render_plot(plot, widget_mode, message=None):
    """
    Used by the display hooks to render a plot according to the
    following policy:

    1. If there is a single frame, render it as a figure.
    2. If in widget mode, render as a widget
    3. Otherwise render it as an animation, falling back to a figure
    if there is an exception.
    """

    figure_format =  OutputMagic.options['fig']
    if figure_format == 'repr': return None
    kwargs = dict(widget_mode=widget_mode,
                  message=message,
                  figure_format = figure_format,
                  holomap_format= OutputMagic.options['holomap'],
                  backend =       OutputMagic.options['backend'],
                  dpi=            OutputMagic.options['dpi'],
                  css=            OutputMagic.options['css'],
                  fps=            OutputMagic.options['fps'])

    renderer = OutputMagic.renderer(dpi=kwargs['dpi'], fps=kwargs['fps'])
    with renderer.state():
        if len(plot) == 1:
            plot.update(0)
            return display_frame(plot, renderer, **kwargs)
        elif widget_mode is not None:
            return display_widgets(plot, renderer, **kwargs)
        else:
            return display_video(plot, renderer, **kwargs)

#===============#
# Display hooks #
#===============#


def display_hook(fn):
    @wraps(fn)
    def wrapped(element):
        optstate = StoreOptions.state(element)
        try:
            widget_mode = OutputMagic.options['widgets']
            # If widget_mode is None, widgets are not being used
            widget_mode = (widget_mode if OutputMagic.options['holomap']
                           in OutputMagic.inbuilt_formats else None)
            html = fn(element,
                      size=OutputMagic.options['size'],
                      max_frames=OutputMagic.options['max_frames'],
                      max_branches = OutputMagic.options['max_branches'],
                      widget_mode = widget_mode)
            notebook_archive.add(element, html=html)
            keys = ['fig', 'holomap', 'size', 'fps', 'dpi']
            filename = OutputMagic.options['filename']
            if filename:
                options = {k:OutputMagic.options[k] for k in keys}
                if options['holomap']  in OutputMagic.inbuilt_formats:
                    options['holomap'] = None

                OutputMagic.renderer(**options).save(element, filename)

            return html
        except Exception as e:
            StoreOptions.state(element, state=optstate)
            if ABBREVIATE_TRACEBACKS:
                info = dict(name=type(e).__name__,
                            message=str(e).replace('\n','<br>'))
                return "<b>{name}</b><br>{message}".format(**info)
            else:
                traceback.print_exc()
    return wrapped


@display_hook
def element_display(element,size, max_frames, max_branches, widget_mode):
    if not isinstance(element, ViewableElement): return None
    if type(element) == Element:                 return None
    info = process_object(element)
    if info: return info

    if element.__class__ not in Store.registry[OutputMagic.backend()]: return None
    plot_class = Store.registry[OutputMagic.backend()][element.__class__]
    element_plot = plot_class(element,
                              **OutputMagic.renderer().plot_options(element, size))

    return render_plot(element_plot, False)


class Warning(param.Parameterized): pass
display_warning = Warning(name='Warning')

@display_hook
def map_display(vmap, size, max_frames, max_branches, widget_mode):
    if not isinstance(vmap, HoloMap): return None

    if not displayable(vmap):
        display_warning.warning("Nesting %ss within a HoloMap makes it difficult "
                                "to access your data or control how it appears; "
                                "we recommend calling .collate() on the HoloMap "
                                "in order to follow the recommended nesting "
                                "structure shown in the Composing Data tutorial"
                                "(http://git.io/vtIQh)" % vmap.type.__name__)
        return display(vmap.collate(), raw=True)

    info = process_object(vmap)
    if info: return info
    if vmap.type not in Store.registry[OutputMagic.backend()]:  return None

    plot_class = Store.registry[OutputMagic.backend()][vmap.type]
    mapplot = plot_class(vmap, **OutputMagic.renderer().plot_options(vmap, size))
    if len(mapplot) == 0:
        return sanitize_HTML(vmap)
    elif len(mapplot) > max_frames:
        max_frame_warning(max_frames)
        return sanitize_HTML(vmap)

    return render_plot(mapplot, widget_mode)


@display_hook
def layout_display(layout, size, max_frames, max_branches, widget_mode):
    if isinstance(layout, AdjointLayout): layout = Layout.from_values(layout)
    if not isinstance(layout, (Layout, NdLayout)): return None

    if not displayable(layout):
        try:
            display_warning.warning(
                "Layout contains HoloMaps which are not nested in the "
                "recommended format for accessing your data; calling "
                ".collate() on these objects will resolve any violations "
                "of the recommended nesting presented in the Composing Data "
                "tutorial (http://git.io/vqs03)")
            expanded = []
            for el in layout.values():
                if isinstance(el, HoloMap) and not displayable(el):
                    collated_layout = Layout.from_values(el.collate())
                    expanded.extend(collated_layout.values())
            layout = Layout(expanded)
        except:
            return undisplayable_info(layout, html=True)

    nframes = len(unique_dimkeys(layout)[1])

    info = process_object(layout)
    if info: return info

    plot_class = Store.registry[OutputMagic.backend()][Layout]
    layoutplot = plot_class(layout,
                            **OutputMagic.renderer().plot_options(layout, size))

    if isinstance(layout, Layout):
        if layout._display == 'auto':
            branches = len(set([path[0] for path in list(layout.data.keys())]))
            if branches > max_branches:
                return '<tt>'+ sanitize_HTML(layout) + '</tt>'
            elif len(layout.data) * nframes > max_frames:
                max_frame_warning(max_frames)
                return '<tt>'+ sanitize_HTML(layout) + '</tt>'

    return render_plot(layoutplot, widget_mode)


@display_hook
def grid_display(grid, size, max_frames, max_branches, widget_mode):
    if not isinstance(grid, GridSpace): return None

    if not displayable(grid):
        return undisplayable_info(grid, html=True)

    info = process_object(grid)
    if info: return info

    plot_class = Store.registry[OutputMagic.backend()][GridSpace]
    gridplot = plot_class(grid, **OutputMagic.renderer().plot_options(grid, size))

    if len(gridplot) > max_frames:
        max_frame_warning(max_frames)
        return sanitize_HTML(grid)

    return render_plot(gridplot, widget_mode)


def display(obj, raw=False):
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
    elif isinstance(obj, HoloMap):
        html = map_display(obj)
    else:
        return repr(obj) if raw else IPython.display.display(obj)
    return html if raw else IPython.display.display(IPython.display.HTML(html))


def pprint_display(obj):
    # If pretty printing is off, return None (will fallback to repr)
    ip = get_ipython()  #  # pyflakes:ignore (in IPython namespace)
    if not ip.display_formatter.formatters['text/plain'].pprint:
        return None
    return display(obj, raw=True)


# display_video output by default, but may be set to first_frame,
# middle_frame or last_frame (e.g. for testing purposes)
render_anim = None

def set_display_hooks(ip):
    html_formatter = ip.display_formatter.formatters['text/html']
    html_formatter.for_type(ViewableElement, pprint_display)
    html_formatter.for_type(UniformNdMapping, pprint_display)
    html_formatter.for_type(AdjointLayout, pprint_display)
    html_formatter.for_type(Layout, pprint_display)
