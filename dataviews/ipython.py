import matplotlib.pyplot as plt
try:    from matplotlib import animation
except: animation = None

from IPython.core.pylabtools import print_figure
from IPython.core import page
from IPython.core.magic import Magics, magics_class, line_magic

import param
from tempfile import NamedTemporaryFile
import textwrap

from dataviews import DataStack, DataLayer
from plots import Plot, GridLayoutPlot, viewmap
from sheetviews import SheetStack, SheetLayer, GridLayout, CoordinateGrid
from views import Stack, View

WARN_MISFORMATTED_DOCSTRINGS = False
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



@magics_class
class ParamMagics(Magics):
    """
    Implements the %params magic which is useful for inspecting
    the parameters of any parameterized class or object. For
    example you can inspect Imagen's Gaussian pattern as follows:

    %params imagen.Gaussian
    """
    def __init__(self, *args, **kwargs):
        super(ParamMagics, self).__init__(*args, **kwargs)
        self.red = '\x1b[1;31m%s\x1b[0m'
        self.blue = '\x1b[1;34m%s\x1b[0m'
        self.cyan = '\x1b[1;36m%s\x1b[0m'
        self.green = '\x1b[1;32m%s\x1b[0m'

    def _get_param_info(self, obj, include_super=True):
        """
        Get the parameter dictionary, the list of modifed parameters
        and the dictionary or parameter values. If include_super is
        true, parameters are collected from the super classes.
        """
        params = dict(obj.params())
        if isinstance(obj,type):
            changed = []
            val_dict = dict((k,p.default) for (k,p) in params.items())
            self_class = obj
        else:
            changed = [name for (name,_) in obj.get_param_values(onlychanged=True)]
            val_dict = dict(obj.get_param_values())
            self_class = obj.__class__

        if not include_super:
            params = dict((k,v) for (k,v) in params.items()
                          if k in self_class.__dict__.keys())

        params.pop('name') # This is already displayed in the title.
        return (params, val_dict, changed)


    def _param_docstrings(self, info, max_col_len=100, only_changed=False):
        """
        Build a string to succinctly display all the parameter
        docstrings in a clean format.
        """
        (params, val_dict, changed) = info
        contents = []
        displayed_params = {}
        for name, p in params.items():
            if only_changed and not (name in changed):
                continue
            displayed_params[name] = p

        right_shift = max(len(name) for name in displayed_params.keys())+2

        for i, name in enumerate(sorted(displayed_params)):
            p = displayed_params[name]
            heading = "%s: " % name
            unindented = textwrap.dedent("< No docstring available >" if p.doc is None else p.doc)

            if (WARN_MISFORMATTED_DOCSTRINGS
                and not unindented.startswith("\n")  and len(unindented.splitlines()) > 1):
                param.main.warning("Multi-line docstring for %r is incorrectly formatted (should start with newline)" % name)
            # Strip any starting newlines
            while unindented.startswith("\n"):
                unindented = unindented[1:]

            lines = unindented.splitlines()
            if len(lines) > 1:
                tail = ['%s%s' % (' '  * right_shift, line) for line in lines[1:]]
                all_lines = [ heading.ljust(right_shift) + lines[0]] + tail
            else:
                all_lines = [ heading.ljust(right_shift) + lines[0]]

            if i % 2:
                contents.extend([self.red %el for el in all_lines])
            else:
                contents.extend([self.blue %el for el in all_lines])

        return "\n".join(contents)


    def _build_table(self, info, order, max_col_len=40, only_changed=False):
        """
        Collect the information about parameters needed for
        tabulation.
        """
        info_dict, bounds_dict = {}, {}
        (params, val_dict, changed) = info
        col_widths = dict((k,0) for k in order)

        for name, p in params.items():
            if only_changed and not (name in changed):
                continue

            constant = 'C' if p.constant else 'V'
            readonly = 'RO' if p.readonly else 'RW'
            allow_None = ' AN' if hasattr(p, 'allow_None') and p.allow_None else ''

            mode = '%s %s%s' % (constant, readonly, allow_None)
            info_dict[name] = {'name': name, 'type':p.__class__.__name__,
                               'mode':mode}

            if hasattr(p, 'bounds'):
                lbound, ubound = (None,None) if p.bounds is None else p.bounds

                mark_lbound, mark_ubound = False, False
                # Use soft_bounds when bounds not defined.
                if hasattr(p, 'get_soft_bounds'):
                    soft_lbound, soft_ubound = p.get_soft_bounds()
                    if lbound is None and soft_lbound is not None:
                        lbound = soft_lbound
                        mark_lbound = True
                    if ubound is None and soft_ubound is not None:
                        ubound = soft_ubound
                        mark_ubound = True

                if (lbound, ubound) != (None,None):
                    bounds_dict[name] = (mark_lbound, mark_ubound)
                    info_dict[name]['bounds'] = '(%s, %s)' % (lbound, ubound)

            value = repr(val_dict[name])
            if len(value) > (max_col_len - 3):
                value = value[:max_col_len-3] + '...'
            info_dict[name]['value'] = value

            for col in info_dict[name]:
                max_width = max([col_widths[col], len(info_dict[name][col])])
                col_widths[col] = max_width

        return self._tabulate(info_dict, col_widths, changed, order, bounds_dict)


    def _tabulate(self, info_dict, col_widths, changed, order, bounds_dict):
        """
        Returns the supplied information as a table of parameter
        information suitable for printing or paging.
        """
        contents, tail = [], []
        column_set = set(k for row in info_dict.values() for k in row)
        columns = [col for col in order if col in column_set]

        title_row = []
        # Column headings
        for i, col in enumerate(columns):
            width = col_widths[col]+2
            col = col.capitalize()
            formatted = col.ljust(width) if i == 0 else col.center(width)
            title_row.append(formatted)
        contents.append(self.blue % ''.join(title_row)+"\n")

        # Print rows
        for row in sorted(info_dict):
            row_list = []
            info = info_dict[row]
            for i,col in enumerate(columns):
                width = col_widths[col]+2
                val = info[col] if (col in info) else ''
                formatted = val.ljust(width) if i==0 else val.center(width)

                if col == 'bounds' and bounds_dict.get(row,False):
                    (mark_lbound, mark_ubound) = bounds_dict[row]
                    lval, uval = formatted.rsplit(',')
                    lspace, lstr = lval.rsplit('(')
                    ustr, uspace = uval.rsplit(')')
                    lbound = lspace + '('+(self.cyan % lstr) if mark_lbound else lval
                    ubound = (self.cyan % ustr)+')'+uspace if mark_ubound else uval
                    formatted = "%s,%s" % (lbound, ubound)
                row_list.append(formatted)

            row_text = ''.join(row_list)
            if row in changed:
                row_text = self.red % row_text

            contents.append(row_text)

        return '\n'.join(contents+tail)

    @line_magic
    def params(self, parameter_s='', namespaces=None):
        """
        The %params line magic accepts a single argument, the
        parameterized object to be inspected.
        """
        order = ['name', 'changed', 'value', 'type', 'bounds', 'mode']
        if parameter_s=='':
            print "Please specify an object to inspect."
            return

        # Beware! Uses IPython internals that may change in future...
        obj = self.shell._object_find(parameter_s)
        if obj.found is False:
            print "Object %r not found in the namespace." % parameter_s
            return

        param_obj = obj.obj
        parameterized_object = isinstance(param_obj, param.Parameterized)
        parameterized_class = (isinstance(param_obj,type)
                               and  issubclass(param_obj,param.Parameterized))

        if not (parameterized_object or parameterized_class):
            print "Object is not a parameterized class or object."
            return

        param_info = self._get_param_info(param_obj, include_super=True)
        table = self._build_table(param_info, order, max_col_len=40,
                                  only_changed=False)

        docstrings = self._param_docstrings(param_info, max_col_len=100, only_changed=False)


        title = 'Parameters of %r' % param_obj.name
        dflt_msg = "Parameters changed from their default values are marked in red."
        heading_line = '=' * len(title)
        heading_text = "%s\n%s\n" % (title, heading_line)
        top_heading = (self.green % heading_text)
        top_heading += "\n%s" % (self.red % dflt_msg)
        top_heading += "\n%s" % (self.cyan % "Soft bound values are marked in cyan.")
        top_heading += '\nC/V= Constant/Variable, RO/RW = ReadOnly/ReadWrite, AN=Allow None'

        heading_text = 'Parameter docstrings:'
        heading_string = "%s\n%s" % (heading_text, '=' * len(heading_text))
        docstring_heading = (self.green % heading_string)
        page.page("%s\n\n%s\n\n%s\n\n%s" % (top_heading, table,
                                            docstring_heading, docstrings))



message = """Welcome to the Imagen IPython extension! (http://ioam.github.io/imagen/)"""

_loaded = False
VIDEO_FORMAT = select_format(['webm','h264','gif'])

def load_ipython_extension(ip, verbose=True):

    if verbose: print message

    global _loaded
    if not _loaded:
        _loaded = True

        ip.register_magics(ParamMagics)

        html_formatter = ip.display_formatter.formatters['text/html']
        html_formatter.for_type_by_name('matplotlib.animation', 'FuncAnimation', animation_display)
        html_formatter.for_type(SheetLayer, view_display)
        html_formatter.for_type(DataLayer, view_display)
        html_formatter.for_type(SheetStack, stack_display)
        html_formatter.for_type(DataStack, stack_display)
        html_formatter.for_type(GridLayout, layout_display)
        html_formatter.for_type(CoordinateGrid, projection_display)

        update_matplotlib_rc()
