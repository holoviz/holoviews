import matplotlib.pyplot as plt
try:    from matplotlib import animation
except: animation = None

from IPython.core.pylabtools import print_figure
from IPython.core import page
try:
    from IPython.core.magic import Magics, magics_class, line_magic, cell_magic, line_cell_magic
except:
    from nose.plugins.skip import SkipTest
    raise SkipTest("IPython extension requires IPython >= 0.13")

import param
from tempfile import NamedTemporaryFile
from functools import wraps
import textwrap, traceback

from dataviews import Stack
from plots import Plot, GridLayoutPlot, viewmap
from sheetviews import GridLayout, CoordinateGrid
from views import View

# Variables controlled via the %view magic
PERCENTAGE_SIZE, FPS, FIGURE_FORMAT  = 100, 20, 'png'

ENABLE_TRACEBACKS=False # To assist with debugging of display hooks

GIF_TAG = "<center><img src='data:image/gif;base64,{b64}'/><center/>"
VIDEO_TAG = """<center><video controls>
 <source src="data:video/{mime_type};base64,{b64}" type="video/{mime_type}">
 Your browser does not support the video tag.
</video><center/>"""

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

WARN_MISFORMATTED_DOCSTRINGS = False

#========#
# Magics #
#========#


@magics_class
class ViewMagic(Magics):
    """
    Magic to allow easy control over the display of dataviews. The
    figure and animation output formats, the animation frame rate and
    figure size can all be controlled.

    Usage: %view [png|svg] [webm|h264|gif[:<fps>]] [<percent size>]
    """

    anim_formats = ['webm','h264','gif']

    def __init__(self, *args, **kwargs):
        super(ViewMagic, self).__init__(*args, **kwargs)
        self.usage_info = "Usage: %view [png|svg] [webm|h264|gif[:<fps>]] [<percent size>]"
        self.usage_info += " (Arguments may be in any order)"

    @classmethod
    def option_completer(cls, k,v):
        return cls.anim_formats + ['png', 'svg']

    def _set_animation_options(self, anim_spec):
        """
        Parse the animation format and fps from the specification string.
        """
        global VIDEO_FORMAT, FPS
        format_choice, fps_str = ((anim_spec, None) if (':' not in anim_spec)
                                  else anim_spec.rsplit(':'))
        if format_choice not in self.anim_formats:
            print "Valid animations types: %s" % ', '.join(self.anim_formats)
            return False
        elif fps_str is None:
            VIDEO_FORMAT = format_choice
            return True
        try:
            fps = int(fps_str)
        except:
            print "Invalid frame rate: '%s'" %  fps_str
            return False

        VIDEO_FORMAT, FPS = format_choice, fps
        if format_choice == 'gif':
            ANIMATION_OPTS['gif'][2]['fps'] = fps
        return True


    def _set_size(self, size_spec):
        global PERCENTAGE_SIZE
        try:     size = int(size_spec)
        except:  size = None

        if (size is None) or (size < 0):
            print "Percentage size must be an integer larger than zero."
            return False
        else:
            PERCENTAGE_SIZE = size
            return True


    def _parse_settings(self, opts):
        global FIGURE_FORMAT
        fig_fmt = [('svg' in opts), ('png' in opts)]
        if all(fig_fmt):
            success = False
            print "Please select either png or svg for static output"
        elif True in fig_fmt:
            figure_format = ['svg', 'png'][fig_fmt.index(True)]
            FIGURE_FORMAT= figure_format
            opts.remove(figure_format)
        elif len(opts) == 0: success = True

        if not len(opts) or len(opts) > 2:
            success = not len(opts)
        elif len(opts) == 1:
            success = (self._set_animation_options(opts[0].lower())
                       if opts[0][0].isalpha() else self._set_size(opts[0]))
        elif sum(el[0].isalpha() for el in opts) in [0,2]:
            success = False
        else:
            (anim, size) = (opts if opts[0][0].isalpha()
                            else (opts[1], opts[0]))
            anim_success = self._set_animation_options(anim.lower())
            size_success = self._set_size(size)
            success =  anim_success and size_success

        return success

    @line_cell_magic
    def view(self, line, cell=None):
        global FIGURE_FORMAT,  VIDEO_FORMAT, PERCENTAGE_SIZE,  FPS
        start_opts = [FIGURE_FORMAT,  VIDEO_FORMAT, PERCENTAGE_SIZE,  FPS]

        opts = line.split()
        success = self._parse_settings(opts)

        if cell is None and success:
            info = (VIDEO_FORMAT.upper(), FIGURE_FORMAT.upper(), PERCENTAGE_SIZE, FPS)
            print "Displaying %s animation and %s figures [%d%% size, %s FPS]" % info
        elif cell and success:
            self.shell.run_cell(cell)
            [FIGURE_FORMAT,  VIDEO_FORMAT, PERCENTAGE_SIZE,  FPS] = start_opts
        else:
            print self.usage_info


class ParamPager(object):
    """
    Callable class that displays information about the supplied
    parameterized object or class in the IPython pager.
    """
    def __init__(self):

        self.order = ['name', 'changed', 'value', 'type', 'bounds', 'mode']
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
                param.main.warning("Multi-line docstring for %r is incorrectly formatted "
                                   " (should start with newline)" % name)
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


    def __call__(self, param_obj):
        """
        Given a parameterized object or class display information
        about the parameters in the IPython pager.
        """
        parameterized_object = isinstance(param_obj, param.Parameterized)
        parameterized_class = (isinstance(param_obj,type)
                               and  issubclass(param_obj,param.Parameterized))

        if not (parameterized_object or parameterized_class):
            print "Object is not a parameterized class or object."
            return

        param_info = self._get_param_info(param_obj, include_super=True)
        table = self._build_table(param_info, self.order, max_col_len=40,
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



@magics_class
class ParamMagics(Magics):
    """
    Implements the %params line magic used to inspect the parameters
    of a parameterized class or object.
    """
    def __init__(self, *args, **kwargs):
        super(ParamMagics, self).__init__(*args, **kwargs)
        self.param_pager = ParamPager()


    @line_magic
    def params(self, parameter_s='', namespaces=None):
        """
        The %params line magic accepts a single argument which is a
        handle on the parameterized object to be inspected. If the
        object can be found in the active namespace, information about
        the object's parameters is displayed in the IPython pager.

        Usage: %params <parameterized class or object>
        """
        if parameter_s=='':
            print "Please specify an object to inspect."
            return

        # Beware! Uses IPython internals that may change in future...
        obj = self.shell._object_find(parameter_s)
        if obj.found is False:
            print "Object %r not found in the namespace." % parameter_s
            return

        return self.param_pager(obj.obj)



@magics_class
class PlotOptsMagic(Magics):
    """
    Implements the %plotopts line magic used to set the plotting
    options on view objects.
    """
    def __init__(self, *args, **kwargs):
        super(PlotOptsMagic, self).__init__(*args, **kwargs)
        self.param_pager = ParamPager()


    @classmethod
    def _get_view(self, shell, token):
        """
        Given a string with the object name, return the view
        object. If the object is a GridLayout with integer indexing,
        apply the indexing to return the reference view object.
        """
        obj =  shell._object_find(token)
        found, indexed = obj.found, False
        # Attempt to index (e.g. for GridLayouts)
        if not found:
            indexed = (token.count('['), token.count(']')) == (1,1)
            if indexed:
                index_split = token.rsplit('[')
                indexing_string = '['+index_split[1]
                try:
                    indices = eval(indexing_string)
                except:
                    indices = False
                    print "Could not evaluate index %s" % indexing_string
                obj =  shell._object_find(index_split[0])
                found = obj.found and isinstance(obj.obj, GridLayout)

        # If the object still hasn't been found in the namespace...
        if not found:
            print "Object %r not found in the namespace." % token
            return False
        if indexed:   return obj.obj[tuple(indices)]
        else:         return obj.obj


    @classmethod
    def _plot_parameter_list(cls, view):
        """
        Lookup the plot type for a given view object and return the
        list of available parameter names and the plot class.
        """
        if not isinstance(view, (View, GridLayout)):
            print "Object %s is not a View" % view.__class__.__name__
            param_list =  []
        plotclass = viewmap.get(view.__class__, None)
        if not plotclass and isinstance(view, GridLayout):
            param_list = GridLayoutPlot.params().keys()
        elif not plotclass:
            print("Could not find appropriate plotting class for view of type %r "
                  % view.__class__.__name__)
            param_list =  []
        else:
            param_list =  plotclass.params().keys()
        return param_list, plotclass

    @classmethod
    def option_completer(cls, k,v):
        """
        Tab completion hook for the %plotopts magic.
        """
        view = cls._get_view(k, v.line.split()[1])
        if view is False: return []
        return ['%s=' % p for p in cls._plot_parameter_list(view)[0]]


    @line_magic
    def plotopts(self, parameter_s=''):
        """
        The %plotopts line magic to set the plotting options on a
        particular view object using keyword-value pairs. If no
        keywords are given, parameter information about the
        corresponding plot type is displayed in the IPython pager.

        Usage: %plotopts <view> [<keyword>=<value>]
        """
        if parameter_s=='':
            print "Please specify a view object to configure."
            return

        split = parameter_s.split()
        # Beware! Uses IPython internals that may change in future...
        obj = self._get_view(self.shell, split[0])
        if obj is False: return

        params, options = obj.params().keys(), {}
        for opt in split[1:]:
            try:
                option = eval("dict(%s)" % opt)
                options.update(option)
            except:
                print "Could not parse option %s" % opt

        allowed_params, plotclass = self._plot_parameter_list(obj)
        mismatches = set(options.keys()) - set(allowed_params)

        if mismatches:
            mismatch_list = ', '.join(repr(el) for el in mismatches)
            print "Parameters %s are not valid for this object"  % mismatch_list
        elif len(split) == 1:
            self.param_pager(plotclass)
        else:
            obj.metadata['plot_opts'] = options


#==================#
# Helper functions #
#==================#

def select_format(format_priority):
    for fmt in format_priority:
        try:
            anim = animation.FuncAnimation(plt.figure(),
                                           lambda x: x, frames=[0,1])
            animate(anim, *ANIMATION_OPTS[fmt])
            return fmt
        except: pass
    return format_priority[-1]


def get_plot_size():
    factor = PERCENTAGE_SIZE / 100.0
    return (Plot.size[0] * factor,
            Plot.size[1] * factor)


def opts(obj):
    extra_opts = obj.metadata.get('plot_opts', {})
    return dict({'size':get_plot_size()}, **extra_opts)


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
    anim = plot.anim(fps=FPS)
    writers = animation.writers.avail
    for fmt in [VIDEO_FORMAT] + ANIMATION_OPTS.keys():
        if ANIMATION_OPTS[fmt][0] in writers:
            try:
                return animate(anim, *ANIMATION_OPTS[fmt])
            except: pass
    return "<b>Could not generate %s animation</b>" % VIDEO_FORMAT


def figure_display(fig, size=None, message=None):
    if size is not None:
        inches = size / float(fig.dpi)
        fig.set_size_inches(inches, inches)

    mime_type = 'svg+xml' if FIGURE_FORMAT.lower()=='svg' else 'png'
    prefix = 'data:image/%s;base64,' % mime_type
    b64 = prefix + print_figure(fig, FIGURE_FORMAT).encode("base64")
    if size is not None:
        html = "<center><img height='%d' width='%d' src='%s'/><center/>" % (size, size, b64)
    else:
        html = "<center><img src='%s' /><center/>" % b64
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


def show_tracebacks(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except:
            if ENABLE_TRACEBACKS:
                traceback.print_exc()
    return wrapped

@show_tracebacks
def animation_display(anim):
    return animate(anim, *ANIMATION_OPTS[VIDEO_FORMAT])

@show_tracebacks
def stack_display(stack, size=256):
    if not isinstance(stack, Stack): return None
    stackplot = viewmap[stack.type](stack, **opts(stack))
    if len(stack) == 1:
        fig = stackplot()
        return figure_display(fig)

    try:    return HTML_video(stackplot, stack)
    except: return figure_fallback(stackplot)

@show_tracebacks
def layout_display(grid, size=256):
    if not isinstance(grid, GridLayout): return None
    grid_size = (grid.shape[1]*get_plot_size()[1],
                 grid.shape[0]*get_plot_size()[0])
    gridplot = GridLayoutPlot(grid, **dict(opts(grid), size=grid_size))
    if len(grid)==1:
        fig =  gridplot()
        return figure_display(fig)

    try:     return HTML_video(gridplot, grid)
    except:  return figure_fallback(gridplot)

@show_tracebacks
def projection_display(grid, size=256):
    if not isinstance(grid, CoordinateGrid): return None
    size_factor = 0.17
    grid_size = (size_factor*grid.shape[1]*get_plot_size()[1],
                 size_factor*grid.shape[0]*get_plot_size()[0])
    gridplot = viewmap[grid.__class__](grid, **dict(opts(grid), size=grid_size))
    if len(grid)==1:
        fig =  gridplot()
        return figure_display(fig)

    try:     return HTML_video(gridplot, grid)
    except:  return figure_fallback(gridplot)

@show_tracebacks
def view_display(view, size=256):
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


        ip.register_magics(ParamMagics)
        ip.register_magics(ViewMagic)
        ip.register_magics(PlotOptsMagic)

        # Configuring tab completion
        ip.set_hook('complete_command', PlotOptsMagic.option_completer, str_key = '%plotopts')
        ip.set_hook('complete_command', ViewMagic.option_completer, str_key = '%view')
        ip.set_hook('complete_command', ViewMagic.option_completer, str_key = '%%view')

        html_formatter = ip.display_formatter.formatters['text/html']
        html_formatter.for_type_by_name('matplotlib.animation', 'FuncAnimation', animation_display)

        html_formatter.for_type(View, view_display)
        html_formatter.for_type(Stack, stack_display)
        html_formatter.for_type(GridLayout, layout_display)
        html_formatter.for_type(CoordinateGrid, projection_display)

        update_matplotlib_rc()
