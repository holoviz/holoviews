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
import textwrap, traceback, itertools

from dataviews import Stack
from plots import Plot, GridLayoutPlot, viewmap
from sheetviews import GridLayout, CoordinateGrid
from views import View, Overlay
from styles import Styles, Style

# Variables controlled via the %view magic
PERCENTAGE_SIZE, FPS, FIGURE_FORMAT  = 100, 20, 'png'

ENABLE_TRACEBACKS=True # To assist with debugging of display hooks

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
        if not isinstance(view, (View, GridLayout, Stack)):
            print "Object %s is not a View" % view.__class__.__name__
            param_list =  []

        if isinstance(view, Stack):
            plotclass = viewmap[view.type]
        else:
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



@magics_class
class StyleMagic(Magics):
    """
    The %style and %%style line and cell magics allow customization of
    display style of dataviews. The %style line magic updates or
    creates a new style in the global StyleMap whereas the %%style
    cell magic sets a custom style that is associated with the
    displayed view object.
    """

    invalid_styles = []
    custom_styles = {}
    show_info = False
    show_labels = False

    @classmethod
    def collect(cls, obj, attr='style'):
        """
        Given a view object, build a dictionary of either the 'style'
        or 'label' attributes across overlays and grid layouts. The
        return value is a dictionary with the collected strings as
        keys and the associated view type as values.
        """
        group = {}
        if isinstance(obj, (Overlay, GridLayout)):
            for subview in obj:
                group.update(cls.collect(subview, attr))
        elif isinstance(obj, Stack):
            key_lists = [cls.collect(el, attr).keys() for el in obj]
            values = set(el for els in key_lists for el in els)
            for val in values:
                group.update({val:obj.type})
        else:
            value = '' if getattr(obj, attr, None) is None else getattr(obj, attr)
            group.update({value:obj})
        return group


    @classmethod
    def _basename(cls, name):
        """
        Strips out the 'Custom' prefix of styles names that have been
        customized with an object identifier string.
        """
        split = name.rsplit('>]_')
        if not name.startswith('Custom'):   return name
        elif len(split) == 2:               return split[1]
        else:
            raise Exception("Invalid style name %s" % name)


    @classmethod
    def _set_style_names(cls, obj, custom_name_map):
        """
        Update the style names on a view to the custom style name when
        there is a basename match as specified by the supplied
        dictionary.
        """
        if isinstance(obj, GridLayout):
            for subview in obj.values():
                cls._set_style_names(subview, custom_name_map)
        elif isinstance(obj.style, list):
            obj.style = [custom_name_map.get(cls._basename(s), s) for s in obj.style]
        elif cls._basename(obj.style) in custom_name_map:
            obj.style = custom_name_map[cls._basename(obj.style)]


    @classmethod
    def set_view_style(cls, obj):
        """
        To be called by the display hook which supplies the view
        object on which the style is to be customized.
        """

        if cls.show_labels:
            labels = cls.collect(obj, 'label').keys()
            info = (len(labels), labels.count(''))
            summary = '%d objects inspected, %d without labels. The set of labels found:<br><br>&emsp;' % info
            label_list = '<br>&emsp;'.join(['<b>%s</b>' % l for l in sorted(set(labels)) if l])
            return summary + label_list

        if not any([cls.invalid_styles, cls.custom_styles, cls.show_info]): return

        styles = cls.collect(obj, 'style')
        # The set of available style basenames in the view object
        available_styles = set(cls._basename(s) for s in styles)
        custom_styles = set(s for s in styles if s.startswith('Custom'))

        mismatch_set = set(cls.custom_styles.keys()) - available_styles
        mismatches = sorted(set(cls.invalid_styles) | mismatch_set)

        if cls.show_info or mismatches:
            return cls._style_info(obj, available_styles, mismatches, custom_styles)

        # The name map associates the style basename to the custom name
        name_mapping = {}
        for name, new_style in cls.custom_styles.items():
            # Create a custom style name for the object
            style_name = 'Custom[<' + obj.name + '>]_' + name
            # Register the new style in the StyleMap
            Styles[style_name] = new_style
            name_mapping[str(name)] = style_name

        # Link the object to the new custom style
        cls._set_style_names(obj, name_mapping)


    @classmethod
    def _style_info(cls, obj, available_styles, mismatches, custom_styles):
        """
        Format the style information as HTML, listing any mismatched
        names, the styles available for manipulation and the list of
        objects with customized styles.
        """
        obj_name = "<b>%s</b>" % obj.__class__.__name__
        if len(available_styles) == 0:
            return "<b>No styles are defined on the current %s</b>" % obj_name

        mismatch_str = ', '.join('<b>%r</b>' % el for el in mismatches)
        unavailable_msg = '%s not in ' % mismatch_str if mismatch_str else ''
        s = "%s %s customizable Styles:<br><br>" % (unavailable_msg, obj_name)
        max_len = max(len(s) for s in available_styles)
        for name in sorted(available_styles):
            padding = '&nbsp;'*(max_len - len(name))
            s += '&emsp;<code><b>%s</b>%s : %r</code><br>' % (name, padding, Styles[name])

        if custom_styles:
            s += '<br>Styles that have been customized for the displayed view:<br><br>'
            custom_names = [style_name.rsplit('>]_')[1] for style_name in custom_styles]
            max_len = max(len(s) for s in custom_names)
            for custom_style, custom_name in zip(custom_styles, custom_names):
                padding = '&nbsp;'*(max_len - len(custom_name))
                s += '&emsp;<code><b>%s</b>%s : %r</code><br>' % (custom_name,
                                                                  padding,
                                                                  Styles[custom_style])
        return s


    def _parse_styles(self, line):
        """
        Parse the arguments to the magic, returning one dictionary of
        updated Style objects for existing styles and a second
        dictionary containing new Style objects for styles that have
        not yet been defined.
        """
        updated_styles, new_styles  = {}, {}
        tokens = line.split()
        if tokens == []:
            return {}, []
        elif not tokens[0][0].isupper():
            raise SyntaxError("First token must be a Style name (a capitalized string)")

        # Split the input by the capitalized tokens
        style_names, settings_list, updates = [], [], []
        for upper, vals in itertools.groupby(tokens, key=lambda x: x[0].isupper()):
            values = list(vals)
            if upper and len(values) != 1:
                raise SyntaxError("Style names should be split by keywords")
            elif upper:
                style_name = values[0]
                matches = Styles.fuzzy_matches(style_name)
                style_names.append(style_name)
                updates.append(style_name in Styles.styles())
            else:
                settings_str = 'dict(' + ', '.join(values) + ')'
                try:     settings_list.append(eval(settings_str))
                except:  raise SyntaxError("Could not parse keywords '%s'" % values)

        for name, kwargs, update  in zip(style_names, settings_list, updates):
            if update:
                updated_styles[name] = Styles[str(name)](**kwargs)
            else:
                new_styles[name] = Styles[str(name)] = Style(**kwargs)
        return updated_styles, new_styles


    def _linemagic(self, line):
        """
        Update or create a Style in the StyleMap.

        Usage: %setstyle [-v] [StyleName] [<keyword>=<value>]

        The -v flag toggles verbose output.
        """
        verbose = False
        if str(line).startswith('-v'):
            verbose = True
            line = line.replace('-v', '')

        info = False
        if str(line).startswith('info'):
            verbose = True
            line = line.replace('info', '')


        updated_styles, new_styles = self._parse_styles(str(line))

        if not updated_styles and not new_styles:
            info = (len(Styles.keys()), len([k for k in Styles.keys() if k.startswith('Custom')]))
            print "There are a total of %d Styles defined including %d custom object styles." % info
            return

        for name, style in updated_styles.items():
            if verbose:
                print "Updated existing style %r : %r" % (name, style)
            Styles[name] = style

        for name, style in new_styles.items():
            if verbose:
                print "Created new style %r : %r" % (name, style)
            Styles[name] = style


    @cell_magic
    def labels(self, line='', cell=None):
        StyleMagic.show_labels = True
        self.shell.run_cell(cell)
        StyleMagic.show_labels = False

    @line_cell_magic
    def style(self, line='', cell=None):
        """
        Set a custom display style uniquely assigned to the views presented.

        Usage: %%style [StyleName] [<keyword>=<value>]

        Keyword-value pairs are assumed to separate the capitalized
        Style names to which they are assigned.
        """

        if cell is None:
            return self._linemagic(line)
        elif not line.strip():
            StyleMagic.show_info=True
        else:
            updated_styles, new_styles = self._parse_styles(str(line))
            (StyleMagic.custom_styles,
             StyleMagic.invalid_styles) = (updated_styles, new_styles.keys())
        # Run the cell in the updated environment
        self.shell.run_cell(cell)
        # Reset the class attributes
        StyleMagic.invalid_styles = []
        StyleMagic.custom_styles = {}
        StyleMagic.show_info=False



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
    invalid_styles = StyleMagic.set_view_style(stack)
    if invalid_styles: return invalid_styles
    stackplot = viewmap[stack.type](stack, **opts(stack))
    if len(stackplot) == 1:
        fig = stackplot()
        return figure_display(fig)

    try:    return HTML_video(stackplot, stack)
    except: return figure_fallback(stackplot)

@show_tracebacks
def layout_display(grid, size=256):
    if not isinstance(grid, GridLayout): return None
    invalid_styles = StyleMagic.set_view_style(grid)
    if invalid_styles: return invalid_styles
    grid_size = (grid.shape[1]*get_plot_size()[1],
                 grid.shape[0]*get_plot_size()[0])
    gridplot = GridLayoutPlot(grid, **dict(opts(grid), size=grid_size))
    if len(gridplot)==1:
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
    invalid_styles = StyleMagic.set_view_style(grid)
    if invalid_styles: return invalid_styles
    gridplot = viewmap[grid.__class__](grid, **dict(opts(grid), size=grid_size))
    if len(gridplot)==1:
        fig =  gridplot()
        return figure_display(fig)

    try:     return HTML_video(gridplot, grid)
    except:  return figure_fallback(gridplot)

@show_tracebacks
def view_display(view, size=256):
    if not isinstance(view, View): return None
    if isinstance(view, Annotation): return None
    invalid_styles = StyleMagic.set_view_style(view)
    if invalid_styles: return invalid_styles
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


all_line_magics = sorted(['%params', '%style', '%view', '%plotopts'])
all_cell_magics = sorted(['%%view', '%%style', '%%labels'])
message = """Welcome to the Dataviews IPython extension! (http://ioam.github.io/imagen/)"""
message += '\nAvailable magics: %s' % ', '.join(all_line_magics + all_cell_magics)

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
        ip.register_magics(StyleMagic)


        # Configuring tab completion
        ip.set_hook('complete_command', PlotOptsMagic.option_completer, str_key = '%plotopts')
        ip.set_hook('complete_command', ViewMagic.option_completer, str_key = '%view')
        ip.set_hook('complete_command', ViewMagic.option_completer, str_key = '%%view')
        ip.set_hook('complete_command', lambda x,v: Styles.styles(), str_key = '%%style')
        ip.set_hook('complete_command', lambda x,v: Styles.styles(), str_key = '%style')


        html_formatter = ip.display_formatter.formatters['text/html']
        html_formatter.for_type_by_name('matplotlib.animation', 'FuncAnimation', animation_display)

        html_formatter.for_type(View, view_display)
        html_formatter.for_type(Stack, stack_display)
        html_formatter.for_type(GridLayout, layout_display)
        html_formatter.for_type(CoordinateGrid, projection_display)

        update_matplotlib_rc()
