import matplotlib as mpl
import matplotlib.pyplot as plt
try:    from matplotlib import animation
except: animation = None

from IPython.core.pylabtools import print_figure
from IPython.core import page
try:
    from IPython.core.magic import Magics, magics_class, cell_magic, line_cell_magic
except:
    from nose.plugins.skip import SkipTest
    raise SkipTest("IPython extension requires IPython >= 0.13")

from param import ipython as param_ext
from tempfile import NamedTemporaryFile
from functools import wraps
import traceback, itertools, string
import base64
import sys

from ..dataviews import Stack, View
from ..plots import Plot, GridLayoutPlot, viewmap, channel_modes
from ..sheetviews import GridLayout, CoordinateGrid, SheetOverlay
from ..views import View, Overlay, Annotation, Layout
from ..options import channels, PlotOpts, StyleOpts, ChannelOpts



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

#========#
# Magics #
#========#

# ANSI color codes for the IPython pager
red   = '\x1b[1;31m%s\x1b[0m'
blue  = '\x1b[1;34m%s\x1b[0m'
green = '\x1b[1;32m%s\x1b[0m'
cyan = '\x1b[1;36m%s\x1b[0m'

# Corresponding HTML color codes
html_red = '#980f00'
html_blue = '#00008e'


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
            print("Valid animations types: %s" % ', '.join(self.anim_formats))
            return False
        elif fps_str is None:
            VIDEO_FORMAT = format_choice
            return True
        try:
            fps = int(fps_str)
        except:
            print("Invalid frame rate: '%s'" %  fps_str)
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
            print("Percentage size must be an integer larger than zero.")
            return False
        else:
            PERCENTAGE_SIZE = size
            return True


    def _parse_settings(self, opts):
        global FIGURE_FORMAT
        fig_fmt = [('svg' in opts), ('png' in opts)]
        if all(fig_fmt):
            success = False
            print("Please select either png or svg for static output")
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
            print("Displaying %s animation and %s figures [%d%% size, %s FPS]" % info)
        elif cell and success:
            self.shell.run_cell(cell)
            [FIGURE_FORMAT,  VIDEO_FORMAT, PERCENTAGE_SIZE,  FPS] = start_opts
        else:
            print(self.usage_info)



@magics_class
class ChannelMagic(Magics):

    custom_channels = {}

    @cell_magic
    def channels(self, line, cell=None):
        """
        The %%channels cell magic allows channel definitions to be
        defined on the displayed SheetOverlay.

        For instance, if you have three SheetViews (R,G and B)
        together in a SheetOverlay with labels 'R_Channel',
        'G_Channel', 'B_Channel' respectively, you can display this
        object as an RGB image using:

        %%channels R_Channel * G_Channel * B_Channel => RGBA []
        R * G * B

        The available operators are defined in the plots.channel_modes
        dictionary and additional arguments to the channel operator
        are supplied via keywords in the square brackets.
        """
        ChannelMagic.custom_channels = self._parse_channels(str(line))
        self.shell.run_cell(cell)
        ChannelMagic.custom_channels = {}


    @classmethod
    def _set_overlay_labels(cls, obj, label):
        """
        Labels on Overlays are used to index channel definitions.
        """
        if isinstance(obj, GridLayout):
            for subview in obj.values():
                cls._set_overlay_labels(subview, label)
        elif isinstance(obj, Stack) and issubclass(obj.type, Overlay):
            for overlay in obj:
                overlay.label = label
        elif isinstance(obj, Overlay):
            obj.label = label


    @classmethod
    def _set_channels(cls, obj, custom_channels, prefix):
        cls._set_overlay_labels(obj, prefix)
        for name, (pattern, params) in custom_channels.items():
            SheetOverlay.channels[prefix + '_' + name] = ChannelOpts(name, pattern,
                                                        **params)


    @classmethod
    def set_channels(cls, obj):
        prefix = 'Custom[<' + obj.name + '>]'
        if cls.custom_channels:
            cls._set_channels(obj, cls.custom_channels, prefix)


    def _parse_channels(self, line):
        """
        Parse the arguments to the magic, returning a dictionary of
        {'channel op name' : ('pattern', kwargs).
        """
        tokens = line.split()
        if tokens == []: return {}

        channel_split = [(el+']') for el in line.rsplit(']') if el.strip()]
        spec_split = [el.rsplit('=>') for el in channel_split]
        channels = {}
        for head, tail in spec_split:
            head = head.strip()
            op_match = [op for op in channel_modes if tail.strip().startswith(op)]
            if len(op_match) != 1:
                raise Exception("Unrecognized channel operation: ", tail.split()[0])
            argument_str = tail.replace(op_match[0],'')
            try:
                eval_str = argument_str.replace('[','dict(').replace(']', ')')
                args = eval(eval_str)
            except:
                raise Exception("Could not evaluate: %s" % argument_str)

            op = op_match[0]
            params = set(p for p in channel_modes[op].params().keys() if p!='name')

            mismatch_keys = set(args.keys()) - params
            if mismatch_keys:
                raise Exception("Parameter(s) %r not accepted by %s operation"
                                % (', '.join(mismatch_keys), op))
            # As string.letters (Python 2) does not exist in Python 3
            letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
            valid_chars = letters + string.digits + '_* '
            if not head.count('*') or any(l not in valid_chars for l in head):
                raise Exception("Invalid characters in overlay pattern specification: %s" % head)

            pattern =  ' * '.join(el.strip() for el in head.rsplit('*'))
            channel_modes[op].instance(**args)
            channels[op] =  (pattern, args)
        return channels


    @classmethod
    def option_completer(cls, k,v):
        """
        Tab completion hook for the %opts and %%opts magic.
        """
        line = v.text_until_cursor
        if line.endswith(']') or (line.count('[') - line.count(']')) % 2:
            line_tail = line[len('%%channels'):]
            op_name = line_tail[::-1].rsplit('[')[1][::-1].strip().split()[-1]
            if op_name in  channel_modes:
                return list(channel_modes[op_name].params().keys())
        else:
            return list(channel_modes.keys())


@magics_class
class OptsMagic(Magics):
    """
    The %opts and %%opts line and cell magics allow customization of
    how dataviews are displayed. The %opts line magic updates or
    creates new options for either StyleOpts (i.e. matplotlib options)
    or in the PlotOpts (plot settings). The %%opts cell magic sets
    custom display options associated on the displayed view object
    which will persist every time that object is displayed.
    """
    # Attributes set by the magic and read when display hooks run
    custom_options = {}
    show_info = False
    show_labels = False

    def __init__(self, *args, **kwargs):
        super(OptsMagic, self).__init__(*args, **kwargs)
        styles_list = [el.style_opts for el in viewmap.values()]
        params_lists = [[k for (k,v) in el.params().items()
                         if not v.constant] for el in viewmap.values()]

        # List of all parameters and styles for tab completion
        OptsMagic.all_styles = sorted(set([s for styles in styles_list for s in styles]))
        OptsMagic.all_params = sorted(set([p for plist in params_lists for p in plist]))


    @classmethod
    def pprint_kws(cls, style):
        return ', '.join("%s=%r" % (k,v) for (k,v) in style.items.items())


    @classmethod
    def collect(cls, obj, attr='style'):
        """
        Given a composite view object, build a dictionary of either
        the 'style' or 'label' attributes across all contained
        atoms. This method works across overlays, grid layouts and
        stacks. The return is a dictionary with the collected string
        values as keys for the the associated view type.
        """
        group = {}
        if isinstance(obj, (Overlay, Layout, GridLayout)):
            for subview in obj:
                group.update(cls.collect(subview, attr))
        elif isinstance(obj, Stack) and not issubclass(obj.type, Overlay):
            key_lists = [list(cls.collect(el, attr).keys()) for el in obj]
            values = set(el for els in key_lists for el in els)
            for val in values:
                group.update({val:obj.type})
        elif isinstance(obj, Stack):
            for subview in obj.last:
                group.update(cls.collect(subview, attr))
        else:
            value = '' if getattr(obj, attr, None) is None else getattr(obj, attr)
            group.update({value:type(obj)})
        return group


    @classmethod
    def _basename(cls, name):
        """
        Strips out the 'Custom' prefix from styles names that have
        been customized by an object identifier.
        """
        split = name.rsplit('>]_')
        if not name.startswith('Custom'):   return name
        elif len(split) == 2:               return split[1]
        else:
            raise Exception("Invalid style name %s" % name)


    @classmethod
    def _set_style_names(cls, obj, custom_name_map):
        """
        Update the style names on a composite view to the custom style
        name for all matches. A match occurs when the basename of the
        view.style is found in the supplied dictionary.
        """
        if isinstance(obj, GridLayout):
            for subview in obj.values():
                cls._set_style_names(subview, custom_name_map)
        elif isinstance(obj.style, list):
            obj.style = [custom_name_map.get(cls._basename(s), s) for s in obj.style]
        else:
            style = cls._basename(obj.style)
            obj.style = custom_name_map.get(style, obj.style)


    @classmethod
    def set_view_options(cls, obj):
        """
        To be called by the display hook which supplies the view
        object to be displayed. Any custom options are defined on the
        object as necessary and if there is an error, an HTML message
        is returned.
        """
        prefix = 'Custom[<' + obj.name + '>]_'

        # Implements the %%labels magic
        if cls.show_labels:
            labels = list(cls.collect(obj, 'label').keys())
            info = (len(labels), labels.count(''))
            summary = ("%d objects inspected, %d without labels. "
                       "The set of labels found:<br><br>&emsp;" % info)
            label_list = '<br>&emsp;'.join(['<b>%s</b>' % l for l in sorted(set(labels)) if l])
            return summary + label_list

        # Nothing to be done
        if not any([cls.custom_options, cls.show_info]): return

        styles = cls.collect(obj, 'style')
        # The set of available style basenames present in the object
        available_styles = set(cls._basename(s) for s in styles)
        custom_styles = set(s for s in styles if s.startswith('Custom'))

        mismatches = set(cls.custom_options.keys()) - (available_styles | set(channel_modes))
        if cls.show_info or mismatches:
            return cls._option_key_info(obj, available_styles, mismatches, custom_styles)

        # Test the options are valid
        error = cls._keyword_info(styles, cls.custom_options)
        if error: return error

        # Link the object to the new custom style
        cls._set_style_names(obj, dict((k, prefix + k) for k in cls.custom_options))
        # Define the Styles in the OptionMaps
        cls._define_options(cls.custom_options, prefix=prefix)


    @classmethod
    def _keyword_info(cls, styles, custom_options):
        """
        Check that the keywords in the StyleOpts or PlotOpts are
        valid. If not, the appropriate HTML error message is returned.
        """
        errmsg = ''
        for key, (plot_kws, style_kws) in custom_options.items():
            for name, viewtype in styles.items():
                plottype = viewmap[viewtype]
                if cls._basename(name) != key: continue
                # Plot options checks
                params = [k for (k,v) in plottype.params().items() if not v.constant]
                mismatched_params = set(plot_kws.keys()) - set(params)
                if mismatched_params:
                    info = (', '.join('<b>%r</b>' % el for el in mismatched_params),
                            '<b>%r</b>' % plottype.name,
                            ', '.join('<b>%s</b>' % el for el in params))
                    errmsg += "Keywords %s not in valid %s plot options: <br>&nbsp;&nbsp;%s" % info

                # Style options checks
                style_opts = plottype.style_opts
                mismatched_opts = set(style_kws.keys()) - set(style_opts)
                if style_opts == [] and mismatched_opts:
                    errmsg += 'No styles accepted by %s. <br>' % plottype.name
                elif mismatched_opts:
                    spacing = '<br><br>' if errmsg else ''
                    info = (spacing,
                            ', '.join('<b>%r</b>' % el for el in mismatched_opts),
                            '<b>%r</b>' % plottype.name,
                            ', '.join('<b>%s</b>' % el for el in style_opts))
                    errmsg += "%sKeywords %s not in valid %s style options: <br>&nbsp;&nbsp;%s" % info
        return errmsg


    @classmethod
    def _option_key_info(cls, obj, available_styles, mismatches, custom_styles):
        """
        Format the information about valid options keys as HTML,
        listing mismatched names and the available keys.
        """
        fmt = '&emsp;<code><font color="%s">%%s</font>%%s : ' % html_red
        fmt+= '<font color="%s">[%%s]</font> %%s</code><br>' % html_blue
        obj_name = "<b>%s</b>" % obj.__class__.__name__
        if len(available_styles) == 0:
            return "<b>No keys are available in the current %s</b>" % obj_name

        mismatch_str = ', '.join('<b>%r</b>' % el for el in mismatches)
        unavailable_msg = '%s not in customizable' % mismatch_str if mismatch_str else 'Customizable'
        s = "%s %s options:<br>" % (unavailable_msg, obj_name)
        max_len = max(len(s) for s in available_styles)
        for name in sorted(available_styles):
            padding = '&nbsp;'*(max_len - len(name))
            s += fmt % (name, padding,
                        cls.pprint_kws(View.options.plotting(name)),
                        cls.pprint_kws(View.options.style(name)))

        if custom_styles:
            s += '<br>Options that have been customized for the displayed view only:<br>'
            custom_names = [style_name.rsplit('>]_')[1] for style_name in custom_styles]
            max_len = max(len(s) for s in custom_names)
            for custom_style, custom_name in zip(custom_styles, custom_names):
                padding = '&nbsp;'*(max_len - len(custom_name))
                s += fmt % (custom_name, padding,
                            cls.pprint_kws(View.options.plotting(custom_style)),
                            cls.pprint_kws(View.options.style(custom_style)))
        return s


    def _parse_keywords(self, line):
        """
        Parse the arguments to the magic, returning a dictionary with
        style name keys and tuples of keywords as values. The first
        element of the tuples are the plot keyword options and the
        second element are the style keyword options.
        """
        tokens = line.split()
        if tokens == []: return {}
        elif not tokens[0][0].isupper():
            raise SyntaxError("First token must be a option name (a capitalized string)")

        # Split the input by the capitalized tokens
        style_names, tuples = [], []
        for upper, vals in itertools.groupby(tokens, key=lambda x: x[0].isupper()):
            values = list(vals)
            if upper and len(values) != 1:
                raise SyntaxError("Options should be split by keywords")
            elif upper:
                style_names.append(values[0])
            else:
                parse_string = ' '.join(values).replace(',', ' ')
                if not parse_string.startswith('[') and parse_string.count(']')==0:
                    plotstr, stylestr = '',  parse_string
                elif [parse_string.count(el) for el in '[]'] != [1,1]:
                    raise SyntaxError("Plot options not supplied in a well formed list.")
                else:
                    split_ind = parse_string.index(']')
                    plotstr = parse_string[1:split_ind]
                    stylestr = parse_string[split_ind+1:]
                try:
                    # Evalute the strings to obtain dictionaries
                    dicts = [eval('dict(%s)' % ', '.join(els))
                             for els in [plotstr.split(), stylestr.split()]]
                    tuples.append(tuple(dicts))
                except:
                    raise SyntaxError("Could not parse keywords from '%s'" % parse_string)

        return dict((k,v) for (k,v) in zip(style_names, tuples) if v != ({},{}))



    @classmethod
    def _define_options(cls, kwarg_map, prefix='', verbose=False):
        """
        Define the style and plot options.
        """
        lens, strs = [0,0,0], []
        for name, (plot_kws, style_kws) in kwarg_map.items():
            plot_update = name in View.options.plotting
            if plot_update and plot_kws:
                View.options[prefix+name] = View.options.plotting[name](**plot_kws)
            elif plot_kws:
                View.options[prefix+name] = PlotOpts(**plot_kws)

            style_update = name in View.options.style
            if style_update and style_kws:
                View.options[prefix+name] = View.options.style[name](**style_kws)
            elif style_kws:
                View.options[prefix+name] = StyleOpts(**style_kws)

            if verbose:
                plotstr = ('[%s]' % cls.pprint_kws(View.options.plotting[name])
                           if name in View.options.plotting else '')
                stylestr = (cls.pprint_kws(View.options.style[name])
                            if name in View.options.style else '')
                strs.append((name+':', plotstr, stylestr))
                lens = [max(len(name)+1, lens[0]),
                        max(len(plotstr), lens[1]),
                        max(len(stylestr),lens[2])]

        if verbose:
            heading = "Plot and Style Options"
            title = '%s\n%s' % (heading, '='*len(heading))
            description = "Each line describes the options associated with a single key:"
            msg = '%s\n\n%s\n\n    %s %s %s\n\n' % (green % title, description,
                                                    red % 'Name:', blue % '[Plot Options]',
                                                    'Style Options')
            for (name, plot_str, style_str) in strs:
                msg += "%s %s %s\n" % (red % name.ljust(lens[0]),
                                       blue % plot_str.ljust(lens[1]),
                                       style_str.ljust(lens[2]))
            page.page(msg)


    @classmethod
    def option_completer(cls, k,v):
        """
        Tab completion hook for the %opts and %%opts magic.
        """
        line = v.text_until_cursor
        if line.endswith(']') or (line.count('[') - line.count(']')) % 2:
            return [el+'=' for el in cls.all_params]
        else:
            return [el+'=' for el in cls.all_styles] + View.options.options()


    def _line_magic(self, line):
        """
        Update or create new options in for the plot or style
        options. Plot options keyword-value pairs, when supplied need
        to be give in square brackets after the option key. Any style
        keywords then following the closing square bracket. The -v
        flag toggles verbose output.

        Usage: %opts [-v] <Key> [ [<keyword>=<value>...]] [<keyword>=<value>...]
        """
        verbose = False
        if str(line).startswith('-v'):
            verbose = True
            line = line.replace('-v', '')

        kwarg_map = self._parse_keywords(str(line))

        if not kwarg_map:
            info = (len(View.options.style.keys()),
                    len([k for k in View.options.style.keys() if k.startswith('Custom')]))
            print("There are %d style options defined (%d custom object styles)." % info)
            info = (len(View.options.plotting.keys()),
                    len([k for k in View.options.plotting.keys() if k.startswith('Custom')]))
            print("There are %d plot options defined (%d custom object plot settings)." % info)
            return

        self._define_options(kwarg_map, verbose=verbose)


    @cell_magic
    def labels(self, line, cell=None):
        """
        Simple magic to see the full list of defined labels for the
        displayed view object.
        """
        if line != '':
            raise Exception("%%labels magics accepts no arguments.")
        OptsMagic.show_labels = True
        self.shell.run_cell(cell)
        OptsMagic.show_labels = False


    @line_cell_magic
    def opts(self, line='', cell=None):
        """
        Set custom display options unique to the displayed view. The
        keyword-value pairs in the square brackets (if present) set
        the plot parameters. Keyword-value pairs outside the square
        brackets are matplotlib style options.

        Usage: %%opts <Key> [ [<keyword>=<value>...] ] [<keyword>=<value>...]

        Multiple keys may be listed, setting plot and style options in
        this way.
        """
        if cell is None:
            return self._line_magic(str(line))
        elif not line.strip():
            OptsMagic.show_info=True
        else:
            OptsMagic.custom_options = self._parse_keywords(str(line))

        # Run the cell in the updated environment
        self.shell.run_cell(cell)
        # Reset the class attributes
        OptsMagic.custom_options = {}
        OptsMagic.show_info=False


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
    anim = plot.anim(fps=FPS)
    writers = animation.writers.avail
    for fmt in [VIDEO_FORMAT] + list(ANIMATION_OPTS.keys()):
        if ANIMATION_OPTS[fmt][0] in writers:
            try:
                return animate(anim, *ANIMATION_OPTS[fmt])
            except: pass
    msg = "<b>Could not generate %s animation</b>" % VIDEO_FORMAT
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

    mime_type = 'svg+xml' if FIGURE_FORMAT.lower()=='svg' else 'png'
    prefix = 'data:image/%s;base64,' % mime_type
    b64 = prefix + base64.b64encode(print_figure(fig, FIGURE_FORMAT)).decode("utf-8")
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
    return animate(anim, *ANIMATION_OPTS[VIDEO_FORMAT])

@display_hook
def stack_display(stack, size=256):
    if not isinstance(stack, Stack): return None
    magic_info = process_view_magics(stack)
    if magic_info: return magic_info
    opts = dict(View.options.plotting(stack).opts, size=get_plot_size())
    stackplot = viewmap[stack.type](stack, **opts)
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
    gridplot = viewmap[grid.__class__](grid, **opts)
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
    fig = viewmap[view.__class__](view, **opts)()
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


all_line_magics = sorted(['%params', '%opts', '%view'])
all_cell_magics = sorted(['%%view', '%%opts', '%%labels'])
message = """Welcome to the Dataviews IPython extension! (http://ioam.github.io/imagen/)"""
message += '\nAvailable magics: %s' % ', '.join(all_line_magics + all_cell_magics)

_loaded = False
VIDEO_FORMAT = select_format(['webm','h264','gif'])

# HTML_video output by default, but may be set to first_frame,
# middle_frame or last_frame (e.g. for testing purposes)

render_anim = HTML_video

def load_ipython_extension(ip, verbose=True):

    if verbose: print(message)

    global _loaded
    if not _loaded:
        _loaded = True

        param_ext.load_ipython_extension(ip, verbose=False)

        ip.register_magics(ViewMagic)
        ip.register_magics(OptsMagic)
        ip.register_magics(ChannelMagic)


        # Configuring tab completion
        ip.set_hook('complete_command', ChannelMagic.option_completer, str_key = '%channels')
        ip.set_hook('complete_command', ChannelMagic.option_completer, str_key = '%%channels')

        ip.set_hook('complete_command', ViewMagic.option_completer, str_key = '%view')
        ip.set_hook('complete_command', ViewMagic.option_completer, str_key = '%%view')

        ip.set_hook('complete_command', OptsMagic.option_completer, str_key = '%%opts')
        ip.set_hook('complete_command', OptsMagic.option_completer, str_key = '%opts')


        html_formatter = ip.display_formatter.formatters['text/html']
        html_formatter.for_type_by_name('matplotlib.animation', 'FuncAnimation', animation_display)

        html_formatter.for_type(View, view_display)
        html_formatter.for_type(Stack, stack_display)
        html_formatter.for_type(Layout, layout_display)
        html_formatter.for_type(GridLayout, layout_display)
        html_formatter.for_type(CoordinateGrid, projection_display)

        update_matplotlib_rc()

def unload_ipython_extension(ip):
    global _loaded
    _loaded = False
