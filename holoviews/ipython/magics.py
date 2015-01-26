import itertools
import string

from IPython.core import page

try:
    from IPython.core.magic import Magics, magics_class, cell_magic, line_cell_magic
except:
    from unittest import SkipTest
    raise SkipTest("IPython extension requires IPython >= 0.13")

from ..core import NdOverlay, Element, HoloMap,\
    AdjointLayout, NdLayout, AxisLayout, LayoutTree, CompositeOverlay
from ..plotting import Plot


#========#
# Magics #
#========#


GIF_TAG = "<center><img src='data:image/gif;base64,{b64}' style='max-width:100%'/><center/>"
VIDEO_TAG = """
<center><video controls style='max-width:100%'>
<source src="data:video/{mime_type};base64,{b64}" type="video/{mime_type}">
Your browser does not support the video tag.
</video><center/>"""


# Set to True to automatically run notebooks.
STORE_HISTORY = False


@magics_class
class ViewMagic(Magics):
    """
    Magic to allow easy control over the display of holoviews. The
    applicable settings are available on the settings attribute.
    """

    # Formats that are always available
    inbuilt_formats= ['auto', 'widgets', 'scrubber']
    # Codec or system-dependent format options
    optional_formats = ['webm','h264', 'gif']

    options = {'backend'     : ['mpl','d3'],
               'fig'         : ['svg', 'png'],
               'holomap'     : inbuilt_formats,
               'widgets'     : ['embed', 'live', 'cached'],
               'fps'         : (0, float('inf')),
               'max_frames'  : (0, float('inf')),
               'max_branches': (0, float('inf')),
               'size'        : (0, 100)}


    defaults = {'backend'     : 'mpl',
                'fig'         : 'png',
                'holomap'     : 'auto',
                'widgets'     : 'embed',
                'fps'         : 20,
                'max_frames'  : 500,
                'max_branches': 2,
                'size'        : 100}

    settings = dict(**defaults)

    # <format name> : (animation writer, mime_type,  anim_kwargs, extra_args, tag)
    ANIMATION_OPTS = {
        'webm': ('ffmpeg', 'webm', {},
                 ['-vcodec', 'libvpx', '-b', '1000k'],
                 VIDEO_TAG),
        'h264': ('ffmpeg', 'mp4', {'codec': 'libx264'},
                 ['-pix_fmt', 'yuv420p'],
                 VIDEO_TAG),
        'gif': ('imagemagick', 'gif', {'fps': 10}, [],
                GIF_TAG),
        'scrubber': ('html', None, {'fps': 5}, None, None)
    }


    def __init__(self, *args, **kwargs):
        self.pprint_width = 30  # Maximum width for pretty printing
        super(ViewMagic, self).__init__(*args, **kwargs)


    @classmethod
    def register_supported_formats(cls, supported_formats):
        "Extend available holomap formats with supported format list"
        if not all(el in cls.optional_formats for el in supported_formats):
            raise AssertionError("Registering format in list %s not in known formats %s"
                                 % (supported_formats, cls.optional_formats))
        cls.options['holomap'] = cls.inbuilt_formats + supported_formats


    def _extract_keywords(self, line, items = {}):
        """
        Given the keyword string, parse a dictionary of options.
        """
        unprocessed = list(reversed(line.split('=')))
        while unprocessed:
            chunk = unprocessed.pop()
            key = None
            if chunk.strip() in self.options:
                key = chunk.strip()
            else:
                raise SyntaxError("Invalid keyword: %s" % chunk.strip())
            # The next chunk may end in a subsequent keyword
            value = unprocessed.pop().strip()
            if len(unprocessed) != 0:
                # Check if a new keyword has begun
                for option in self.options:
                    if value.endswith(option):
                        value = value[:-len(option)].strip()
                        unprocessed.append(option)
                        break
                else:
                    raise SyntaxError("Invalid keyword: %s" % value.split()[-1])
            keyword = '%s=%s' % (key, value)
            try:
                items.update(eval('dict(%s)' % keyword))
            except:
                raise SyntaxError("Could not evaluate keyword: %s" % keyword)
        return items


    def _validate(self, settings):
        "Validation of edge cases and incompatible settings"
        if settings['backend'] == 'd3':
            try:      import mpld3 # pyflakes:ignore (Testing optional import)
            except:
                raise ValueError("Cannot use d3 backend without mpld3. "
                                 "Please select a different backend")
            allowed = ['scrubber', 'widget']
            if settings['holomap'] not in d3_allowed:
                raise ValueError("The D3 backend only supports holomap options %r" % allowed)

        if (settings['holomap']=='widgets'
            and settings['widgets']!='embed'
            and settings['fig']=='svg'):
            raise ValueError("SVG mode not supported by widgets unless in embed mode")
        return settings


    def get_settings(self, line, settings={}):
        "Given a keyword specification line, validated and compute settings"
        items = self._extract_keywords(line, {})
        for keyword in self.options:
            if keyword in items:
                value = items[keyword]
                allowed = self.options[keyword]
                if isinstance(allowed, list) and value not in allowed:
                    raise ValueError("Value %r for key %r not one of %s"
                                     % (value, keyword, allowed))
                elif isinstance(allowed, tuple):
                    if not (allowed[0] <= value <= allowed[1]):
                        raise ValueError("Value %r for key %r not between %s and %s"
                                         % (keyword,value)+allowed)
                settings[keyword] = value
            else:
                settings[keyword] = self.defaults[keyword]
        return self._validate(settings)


    @classmethod
    def option_completer(cls, k,v):
        raw_line = v.text_until_cursor
        line = raw_line.replace('%view','')

        # Find the last element class mentioned
        completion_key = None
        tokens = [t for els in reversed(line.split('=')) for t in els.split()]
        cls.LINE = tokens
        for token in tokens:
            if token.strip() in cls.options:
                completion_key = token.strip()
                break
        values = [repr(el) for el in cls.options.get(completion_key, [])
                  if not isinstance(el, tuple)]
        return values + [el+'=' for el in cls.options.keys()]


    def pprint(self):
        """
        Pretty print the current view settings with a maximum width of
        self.pprint_width.
        """
        elements = ["%view"]
        lines, current, count = [], '', 0
        for k,v in sorted(ViewMagic.settings.items()):
            keyword = '%s=%r' % (k,v)
            if len(current) + len(keyword) > self.pprint_width:
                print ('%view' if count==0 else '      ')  + current
                count += 1
                current = keyword
            else:
                current += ' '+ keyword
        else:
            print ('%view' if count==0 else '      ')  + current


    def print_usage_info(self):
        print "The view magic is called with space separated keywords."
        print "Tab completion is available for these keywords:\n\t%s" % self.options.keys()


    @line_cell_magic
    def view(self, line, cell=None):
        "Magic for setting holoview display options"
        if line.strip() == '':
            self.print_usage_info()
            return

        restore_copy = dict(**self.settings)
        try:
            settings = self.get_settings(line)
            ViewMagic.settings = settings
            # Inform writer of chosen fps
            if settings['holomap'] in ['gif', 'scrubber']:
                self.ANIMATION_OPTS[settings['holomap']][2]['fps'] = settings['fps']
            success = True
        except Exception as e:
            print 'SyntaxError: %s\n' % str(e)
            print "For more information call the %view magic without arguments."
            return

        if cell is None:
            self.pprint()
        else:
            self.shell.run_cell(cell, store_history=STORE_HISTORY)
            self.settings = restore_copy




@magics_class
class ChannelMagic(Magics):

    custom_channels = {}

    @cell_magic
    def channels(self, line, cell=None):
        """
        The %%channels cell magic allows channel definitions to be
        defined on the displayed Overlay.

        For instance, if you have three Matrix Views (R,G and B)
        together in a Overlay with labels 'R_Channel',
        'G_Channel', 'B_Channel' respectively, you can display this
        object as an RGB image using:

        %%channels R_Channel * G_Channel * B_Channel => RGBA []
        R * G * B

        The available operators are defined in the modes dictionary of
        ChannelOpts and additional arguments to the channel operator
        are supplied via keywords in the square brackets.
        """
        ChannelMagic.custom_channels = self._parse_channels(str(line))
        self.shell.run_cell(cell, store_history=STORE_HISTORY)
        ChannelMagic.custom_channels = {}


    @classmethod
    def _set_overlay_labels(cls, obj, label):
        """
        Labels on Overlays are used to index channel definitions.
        """
        if isinstance(obj, (AdjointLayout, AxisLayout, NdLayout)):
            for subview in obj:
                cls._set_overlay_labels(subview, label)
        elif isinstance(obj, HoloMap) and issubclass(obj.type, NdOverlay):
            for overlay in obj:
                overlay.relabel(label)
        elif isinstance(obj, NdOverlay):
            obj.relabel(label)


    @classmethod
    def _set_channels(cls, obj, custom_channels, prefix):
        cls._set_overlay_labels(obj, prefix)
        for name, (pattern, params) in custom_channels.items():
            CompositeOverlay.channels[prefix + '_' + name] = ChannelOpts(name, pattern,
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
            op_match = [op for op in channel_ops if tail.strip().startswith(op)]
            if len(op_match) != 1:
                raise Exception("Unrecognized channel operation: ", tail.split()[0])
            argument_str = tail.replace(op_match[0],'')
            try:
                eval_str = argument_str.replace('[','dict(').replace(']', ')')
                args = eval(eval_str)
            except:
                raise Exception("Could not evaluate: %s" % argument_str)

            op = op_match[0]
            params = set(p for p in channel_ops[op].params().keys() if p!='name')

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
            channel_ops[op].instance(**args)
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
            if op_name in  channel_ops:
                return list(channel_ops[op_name].params().keys())
        else:
            return list(channel_ops.keys())


# ANSI color codes for the IPython pager
red   = '\x1b[1;31m%s\x1b[0m'
blue  = '\x1b[1;34m%s\x1b[0m'
green = '\x1b[1;32m%s\x1b[0m'
cyan = '\x1b[1;36m%s\x1b[0m'

# Corresponding HTML color codes
html_red = '#980f00'
html_blue = '#00008e'



@magics_class
class OptsMagic(Magics):
    """
    The %opts and %%opts line and cell magics allow customization of
    how holoviews are displayed. The %opts line magic updates or
    creates new options for either StyleOpts (i.e. matplotlib options)
    or in the PlotOpts (plot settings). The %%opts cell magic sets
    custom display options associated on the displayed element object
    which will persist every time that object is displayed.
    """
    # Attributes set by the magic and read when display hooks run
    custom_options = {}
    show_info = False
    show_labels = False

    def __init__(self, *args, **kwargs):
        super(OptsMagic, self).__init__(*args, **kwargs)
        styles_list = [el.style_opts for el in Plot.defaults.values()]
        params_lists = [[k for (k,v) in el.params().items()
                         if not v.constant] for el in Plot.defaults.values()]

        # List of all parameters and styles for tab completion
        OptsMagic.all_styles = sorted(set([s for styles in styles_list for s in styles]))
        OptsMagic.all_params = sorted(set([p for plist in params_lists for p in plist]))


    @classmethod
    def pprint_kws(cls, style):
        return ', '.join("%s=%r" % (k,v) for (k,v) in sorted(style.items.items()))


    @classmethod
    def collect(cls, obj, attr='style'):
        """
        Given a composite element object, build a dictionary of either
        the 'style' or 'label' attributes across all contained
        atoms. This method works across overlays, grid layouts and
        maps. The return is a dictionary with the collected string
        values as keys for the the associated element type.
        """
        group = {}
        if isinstance(obj, (CompositeOverlay, AdjointLayout, AxisLayout, NdLayout, LayoutTree)):
            for subview in obj:
                group.update(cls.collect(subview, attr))
            if isinstance(obj, (AdjointLayout, NdOverlay)):
                return group

        if isinstance(obj, HoloMap) and not issubclass(obj.type, CompositeOverlay):
            key_lists = [list(cls.collect(el, attr).keys()) for el in obj]
            values = set(el for els in key_lists for el in els)
            for val in values:
                group.update({val:obj.type})
        elif isinstance(obj, HoloMap):
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
        Update the style names on a composite element to the custom style
        name for all matches. A match occurs when the basename of the
        element.style is found in the supplied dictionary.
        """
        if isinstance(obj, (AdjointLayout, AxisLayout, NdLayout, LayoutTree)):
            for subview in obj:
                cls._set_style_names(subview, custom_name_map)
            if isinstance(obj, AdjointLayout):
                return

        if isinstance(obj.style, list) and not isinstance(obj, AdjointLayout):
            obj.style = [custom_name_map.get(cls._basename(s), s) for s in obj.style]
        else:
            style = cls._basename(obj.style)
            obj.style = custom_name_map.get(style, obj.style)


    @classmethod
    def set_view_options(cls, obj):
        """
        To be called by the display hook which supplies the element
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

        mismatches = set(cls.custom_options.keys()) - (available_styles | set(channel_ops))
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
                plottype = Plot.defaults.get(viewtype, None)
                if plottype is None: continue
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
                        cls.pprint_kws(Element.options.plotting(name)),
                        cls.pprint_kws(Element.options.style(name)))

        if custom_styles:
            s += '<br>Options that have been customized for the displayed element only:<br>'
            custom_names = [style_name.rsplit('>]_')[1] for style_name in custom_styles]
            max_len = max(len(s) for s in custom_names)
            for custom_name, custom_style  in sorted(zip(custom_names, custom_styles)):
                padding = '&nbsp;'*(max_len - len(custom_name))
                s += fmt % (custom_name, padding,
                            cls.pprint_kws(Element.options.plotting(custom_style)),
                            cls.pprint_kws(Element.options.style(custom_style)))
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
            plot_update = name in Element.options.plotting
            if plot_update and plot_kws:
                Element.options[prefix+name] = Element.options.plotting[name](**plot_kws)
            elif plot_kws:
                Element.options[prefix+name] = PlotOpts(**plot_kws)

            style_update = name in Element.options.style
            if style_update and style_kws:
                Element.options[prefix+name] = Element.options.style[name](**style_kws)
            elif style_kws:
                Element.options[prefix+name] = StyleOpts(**style_kws)

            if verbose:
                plotstr = ('[%s]' % cls.pprint_kws(Element.options.plotting[name])
                           if name in Element.options.plotting else '')
                stylestr = (cls.pprint_kws(Element.options.style[name])
                            if name in Element.options.style else '')
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
            return [el+'=' for el in cls.all_styles] + Element.options.options()


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
            info = (len(Element.options.style.keys()),
                    len([k for k in Element.options.style.keys() if k.startswith('Custom')]))
            print("There are %d style options defined (%d custom object styles)." % info)
            info = (len(Element.options.plotting.keys()),
                    len([k for k in Element.options.plotting.keys() if k.startswith('Custom')]))
            print("There are %d plot options defined (%d custom object plot settings)." % info)
            return

        self._define_options(kwarg_map, verbose=verbose)


    @cell_magic
    def labels(self, line, cell=None):
        """
        Simple magic to see the full list of defined labels for the
        displayed element object.
        """
        if line != '':
            raise Exception("%%labels magics accepts no arguments.")
        OptsMagic.show_labels = True
        self.shell.run_cell(cell, store_history=STORE_HISTORY)
        OptsMagic.show_labels = False


    @line_cell_magic
    def opts(self, line='', cell=None):
        """
        Set custom display options unique to the displayed element. The
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
        self.shell.run_cell(cell, store_history=STORE_HISTORY)
        # Reset the class attributes
        OptsMagic.custom_options = {}
        OptsMagic.show_info=False



def load_magics(ip):

    ip.register_magics(ViewMagic)
    #ip.register_magics(OptsMagic)
    #ip.register_magics(ChannelMagic)

    # Configuring tab completion
    ip.set_hook('complete_command', ChannelMagic.option_completer, str_key = '%channels')
    ip.set_hook('complete_command', ChannelMagic.option_completer, str_key = '%%channels')

    ip.set_hook('complete_command', ViewMagic.option_completer, str_key = '%view')
    ip.set_hook('complete_command', ViewMagic.option_completer, str_key = '%%view')

    ip.set_hook('complete_command', OptsMagic.option_completer, str_key = '%%opts')
    ip.set_hook('complete_command', OptsMagic.option_completer, str_key = '%opts')
