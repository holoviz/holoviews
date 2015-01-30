import itertools
import string

from IPython.core import page

try:
    from IPython.core.magic import Magics, magics_class, cell_magic, line_magic, line_cell_magic
except:
    from unittest import SkipTest
    raise SkipTest("IPython extension requires IPython >= 0.13")

from ..core import NdOverlay, Element, HoloMap,\
    AdjointLayout, NdLayout, AxisLayout, LayoutTree, CompositeOverlay

from ..core.options import OptionTree, Options, OptionError
from ..plotting import Plot

from collections import OrderedDict
from IPython.display import display, HTML

from ..operation import Channel

#========#
# Magics #
#========#


try:
    import pyparsing
except ImportError:
    pyparsing = None
else:
    from holoviews.ipython.parser import ChannelSpec
    from holoviews.ipython.parser import OptsSpec



GIF_TAG = "<center><img src='data:image/gif;base64,{b64}' style='max-width:100%'/><center/>"
VIDEO_TAG = """
<center><video controls style='max-width:100%'>
<source src="data:video/{mime_type};base64,{b64}" type="video/{mime_type}">
Your browser does not support the video tag.
</video><center/>"""


# Set to True to automatically run notebooks.
STORE_HISTORY = False


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
    Magic for easy customising of display options.
    Consult %%view? for more information.
    """
    # Formats that are always available
    inbuilt_formats= ['auto', 'widgets', 'scrubber']
    # Codec or system-dependent format options
    optional_formats = ['webm','h264', 'gif']

    allowed = {'backend'     : ['mpl','d3'],
               'fig'         : ['svg', 'png'],
               'holomap'     : inbuilt_formats,
               'widgets'     : ['embed', 'live', 'cached'],
               'fps'         : (0, float('inf')),
               'max_frames'  : (0, float('inf')),
               'max_branches': (0, float('inf')),
               'size'        : (0, float('inf')),
               'charwidth'   : (0, float('inf'))}

    defaults = OrderedDict([('backend'     , 'mpl'),
                            ('fig'         , 'png'),
                            ('holomap'     , 'auto'),
                            ('widgets'     , 'embed'),
                            ('fps'         , 20),
                            ('max_frames'  , 500),
                            ('max_branches', 2),
                            ('size'        , 100),
                            ('charwidth'   , 80)])

    options = OrderedDict(defaults.items())

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
        super(ViewMagic, self).__init__(*args, **kwargs)
        self.view.__func__.__doc__ = self._generate_docstring()


    @classmethod
    def register_supported_formats(cls, supported_formats):
        "Extend available holomap formats with supported format list"
        if not all(el in cls.optional_formats for el in supported_formats):
            raise AssertionError("Registering format in list %s not in known formats %s"
                                 % (supported_formats, cls.optional_formats))
        cls.allowed['holomap'] = cls.inbuilt_formats + supported_formats


    @classmethod
    def _generate_docstring(cls):
        intro = ["Magic for setting holoview display options.",
                 "Arguments are supplied as a series of keywords in any order:", '']
        backend = "backend      : The backend used by holoviews %r"  % cls.allowed['backend']
        fig =     "fig          : The static figure format %r" % cls.allowed['fig']
        holomap = "holomap      : The display type for holomaps %r" % cls.allowed['holomap']
        widgets = "widgets      : The widget mode for widgets %r" % cls.allowed['widgets']
        fps =    ("fps          : The frames per second for animations (default %r)"
                  % cls.defaults['widgets'])
        frames=  ("max_frames   : The max number of frames rendered (default %r)"
                  % cls.defaults['max_frames'])
        branches=("max_branches : The max number of LayoutTree branches rendered (default %r)"
                  % cls.defaults['max_branches'])
        size =   ("size         : The percentage size of displayed output (default %r)"
                  % cls.defaults['size'])
        chars =  ("charwidth    : The max character width view magic options display (default %r)"
                  % cls.defaults['charwidth'])

        descriptions = [backend, fig, holomap, widgets, fps, frames, branches, size, chars]
        return '\n'.join(intro + descriptions)


    def _extract_keywords(self, line, items):
        """
        Given the keyword string, parse a dictionary of options.
        """
        unprocessed = list(reversed(line.split('=')))
        while unprocessed:
            chunk = unprocessed.pop()
            key = None
            if chunk.strip() in self.allowed:
                key = chunk.strip()
            else:
                raise SyntaxError("Invalid keyword: %s" % chunk.strip())
            # The next chunk may end in a subsequent keyword
            value = unprocessed.pop().strip()
            if len(unprocessed) != 0:
                # Check if a new keyword has begun
                for option in self.allowed:
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


    def _validate(self, options):
        "Validation of edge cases and incompatible options"
        if options['backend'] == 'd3':
            try:      import mpld3 # pyflakes:ignore (Testing optional import)
            except:
                raise ValueError("Cannot use d3 backend without mpld3. "
                                 "Please select a different backend")
            allowed = ['scrubber', 'widgets', 'auto']
            if options['holomap'] not in allowed:
                raise ValueError("The D3 backend only supports holomap options %r" % allowed)

        if (options['holomap']=='widgets'
            and options['widgets']!='embed'
            and options['fig']=='svg'):
            raise ValueError("SVG mode not supported by widgets unless in embed mode")
        return options


    def get_options(self, line, options):
        "Given a keyword specification line, validated and compute options"
        items = self._extract_keywords(line, OrderedDict())
        for keyword in self.defaults:
            if keyword in items:
                value = items[keyword]
                allowed = self.allowed[keyword]
                if isinstance(allowed, list) and value not in allowed:
                    raise ValueError("Value %r for key %r not one of %s"
                                     % (value, keyword, allowed))
                elif isinstance(allowed, tuple):
                    if not (allowed[0] <= value <= allowed[1]):
                        info = (keyword,value)+allowed
                        raise ValueError("Value %r for key %r not between %s and %s" % info)
                options[keyword] = value
            else:
                options[keyword] = self.defaults[keyword]
        return self._validate(options)


    @classmethod
    def option_completer(cls, k,v):
        raw_line = v.text_until_cursor
        line = raw_line.replace('%view','')

        # Find the last element class mentioned
        completion_key = None
        tokens = [t for els in reversed(line.split('=')) for t in els.split()]

        for token in tokens:
            if token.strip() in cls.allowed:
                completion_key = token.strip()
                break
        values = [repr(el) for el in cls.allowed.get(completion_key, [])
                  if not isinstance(el, tuple)]

        return values + [el+'=' for el in cls.allowed.keys()]


    def pprint(self):
        """
        Pretty print the current view options with a maximum width of
        self.pprint_width.
        """
        elements = ["%view"]
        lines, current, count = [], '', 0
        for k,v in ViewMagic.options.items():
            keyword = '%s=%r' % (k,v)
            if len(current) + len(keyword) > self.options['charwidth']:
                print ('%view' if count==0 else '      ')  + current
                count += 1
                current = keyword
            else:
                current += ' '+ keyword
        else:
            print ('%view' if count==0 else '      ')  + current


    @line_cell_magic
    def view(self, line, cell=None):
        if line.strip() == '':
            self.pprint()
            print "\nFor help with the %view magic, call %view?"
            return

        restore_copy = OrderedDict(ViewMagic.options.items())
        try:
            options = self.get_options(line, OrderedDict())
            ViewMagic.options = options
            # Inform writer of chosen fps
            if options['holomap'] in ['gif', 'scrubber']:
                self.ANIMATION_OPTS[options['holomap']][2]['fps'] = options['fps']
        except Exception as e:
            print 'Error: %s' % str(e)
            print "For help with the %view magic, call %view?\n"
            return

        if cell is not None:
            self.shell.run_cell(cell, store_history=STORE_HISTORY)
            ViewMagic.options = restore_copy



@magics_class
class ChannelMagic(Magics):
    """
    Magic allowing easy definition of channel operations.
    Consult %%channels? for more information.
    """

    def __init__(self, *args, **kwargs):
        super(ChannelMagic, self).__init__(*args, **kwargs)
        lines = ['The %channels line magic is used to define channel operations.']
        self.channels.__func__.__doc__ = '\n'.join(lines + [ChannelSpec.__doc__])


    @line_magic
    def channels(self, line):
        defined_values = [op.value for op in Channel.definitions]
        if line.strip():
            for definition in ChannelSpec.parse(line.strip()):
                if definition.value in defined_values:
                    print("Channel definition ignored as value %r already defined" % definition.value)

                group = {'style':Options(), 'style':Options(), 'norm':Options()}
                type_name = definition.operation.output_type.__name__
                Plot.options[type_name + '.' + definition.value] = group
                Channel.definitions.append(definition)
        else:
            print "For help with the %channels magic, call %channels?\n"


    @classmethod
    def option_completer(cls, k,v):
        line = v.text_until_cursor
        operation_openers = [op.__name__+'(' for op in Channel.operations]

        op_declared = any(op in line for op in operation_openers)
        if not op_declared:
            return operation_openers
        if op_declared and ')' not in line:
            return [')']
        elif line.split(')')[1].strip() and ('[' not in line):
            return ['[']
        elif '[' in line:
            return [']']


class OptsCompleter(object):
    """
    Implements the TAB-completion for the %%opts magic.
    """
    _completions = {} # Contains valid plot and style keywords per Element

    @classmethod
    def setup_completer(cls):
        "Get the dictionary of valid completions"
        if len(cls._completions) != 0: return cls._completions
        for element in Plot.options.children:
            options = Plot.options[element]
            plotkws = options['plot'].allowed_keywords
            stylekws = options['style'].allowed_keywords
            cls._completions[element] = (plotkws, stylekws if stylekws else [])
        return cls._completions


    @classmethod
    def option_completer(cls, k,v):
        "Tab completion hook for the %%opts cell magic."
        line = v.text_until_cursor

        completions = cls.setup_completer()
        channel_defs = {el.value:el.operation.output_type.__name__
                        for el in Channel.definitions}

        # Find the last element class mentioned
        completion_key = None
        for token in [t for t in reversed(line.replace('.', ' ').split())]:
            if token in completions:
                completion_key = token
                break
            # Attempting to match channel definitions
            if token in channel_defs:
                completion_key = channel_defs[token]
                break

        if not completion_key:
            return completions.keys() + channel_defs.keys()

        if line.endswith(']') or (line.count('[') - line.count(']')) % 2:
            kws = completions[completion_key][0]
            return [kw+'=' for kw in kws]

        if line.endswith('}') or (line.count('{') - line.count('}')) % 2:
            return ['-groupwise', '-mapwise']

        style_completions = [kw+'=' for kw in completions[completion_key][1]]
        if line.endswith(')') or (line.count('(') - line.count(')')) % 2:
            return style_completions
        return style_completions + completions.keys() + channel_defs.keys()




@magics_class
class OptsMagic(Magics):
    """
    Magic for easy customising of normalization, plot and style options.
    Consult %%opts? for more information.
    """

    error_message = None
    next_id = None

    @classmethod
    def process_view(cls, obj):
        """
        To be called by the display hook which supplies the element to
        be displayed. Any customisation of the object can then occur
        before final display. If there is any error, a HTML message
        may be returned. If None is returned, display will proceed as
        normal.
        """
        if cls.error_message:
            return cls.error_message
        if cls.next_id is not None:
            assert cls.next_id in Plot.custom_options, 'RealityError'
            obj.traverse(lambda o: setattr(o, 'id', cls.next_id),
                         specs=cls.applied_keys)
            cls.next_id = None
            cls.applied_keys = None
        return None


    @classmethod
    def _format_options_error(cls, err):
        info = (err.invalid_keyword, err.group_name, ', '.join(err.allowed_keywords))
        return "Keyword <b>%r</b> not one of following %s options:<br><br><b>%s</b>" % info


    @classmethod
    def customize_tree(cls, spec, options):
        """
        Returns a customized copy of the Plot.options OptionsTree object.
        """
        for key in sorted(spec.keys()):
            try:
                options[str(key)] = spec[key]
            except OptionError as e:
                cls.error_message = cls._format_options_error(e)
                return None
        return options

    @classmethod
    def register_custom_spec(cls, spec, obj):
        ids = Plot.custom_options.keys()
        max_id = max(ids) if len(ids)>0 else -1
        options = OptionTree(items=Plot.options.data.items(),
                             groups=Plot.options.groups)
        custom_tree = cls.customize_tree(spec, options)
        if custom_tree is not None:
            Plot.custom_options[max_id+1] = custom_tree
            cls.next_id = max_id+1
            cls.applied_keys = spec.keys()
        else:
            cls.next_id = None

    @classmethod
    def expand_channel_keys(cls, spec):
        """
        Expands channel definition keys into {type}.{value} keys. For
        instance a channel operation returning a value string 'Image'
        of element type RGBA expands to 'RGBA.Image'.
        """
        expanded_spec={}
        channel_defs = {el.value:el.operation.output_type.__name__
                        for el in Channel.definitions}
        for key, val in spec.items():
            if key not in channel_defs:
                expanded_spec[key] = val
            else:
                type_name = channel_defs[key]
                expanded_spec[str(type_name+'.'+key)] = val
        return expanded_spec


    @line_cell_magic
    def opts(self, line='', cell=None):
        """
        The opts line/cell magic with tab-completion.

        %%opts [ [path] [normalization] [plotting options] [style options]]+

        path:             A dotted type.value.label specification
                          (e.g. Matrix.Grayscale.Photo)

        normalization:    List of normalization options delimited by braces.
                          One of | -groupwise | -mapwise | +groupwise | +mapwise |
                          E.g. { -groupwise -mapwise }

        plotting options: List of plotting option keywords delimited by
                          square brackets. E.g. [show_title=False]

        style options:    List of style option keywords delimited by
                          parentheses. E.g. (lw=10 marker='+')

        Note that commas between keywords are optional (not
        recommended) and that keywords must end in '=' without a
        separating space.

        More information may be found in the class docstring of
        ipython.parser.OptsSpec.
        """
        get_object = None
        try:
            spec = OptsSpec.parse(line)
            spec = self.expand_channel_keys(spec)
        except SyntaxError:
            display(HTML("<b>Invalid syntax</b>: Consult <tt>%%opts?</tt> for more information."))
            return

        self.register_custom_spec(spec, None)
        if cell:
            self.shell.run_cell(cell, store_history=STORE_HISTORY)
        else:
            retval = self.customize_tree(spec, Plot.options)
            if retval is None:
                display(HTML(OptsMagic.error_message))
        OptsMagic.error_message = None


def load_magics(ip):
    ip.register_magics(ViewMagic)

    if pyparsing is None:  print("%opts magic unavailable (pyparsing cannot be imported)")
    else: ip.register_magics(OptsMagic)

    if pyparsing is None: print("%channels magic unavailable (pyparsing cannot be imported)")
    else: ip.register_magics(ChannelMagic)


    # Configuring tab completion
    ip.set_hook('complete_command', ChannelMagic.option_completer, str_key = '%channels')

    ip.set_hook('complete_command', ViewMagic.option_completer, str_key = '%view')
    ip.set_hook('complete_command', ViewMagic.option_completer, str_key = '%%view')

    OptsCompleter.setup_completer()
    ip.set_hook('complete_command', OptsCompleter.option_completer, str_key = '%%opts')
    ip.set_hook('complete_command', OptsCompleter.option_completer, str_key = '%opts')
