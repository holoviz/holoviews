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

from ..core.settings import SettingsTree, Settings, SettingsError
from ..plotting import Plot

from collections import OrderedDict
from IPython.display import display, HTML
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
               'size'        : (0, 100),
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

    settings = OrderedDict(defaults.items())

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
        cls.options['holomap'] = cls.inbuilt_formats + supported_formats


    @classmethod
    def _generate_docstring(cls):
        intro = ["Magic for setting holoview display options.",
                 "Arguments are supplied as a series of keywords in any order:", '']
        backend = "backend      : The backend used by holoviews %r"  % cls.options['backend']
        fig =     "fig          : The static figure format %r" % cls.options['fig']
        holomap = "holomap      : The display type for holomaps %r" % cls.options['holomap']
        widgets = "widgets      : The widget mode for widgets %r" % cls.options['widgets']
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
            allowed = ['scrubber', 'widget', 'auto']
            if settings['holomap'] not in allowed:
                raise ValueError("The D3 backend only supports holomap options %r" % allowed)

        if (settings['holomap']=='widgets'
            and settings['widgets']!='embed'
            and settings['fig']=='svg'):
            raise ValueError("SVG mode not supported by widgets unless in embed mode")
        return settings


    def get_settings(self, line, settings):
        "Given a keyword specification line, validated and compute settings"
        items = self._extract_keywords(line, OrderedDict())
        for keyword in self.defaults:
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
        for k,v in ViewMagic.settings.items():
            keyword = '%s=%r' % (k,v)
            if len(current) + len(keyword) > self.settings['charwidth']:
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

        restore_copy = OrderedDict(self.settings.items())
        try:
            settings = self.get_settings(line, OrderedDict())
            ViewMagic.settings = settings
            # Inform writer of chosen fps
            if settings['holomap'] in ['gif', 'scrubber']:
                self.ANIMATION_OPTS[settings['holomap']][2]['fps'] = settings['fps']
            success = True
        except Exception as e:
            print 'SyntaxError: %s\n' % str(e)
            print "For help with the %view magic, call %view?\n"
            return

        if cell is not None:
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


class OptsCompleter(object):
    """
    Implements the TAB-completer for the %%opts magic.
    """
    _completions = {} # Contains valid plot and style keywords per Element

    @classmethod
    def setup_completer(cls):
        "Get the dictionary of valid completions"
        if len(cls._completions) != 0: return cls._completions
        for element in Plot.settings.children:
            settings = Plot.settings[element]
            plotkws = settings['plot'].allowed_keywords
            stylekws = settings['style'].allowed_keywords
            cls._completions[element] = (plotkws, stylekws if stylekws else [])
        return cls._completions


    @classmethod
    def option_completer(cls, k,v):
        "Tab completion hook for the %%opts cell magic."
        completions = cls.setup_completer()
        line = v.text_until_cursor
        # Find the last element class mentioned
        completion_key = None
        for token in [t for t in reversed(line.replace('.', ' ').split())]:
            if token in completions:
                completion_key = token
                break

        if not completion_key:
            return completions.keys()

        if line.endswith(']') or (line.count('[') - line.count(']')) % 2:
            kws = completions[completion_key][0]
            return [kw+'=' for kw in kws]

        style_completions = [kw+'=' for kw in completions[completion_key][1]]
        if line.endswith(')') or (line.count('(') - line.count(')')) % 2:
            return style_completions
        return style_completions + completions.keys()


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
            assert cls.next_id in Plot.custom_settings, 'RealityError'
            obj.traverse(lambda o: setattr(o, 'id', cls.next_id))
            cls.next_id = None
        return None


    @classmethod
    def _format_settings_error(cls, err):
        info = (err.invalid_keyword, err.group_name, ', '.join(err.allowed_keywords))
        return "Keyword <b>%r</b> not one of following %s options:<br><br><b>%s</b>" % info


    @classmethod
    def customize_tree(cls, spec, settings):
        """
        Returns a customized copy of the Plot.settings SettingTree object.
        """
        for key in sorted(spec.keys()):
            try:
                settings[str(key)] = spec[key]
            except SettingsError as e:
                cls.error_message = cls._format_settings_error(e)
                return None
        return settings

    @classmethod
    def register_custom_spec(cls, spec, obj):
        ids = Plot.custom_settings.keys()
        max_id = max(ids) if len(ids)>0 else -1
        settings = SettingsTree(items=Plot.settings.data.items(),
                                groups=Plot.settings.groups)
        custom_tree = cls.customize_tree(spec, settings)
        if custom_tree is not None:
            Plot.custom_settings[max_id+1] = custom_tree
            cls.next_id = max_id+1
        else:
            cls.next_id = None


    @line_cell_magic
    def opts(self, line='', cell=None):
        from holoviews.ipython.parser import OptsSpec
        get_object = None
        spec = OptsSpec.parse(line)

        self.register_custom_spec(spec, None)
        if cell:
            self.shell.run_cell(cell, store_history=STORE_HISTORY)
        else:
            retval = self.customize_tree(spec, Plot.settings)
            if retval is None:
                display(HTML(OptsMagic.error_message))
        OptsMagic.error_message = None


def load_magics(ip):

    ip.register_magics(ViewMagic)
    ip.register_magics(OptsMagic)
    #ip.register_magics(ChannelMagic)

    # Configuring tab completion
    ip.set_hook('complete_command', ChannelMagic.option_completer, str_key = '%channels')
    ip.set_hook('complete_command', ChannelMagic.option_completer, str_key = '%%channels')

    ip.set_hook('complete_command', ViewMagic.option_completer, str_key = '%view')
    ip.set_hook('complete_command', ViewMagic.option_completer, str_key = '%%view')

    OptsCompleter.setup_completer()
    ip.set_hook('complete_command', OptsCompleter.option_completer, str_key = '%%opts')
    ip.set_hook('complete_command', OptsCompleter.option_completer, str_key = '%opts')
