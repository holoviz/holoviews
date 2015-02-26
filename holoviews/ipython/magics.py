import string
import time
import os
try:
    from IPython.core.magic import Magics, magics_class, cell_magic, line_magic, line_cell_magic
    from IPython.core.pylabtools import print_figure
except:
    from unittest import SkipTest
    raise SkipTest("IPython extension requires IPython >= 0.13")

from ..core import OrderedDict
from ..core.options import OptionTree, Options, OptionError, Store, save_options

from IPython.display import display, HTML

from ..operation import Compositor

#========#
# Magics #
#========#


try:
    import pyparsing
except ImportError:
    pyparsing = None
else:
    from holoviews.ipython.parser import CompositorSpec
    from holoviews.ipython.parser import OptsSpec


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
class OutputMagic(Magics):
    """
    Magic for easy customising of display options.
    Consult %%output? for more information.
    """
    # Formats that are always available
    inbuilt_formats= ['auto', 'widgets', 'scrubber']
    # Codec or system-dependent format options
    optional_formats = ['webm','h264', 'gif']

    # Lists: strict options, Set: suggested options, Tuple: numeric bounds.
    allowed = {'backend'     : ['mpl','d3'],
               'fig'         : ['svg', 'png'],
               'holomap'     : inbuilt_formats,
               'widgets'     : ['embed', 'live', 'cached'],
               'fps'         : (0, float('inf')),
               'max_frames'  : (0, float('inf')),
               'max_branches': (0, float('inf')),
               'size'        : (0, float('inf')),
               'dpi'         : (1, float('inf')),
               'charwidth'   : (0, float('inf')),
               'filename'   : {None}}

    defaults = OrderedDict([('backend'     , 'mpl'),
                            ('fig'         , 'png'),
                            ('holomap'     , 'auto'),
                            ('widgets'     , 'embed'),
                            ('fps'         , 20),
                            ('max_frames'  , 500),
                            ('max_branches', 2),
                            ('size'        , 100),
                            ('dpi'         , 72),
                            ('charwidth'   , 80),
                            ('filename'    , None)])

    options = OrderedDict(defaults.items())
    _obj = None

    def __init__(self, *args, **kwargs):
        super(OutputMagic, self).__init__(*args, **kwargs)
        self.output.__func__.__doc__ = self._generate_docstring()


    @classmethod
    def register_supported_formats(cls, supported_formats):
        "Extend available holomap formats with supported format list"
        if not all(el in cls.optional_formats for el in supported_formats):
            raise AssertionError("Registering format in list %s not in known formats %s"
                                 % (supported_formats, cls.optional_formats))
        cls.allowed['holomap'] = cls.inbuilt_formats + supported_formats


    @classmethod
    def _generate_docstring(cls):
        intro = ["Magic for setting HoloViews display options.",
                 "Arguments are supplied as a series of keywords in any order:", '']
        backend = "backend      : The backend used by HoloViews %r"  % cls.allowed['backend']
        fig =     "fig          : The static figure format %r" % cls.allowed['fig']
        holomap = "holomap      : The display type for holomaps %r" % cls.allowed['holomap']
        widgets = "widgets      : The widget mode for widgets %r" % cls.allowed['widgets']
        fps =    ("fps          : The frames per second for animations (default %r)"
                  % cls.defaults['widgets'])
        frames=  ("max_frames   : The max number of frames rendered (default %r)"
                  % cls.defaults['max_frames'])
        branches=("max_branches : The max number of Layout branches rendered (default %r)"
                  % cls.defaults['max_branches'])
        size =   ("size         : The percentage size of displayed output (default %r)"
                  % cls.defaults['size'])
        dpi =    ("dpi          : The rendered dpi of the figure (default %r)"
                  % cls.defaults['dpi'])
        chars =  ("charwidth    : The max character width for displaying the output magic (default %r)"
                  % cls.defaults['charwidth'])
        fname =  ("filename    : The filename of the saved output, if any (default %r)"
                  % cls.defaults['filename'])
        descriptions = [backend, fig, holomap, widgets, fps, frames, branches, size, dpi, chars, fname]
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

        if save_options.parse_fields(options['filename']) != []:
            raise ValueError("Please specify a valid filename without any format fields "
                             "(no braces allowed)")
        return options


    def get_options(self, line, options):
        "Given a keyword specification line, validated and compute options"
        items = self._extract_keywords(line, OrderedDict())
        for keyword in self.defaults:
            if keyword in items:
                value = items[keyword]
                allowed = self.allowed[keyword]
                if isinstance(allowed, set):  pass
                elif isinstance(allowed, list) and value not in allowed:
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
        line = raw_line.replace('%output','')

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
        elements = ["%output"]
        lines, current, count = [], '', 0
        for k,v in OutputMagic.options.items():
            keyword = '%s=%r' % (k,v)
            if len(current) + len(keyword) > self.options['charwidth']:
                print(('%output' if count==0 else '      ')  + current)
                count += 1
                current = keyword
            else:
                current += ' '+ keyword
        else:
            print(('%output' if count==0 else '      ')  + current)


    @line_cell_magic
    def output(self, line, cell=None):
        line = line.split('#')[0].strip()
        if line == '':
            self.pprint()
            print("\nFor help with the %output magic, call %output?")
            return

        restore_copy = OrderedDict(OutputMagic.options.items())
        try:
            options = self.get_options(line, OrderedDict())
            OutputMagic.options = options
            # Inform writer of chosen fps
            if options['holomap'] in ['gif', 'scrubber']:
                self.ANIMATION_OPTS[options['holomap']][2]['fps'] = options['fps']
        except Exception as e:
            print('Error: %s' % str(e))
            print("For help with the %output magic, call %output?\n")
            return

        if cell is not None:
            self.shell.run_cell(cell, store_history=STORE_HISTORY)
            OutputMagic.options = restore_copy
        self._obj=None

    @classmethod
    def register_object(cls, obj):
        cls._obj = obj

    @classmethod
    def save_fig(cls, fig, fig_format, dpi):
        filename = save_options.filename(fig_format, cls._obj,
                                         default=cls.options['filename'])
        if filename is None: return

        figure_data = print_figure(fig, fig_format, dpi=dpi)
        if save_options._digest(figure_data): return
        with open(filename, 'w') as f:
            f.write(figure_data)

    @classmethod
    def save_anim(cls, anim, mime_type, writer, dpi, **anim_kwargs):
        filename = save_options.filename(mime_type,cls._obj,
                                         default=cls.options['filename'])
        if filename is None: return
        anim.save(filename, writer=writer, dpi=dpi, **anim_kwargs)



@magics_class
class CompositorMagic(Magics):
    """
    Magic allowing easy definition of compositor operations.
    Consult %compositor? for more information.
    """

    def __init__(self, *args, **kwargs):
        super(CompositorMagic, self).__init__(*args, **kwargs)
        lines = ['The %compositor line magic is used to define compositors.']
        self.compositor.__func__.__doc__ = '\n'.join(lines + [CompositorSpec.__doc__])


    @line_magic
    def compositor(self, line):
        if line.strip():
            for definition in CompositorSpec.parse(line.strip()):
                group = {'style':Options(), 'style':Options(), 'norm':Options()}
                type_name = definition.output_type.__name__
                Store.options[type_name + '.' + definition.group] = group
                Compositor.register(definition)
        else:
            print("For help with the %compositor magic, call %compositor?\n")


    @classmethod
    def option_completer(cls, k,v):
        line = v.text_until_cursor
        operation_openers = [op.__name__+'(' for op in Compositor.operations]

        modes = ['data', 'display']
        op_declared = any(op in line for op in operation_openers)
        mode_declared = any(mode in line for mode in modes)
        if not mode_declared:
            return modes
        elif not op_declared:
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
        for element in Store.options.keys():
            try:
                options = Store.options[element]
                plotkws = options['plot'].allowed_keywords
                stylekws = options['style'].allowed_keywords
                dotted = '.'.join(element)
                cls._completions[dotted] = (plotkws, stylekws if stylekws else [])
            except KeyError:
                pass
        return cls._completions

    @classmethod
    def dotted_completion(cls, line, sorted_keys, compositor_defs):
        """
        Supply the appropriate key in Store.options and supply
        suggestions for further completion.
        """
        completion_key, suggestions = None, []
        tokens = [t for t in reversed(line.replace('.', ' ').split())]
        for i, token in enumerate(tokens):
            key_checks =[]
            if i >= 0:  # Undotted key
                key_checks.append(tokens[i])
            if i >= 1:  # Single dotted key
                key_checks.append('.'.join([key_checks[-1], tokens[i-1]]))
            if i >= 2:  # Double dotted key
                key_checks.append('.'.join([key_checks[-1], tokens[i-2]]))
            # Check for longest potential dotted match first
            for key in reversed(key_checks):
                if key in sorted_keys:
                    completion_key = key
                    depth = completion_key.count('.')
                    suggestions = [k.split('.')[depth+1] for k in sorted_keys
                                   if k.startswith(completion_key+'.')]
                    break
            # Attempting to match compositor definitions
            if token in compositor_defs:
                completion_key = compositor_defs[token]
                break
        return completion_key, suggestions

    @classmethod
    def option_completer(cls, k,v):
        "Tab completion hook for the %%opts cell magic."
        line = v.text_until_cursor

        completions = cls.setup_completer()
        sorted_keys = sorted(completions.keys())
        type_keys = [k for k in sorted_keys if ('.' not in k)]

        compositor_defs = {el.group:el.output_type.__name__
                           for el in Compositor.definitions}

        completion_key, suggestions = cls.dotted_completion(line, sorted_keys, compositor_defs)

        if suggestions and line.endswith('.'):
            return ['.'.join([completion_key, el]) for el in suggestions]
        elif not completion_key:
            return type_keys + compositor_defs.keys()

        if line.endswith(']') or (line.count('[') - line.count(']')) % 2:
            kws = completions[completion_key][0]
            return [kw+'=' for kw in kws]

        if line.endswith('}') or (line.count('{') - line.count('}')) % 2:
            return ['-groupwise', '-mapwise']

        style_completions = [kw+'=' for kw in completions[completion_key][1]]
        if line.endswith(')') or (line.count('(') - line.count(')')) % 2:
            return style_completions
        return style_completions + type_keys + compositor_defs.keys()



@magics_class
class OptsMagic(Magics):
    """
    Magic for easy customising of normalization, plot and style options.
    Consult %%opts? for more information.
    """
    error_message = None # If not None, the error message that will be displayed
    next_id = None       # Next id to propagate, binding displayed object together.
    applied_keys = []    # Path specs selecting the objects to be given a new id

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
            assert cls.next_id in Store.custom_options, 'RealityError'
            obj.traverse(lambda o: setattr(o, 'id', cls.next_id),
                         specs=cls.applied_keys)
            cls.next_id = None
            cls.applied_keys = []
        return None


    @classmethod
    def _format_options_error(cls, err):
        info = (err.invalid_keyword, err.group_name, ', '.join(err.allowed_keywords))
        return "Keyword <b>%r</b> not one of following %s options:<br><br><b>%s</b>" % info


    @classmethod
    def customize_tree(cls, spec, options):
        """
        Returns a customized copy of the Store.options OptionsTree object.
        """
        for key in sorted(spec.keys()):
            try:
                options[str(key)] = spec[key]
            except OptionError as e:
                cls.error_message = cls._format_options_error(e)
                return None
        return options

    @classmethod
    def register_custom_spec(cls, spec, cellmagic):
        ids = Store.custom_options.keys()
        max_id = max(ids) if len(ids)>0 else -1
        options = OptionTree(items=Store.options.data.items(),
                             groups=Store.options.groups)
        custom_tree = cls.customize_tree(spec, options)
        if custom_tree is not None:
            Store.custom_options[max_id+1] = custom_tree
        if cellmagic:
            cls.next_id = max_id+1
            cls.applied_keys += spec.keys()
        else:
            cls.next_id = None
            cls.applied_keys = []

    @classmethod
    def expand_compositor_keys(cls, spec):
        """
        Expands compositor definition keys into {type}.{group}
        keys. For instance a compositor operation returning a group
        string 'Image' of element type RGB expands to 'RGB.Image'.
        """
        expanded_spec={}
        compositor_defs = {el.group:el.output_type.__name__
                           for el in Compositor.definitions}
        for key, val in spec.items():
            if key not in compositor_defs:
                expanded_spec[key] = val
            else:
                cls.applied_keys = ['Overlay'] # Send id to Overlays
                type_name = compositor_defs[key]
                expanded_spec[str(type_name+'.'+key)] = val
        return expanded_spec


    @line_cell_magic
    def opts(self, line='', cell=None):
        """
        The opts line/cell magic with tab-completion.

        %%opts [ [path] [normalization] [plotting options] [style options]]+

        path:             A dotted type.group.label specification
                          (e.g. Image.Grayscale.Photo)

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
            spec = self.expand_compositor_keys(spec)
        except SyntaxError:
            display(HTML("<b>Invalid syntax</b>: Consult <tt>%%opts?</tt> for more information."))
            return

        self.register_custom_spec(spec, cell is not None)
        if cell:
            self.shell.run_cell(cell, store_history=STORE_HISTORY)
        else:
            retval = self.customize_tree(spec, Store.options)
            if retval is None:
                display(HTML(OptsMagic.error_message))
        OptsMagic.error_message = None



@magics_class
class TimerMagic(Magics):
    """
    A line magic for measuring the execution time of multiple cells.

    After you start/reset the timer with '%timer start' you may view
    elapsed time with any subsequent calls to %timer.
    """

    start_time = None

    @staticmethod
    def elapsed_time():
        elapsed = time.time() -  TimerMagic.start_time
        minutes = elapsed // 60
        hours = minutes // 60
        seconds = elapsed % 60
        return "Timer elapsed: %02d:%02d:%02d" % (hours, minutes, seconds)

    @classmethod
    def option_completer(cls, k,v):
        return ['start']

    @line_magic
    def timer(self, line=''):
        """
        Timer magic to print initial date/time information and
        subsequent elapsed time intervals.

        To start the timer, run:

        %timer start

        This will print the start date and time.

        Subsequent calls to %timer will print the elapsed time
        relative to the time when %timer start was called. Subsequent
        calls to %timer start may also be used to reset the timer.
        """

        if line.strip() not in ['', 'start']:
            print("Invalid argument to %timer. For more information consult %timer?")
            return
        elif line.strip() == 'start':
            TimerMagic.start_time = time.time()
            timestamp = time.strftime("%Y/%m/%d %H:%M:%S")
            print("Timer start: %s" % timestamp)
            return
        elif self.start_time is None:
            print("Please start timer with %timer start. For more information consult %timer?")
        else:
            print(self.elapsed_time())


def load_magics(ip):
    ip.register_magics(TimerMagic)
    ip.register_magics(OutputMagic)

    if pyparsing is None:  print("%opts magic unavailable (pyparsing cannot be imported)")
    else: ip.register_magics(OptsMagic)

    if pyparsing is None: print("%compositor magic unavailable (pyparsing cannot be imported)")
    else: ip.register_magics(CompositorMagic)


    # Configuring tab completion
    ip.set_hook('complete_command', TimerMagic.option_completer, str_key = '%timer')
    ip.set_hook('complete_command', CompositorMagic.option_completer, str_key = '%compositor')

    ip.set_hook('complete_command', OutputMagic.option_completer, str_key = '%output')
    ip.set_hook('complete_command', OutputMagic.option_completer, str_key = '%%output')

    OptsCompleter.setup_completer()
    ip.set_hook('complete_command', OptsCompleter.option_completer, str_key = '%%opts')
    ip.set_hook('complete_command', OptsCompleter.option_completer, str_key = '%opts')
