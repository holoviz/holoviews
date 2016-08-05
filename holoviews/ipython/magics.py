import time
from collections import defaultdict

try:
    from IPython.core.magic import Magics, magics_class, line_magic, line_cell_magic
except:
    from unittest import SkipTest
    raise SkipTest("IPython extension requires IPython >= 0.13")

from ..core import OrderedDict
from ..core import util
from ..core.options import Options, OptionError, Store, StoreOptions
from ..core.pprint import InfoPrinter

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

from IPython.core import page
InfoPrinter.store = Store

class OptionsMagic(Magics):
    """
    Base class for magics that are used to specified collections of
    keyword options.
    """
    # Dictionary from keywords to allowed bounds/values
    allowed = {'charwidth'   : (0, float('inf'))}
    defaults = OrderedDict([('charwidth'   , 80)])  # Default keyword values.
    options =  OrderedDict(defaults.items()) # Current options

    # Callables accepting (value, keyword, allowed) for custom exceptions
    custom_exceptions = {}

    # Hidden. Options that won't tab complete (for backward compatibility)
    hidden = {}


    @classmethod
    def update_options(cls, options, items):
        """
        Allows updating options depending on class attributes
        and unvalidated options.
        """
        pass

    @classmethod
    def get_options(cls, line, options, linemagic):
        "Given a keyword specification line, validated and compute options"
        items = cls._extract_keywords(line, OrderedDict())
        options = cls.update_options(options, items)
        for keyword in cls.defaults:
            if keyword in items:
                value = items[keyword]
                allowed = cls.allowed[keyword]
                if isinstance(allowed, set):  pass
                elif isinstance(allowed, dict):
                    if not isinstance(value, dict):
                        raise ValueError("Value %r not a dict type" % value)
                    disallowed = set(value.keys()) - set(allowed.keys())
                    if disallowed:
                        raise ValueError("Keywords %r for %r option not one of %s"
                                         % (disallowed, keyword, allowed))
                    wrong_type = {k: v for k, v in value.items()
                                  if not isinstance(v, allowed[k])}
                    if wrong_type:
                        errors = []
                        for k,v in wrong_type.items():
                            errors.append("Value %r for %r option's %r attribute not of type %r" %
                                          (v, keyword, k, allowed[k]))
                        raise ValueError('\n'.join(errors))
                elif isinstance(allowed, list) and value not in allowed:
                    if keyword in cls.custom_exceptions:
                        cls.custom_exceptions[keyword](value, keyword, allowed)
                    else:
                        raise ValueError("Value %r for key %r not one of %s"
                                         % (value, keyword, allowed))
                elif isinstance(allowed, tuple):
                    if not (allowed[0] <= value <= allowed[1]):
                        info = (keyword,value)+allowed
                        raise ValueError("Value %r for key %r not between %s and %s" % info)
                options[keyword] = value

        return cls._validate(options, items, linemagic)

    @classmethod
    def _validate(cls, options, items, linemagic):
        "Allows subclasses to check options are valid."
        raise NotImplementedError("OptionsMagic is an abstract base class.")

    @classmethod
    def option_completer(cls, k,v):
        raw_line = v.text_until_cursor
        line = raw_line.replace(cls.magic_name,'')

        # Find the last element class mentioned
        completion_key = None
        tokens = [t for els in reversed(line.split('=')) for t in els.split()]

        for token in tokens:
            if token.strip() in cls.allowed:
                completion_key = token.strip()
                break

        values = [val for val in cls.allowed.get(completion_key, [])
                  if val not in cls.hidden.get(completion_key, [])]
        vreprs = [repr(el) for el in values if not isinstance(el, tuple)]
        return vreprs + [el+'=' for el in cls.allowed.keys()]

    @classmethod
    def pprint(cls):
        """
        Pretty print the current element options with a maximum width of
        cls.pprint_width.
        """
        current, count = '', 0
        for k,v in cls.options.items():
            keyword = '%s=%r' % (k,v)
            if len(current) + len(keyword) > cls.options['charwidth']:
                print((cls.magic_name if count==0 else '      ')  + current)
                count += 1
                current = keyword
            else:
                current += ' '+ keyword
        else:
            print((cls.magic_name if count==0 else '      ')  + current)


    @classmethod
    def _extract_keywords(cls, line, items):
        """
        Given the keyword string, parse a dictionary of options.
        """
        unprocessed = list(reversed(line.split('=')))
        while unprocessed:
            chunk = unprocessed.pop()
            key = None
            if chunk.strip() in cls.allowed:
                key = chunk.strip()
            else:
                raise SyntaxError("Invalid keyword: %s" % chunk.strip())
            # The next chunk may end in a subsequent keyword
            value = unprocessed.pop().strip()
            if len(unprocessed) != 0:
                # Check if a new keyword has begun
                for option in cls.allowed:
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



def list_backends():
    backends = []
    for backend in Store.renderers:
        backends.append(backend)
        renderer = Store.renderers[backend]
        modes = [mode for mode in renderer.params('mode').objects if mode  != 'default']
        backends += ['%s:%s' % (backend, mode) for mode in modes]
    return backends


def list_formats(format_type, backend=None):
    """
    Returns list of supported formats for a particular
    backend.
    """
    if backend is None:
        backend = Store.current_backend
        mode = Store.renderers[backend].mode if backend in Store.renderers else None
    else:
        split = backend.split(':')
        backend, mode = split if len(split)==2 else (split[0], 'default')

    if backend in Store.renderers:
        return Store.renderers[backend].mode_formats[format_type][mode]
    else:
        return []


@magics_class
class OutputMagic(OptionsMagic):
    """
    Magic for easy customising of display options.
    Consult %%output? for more information.
    """

    magic_name = '%output'

    # Lists: strict options, Set: suggested options, Tuple: numeric bounds.
    allowed = {'backend'     : list_backends(),
               'fig'         : list_formats('fig'),
               'holomap'     : list_formats('holomap'),
               'widgets'     : ['embed', 'live'],
               'fps'         : (0, float('inf')),
               'max_frames'  : (0, float('inf')),
               'max_branches': (0, float('inf')),
               'size'        : (0, float('inf')),
               'dpi'         : (1, float('inf')),
               'charwidth'   : (0, float('inf')),
               'filename'    : {None},
               'info'        : [True, False],
               'css'         : {k: util.basestring
                                for k in ['width', 'height', 'padding', 'margin',
                                          'max-width', 'min-width', 'max-height',
                                          'min-height', 'outline', 'float']}}

    defaults = OrderedDict([('backend'     , 'matplotlib'),
                            ('fig'         , 'png'),
                            ('holomap'     , 'widgets'),
                            ('widgets'     , 'embed'),
                            ('fps'         , 20),
                            ('max_frames'  , 500),
                            ('max_branches', 2),
                            ('size'        , 100),
                            ('dpi'         , 72),
                            ('charwidth'   , 80),
                            ('filename'    , None),
                            ('info'        , False),
                            ('css'         , {})])

    options = OrderedDict(defaults.items())
    _backend_options = defaultdict(dict)

    # Used to disable info output in testing
    _disable_info_output = False

    #==========================#
    # Backend state management #
    #==========================#

    last_backend = None

    def missing_dependency_exception(value, keyword, allowed):
        raise Exception("Format %r does not appear to be supported." % value)

    custom_exceptions = {'holomap':missing_dependency_exception}

    # Counter for nbagg figures
    nbagg_counter = 0

    def __init__(self, *args, **kwargs):
        super(OutputMagic, self).__init__(*args, **kwargs)
        self.output.__func__.__doc__ = self._generate_docstring()


    @classmethod
    def info(cls, obj):
        if cls.options['info'] and not cls._disable_info_output:
            page.page(InfoPrinter.info(obj, ansi=True))


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
        page =  ("info    : The information to page about the displayed objects (default %r)"
                  % cls.defaults['info'])
        css =   ("css     : Optional css style attributes to apply to the figure image tag")

        descriptions = [backend, fig, holomap, widgets, fps, frames, branches, size, dpi, chars, fname, page, css]
        return '\n'.join(intro + descriptions)


    @classmethod
    def _validate(cls, options, items, linemagic):
        "Validation of edge cases and incompatible options"

        if 'html' in Store.display_formats:
            pass
        elif 'fig' in items and items['fig'] not in Store.display_formats:
            msg = ("Output magic requesting figure format %r " % items['fig']
                   + "not in display formats %r" % Store.display_formats)
            display(HTML("<b>Warning:</b> %s" % msg))

        backend = Store.current_backend
        return Store.renderers[backend].validate(options)


    @line_cell_magic
    def output(self, line, cell=None):
        line = line.split('#')[0].strip()
        if line == '':
            self.pprint()
            print("\nFor help with the %output magic, call %output?")
            return

        restore_copy = OrderedDict(OutputMagic.options.items())
        try:
            options = OrderedDict(OutputMagic.options.items())
            new_options = self.get_options(line, options, cell is None)
            self._set_render_options(new_options)
            OutputMagic.options = new_options
        except Exception as e:
            self.update_options(options, {'backend': restore_copy['backend']})
            OutputMagic.options = restore_copy
            self._set_render_options(restore_copy)
            print('Error: %s' % str(e))
            print("For help with the %output magic, call %output?\n")
            return

        if cell is not None:
            self.shell.run_cell(cell, store_history=STORE_HISTORY)
            self.update_options(options, {'backend': restore_copy['backend']})
            OutputMagic.options = restore_copy
            self._set_render_options(restore_copy)


    @classmethod
    def update_options(cls, options, items):
        """
        Switch default options and backend if new backend
        is supplied in items.
        """
        backend = items.get('backend', '')
        prev_backend = Store.current_backend
        renderer = Store.renderers[Store.current_backend]
        prev_backend += ':%s' % renderer.mode
        if not backend or backend == prev_backend:
            return options

        cls._backend_options[prev_backend] = cls.options

        backend_options = cls._backend_options[backend]
        for p in ['fig', 'holomap']:
            opts = list_formats(p, backend)
            cls.allowed[p] = opts
            cls.defaults[p] = opts[0]
            if p not in backend_options:
                backend_options[p] = opts[0]

        backend = backend.split(':')[0]
        render_params = ['fig', 'holomap', 'size', 'fps', 'dpi', 'css']
        for p in render_params:
            if p in backend_options:
                opt = backend_options[p]
                cls.defaults[p] = opt
            else:
                opt = cls.defaults[p]
                backend_options[p] = opt

        for opt in options:
            if opt not in backend_options:
                backend_options[opt] = options[opt]

        cls.set_backend(backend)
        return backend_options


    @classmethod
    def initialize(cls):
        backend = cls.options.get('backend', cls.defaults['backend'])
        if backend in Store.renderers:
            cls.options = dict(cls.defaults)
            cls._set_render_options(cls.defaults)
            cls.set_backend(backend)
        else:
            cls.options['backend'] = None
            cls.defaults['backend'] = None
            cls.set_backend(None)

    @classmethod
    def set_backend(cls, backend):
        cls.last_backend = Store.current_backend
        Store.current_backend = backend


    @classmethod
    def _set_render_options(cls, options):
        """
        Set options on current Renderer.
        """
        split = options['backend'].split(':')
        backend, mode = split if len(split)==2 else (split[0], 'default')
        renderer = Store.renderers[backend]
        render_params = ['fig', 'holomap', 'size', 'fps', 'dpi', 'css']
        render_options = {k: options[k] for k in render_params}
        renderer.set_param(**dict(render_options, widget_mode=options['widgets'],
                                  mode=mode))


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
            for definition in CompositorSpec.parse(line.strip(), ns=self.shell.user_ns):
                group = {'style':Options(), 'plot': Options(), 'norm':Options()}
                type_name = definition.output_type.__name__
                Store.options()[type_name + '.' + definition.group] = group
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
        try:
            for element in Store.options().keys():
                options = Store.options()['.'.join(element)]
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
                    return completion_key, suggestions
            # Attempting to match compositor definitions
            if token in compositor_defs:
                completion_key = compositor_defs[token]
                break
        return completion_key, suggestions

    @classmethod
    def _inside_delims(cls, line, opener, closer):
        return (line.count(opener) - line.count(closer)) % 2

    @classmethod
    def option_completer(cls, k,v):
        "Tab completion hook for the %%opts cell magic."
        line = v.text_until_cursor
        completions = cls.setup_completer()
        compositor_defs = {el.group:el.output_type.__name__
                           for el in Compositor.definitions}
        return cls.line_completer(line, completions, compositor_defs)

    @classmethod
    def line_completer(cls, line, completions, compositor_defs):
        sorted_keys = sorted(completions.keys())
        type_keys = [key for key in sorted_keys if ('.' not in key)]
        completion_key, suggestions = cls.dotted_completion(line, sorted_keys, compositor_defs)

        verbose_openers = ['style(', 'plot[', 'norm{']
        if suggestions and line.endswith('.'):
            return ['.'.join([completion_key, el]) for el in suggestions]
        elif not completion_key:
            return type_keys + list(compositor_defs.keys()) + verbose_openers

        if cls._inside_delims(line,'[', ']'):
            return [kw+'=' for kw in completions[completion_key][0]]

        if cls._inside_delims(line, '{', '}'):
            return ['+axiswise', '+framewise']

        style_completions = [kw+'=' for kw in completions[completion_key][1]]
        if cls._inside_delims(line, '(', ')'):
            return style_completions

        return type_keys + list(compositor_defs.keys()) + verbose_openers



@magics_class
class OptsMagic(Magics):
    """
    Magic for easy customising of normalization, plot and style options.
    Consult %%opts? for more information.
    """
    error_message = None # If not None, the error message that will be displayed
    opts_spec = None       # Next id to propagate, binding displayed object together.

    @classmethod
    def process_element(cls, obj):
        """
        To be called by the display hook which supplies the element to
        be displayed. Any customisation of the object can then occur
        before final display. If there is any error, a HTML message
        may be returned. If None is returned, display will proceed as
        normal.
        """
        if cls.error_message:
            return cls.error_message
        if cls.opts_spec is not None:
            StoreOptions.set_options(obj, cls.opts_spec)
            cls.opts_spec = None
        return None


    @classmethod
    def _format_options_error(cls, err):
        info = (err.invalid_keyword, err.group_name, ', '.join(err.allowed_keywords))
        return "Keyword <b>%r</b> not one of following %s options:<br><br><b>%s</b>" % info

    @classmethod
    def register_custom_spec(cls, spec):
        spec, _ = StoreOptions.expand_compositor_keys(spec)
        try:
            StoreOptions.validate_spec(spec)
        except OptionError as e:
            cls.error_message = cls._format_options_error(e)
            return None

        cls.opts_spec = spec

    @classmethod
    def _partition_lines(cls, line, cell):
        """
        Check the code for additional use of %%opts. Enables
        multi-line use of %%opts in a single call to the magic.
        """
        if cell is None: return (line, cell)
        specs, code = [line], []
        for line in cell.splitlines():
            if line.strip().startswith('%%opts'):
                specs.append(line.strip()[7:])
            else:
                code.append(line)
        return ' '.join(specs), '\n'.join(code)


    @line_cell_magic
    def opts(self, line='', cell=None):
        """
        The opts line/cell magic with tab-completion.

        %%opts [ [path] [normalization] [plotting options] [style options]]+

        path:             A dotted type.group.label specification
                          (e.g. Image.Grayscale.Photo)

        normalization:    List of normalization options delimited by braces.
                          One of | -axiswise | -framewise | +axiswise | +framewise |
                          E.g. { +axiswise +framewise }

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
        line, cell = self._partition_lines(line, cell)
        try:
            spec = OptsSpec.parse(line, ns=self.shell.user_ns)
        except SyntaxError:
            display(HTML("<b>Invalid syntax</b>: Consult <tt>%%opts?</tt> for more information."))
            return

        if cell:
            self.register_custom_spec(spec)
            # Process_element is invoked when the cell is run.
            self.shell.run_cell(cell, store_history=STORE_HISTORY)
        else:
            try:
                StoreOptions.validate_spec(spec)
            except OptionError as e:
                OptsMagic.error_message = None
                display(HTML(self._format_options_error(e)))
                return

            StoreOptions.apply_customizations(spec, Store.options())
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
        seconds = time.time() -  TimerMagic.start_time
        minutes = seconds // 60
        hours = minutes // 60
        return "Timer elapsed: %02d:%02d:%02d" % (hours, minutes % 60, seconds % 60)

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
