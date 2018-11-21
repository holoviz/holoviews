import os, sys, inspect, shutil

import param

from ..core import DynamicMap, HoloMap, Dimensioned, ViewableElement, StoreOptions, Store
from ..core.options import options_policy, Keywords, Options
from ..core.operation import Operation
from ..core.util import Aliases, basestring, merge_option_dicts  # noqa (API import)
from ..core.operation import OperationCallable
from ..core.spaces import Callable
from ..core import util
from ..streams import Stream
from .settings import OutputSettings, list_formats, list_backends

Store.output_settings = OutputSettings



def examples(path='holoviews-examples', verbose=False, force=False, root=__file__):
    """
    Copies the notebooks to the supplied path.
    """
    filepath = os.path.abspath(os.path.dirname(root))
    example_dir = os.path.join(filepath, './examples')
    if not os.path.exists(example_dir):
        example_dir = os.path.join(filepath, '../examples')
    if os.path.exists(path):
        if not force:
            print('%s directory already exists, either delete it or set the force flag' % path)
            return
        shutil.rmtree(path)
    ignore = shutil.ignore_patterns('.ipynb_checkpoints','*.pyc','*~')
    tree_root = os.path.abspath(example_dir)
    if os.path.isdir(tree_root):
        shutil.copytree(tree_root, path, ignore=ignore, symlinks=True)
    else:
        print('Cannot find %s' % tree_root)



class opts(param.ParameterizedFunction):
    """
    Utility function to set options either at the global level or on a
    specific object.

    To set opts globally use:

    opts(options)

    Where options may be an options specification string (as accepted by
    the %opts magic) or an options specifications dictionary.

    For instance:

    opts("Curve (color='k')") # Or equivalently
    opts({'Curve': {'style': {'color':'k'}}})

    To set opts on a specific object, just supply it as the second
    argument:

    opts(options, obj)

    For instance:

    curve = hv.Curve([1,2,3])
    opts("Curve (color='k')", curve) # Or equivalently
    opts({'Curve': {'style': {'color':'k'}}}, curve)

    These two modes are equivalent to the %opts line magic and the
    %%opts cell magic respectively.
    """

    __original_docstring__ = None

    strict = param.Boolean(default=False, doc="""
       Whether to be strict about the options specification. If not set
       to strict (default), any invalid keywords are simply skipped. If
       strict, invalid keywords prevent the options being applied.""")

    def __call__(self, *args, **params):

        if args and set(params.keys()) - set(['strict']):
            raise TypeError('When used with positional arguments, hv.opts accepts only strings and dictionaries, not keywords.')
        if params and not args:
            return Options(**params)

        p = param.ParamOverrides(self, params)
        if len(args) not in [1,2]:
            raise TypeError('The opts utility accepts one or two positional arguments.')
        elif len(args) == 1:
            options, obj = args[0], None
        elif len(args) == 2:
            (options, obj) = args

        if isinstance(options, basestring):
            from .parser import OptsSpec
            try:     ns = get_ipython().user_ns  # noqa
            except:  ns = globals()
            options = OptsSpec.parse(options, ns=ns)


        errmsg = StoreOptions.validation_error_message(options)
        if errmsg:
            sys.stderr.write(errmsg)
            if p.strict:
                sys.stderr.write(' Options specification will not be applied.')
                if obj: return obj
                else:   return

        if obj is None:
            with options_policy(skip_invalid=True, warn_on_skip=False):
                StoreOptions.apply_customizations(options, Store.options())
        elif not isinstance(obj, Dimensioned):
            return obj
        else:
            return StoreOptions.set_options(obj, options)


    @classmethod
    def expand_options(cls, options, backend=None):
        """
        Validates and expands a dictionaries of options indexed by
        type[.group][.label] keys into separate style, plot and norm
        options.

            opts.expand_options({'Image': dict(cmap='viridis', show_title=False)})

        returns

            {'Image': {'plot': dict(show_title=False), 'style': dict(cmap='viridis')}}
        """
        current_backend = Store.current_backend
        try:
            backend_options = Store.options(backend=backend or current_backend)
        except KeyError as e:
            raise Exception('The %s backend is not loaded. Please load the backend using hv.extension.' % str(e))
        expanded = {}
        if isinstance(options, list):
            merged_options = {}
            for obj in options:
                if isinstance(obj,dict):
                    new_opts = obj
                else:
                    new_opts = {obj.key: obj.kwargs}

                merged_options = merge_option_dicts(merged_options, new_opts)
            options = merged_options

        for objspec, options in options.items():
            objtype = objspec.split('.')[0]
            if objtype not in backend_options:
                raise ValueError('%s type not found, could not apply options.'
                                 % objtype)
            obj_options = backend_options[objtype]
            expanded[objspec] = {g: {} for g in obj_options.groups}
            for opt, value in options.items():
                found = False
                valid_options = []
                for g, group_opts in sorted(obj_options.groups.items()):
                    if opt in group_opts.allowed_keywords:
                        expanded[objspec][g][opt] = value
                        found = True
                        break
                    valid_options += group_opts.allowed_keywords
                if found: continue
                cls._options_error(opt, objtype, backend, valid_options)
        return expanded


    @classmethod
    def _options_error(cls, opt, objtype, backend, valid_options):
        """
        Generates an error message for an invalid option suggesting
        similar options through fuzzy matching.
        """
        current_backend = Store.current_backend
        loaded_backends = Store.loaded_backends()
        kws = Keywords(values=valid_options)
        matches = sorted(kws.fuzzy_match(opt))
        if backend is not None:
            if matches:
                raise ValueError('Unexpected option %r for %s types '
                                 'when using the %r extension. Similar '
                                 'options are: %s.' %
                                 (opt, objtype, backend, matches))
            else:
                raise ValueError('Unexpected option %r for %s types '
                                 'when using the %r extension. No '
                                 'similar options founds.' %
                                 (opt, objtype, backend))

        # Check option is invalid for all backends
        found = []
        for lb in [b for b in loaded_backends if b != backend]:
            lb_options = Store.options(backend=lb).get(objtype)
            if lb_options is None:
                continue
            for g, group_opts in lb_options.groups.items():
                if opt in group_opts.allowed_keywords:
                    found.append(lb)
        if found:
            param.main.warning('Option %r for %s type not valid '
                               'for selected backend (%r). Option '
                               'only applies to following backends: %r' %
                               (opt, objtype, current_backend, found))
            return

        if matches:
            raise ValueError('Unexpected option %r for %s types '
                             'across all extensions. Similar options '
                             'for current extension (%r) are: %s.' %
                             (opt, objtype, current_backend, matches))
        else:
            raise ValueError('Unexpected option %r for %s types '
                             'across all extensions. No similar options '
                             'found.' % (opt, objtype))

    @classmethod
    def _completer_reprs(cls, options, namespace=None):
        """
        Given a list of Option objects (such as those returned from
        OptsSpec.parse_options) or an %opts or %%opts magic string,
        return a list of corresponding completer reprs. The namespace is
        typically given as 'hv' if fully qualified namespaces are
        desired.
        """
        if isinstance(options, basestring):
            from .parser import OptsSpec
            try:     ns = get_ipython().user_ns  # noqa
            except:  ns = globals()
            options = options.replace('%%opts','').replace('%opts','')
            options = OptsSpec.parse_options(options, ns=ns)


        reprs = []
        ns = '{namespace}.'.format(namespace=namespace) if namespace else ''
        for option in options:
            kws = ', '.join('%s=%r' % (k,option.kwargs[k]) for k in sorted(option.kwargs))
            if '.' in option.key:
                element = option.key.split('.')[0]
                spec = repr('.'.join(option.key.split('.')[1:])) + ', '
            else:
                element = option.key
                spec = ''

            opts_format = '{ns}opts.{element}({spec}{kws})'
            reprs.append(opts_format.format(ns=ns, spec=spec, kws=kws))
        return reprs

    @classmethod
    def _build_completer(cls, element, allowed):
        def fn(cls, spec=None, **kws):
            spec = element if spec is None else '%s.%s' % (element, spec)
            return Options(spec, **kws)

        kws = ', '.join('{opt}=None'.format(opt=opt) for opt in sorted(allowed))
        fn.__doc__ = '{element}({kws})'.format(element=element, kws=kws)
        return classmethod(fn)

    @classmethod
    def _update_backend(cls, backend):

        if cls.__original_docstring__ is None:
            cls.__original_docstring__ = cls.__doc__

        if backend not in Store.loaded_backends():
            return

        backend_options = Store.options(backend)
        all_keywords = set()
        for element in backend_options.keys():
            if '.' in element: continue
            element_keywords = []
            options = backend_options['.'.join(element)]
            for group in Options._option_groups:
                element_keywords.extend(options[group].allowed_keywords)

            all_keywords |= set(element_keywords)
            with param.logging_level('CRITICAL'):
                setattr(cls, element[0],
                        cls._build_completer(element[0],
                                             element_keywords))

        kws = ', '.join('{opt}=None'.format(opt=opt) for opt in sorted(all_keywords))
        old_doc = cls.__original_docstring__.replace('params(strict=Boolean, name=String)','')
        cls.__doc__ = '\n    opts({kws})'.format(kws=kws) + old_doc


Store._backend_switch_hooks.append(opts._update_backend)


class output(param.ParameterizedFunction):
    """
    Utility function to set output either at the global level or on a
    specific object.

    To set output globally use:

    output(options)

    Where options may be an options specification string (as accepted by
    the %opts magic) or an options specifications dictionary.

    For instance:

    output("backend='bokeh'") # Or equivalently
    output(backend='bokeh')

    To set save output from a specific object do disk using the
    'filename' argument, you can supply the object as the first
    positional argument and supply the filename keyword:

    curve = hv.Curve([1,2,3])
    output(curve, filename='curve.png')

    For compatibility with the output magic, you can supply the object
    as the second argument after the string specification:

    curve = hv.Curve([1,2,3])
    output("filename='curve.png'", curve)

    These two modes are equivalent to the %output line magic and the
    %%output cell magic respectively. Note that only the filename
    argument is supported when supplying an object and all other options
    are ignored.
    """

    filename_warning = param.Boolean(default=True, doc="""
       Whether to warn if the output utility is called on an object and
       a filename is not given (in which case the utility has no
       effect)""" )

    def __call__(self, *args, **options):
        warn = options.pop('filename_warning', self.filename_warning)
        help_prompt = 'For help with hv.util.output call help(hv.util.output)'
        line, obj = None,None
        if len(args) > 2:
            raise TypeError('The opts utility accepts one or two positional arguments.')
        if len(args) == 1 and options:
            obj = args[0]
        elif len(args) == 1:
            line = args[0]
        elif len(args) == 2:
            (line, obj) = args

        if isinstance(obj, Dimensioned):
            if line:
                options = Store.output_settings.extract_keywords(line, {})
            for k in options.keys():
                if k not in Store.output_settings.allowed:
                    raise KeyError('Invalid keyword: %s' % k)
            if 'filename' in options:
                def save_fn(obj, renderer): renderer.save(obj, options['filename'])
                Store.output_settings.output(line=line, cell=obj, cell_runner=save_fn,
                                             help_prompt=help_prompt, **options)
            elif warn:
                self.warning("hv.output not supplied a filename to export the "
                             "given object. This call will have no effect." )
            return obj
        elif obj is not None:
            return obj
        else:
            Store.output_settings.output(line=line, help_prompt=help_prompt, **options)

output.__doc__ = Store.output_settings._generate_docstring()


def renderer(name):
    """
    Helper utility to access the active renderer for a given extension.
    """
    try:
        if name not in Store.renderers:
            extension(name)
        return Store.renderers[name]
    except ImportError:
        msg = ('Could not find a {name!r} renderer, available renderers are: {available}.')
        available = ', '.join(repr(k) for k in Store.renderers)
        raise ImportError(msg.format(name=name, available=available))


class extension(param.ParameterizedFunction):
    """
    Helper utility used to load holoviews extensions. These can be
    plotting extensions, element extensions or anything else that can be
    registered to work with HoloViews.
    """

    # Mapping between backend name and module name
    _backends = {'matplotlib': 'mpl',
                 'bokeh': 'bokeh',
                 'plotly': 'plotly'}

    def __call__(self, *args, **params):
        # Get requested backends
        config = params.pop('config', {})
        util.config.set_param(**config)
        imports = [(arg, self._backends[arg]) for arg in args
                   if arg in self._backends]
        for p, val in sorted(params.items()):
            if p in self._backends:
                imports.append((p, self._backends[p]))
        if not imports:
            args = ['matplotlib']
            imports = [('matplotlib', 'mpl')]

        args = list(args)
        selected_backend = None
        for backend, imp in imports:
            try:
                __import__(backend)
            except:
                self.warning("%s could not be imported, ensure %s is installed."
                             % (backend, backend))
            try:
                __import__('holoviews.plotting.%s' % imp)
                if selected_backend is None:
                    selected_backend = backend
            except util.VersionError as e:
                self.warning("HoloViews %s extension could not be loaded. "
                             "The installed %s version %s is less than "
                             "the required version %s." %
                             (backend, backend, e.version, e.min_version))
            except Exception as e:
                self.warning("Holoviews %s extension could not be imported, "
                             "it raised the following exception: %s('%s')" %
                             (backend, type(e).__name__, e))
            finally:
                Store.output_settings.allowed['backend'] = list_backends()
                Store.output_settings.allowed['fig'] = list_formats('fig', backend)
                Store.output_settings.allowed['holomap'] = list_formats('holomap', backend)

        if selected_backend is None:
            raise ImportError('None of the backends could be imported')
        Store.set_current_backend(selected_backend)


def save(obj, filename, fmt='auto', backend=None, **kwargs):
    """
    Saves the supplied object to file.

    The available output formats depend on the backend being used. By
    default and if the filename is a string the output format will be
    inferred from the file extension. Otherwise an explicit format
    will need to be specified. For ambiguous file extensions such as
    html it may be necessary to specify an explicit fmt to override
    the default, e.g. in the case of 'html' output the widgets will
    default to fmt='widgets', which may be changed to scrubber widgets
    using fmt='scrubber'.

    Arguments
    ---------
    obj: HoloViews object
        The HoloViews object to save to file
    filename: string or IO object
        The filename or BytesIO/StringIO object to save to
    fmt: string
        The format to save the object as, e.g. png, svg, html, or gif
        and if widgets are desired either 'widgets' or 'scrubber'
    backend: string
        A valid HoloViews rendering backend, e.g. bokeh or matplotlib
    **kwargs: dict
        Additional keyword arguments passed to the renderer,
        e.g. fps for animations
    """
    backend = backend or Store.current_backend
    renderer_obj = renderer(backend)
    if kwargs:
        renderer_obj = renderer_obj.instance(**kwargs)
    if isinstance(filename, basestring):
        supported = [mfmt for tformats in renderer_obj.mode_formats.values()
                     for mformats in tformats.values() for mfmt in mformats]
        formats = filename.split('.')
        if fmt == 'auto' and formats and formats[-1] != 'html':
            fmt = formats[-1]
        if formats[-1] in supported:
            filename = '.'.join(formats[:-1])
    return renderer_obj.save(obj, filename, fmt=fmt)


def render(obj, backend=None, **kwargs):
    """
    Renders the HoloViews object to the corresponding object in the
    specified backend, e.g. a Matplotlib or Bokeh figure.

    The backend defaults to the currently declared default
    backend. The resulting object can then be used with other objects
    in the specified backend. For instance, if you want to make a
    multi-part Bokeh figure using a plot type only available in
    HoloViews, you can use this function to return a Bokeh figure that
    you can use like any hand-constructed Bokeh figure in a Bokeh
    layout.

    Arguments
    ---------
    obj: HoloViews object
        The HoloViews object to render
    backend: string
        A valid HoloViews rendering backend
    **kwargs: dict
        Additional keyword arguments passed to the renderer,
        e.g. fps for animations

    Returns
    -------
    renderered:
        The rendered representation of the HoloViews object, e.g.
        if backend='matplotlib' a matplotlib Figure or FuncAnimation
    """
    backend = backend or Store.current_backend
    renderer_obj = renderer(backend)
    if kwargs:
        renderer_obj = renderer_obj.instance(**kwargs)
    plot = renderer_obj.get_plot(obj)
    if backend == 'matplotlib' and len(plot) > 1:
        return plot.anim(fps=renderer_obj.fps)
    return renderer_obj.get_plot(obj).state


class Dynamic(param.ParameterizedFunction):
    """
    Dynamically applies a callable to the Elements in any HoloViews
    object. Will return a DynamicMap wrapping the original map object,
    which will lazily evaluate when a key is requested. By default
    Dynamic applies a no-op, making it useful for converting HoloMaps
    to a DynamicMap.

    Any supplied kwargs will be passed to the callable and any streams
    will be instantiated on the returned DynamicMap. If the supplied
    operation is a method on a parameterized object which was
    decorated with parameter dependencies Dynamic will automatically
    create a stream to watch the parameter changes. This default
    behavior may be disabled by setting watch=False.
    """

    operation = param.Callable(default=lambda x: x, doc="""
        Operation or user-defined callable to apply dynamically""")

    kwargs = param.Dict(default={}, doc="""
        Keyword arguments passed to the function.""")

    link_inputs = param.Boolean(default=True, doc="""
         If Dynamic is applied to another DynamicMap, determines whether
         linked streams attached to its Callable inputs are
         transferred to the output of the utility.

         For example if the Dynamic utility is applied to a DynamicMap
         with an RangeXY, this switch determines whether the
         corresponding visualization should update this stream with
         range changes originating from the newly generated axes.""")

    shared_data = param.Boolean(default=False, doc="""
        Whether the cloned DynamicMap will share the same cache.""")

    streams = param.List(default=[], doc="""
        List of streams to attach to the returned DynamicMap""")

    def __call__(self, map_obj, **params):
        watch = params.pop('watch', True)
        self.p = param.ParamOverrides(self, params)
        callback = self._dynamic_operation(map_obj)
        streams = self._get_streams(map_obj, watch)
        if isinstance(map_obj, DynamicMap):
            dmap = map_obj.clone(callback=callback, shared_data=self.p.shared_data,
                                 streams=streams)
        else:
            dmap = self._make_dynamic(map_obj, callback, streams)
        return dmap


    def _get_streams(self, map_obj, watch=True):
        """
        Generates a list of streams to attach to the returned DynamicMap.
        If the input is a DynamicMap any streams that are supplying values
        for the key dimension of the input are inherited. And the list
        of supplied stream classes and instances are processed and
        added to the list.
        """
        streams = []
        for stream in self.p.streams:
            if inspect.isclass(stream) and issubclass(stream, Stream):
                stream = stream()
            elif not isinstance(stream, Stream):
                raise ValueError('Streams must be Stream classes or instances')
            if isinstance(self.p.operation, Operation):
                updates = {k: self.p.operation.p.get(k) for k, v in stream.contents.items()
                           if v is None and k in self.p.operation.p}
                if updates:
                    reverse = {v: k for k, v in stream._rename.items()}
                    stream.update(**{reverse.get(k, k): v for k, v in updates.items()})
            streams.append(stream)
        if isinstance(map_obj, DynamicMap):
            dim_streams = util.dimensioned_streams(map_obj)
            streams = list(util.unique_iterator(streams + dim_streams))

        # If callback is a parameterized method and watch is disabled add as stream
        param_watch_support = util.param_version >= '1.8.0' and watch
        if util.is_param_method(self.p.operation) and param_watch_support:
            streams.append(self.p.operation)
        valid, invalid = Stream._process_streams(streams)
        if invalid:
            msg = ('The supplied streams list contains objects that '
                   'are not Stream instances: {objs}')
            raise TypeError(msg.format(objs = ', '.join('%r' % el for el in invalid)))
        return valid


    def _process(self, element, key=None):
        if isinstance(self.p.operation, Operation):
            kwargs = {k: v for k, v in self.p.kwargs.items()
                      if k in self.p.operation.params()}
            return self.p.operation.process_element(element, key, **kwargs)
        else:
            return self.p.operation(element, **self.p.kwargs)


    def _dynamic_operation(self, map_obj):
        """
        Generate function to dynamically apply the operation.
        Wraps an existing HoloMap or DynamicMap.
        """
        if not isinstance(map_obj, DynamicMap):
            def dynamic_operation(*key, **kwargs):
                self.p.kwargs.update(kwargs)
                obj = map_obj[key] if isinstance(map_obj, HoloMap) else map_obj
                return self._process(obj, key)
        else:
            def dynamic_operation(*key, **kwargs):
                self.p.kwargs.update(kwargs)
                return self._process(map_obj[key], key)
        if isinstance(self.p.operation, Operation):
            return OperationCallable(dynamic_operation, inputs=[map_obj],
                                     link_inputs=self.p.link_inputs,
                                     operation=self.p.operation)
        else:
            return Callable(dynamic_operation, inputs=[map_obj],
                            link_inputs=self.p.link_inputs)


    def _make_dynamic(self, hmap, dynamic_fn, streams):
        """
        Accepts a HoloMap and a dynamic callback function creating
        an equivalent DynamicMap from the HoloMap.
        """
        if isinstance(hmap, ViewableElement):
            return DynamicMap(dynamic_fn, streams=streams)
        dim_values = zip(*hmap.data.keys())
        params = util.get_param_values(hmap)
        kdims = [d(values=list(util.unique_iterator(values))) for d, values in
                 zip(hmap.kdims, dim_values)]
        return DynamicMap(dynamic_fn, streams=streams, **dict(params, kdims=kdims))
