import os, sys, inspect, shutil

import param

from ..core import DynamicMap, HoloMap, Dimensioned, ViewableElement, StoreOptions, Store
from ..core.options import options_policy
from ..core.operation import Operation
from ..core.util import Aliases, basestring  # noqa (API import)
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

    strict = param.Boolean(default=False, doc="""
       Whether to be strict about the options specification. If not set
       to strict (default), any invalid keywords are simply skipped. If
       strict, invalid keywords prevent the options being applied.""")

    def __call__(self, *args, **params):
        p = param.ParamOverrides(self, params)
        if len(args) not in [1,2]:
            raise TypeError('The opts utility accepts one or two positional arguments.')
        elif len(args) == 1:
            options, obj = args[0], None
        elif len(args) == 2:
            (options, obj) = args

        if isinstance(options, basestring):
            from .parser import OptsSpec
            options = OptsSpec.parse(options)


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
        Expands a flat dictionary of options organized by the type
        of object into the appropriate groups.
        """
        backend = backend or Store.current_backend
        backend_options = Store.options(backend=backend)
        groups = set(backend_options.groups.keys())
        expanded = {}
        for objtype, options in options.items():
            if objtype not in backend_options:
                raise ValueError('%s type not found, could not apply options.' % objtype)
            obj_options = backend_options[objtype]
            expanded[objtype] = {g: {} for g in obj_options.groups}
            for opt, value in options.items():
                found = False
                for g, group_opts in obj_options.groups.items():
                    if opt in group_opts.allowed_keywords:
                        expanded[objtype][g][opt] = value
                        found = True
                if not found:
                    raise ValueError('%s option is not valid for %s types '
                                     'on %s backend.' % (opt, objtype, backend))
        return expanded


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
                self.warning("%s is not available, ensure %s is installed "
                             "to activate %s extension." % (backend, backend))
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
        Store.current_backend = selected_backend


class Dynamic(param.ParameterizedFunction):
    """
    Dynamically applies a callable to the Elements in any HoloViews
    object. Will return a DynamicMap wrapping the original map object,
    which will lazily evaluate when a key is requested. By default
    Dynamic applies a no-op, making it useful for converting HoloMaps
    to a DynamicMap.

    Any supplied kwargs will be passed to the callable and any streams
    will be instantiated on the returned DynamicMap.
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
        self.p = param.ParamOverrides(self, params)
        callback = self._dynamic_operation(map_obj)
        streams = self._get_streams(map_obj)
        if isinstance(map_obj, DynamicMap):
            dmap = map_obj.clone(callback=callback, shared_data=self.p.shared_data,
                                 streams=streams)
        else:
            dmap = self._make_dynamic(map_obj, callback, streams)
        return dmap


    def _get_streams(self, map_obj):
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
        return streams


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
