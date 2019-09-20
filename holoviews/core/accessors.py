"""
Module for accessor objects for viewable HoloViews objects.
"""
from __future__ import absolute_import, unicode_literals

from collections import OrderedDict
from types import FunctionType

import param

from . import util
from .pprint import PrettyPrinter


class Apply(object):
    """
    Utility to apply a function or operation to all viewable elements
    inside the object.
    """

    def __init__(self, obj, mode=None):
        self._obj = obj

    def __call__(self, function, streams=[], link_inputs=True, dynamic=None, **kwargs):
        """Applies a function to all (Nd)Overlay or Element objects.

        Any keyword arguments are passed through to the function. If
        keyword arguments are instance parameters, or streams are
        supplied the returned object will dynamically update in
        response to changes in those objects.

        Args:
            function: A callable function
                The function will be passed the return value of the
                DynamicMap as the first argument and any supplied
                stream values or keywords as additional keyword
                arguments.
            streams (list, optional): A list of Stream objects
                The Stream objects can dynamically supply values which
                will be passed to the function as keywords.
            link_inputs (bool, optional): Whether to link the inputs
                Determines whether Streams and Links attached to
                original object will be inherited.
            dynamic (bool, optional): Whether to make object dynamic
                By default object is made dynamic if streams are
                supplied, an instance parameter is supplied as a
                keyword argument, or the supplied function is a
                parameterized method.
            kwargs (dict, optional): Additional keyword arguments
                Keyword arguments which will be supplied to the
                function.

        Returns:
            A new object where the function was applied to all
            contained (Nd)Overlay or Element objects.
        """
        from .dimension import ViewableElement
        from .spaces import HoloMap, DynamicMap
        from ..util import Dynamic

        if isinstance(self._obj, DynamicMap) and dynamic == False:
            samples = tuple(d.values for d in self._obj.kdims)
            if not all(samples):
                raise ValueError('Applying a function to a DynamicMap '
                                 'and setting dynamic=False is only '
                                 'possible if key dimensions define '
                                 'a discrete parameter space.')
            return HoloMap(self._obj[samples]).apply(
                function, streams, link_inputs, dynamic, **kwargs)

        if isinstance(function, util.basestring):
            args = kwargs.pop('_method_args', ())
            method_name = function
            def function(object, **kwargs):
                method = getattr(object, method_name, None)
                if method is None:
                    raise AttributeError('Applied method %s does not exist.'
                                         'When declaring a method to apply '
                                         'as a string ensure a corresponding '
                                         'method exists on the object.' %
                                         method_name)
                return method(*args, **kwargs)

        applies = isinstance(self._obj, (ViewableElement, HoloMap))
        params = {p: val for p, val in kwargs.items()
                  if isinstance(val, param.Parameter)
                  and isinstance(val.owner, param.Parameterized)}

        dependent_kws = any(
            (isinstance(val, FunctionType) and hasattr(val, '_dinfo')) or
            util.is_param_method(val, has_deps=True) for val in kwargs.values()
        )

        if dynamic is None:
            dynamic = (bool(streams) or isinstance(self._obj, DynamicMap) or
                       util.is_param_method(function, has_deps=True) or
                       params or dependent_kws)

        if applies and dynamic:
            return Dynamic(self._obj, operation=function, streams=streams,
                           kwargs=kwargs, link_inputs=link_inputs)
        elif applies:
            inner_kwargs = util.resolve_dependent_kwargs(kwargs)
            if hasattr(function, 'dynamic'):
                inner_kwargs['dynamic'] = False
            return function(self._obj, **inner_kwargs)
        elif self._obj._deep_indexable:
            mapped = []
            for k, v in self._obj.data.items():
                new_val = v.apply(function, dynamic=dynamic, streams=streams,
                                  link_inputs=link_inputs, **kwargs)
                if new_val is not None:
                    mapped.append((k, new_val))
            return self._obj.clone(mapped, link=link_inputs)

        
    def aggregate(self, dimensions=None, function=None, spreadfn=None, **kwargs):
        """Applies a aggregate function to all ViewableElements.

        See :py:meth:`Dimensioned.opts` and :py:meth:`Apply.__call__`
        for more information.
        """
        kwargs['_method_args'] = (dimensions, function, spreadfn)
        return self.__call__('aggregate', **kwargs)

    def opts(self, *args, **kwargs):
        """Applies options to all ViewableElement objects.

        See :py:meth:`Dimensioned.opts` and :py:meth:`Apply.__call__`
        for more information.
        """
        kwargs['_method_args'] = args
        return self.__call__('opts', **kwargs)

    def reduce(self, dimensions=[], function=None, spreadfn=None, **kwargs):
        """Applies a reduce function to all ViewableElement objects.

        See :py:meth:`Dimensioned.opts` and :py:meth:`Apply.__call__`
        for more information.
        """
        kwargs['_method_args'] = (dimensions, function, spreadfn)
        return self.__call__('reduce', **kwargs)

    def select(self, **kwargs):
        """Applies a selection to all ViewableElement objects.

        See :py:meth:`Dimensioned.opts` and :py:meth:`Apply.__call__`
        for more information.
        """
        return self.__call__('select', **kwargs)



class Redim(object):
    """
    Utility that supports re-dimensioning any HoloViews object via the
    redim method.
    """

    def __init__(self, obj, mode=None):
        self._obj = obj
        # Can be 'dataset', 'dynamic' or None
        self.mode = mode

    def __str__(self):
        return "<holoviews.core.dimension.redim method>"

    @classmethod
    def replace_dimensions(cls, dimensions, overrides):
        """Replaces dimensions in list with dictionary of overrides.

        Args:
            dimensions: List of dimensions
            overrides: Dictionary of dimension specs indexed by name

        Returns:
            list: List of dimensions with replacements applied
        """
        from .dimension import Dimension
        
        replaced = []
        for d in dimensions:
            if d.name in overrides:
                override = overrides[d.name]
            elif d.label in overrides:
                override = overrides[d.label]
            else:
                override = None

            if override is None:
                replaced.append(d)
            elif isinstance(override, (util.basestring, tuple)):
                replaced.append(d.clone(override))
            elif isinstance(override, Dimension):
                replaced.append(override)
            elif isinstance(override, dict):
                replaced.append(d.clone(override.get('name',None),
                                        **{k:v for k,v in override.items() if k != 'name'}))
            else:
                raise ValueError('Dimension can only be overridden '
                                 'with another dimension or a dictionary '
                                 'of attributes')
        return replaced


    def _filter_cache(self, dmap, kdims):
        """
        Returns a filtered version of the DynamicMap cache leaving only
        keys consistently with the newly specified values
        """
        filtered = []
        for key, value in dmap.data.items():
            if not any(kd.values and v not in kd.values for kd, v in zip(kdims, key)):
                filtered.append((key, value))
        return filtered


    def __call__(self, specs=None, **dimensions):
        """
        Replace dimensions on the dataset and allows renaming
        dimensions in the dataset. Dimension mapping should map
        between the old dimension name and a dictionary of the new
        attributes, a completely new dimension or a new string name.
        """
        obj = self._obj
        redimmed = obj
        if obj._deep_indexable and self.mode != 'dataset':
            deep_mapped = [(k, v.redim(specs, **dimensions))
                           for k, v in obj.items()]
            redimmed = obj.clone(deep_mapped)

        if specs is not None:
            if not isinstance(specs, list):
                specs = [specs]
            matches = any(obj.matches(spec) for spec in specs)
            if self.mode != 'dynamic' and not matches:
                return redimmed

        kdims = self.replace_dimensions(obj.kdims, dimensions)
        vdims = self.replace_dimensions(obj.vdims, dimensions)
        zipped_dims = zip(obj.kdims+obj.vdims, kdims+vdims)
        renames = {pk.name: nk for pk, nk in zipped_dims if pk != nk}

        if self.mode == 'dataset':
            data = obj.data
            if renames:
                data = obj.interface.redim(obj, renames)
            clone = obj.clone(data, kdims=kdims, vdims=vdims)
            if self._obj.dimensions(label='name') == clone.dimensions(label='name'):
                # Ensure that plot_id is inherited as long as dimension
                # name does not change
                clone._plot_id = self._obj._plot_id
            return clone

        if self.mode != 'dynamic':
            return redimmed.clone(kdims=kdims, vdims=vdims)

        from ..util import Dynamic
        def dynamic_redim(obj, **dynkwargs):
            return obj.redim(specs, **dimensions)
        dmap = Dynamic(obj, streams=obj.streams, operation=dynamic_redim)
        dmap.data = OrderedDict(self._filter_cache(redimmed, kdims))
        with util.disable_constant(dmap):
            dmap.kdims = kdims
            dmap.vdims = vdims
        return dmap


    def _redim(self, name, specs, **dims):
        dimensions = {k:{name:v} for k,v in dims.items()}
        return self(specs, **dimensions)

    def cyclic(self, specs=None, **values):
        return self._redim('cyclic', specs, **values)

    def value_format(self, specs=None, **values):
        return self._redim('value_format', specs, **values)

    def range(self, specs=None, **values):
        return self._redim('range', specs, **values)

    def label(self, specs=None, **values):
        for k, v in values.items():
            dim = self._obj.get_dimension(k)
            if dim and dim.name != dim.label and dim.label != v:
                raise ValueError('Cannot override an existing Dimension label')
        return self._redim('label', specs, **values)

    def soft_range(self, specs=None, **values):
        return self._redim('soft_range', specs, **values)

    def type(self, specs=None, **values):
        return self._redim('type', specs, **values)

    def step(self, specs=None, **values):
        return self._redim('step', specs, **values)

    def default(self, specs=None, **values):
        return self._redim('default', specs, **values)

    def unit(self, specs=None, **values):
        return self._redim('unit', specs, **values)

    def values(self, specs=None, **ranges):
        return self._redim('values', specs, **ranges)



class Opts(object):

    def __init__(self, obj, mode=None):
        self._mode = mode
        self._obj = obj


    def get(self, group=None, backend=None):
        """Returns the corresponding Options object.

        Args:
            group: The options group. Flattens across groups if None.
            backend: Current backend if None otherwise chosen backend.

        Returns:
            Options object associated with the object containing the
            applied option keywords.
        """
        from .options import Store, Options
        keywords = {}
        groups = Options._option_groups if group is None else [group]
        backend = backend if backend else Store.current_backend
        for group in groups:
            optsobj = Store.lookup_options(backend, self._obj, group)
            keywords = dict(keywords, **optsobj.kwargs)
        return Options(**keywords)


    def __call__(self, *args, **kwargs):
        """Applies nested options definition.

        Applies options on an object or nested group of objects in a
        flat format. Unlike the .options method, .opts modifies the
        options in place by default. If the options are to be set
        directly on the object a simple format may be used, e.g.:

            obj.opts(cmap='viridis', show_title=False)

        If the object is nested the options must be qualified using
        a type[.group][.label] specification, e.g.:

            obj.opts('Image', cmap='viridis', show_title=False)

        or using:

            obj.opts({'Image': dict(cmap='viridis', show_title=False)})

        Args:
            *args: Sets of options to apply to object
                Supports a number of formats including lists of Options
                objects, a type[.group][.label] followed by a set of
                keyword options to apply and a dictionary indexed by
                type[.group][.label] specs.
            backend (optional): Backend to apply options to
                Defaults to current selected backend
            clone (bool, optional): Whether to clone object
                Options can be applied in place with clone=False
            **kwargs: Keywords of options
                Set of options to apply to the object

        For backwards compatibility, this method also supports the
        option group semantics now offered by the hv.opts.apply_groups
        utility. This usage will be deprecated and for more
        information see the apply_options_type docstring.

        Returns:
            Returns the object or a clone with the options applied
        """
        if self._mode is None:
            apply_groups, _, _ = util.deprecated_opts_signature(args, kwargs)
            if apply_groups and util.config.future_deprecations:
                msg = ("Calling the .opts method with options broken down by options "
                       "group (i.e. separate plot, style and norm groups) is deprecated. "
                       "Use the .options method converting to the simplified format "
                       "instead or use hv.opts.apply_groups for backward compatibility.")
                param.main.warning(msg)

        return self._dispatch_opts( *args, **kwargs)

    def _dispatch_opts(self, *args, **kwargs):
        if self._mode is None:
            return self._base_opts(*args, **kwargs)
        elif self._mode == 'holomap':
            return self._holomap_opts(*args, **kwargs)
        elif self._mode == 'dynamicmap':
            return self._dynamicmap_opts(*args, **kwargs)

    def clear(self, clone=False):
        """Clears any options applied to the object.

        Args:
            clone: Whether to return a cleared clone or clear inplace

        Returns:
            The object cleared of any options applied to it
        """
        return self._obj.opts(clone=clone)

    def info(self, show_defaults=False):
        """Prints a repr of the object including any applied options.

        Args:
            show_defaults: Whether to include default options
        """
        pprinter = PrettyPrinter(show_options=True, show_defaults=show_defaults)
        print(pprinter.pprint(self._obj))

    def _holomap_opts(self, *args, **kwargs):
        clone = kwargs.pop('clone', None)
        apply_groups, _, _ = util.deprecated_opts_signature(args, kwargs)
        data = OrderedDict([(k, v.opts(*args, **kwargs))
                             for k, v in self._obj.data.items()])

        # By default do not clone in .opts method
        if (apply_groups if clone is None else clone):
            return self._obj.clone(data)
        else:
            self._obj.data = data
            return self._obj

    def _dynamicmap_opts(self, *args, **kwargs):
        from ..util import Dynamic

        clone = kwargs.get('clone', None)
        apply_groups, _, _ = util.deprecated_opts_signature(args, kwargs)
        # By default do not clone in .opts method
        clone = (apply_groups if clone is None else clone)

        obj = self._obj if clone else self._obj.clone()
        dmap = Dynamic(obj, operation=lambda obj, **dynkwargs: obj.opts(*args, **kwargs),
                       streams=self._obj.streams, link_inputs=True)
        if not clone:
            with util.disable_constant(self._obj):
                obj.callback = self._obj.callback
                self._obj.callback = dmap.callback
            dmap = self._obj
            dmap.data = OrderedDict([(k, v.opts(*args, **kwargs))
                                     for k, v in self._obj.data.items()])
        return dmap


    def _base_opts(self, *args, **kwargs):
        apply_groups, options, new_kwargs = util.deprecated_opts_signature(args, kwargs)

        # By default do not clone in .opts method
        clone = kwargs.get('clone', None)
        if apply_groups:
            from ..util import opts
            if options is not None:
                kwargs['options'] = options
            return opts.apply_groups(self._obj, **dict(kwargs, **new_kwargs))

        kwargs['clone'] = False if clone is None else clone
        return self._obj.options(*args, **kwargs)
