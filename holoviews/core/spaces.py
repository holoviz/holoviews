import itertools
import types
from numbers import Number
from itertools import groupby
from functools import partial
from contextlib import contextmanager
from inspect import ArgSpec

import numpy as np
import param

from . import traversal, util
from .dimension import OrderedDict, Dimension, ViewableElement, redim
from .layout import Layout, AdjointLayout, NdLayout, Empty
from .ndmapping import UniformNdMapping, NdMapping, item_check
from .overlay import Overlay, CompositeOverlay, NdOverlay, Overlayable
from .options import Store, StoreOptions
from ..streams import Stream



class HoloMap(UniformNdMapping, Overlayable):
    """
    A HoloMap can hold any number of DataLayers indexed by a list of
    dimension values. It also has a number of properties, which can find
    the x- and y-dimension limits and labels.
    """

    data_type = (ViewableElement, NdMapping, Layout)

    def overlay(self, dimensions=None, **kwargs):
        """
        Splits the UniformNdMapping along a specified number of dimensions and
        overlays items in the split out Maps.

        Shows all HoloMap data When no dimensions are specified.
        """
        dimensions = self._valid_dimensions(dimensions)
        if len(dimensions) == self.ndims:
            with item_check(False):
                return NdOverlay(self, **kwargs).reindex(dimensions)
        else:
            dims = [d for d in self.kdims if d not in dimensions]
            return self.groupby(dims, group_type=NdOverlay, **kwargs)


    def grid(self, dimensions=None, **kwargs):
        """
        GridSpace takes a list of one or two dimensions, and lays out the containing
        Views along these axes in a GridSpace.

        Shows all HoloMap data When no dimensions are specified.
        """
        dimensions = self._valid_dimensions(dimensions)
        if len(dimensions) == self.ndims:
            with item_check(False):
                return GridSpace(self, **kwargs).reindex(dimensions)
        return self.groupby(dimensions, container_type=GridSpace, **kwargs)


    def layout(self, dimensions=None, **kwargs):
        """
        GridSpace takes a list of one or two dimensions, and lays out the containing
        Views along these axes in a GridSpace.

        Shows all HoloMap data When no dimensions are specified.
        """
        dimensions = self._valid_dimensions(dimensions)
        if len(dimensions) == self.ndims:
            with item_check(False):
                return NdLayout(self, **kwargs).reindex(dimensions)
        return self.groupby(dimensions, container_type=NdLayout, **kwargs)


    def split_overlays(self):
        """
        Given a UniformNdMapping of Overlays of N layers, split out the layers into
        N separate Maps.
        """
        if not issubclass(self.type, CompositeOverlay):
            return None, self.clone()

        item_maps = OrderedDict()
        for k, overlay in self.data.items():
            for key, el in overlay.items():
                if key not in item_maps:
                    item_maps[key] = [(k, el)]
                else:
                    item_maps[key].append((k, el))

        maps, keys = [], []
        for k, layermap in item_maps.items():
            maps.append(self.clone(layermap))
            keys.append(k)
        return keys, maps


    def _dimension_keys(self):
        """
        Helper for __mul__ that returns the list of keys together with
        the dimension labels.
        """
        return [tuple(zip([d.name for d in self.kdims], [k] if self.ndims == 1 else k))
                for k in self.keys()]


    def _dynamic_mul(self, dimensions, other, keys):
        """
        Implements dynamic version of overlaying operation overlaying
        DynamicMaps and HoloMaps where the key dimensions of one is
        a strict superset of the other.
        """
        # If either is a HoloMap compute Dimension values
        if not isinstance(self, DynamicMap) or not isinstance(other, DynamicMap):
            keys = sorted((d, v) for k in keys for d, v in k)
            grouped =  dict([(g, [v for _, v in group])
                             for g, group in groupby(keys, lambda x: x[0])])
            dimensions = [d(values=grouped[d.name]) for d in dimensions]
            map_obj = None

        # Combine streams
        map_obj = self if isinstance(self, DynamicMap) else other
        if isinstance(self, DynamicMap) and isinstance(other, DynamicMap):
            self_streams = util.dimensioned_streams(self)
            other_streams = util.dimensioned_streams(other)
            streams = list(util.unique_iterator(self_streams+other_streams))
        else:
            streams = map_obj.streams

        def dynamic_mul(*key, **kwargs):
            key_map = {d.name: k for d, k in zip(dimensions, key)}
            layers = []
            try:
                self_el = self.select(HoloMap, **key_map) if self.kdims else self[()]
                layers.append(self_el)
            except KeyError:
                pass
            try:
                other_el = other.select(HoloMap, **key_map) if other.kdims else other[()]
                layers.append(other_el)
            except KeyError:
                pass
            return Overlay(layers)
        callback = Callable(dynamic_mul, inputs=[self, other])
        callback._is_overlay = True
        if map_obj:
            return map_obj.clone(callback=callback, shared_data=False,
                                 kdims=dimensions, streams=streams)
        else:
            return DynamicMap(callback=callback, kdims=dimensions,
                              streams=streams)


    def __mul__(self, other):
        """
        The mul (*) operator implements overlaying of different Views.
        This method tries to intelligently overlay Maps with differing
        keys. If the UniformNdMapping is mulled with a simple
        ViewableElement each element in the UniformNdMapping is
        overlaid with the ViewableElement. If the element the
        UniformNdMapping is mulled with is another UniformNdMapping it
        will try to match up the dimensions, making sure that items
        with completely different dimensions aren't overlaid.
        """
        if isinstance(other, HoloMap):
            self_set = {d.name for d in self.kdims}
            other_set = {d.name for d in other.kdims}

            # Determine which is the subset, to generate list of keys and
            # dimension labels for the new view
            self_in_other = self_set.issubset(other_set)
            other_in_self = other_set.issubset(self_set)
            dims = [other.kdims, self.kdims] if self_in_other else [self.kdims, other.kdims]
            dimensions = util.merge_dimensions(dims)

            if self_in_other and other_in_self: # superset of each other
                keys = self._dimension_keys() + other._dimension_keys()
                super_keys = util.unique_iterator(keys)
            elif self_in_other: # self is superset
                dimensions = other.kdims
                super_keys = other._dimension_keys()
            elif other_in_self: # self is superset
                super_keys = self._dimension_keys()
            else: # neither is superset
                raise Exception('One set of keys needs to be a strict subset of the other.')

            if isinstance(self, DynamicMap) or isinstance(other, DynamicMap):
                return self._dynamic_mul(dimensions, other, super_keys)

            items = []
            for dim_keys in super_keys:
                # Generate keys for both subset and superset and sort them by the dimension index.
                self_key = tuple(k for p, k in sorted(
                    [(self.get_dimension_index(dim), v) for dim, v in dim_keys
                     if dim in self.kdims]))
                other_key = tuple(k for p, k in sorted(
                    [(other.get_dimension_index(dim), v) for dim, v in dim_keys
                     if dim in other.kdims]))
                new_key = self_key if other_in_self else other_key
                # Append SheetOverlay of combined items
                if (self_key in self) and (other_key in other):
                    items.append((new_key, self[self_key] * other[other_key]))
                elif self_key in self:
                    items.append((new_key, Overlay([self[self_key]])))
                else:
                    items.append((new_key, Overlay([other[other_key]])))
            return self.clone(items, kdims=dimensions, label=self._label, group=self._group)
        elif isinstance(other, self.data_type):
            if isinstance(self, DynamicMap):
                def dynamic_mul(*args, **kwargs):
                    element = self[args]
                    return element * other
                callback = Callable(dynamic_mul, inputs=[self, other])
                callback._is_overlay = True
                return self.clone(shared_data=False, callback=callback,
                                  streams=[])
            items = [(k, v * other) for (k, v) in self.data.items()]
            return self.clone(items, label=self._label, group=self._group)
        else:
            return NotImplemented


    def __add__(self, obj):
        return Layout.from_values([self, obj])


    def __lshift__(self, other):
        if isinstance(other, (ViewableElement, UniformNdMapping, Empty)):
            return AdjointLayout([self, other])
        elif isinstance(other, AdjointLayout):
            return AdjointLayout(other.data+[self])
        else:
            raise TypeError('Cannot append {0} to a AdjointLayout'.format(type(other).__name__))


    def collate(self, merge_type=None, drop=[], drop_constant=False):
        """
        Collation allows collapsing nested HoloMaps by merging
        their dimensions. In the simple case a HoloMap containing
        other HoloMaps can easily be joined in this way. However
        collation is particularly useful when the objects being
        joined are deeply nested, e.g. you want to join multiple
        Layouts recorded at different times, collation will return
        one Layout containing HoloMaps indexed by Time. Changing
        the merge_type will allow merging the outer Dimension
        into any other UniformNdMapping type.

        Specific dimensions may be dropped if they are redundant
        by supplying them in a list. Enabling drop_constant allows
        ignoring any non-varying dimensions during collation.
        """
        from .element import Collator
        merge_type=merge_type if merge_type else self.__class__
        return Collator(self, merge_type=merge_type, drop=drop,
                        drop_constant=drop_constant)()


    def collapse(self, dimensions=None, function=None, spreadfn=None, **kwargs):
        """
        Allows collapsing one of any number of key dimensions
        on the HoloMap. Homogeneous Elements may be collapsed by
        supplying a function, inhomogeneous elements are merged.
        """
        if not dimensions:
            dimensions = self.kdims
        if not isinstance(dimensions, list): dimensions = [dimensions]
        if self.ndims > 1 and len(dimensions) != self.ndims:
            groups = self.groupby([dim for dim in self.kdims
                                   if dim not in dimensions])
        elif all(d in self.kdims for d in dimensions):
            groups = HoloMap([(0, self)])
        else:
            raise KeyError("Supplied dimensions not found.")

        collapsed = groups.clone(shared_data=False)
        for key, group in groups.items():
            group_data = [el.data for el in group]
            args = (group_data, function, group.last.kdims)
            if hasattr(group.last, 'interface'):
                col_data = group.type(group.table().aggregate(group.last.kdims, function, spreadfn, **kwargs))

            else:
                data = group.type.collapse_data(*args, **kwargs)
                col_data = group.last.clone(data)
            collapsed[key] = col_data
        return collapsed if self.ndims > 1 else collapsed.last


    def sample(self, samples=[], bounds=None, **sample_values):
        """
        Sample each Element in the UniformNdMapping by passing either a list of
        samples or a tuple specifying the number of regularly spaced
        samples per dimension. Alternatively, a single sample may be
        requested using dimension-value pairs. Optionally, the bounds
        argument can be used to specify the bounding extent from which
        the coordinates are to regularly sampled. Regular sampling
        assumes homogeneous and regularly sampled data.

        For 1D sampling, the shape is simply as the desired number of
        samples (and not a tuple). The bounds format for 1D sampling
        is the tuple (lower, upper) and the tuple (left, bottom,
        right, top) for 2D sampling.
        """
        dims = self.last.ndims
        if isinstance(samples, tuple) or np.isscalar(samples):
            if dims == 1:
                xlim = self.last.range(0)
                lower, upper = (xlim[0], xlim[1]) if bounds is None else bounds
                edges = np.linspace(lower, upper, samples+1)
                linsamples = [(l+u)/2.0 for l,u in zip(edges[:-1], edges[1:])]
            elif dims == 2:
                (rows, cols) = samples
                if bounds:
                    (l,b,r,t) = bounds
                else:
                    l, r = self.last.range(0)
                    b, t = self.last.range(1)

                xedges = np.linspace(l, r, cols+1)
                yedges = np.linspace(b, t, rows+1)
                xsamples = [(lx+ux)/2.0 for lx,ux in zip(xedges[:-1], xedges[1:])]
                ysamples = [(ly+uy)/2.0 for ly,uy in zip(yedges[:-1], yedges[1:])]

                Y,X = np.meshgrid(ysamples, xsamples)
                linsamples = list(zip(X.flat, Y.flat))
            else:
                raise NotImplementedError("Regular sampling not implemented "
                                          "for high-dimensional Views.")

            samples = list(util.unique_iterator(self.last.closest(linsamples)))

        sampled = self.clone([(k, view.sample(samples, closest=False,
                                              **sample_values))
                              for k, view in self.data.items()])
        return sampled.table()


    def reduce(self, dimensions=None, function=None, **reduce_map):
        """
        Reduce each Element in the HoloMap using a function supplied
        via the kwargs, where the keyword has to match a particular
        dimension in the Elements.
        """
        from ..element import Table
        reduced_items = [(k, v.reduce(dimensions, function, **reduce_map))
                         for k, v in self.items()]
        if not isinstance(reduced_items[0][1], Table):
            params = dict(util.get_param_values(self.last),
                          kdims=self.kdims, vdims=self.last.vdims)
            return Table(reduced_items, **params)
        return self.clone(reduced_items).table()


    def relabel(self, label=None, group=None, depth=1):
        # Identical to standard relabel method except for default depth of 1
        return super(HoloMap, self).relabel(label=label, group=group, depth=depth)


    def hist(self, num_bins=20, bin_range=None, adjoin=True, individually=True, **kwargs):
        histmaps = [self.clone(shared_data=False) for _ in
                    kwargs.get('dimension', range(1))]

        if individually:
            map_range = None
        else:
            if 'dimension' not in kwargs:
                raise Exception("Please supply the dimension to compute a histogram for.")
            map_range = self.range(kwargs['dimension'])
        bin_range = map_range if bin_range is None else bin_range
        style_prefix = 'Custom[<' + self.name + '>]_'
        if issubclass(self.type, (NdOverlay, Overlay)) and 'index' not in kwargs:
            kwargs['index'] = 0
        for k, v in self.data.items():
            hists = v.hist(adjoin=False, bin_range=bin_range,
                           individually=individually, num_bins=num_bins,
                           style_prefix=style_prefix, **kwargs)
            if isinstance(hists, Layout):
                for i, hist in enumerate(hists):
                    histmaps[i][k] = hist
            else:
                histmaps[0][k] = hists

        if adjoin:
            layout = self
            for hist in histmaps:
                layout = (layout << hist)
            if issubclass(self.type, (NdOverlay, Overlay)):
                layout.main_layer = kwargs['index']
            return layout
        else:
            if len(histmaps) > 1:
                return Layout.from_values(histmaps)
            else:
                return histmaps[0]


class Callable(param.Parameterized):
    """
    Callable allows wrapping callbacks on one or more DynamicMaps
    allowing their inputs (and in future outputs) to be defined.
    This makes it possible to wrap DynamicMaps with streams and
    makes it possible to traverse the graph of operations applied
    to a DynamicMap.

    Additionally, if the memoize attribute is True, a Callable will
    memoize the last returned value based on the arguments to the
    function and the state of all streams on its inputs, to avoid
    calling the function unnecessarily. Note that because memoization
    includes the streams found on the inputs it may be disabled if the
    stream requires it and is triggering.

    A Callable may also specify a stream_mapping which specifies the
    objects that are associated with interactive (i.e linked) streams
    when composite objects such as Layouts are returned from the
    callback. This is required for building interactive, linked
    visualizations (for the backends that support them) when returning
    Layouts, NdLayouts or GridSpace objects. When chaining multiple
    DynamicMaps into a pipeline, the link_inputs parameter declares
    whether the visualization generated using this Callable will
    inherit the linked streams. This parameter is used as a hint by
    the applicable backend.

    The mapping should map from an appropriate key to a list of
    streams associated with the selected object. The appropriate key
    may be a type[.group][.label] specification for Layouts, an
    integer index or a suitable NdLayout/GridSpace key. For more
    information see the DynamicMap tutorial at holoviews.org.
    """

    callable = param.Callable(default=None, constant=True, doc="""
         The callable function being wrapped.""")

    inputs = param.List(default=[], constant=True, doc="""
         The list of inputs the callable function is wrapping. Used
         to allow deep access to streams in chained Callables.""")

    link_inputs = param.Boolean(default=True, doc="""
         If the Callable wraps around other DynamicMaps in its inputs,
         determines whether linked streams attached to the inputs are
         transferred to the objects returned by the Callable.

         For example the Callable wraps a DynamicMap with an RangeXY
         stream, this switch determines whether the corresponding
         visualization should update this stream with range changes
         originating from the newly generated axes.""")

    memoize = param.Boolean(default=True, doc="""
         Whether the return value of the callable should be memoized
         based on the call arguments and any streams attached to the
         inputs.""")

    stream_mapping = param.Dict(default={}, constant=True, doc="""
         Defines how streams should be mapped to objects returned by
         the Callable, e.g. when it returns a Layout.""")

    def __init__(self, callable, **params):
        super(Callable, self).__init__(callable=callable,
                                       **dict(params, name=util.callable_name(callable)))
        self._memoized = {}
        self._is_overlay = False
        self.args = None
        self.kwargs = None
        self._stream_memoization = self.memoize

    @property
    def argspec(self):
        return util.argspec(self.callable)

    @property
    def noargs(self):
        "Returns True if the callable takes no arguments"
        noargs = ArgSpec(args=[], varargs=None, keywords=None, defaults=None)
        return self.argspec == noargs


    def clone(self, callable=None, **overrides):
        """
        Allows making a copy of the Callable optionally overriding
        the callable and other parameters.
        """
        old = {k: v for k, v in self.get_param_values()
               if k not in ['callable', 'name']}
        params = dict(old, **overrides)
        callable = self.callable if callable is None else callable
        return self.__class__(callable, **params)


    def __call__(self, *args, **kwargs):
        # Nothing to do for callbacks that accept no arguments
        kwarg_hash = kwargs.pop('memoization_hash', ())
        (self.args, self.kwargs) = (args, kwargs)
        if not args and not kwargs: return self.callable()
        inputs = [i for i in self.inputs if isinstance(i, DynamicMap)]
        streams = []
        for stream in [s for i in inputs for s in get_nested_streams(i)]:
            if stream not in streams: streams.append(stream)

        memoize = self._stream_memoization and not any(s.transient and s._triggering for s in streams)
        values = tuple(tuple(sorted(s.hashkey.items())) for s in streams)
        key = args + kwarg_hash + values

        hashed_key = util.deephash(key) if self.memoize else None
        if hashed_key is not None and memoize and hashed_key in self._memoized:
            return self._memoized[hashed_key]

        if self.argspec.varargs is not None:
            # Missing information on positional argument names, cannot promote to keywords
            pass
        elif len(args) != 0: # Turn positional arguments into keyword arguments
            pos_kwargs = {k:v for k,v in zip(self.argspec.args, args)}
            ignored = range(len(self.argspec.args),len(args))
            if len(ignored):
                self.warning('Ignoring extra positional argument %s'
                             % ', '.join('%s' % i for i in ignored))
            clashes = set(pos_kwargs.keys()) & set(kwargs.keys())
            if clashes:
                self.warning('Positional arguments %r overriden by keywords'
                             % list(clashes))
            args, kwargs = (), dict(pos_kwargs, **kwargs)

        try:
            ret = self.callable(*args, **kwargs)
        except KeyError:
            # KeyError is caught separately because it is used to signal
            # invalid keys on DynamicMap and should not warn
            raise
        except:
            posstr = ', '.join(['%r' % el for el in self.args]) if self.args else ''
            kwstr = ', '.join('%s=%r' % (k,v) for k,v in self.kwargs.items())
            argstr = ', '.join([el for el in [posstr, kwstr] if el])
            message = ("Exception raised in callable '{name}' of type '{ctype}'.\n"
                       "Invoked as {name}({argstr})")
            self.warning(message.format(name=self.name,
                                        ctype = type(self.callable).__name__,
                                        argstr=argstr))
            raise

        if hashed_key is not None:
            self._memoized = {hashed_key : ret}
        return ret



class Generator(Callable):
    """
    Generators are considered a special case of Callable that accept no
    arguments and never memoize.
    """

    callable = param.ClassSelector(default=None, class_ = types.GeneratorType,
                                   constant=True, doc="""
         The generator that is wrapped by this Generator.""")

    @property
    def argspec(self):
        return ArgSpec(args=[], varargs=None, keywords=None, defaults=None)

    def __call__(self):
        try:
            return next(self.callable)
        except StopIteration:
            raise
        except Exception:
            msg = 'Generator {name} raised the following exception:'
            self.warning(msg.format(name=self.name))
            raise


def get_nested_dmaps(dmap):
    """
    Get all DynamicMaps referenced by the supplied DynamicMap's callback.
    """
    if not isinstance(dmap, DynamicMap):
        return []
    dmaps = [dmap]
    for o in dmap.callback.inputs:
        dmaps.extend(get_nested_dmaps(o))
    return list(set(dmaps))


def get_nested_streams(dmap):
    """
    Get all (potentially nested) streams from DynamicMap with Callable
    callback.
    """
    return list({s for dmap in get_nested_dmaps(dmap) for s in dmap.streams})


@contextmanager
def dynamicmap_memoization(callable_obj, streams):
    """
    Determine whether the Callable should have memoization enabled
    based on the supplied streams (typically by a
    DynamicMap). Memoization is disabled if any of the streams require
    it it and are currently in a triggered state.
    """
    memoization_state = bool(callable_obj._stream_memoization)
    callable_obj._stream_memoization &= not any(s.transient and s._triggering for s in streams)
    try:
        yield
    except:
        raise
    finally:
        callable_obj._stream_memoization = memoization_state



class periodic(object):
    """
    Implements the utility of the same name on DynamicMap.

    Used to defined periodic event updates that can be started and
    stopped.
    """
    _periodic_util = util.periodic

    def __init__(self, dmap):
        self.dmap = dmap
        self.instance = None

    def __call__(self, period, count=None, param_fn=None, timeout=None, block=True):
        """
        Run a non-blocking loop that updates the stream parameters using
        the event method. Runs count times with the specified period. If
        count is None, runs indefinitely.

        If param_fn is not specified, the event method is called without
        arguments. If it is specified, it must be a callable accepting a
        single argument (the iteration count, starting at 1) that
        returns a dictionary of the new stream values to be passed to
        the event method.
        """

        if self.instance is not None and not self.instance.completed:
            raise RuntimeError('Periodic process already running. '
                               'Wait until it completes or call '
                               'stop() before running a new periodic process')
        def inner(i):
            kwargs = {} if param_fn is None else param_fn(i)
            self.dmap.event(**kwargs)

        instance = self._periodic_util(period, count, inner,
                                       timeout=timeout, block=block)
        instance.start()
        self.instance= instance

    def stop(self):
        "Stop the periodic process."
        self.instance.stop()

    def __str__(self):
        return "<holoviews.core.spaces.periodic method>"



class DynamicMap(HoloMap):
    """
    A DynamicMap is a type of HoloMap where the elements are dynamically
    generated by a callable. The callable is invoked with values
    associated with the key dimensions or with values supplied by stream
    parameters.
    """

    # Declare that callback is a positional parameter (used in clone)
    __pos_params = ['callback']

    kdims = param.List(default=[], constant=True, doc="""
        The key dimensions of a DynamicMap map to the arguments of the
        callback. This mapping can be by position or by name.""")

    callback = param.ClassSelector(class_=Callable, constant=True, doc="""
        The callable used to generate the elements. The arguments to the
        callable includes any number of declared key dimensions as well
        as any number of stream parameters defined on the input streams.

        If the callable is an instance of Callable it will be used
        directly, otherwise it will be automatically wrapped in one.""")

    streams = param.List(default=[], constant=True, doc="""
       List of Stream instances to associate with the DynamicMap. The
       set of parameter values across these streams will be supplied as
       keyword arguments to the callback when the events are received,
       updating the streams.""" )

    cache_size = param.Integer(default=500, doc="""
       The number of entries to cache for fast access. This is an LRU
       cache where the least recently used item is overwritten once
       the cache is full.""")

    def __init__(self, callback, initial_items=None, **params):

        if isinstance(callback, types.GeneratorType):
            callback = Generator(callback)
        elif not isinstance(callback, Callable):
            callback = Callable(callback)

        if 'sampled' in params:
            self.warning('DynamicMap sampled parameter is deprecated '
                         'and no longer needs to be specified.')
            del params['sampled']

        super(DynamicMap, self).__init__(initial_items, callback=callback, **params)
        invalid = [s for s in self.streams if not isinstance(s, Stream)]
        if invalid:
            msg = ('The supplied streams list contains objects that '
                   'are not Stream instances: {objs}')
            raise TypeError(msg.format(objs = ', '.join('%r' % el for el in invalid)))


        if self.callback.noargs:
            prefix = 'DynamicMaps using generators (or callables without arguments)'
            if self.kdims:
                raise Exception(prefix + ' must be declared without key dimensions')
            if len(self.streams)> 1:
                raise Exception(prefix + ' must have either streams=[] or a single, '
                                + 'stream instance without any stream parameters')
            if util.stream_parameters(self.streams) != []:
                raise Exception(prefix + ' cannot accept any stream parameters')

        self._posarg_keys = util.validate_dynamic_argspec(self.callback,
                                                          self.kdims,
                                                          self.streams)
        # Set source to self if not already specified
        for stream in self.streams:
            if stream.source is None:
                stream.source = self
        self.redim = redim(self, mode='dynamic')
        self.periodic = periodic(self)

    @property
    def unbounded(self):
        """
        Returns a list of key dimensions that are unbounded, excluding
        stream parameters. If any of theses key dimensions are
        unbounded, the DynamicMap as a whole is also unbounded.
        """
        unbounded_dims = []
        # Dimensioned streams do not need to be bounded
        stream_params = set(util.stream_parameters(self.streams))
        for kdim in self.kdims:
            if str(kdim) in stream_params:
                continue
            if kdim.values:
                continue
            if None in kdim.range:
                unbounded_dims.append(str(kdim))
        return unbounded_dims

    def _initial_key(self):
        """
        Construct an initial key for based on the lower range bounds or
        values on the key dimensions.
        """
        key = []
        undefined = []
        stream_params = set(util.stream_parameters(self.streams))
        for kdim in self.kdims:
            if str(kdim) in stream_params:
                key.append(None)
            elif kdim.values:
                key.append(kdim.values[0])
            elif kdim.range[0] is not None:
                key.append(kdim.range[0])
            else:
                undefined.append(kdim)
        if undefined:
            msg = ('Dimension(s) {undefined_dims} do not specify range or values needed '
                   'to generate initial key')
            undefined_dims = ', '.join(['%r' % str(dim) for dim in undefined])
            raise KeyError(msg.format(undefined_dims=undefined_dims))

        return tuple(key)


    def _validate_key(self, key):
        """
        Make sure the supplied key values are within the bounds
        specified by the corresponding dimension range and soft_range.
        """
        if key == () and len(self.kdims) == 0: return ()
        key = util.wrap_tuple(key)
        assert len(key) == len(self.kdims)
        for ind, val in enumerate(key):
            kdim = self.kdims[ind]
            low, high = util.max_range([kdim.range, kdim.soft_range])
            if low is not np.NaN:
                if val < low:
                    raise KeyError("Key value %s below lower bound %s"
                                   % (val, low))
            if high is not np.NaN:
                if val > high:
                    raise KeyError("Key value %s above upper bound %s"
                                   % (val, high))

    def event(self, **kwargs):
        """
        This method allows any of the available stream parameters
        (renamed as appropriate) to be updated in an event.
        """
        if self.callback.noargs and self.streams == []:
            self.warning('No streams declared. To update a DynamicMaps using '
                         'generators (or callables without arguments) use streams=[Next()]')
            return
        if self.streams == []:
            self.warning('No streams on DynamicMap, calling event will have no effect')
            return

        stream_params = set(util.stream_parameters(self.streams))
        invalid = [k for k in kwargs.keys() if k not in stream_params]
        if invalid:
            msg = 'Key(s) {invalid} do not correspond to stream parameters'
            raise KeyError(msg.format(invalid = ', '.join('%r' % i for i in invalid)))

        for stream in self.streams:
            applicable_kws = {k:v for k,v in kwargs.items()
                              if k in set(stream.contents.keys())}
            rkwargs = util.rename_stream_kwargs(stream, applicable_kws, reverse=True)
            stream.update(**rkwargs)

        Stream.trigger(self.streams)

    def _style(self, retval):
        """
        Use any applicable OptionTree of the DynamicMap to apply options
        to the return values of the callback.
        """
        if self.id not in Store.custom_options():
            return retval
        spec = StoreOptions.tree_to_dict(Store.custom_options()[self.id])
        return retval.opts(spec)


    def _execute_callback(self, *args):
        """
        Execute the callback, validating both the input key and output
        key where applicable.
        """
        self._validate_key(args)      # Validate input key

        # Additional validation needed to ensure kwargs don't clash
        kdims = [kdim.name for kdim in self.kdims]
        kwarg_items = [s.contents.items() for s in self.streams]
        hash_items = tuple(tuple(sorted(s.hashkey.items())) for s in self.streams)+args
        flattened = [(k,v) for kws in kwarg_items for (k,v) in kws
                     if k not in kdims]

        if self._posarg_keys:
            kwargs = dict(flattened, **dict(zip(self._posarg_keys, args)))
            args = ()
        else:
            kwargs = dict(flattened)
        if not isinstance(self.callback, Generator):
            kwargs['memoization_hash'] = hash_items

        with dynamicmap_memoization(self.callback, self.streams):
            retval = self.callback(*args, **kwargs)
        return self._style(retval)


    def opts(self, options=None, backend=None, **kwargs):
        """
        Applies options on an object or nested group of objects in a
        by options group returning a new object with the options
        applied. If the options are to be set directly on the object a
        simple format may be used, e.g.:

            obj.opts(style={'cmap': 'viridis'}, plot={'show_title': False})

        If the object is nested the options must be qualified using
        a type[.group][.label] specification, e.g.:

            obj.opts({'Image': {'plot':  {'show_title': False},
                                'style': {'cmap': 'viridis}}})

        If no opts are supplied all options on the object will be reset.
        """
        from ..util import Dynamic
        dmap = Dynamic(self, operation=lambda obj, **dynkwargs: obj.opts(options, backend, **kwargs),
                       streams=self.streams, link_inputs=True)
        dmap.data = OrderedDict([(k, v.opts(options, **kwargs))
                                 for k, v in self.data.items()])
        return dmap


    def options(self, options=None, backend=None, **kwargs):
        """
        Applies options on an object or nested group of objects in a
        flat format returning a new object with the options
        applied. If the options are to be set directly on the object a
        simple format may be used, e.g.:

            obj.options(cmap='viridis', show_title=False)

        If the object is nested the options must be qualified using
        a type[.group][.label] specification, e.g.:

            obj.options('Image', cmap='viridis', show_title=False)

        or using:

            obj.options({'Image': dict(cmap='viridis', show_title=False)})

        If no options are supplied all options on the object will be reset.
        """
        from ..util import Dynamic
        dmap = Dynamic(self, operation=lambda obj, **dynkwargs: obj.options(options, backend, **kwargs),
                       streams=self.streams, link_inputs=True)
        dmap.data = OrderedDict([(k, v.options(options, backend, **kwargs))
                                 for k, v in self.data.items()])
        return dmap


    def clone(self, data=None, shared_data=True, new_type=None, link_inputs=True,
              *args, **overrides):
        """
        Clone method to adapt the slightly different signature of
        DynamicMap that also overrides Dimensioned clone to avoid
        checking items if data is unchanged.
        """
        if data is None and shared_data:
            data = self.data
            overrides['plot_id'] = self._plot_id
        clone = super(UniformNdMapping, self).clone(overrides.pop('callback', self.callback),
                                                    shared_data, new_type,
                                                    *(data,) + args, **overrides)

        # Ensure the clone references this object to ensure
        # stream sources are inherited
        if clone.callback is self.callback:
            with util.disable_constant(clone):
                clone.callback = clone.callback.clone(inputs=[self],
                                                      link_inputs=link_inputs)
        return clone


    def reset(self):
        """
        Return a cleared dynamic map with a cleared cached
        """
        self.data = OrderedDict()
        return self


    def _cross_product(self, tuple_key, cache, data_slice):
        """
        Returns a new DynamicMap if the key (tuple form) expresses a
        cross product, otherwise returns None. The cache argument is a
        dictionary (key:element pairs) of all the data found in the
        cache for this key.

        Each key inside the cross product is looked up in the cache
        (self.data) to check if the appropriate element is
        available. Otherwise the element is computed accordingly.

        The data_slice may specify slices into each value in the
        the cross-product.
        """
        if not any(isinstance(el, (list, set)) for el in tuple_key):
            return None
        if len(tuple_key)==1:
            product = tuple_key[0]
        else:
            args = [set(el) if isinstance(el, (list,set))
                    else set([el]) for el in tuple_key]
            product = itertools.product(*args)

        data = []
        for inner_key in product:
            key = util.wrap_tuple(inner_key)
            if key in cache:
                val = cache[key]
            else:
                val = self._execute_callback(*key)
            if data_slice:
                val = self._dataslice(val, data_slice)
            data.append((key, val))
        product = self.clone(data)

        if data_slice:
            from ..util import Dynamic
            dmap = Dynamic(self, operation=lambda obj, **dynkwargs: obj[data_slice],
                           streams=self.streams)
            dmap.data = product.data
            return dmap
        return product


    def _slice_bounded(self, tuple_key, data_slice):
        """
        Slices bounded DynamicMaps by setting the soft_ranges on
        key dimensions and applies data slice to cached and dynamic
        values.
        """
        slices = [el for el in tuple_key if isinstance(el, slice)]
        if any(el.step for el in slices):
            raise Exception("DynamicMap slices cannot have a step argument")
        elif len(slices) not in [0, len(tuple_key)]:
            raise Exception("Slices must be used exclusively or not at all")
        elif not slices:
            return None

        sliced = self.clone(self)
        for i, slc in enumerate(tuple_key):
            (start, stop) = slc.start, slc.stop
            if start is not None and start < sliced.kdims[i].range[0]:
                raise Exception("Requested slice below defined dimension range.")
            if stop is not None and stop > sliced.kdims[i].range[1]:
                raise Exception("Requested slice above defined dimension range.")
            sliced.kdims[i].soft_range = (start, stop)
        if data_slice:
            if not isinstance(sliced, DynamicMap):
                return self._dataslice(sliced, data_slice)
            else:
                from ..util import Dynamic
                if len(self):
                    slices = [slice(None) for _ in range(self.ndims)] + list(data_slice)
                    sliced = super(DynamicMap, sliced).__getitem__(tuple(slices))
                dmap = Dynamic(self, operation=lambda obj, **dynkwargs: obj[data_slice],
                               streams=self.streams)
                dmap.data = sliced.data
                return dmap
        return sliced


    def __getitem__(self, key):
        """
        Return an element for any key chosen key. Also allows for usual
        deep slicing semantics by slicing values in the cache and
        applying the deep slice to newly generated values.
        """
        # Split key dimensions and data slices
        sample = False
        if key is Ellipsis:
            return self
        elif isinstance(key, (list, set)) and all(isinstance(v, tuple) for v in key):
            map_slice, data_slice = key, ()
            sample = True
        else:
            map_slice, data_slice = self._split_index(key)
        tuple_key = util.wrap_tuple_streams(map_slice, self.kdims, self.streams)

        # Validation
        if not sample:
            sliced = self._slice_bounded(tuple_key, data_slice)
            if sliced is not None:
                return sliced

        # Cache lookup
        try:
            dimensionless = util.dimensionless_contents(get_nested_streams(self),
                                                        self.kdims, no_duplicates=False)
            empty = util.stream_parameters(self.streams) == [] and self.kdims==[]
            if dimensionless or empty:
                raise KeyError('Using dimensionless streams disables DynamicMap cache')
            cache = super(DynamicMap,self).__getitem__(key)
        except KeyError:
            cache = None

        # If the key expresses a cross product, compute the elements and return
        product = self._cross_product(tuple_key, cache.data if cache else {}, data_slice)
        if product is not None:
            return product

        # Not a cross product and nothing cached so compute element.
        if cache is not None: return cache
        val = self._execute_callback(*tuple_key)
        if data_slice:
            val = self._dataslice(val, data_slice)
        self._cache(tuple_key, val)
        return val


    def select(self, selection_specs=None, **kwargs):
        """
        Allows slicing or indexing into the DynamicMap objects by
        supplying the dimension and index/slice as key value
        pairs. Select descends recursively through the data structure
        applying the key dimension selection and applies to dynamically
        generated items by wrapping the callback.

        The selection may also be selectively applied to specific
        objects by supplying the selection_specs as an iterable of
        type.group.label specs, types or functions.
        """
        if selection_specs is not None and not isinstance(selection_specs, (list, tuple)):
            selection_specs = [selection_specs]
        selection = super(DynamicMap, self).select(selection_specs, **kwargs)
        def dynamic_select(obj, **dynkwargs):
            if selection_specs is not None:
                matches = any(obj.matches(spec) for spec in selection_specs)
            else:
                matches = True
            if matches:
                return obj.select(**kwargs)
            return obj

        if not isinstance(selection, DynamicMap):
            return dynamic_select(selection)
        else:
            from ..util import Dynamic
            dmap = Dynamic(self, operation=dynamic_select, streams=self.streams)
            dmap.data = selection.data
            return dmap
            


    def _cache(self, key, val):
        """
        Request that a key/value pair be considered for caching.
        """
        cache_size = (1 if util.dimensionless_contents(self.streams, self.kdims)
                      else self.cache_size)
        if len(self) >= cache_size:
            first_key = next(k for k in self.data)
            self.data.pop(first_key)
        self[key] = val


    def map(self, map_fn, specs=None, clone=True, link_inputs=True):
        """
        Recursively replaces elements using a map function when the
        specification applies. Extends regular map with functionality
        to dynamically apply functions. By default all streams are
        still linked to the mapped object, to disable linked streams
        set linked_inputs=False.
        """
        deep_mapped = super(DynamicMap, self).map(map_fn, specs, clone)
        if isinstance(deep_mapped, type(self)):
            from ..util import Dynamic
            def apply_map(obj, **dynkwargs):
                return obj.map(map_fn, specs, clone)
            dmap = Dynamic(self, operation=apply_map, streams=self.streams,
                           link_inputs=link_inputs)
            dmap.data = deep_mapped.data
            return dmap
        return deep_mapped


    def relabel(self, label=None, group=None, depth=1):
        """
        Assign a new label and/or group to an existing LabelledData
        object, creating a clone of the object with the new settings.
        """
        relabelled = super(DynamicMap, self).relabel(label, group, depth)
        if depth > 0:
            from ..util import Dynamic
            def dynamic_relabel(obj, **dynkwargs):
                return obj.relabel(group=group, label=label, depth=depth-1)
            dmap = Dynamic(self, streams=self.streams, operation=dynamic_relabel)
            dmap.data = relabelled.data
            with util.disable_constant(dmap):
                dmap.group = relabelled.group
                dmap.label = relabelled.label
            return dmap
        return relabelled


    def collate(self):
        """
        Collation allows reorganizing DynamicMaps with invalid nesting
        hierarchies. This is particularly useful when defining
        DynamicMaps returning an (Nd)Layout or GridSpace
        types. Collating will split the DynamicMap into individual
        DynamicMaps for each item in the container. Note that the
        composite object has to be of consistent length and types for
        this to work correctly.
        """
        # Initialize
        if self.last is not None:
            initialized = self
        else:
            initialized = self.clone()
            initialized[initialized._initial_key()]

        if not isinstance(initialized.last, (Layout, NdLayout, GridSpace)):
            return self

        container = initialized.last.clone(shared_data=False)

        # Get stream mapping from callback
        remapped_streams = []
        streams = self.callback.stream_mapping
        for i, (k, v) in enumerate(initialized.last.data.items()):
            vstreams = streams.get(i, [])
            if not vstreams:
                if isinstance(initialized.last, Layout):
                    for l in range(len(k)):
                        path = '.'.join(k[:l])
                        if path in streams:
                            vstreams = streams[path]
                            break
                else:
                    vstreams = streams.get(k, [])
            if any(s in remapped_streams for s in vstreams):
                raise ValueError(
                    "The stream_mapping supplied on the Callable "
                    "is ambiguous please supply more specific Layout "
                    "path specs.")
            remapped_streams += vstreams

            # Define collation callback
            def collation_cb(*args, **kwargs):
                return self[args][kwargs['selection_key']]
            callback = Callable(partial(collation_cb, selection_key=k),
                                inputs=[self])
            vdmap = self.clone(callback=callback, shared_data=False,
                               streams=vstreams)

            # Remap source of streams
            for stream in vstreams:
                if stream.source is self:
                    stream.source = vdmap
            container[k] = vdmap

        unmapped_streams = [repr(stream) for stream in self.streams
                            if (stream.source is self) and
                            (stream not in remapped_streams)
                            and stream.linked]
        if unmapped_streams:
            raise ValueError(
                'The following streams are set to be automatically '
                'linked to a plot, but no stream_mapping specifying '
                'which item in the (Nd)Layout to link it to was found:\n%s'
                % ', '.join(unmapped_streams)
            )
        return container


    def groupby(self, dimensions=None, container_type=None, group_type=None, **kwargs):
        """
        Implements a dynamic version of a groupby, which will
        intelligently expand either the inner or outer dimensions
        depending on whether the container_type or group_type is dynamic.

        To apply a groupby to a DynamicMap the dimensions, which are
        expanded into a non-dynamic type must define a fixed sampling
        via the values attribute.

        Using the dynamic groupby makes it incredibly easy to generate
        dynamic views into a high-dimensional space while taking
        advantage of the capabilities of NdOverlay, GridSpace and
        NdLayout types to visualize more than one Element at a time.
        """
        if dimensions is None:
            dimensions = self.kdims
        if not isinstance(dimensions, (list, tuple)):
            dimensions = [dimensions]

        container_type = container_type if container_type else type(self)
        group_type = group_type if group_type else type(self)

        outer_kdims = [self.get_dimension(d) for d in dimensions]
        inner_kdims = [d for d in self.kdims if not d in outer_kdims]

        outer_dynamic = issubclass(container_type, DynamicMap)
        inner_dynamic = issubclass(group_type, DynamicMap)

        if ((not outer_dynamic and any(not d.values for d in outer_kdims)) or
            (not inner_dynamic and any(not d.values for d in inner_kdims))):
            raise Exception('Dimensions must specify sampling via '
                            'values to apply a groupby')

        if outer_dynamic:
            def outer_fn(*outer_key, **dynkwargs):
                if inner_dynamic:
                    def inner_fn(*inner_key, **dynkwargs):
                        outer_vals = zip(outer_kdims, util.wrap_tuple(outer_key))
                        inner_vals = zip(inner_kdims, util.wrap_tuple(inner_key))
                        inner_sel = [(k.name, v) for k, v in inner_vals]
                        outer_sel = [(k.name, v) for k, v in outer_vals]
                        return self.select(**dict(inner_sel+outer_sel))
                    return self.clone([], callback=inner_fn, kdims=inner_kdims)
                else:
                    dim_vals = [(d.name, d.values) for d in inner_kdims]
                    dim_vals += [(d.name, [v]) for d, v in
                                   zip(outer_kdims, util.wrap_tuple(outer_key))]
                    with item_check(False):
                        selected = HoloMap(self.select(**dict(dim_vals)))
                        return group_type(selected.reindex(inner_kdims))
            if outer_kdims:
                return self.clone([], callback=outer_fn, kdims=outer_kdims)
            else:
                return outer_fn(())
        else:
            outer_product = itertools.product(*[self.get_dimension(d).values
                                                for d in dimensions])
            groups = []
            for outer in outer_product:
                outer_vals = [(d.name, [o]) for d, o in zip(outer_kdims, outer)]
                if inner_dynamic or not inner_kdims:
                    def inner_fn(outer_vals, *key, **dynkwargs):
                        inner_dims = zip(inner_kdims, util.wrap_tuple(key))
                        inner_vals = [(d.name, k) for d, k in inner_dims]
                        return self.select(**dict(outer_vals+inner_vals)).last
                    if inner_kdims:
                        group = self.clone(callback=partial(inner_fn, outer_vals),
                                           kdims=inner_kdims)
                    else:
                        group = inner_fn(outer_vals, ())
                    groups.append((outer, group))
                else:
                    inner_vals = [(d.name, self.get_dimension(d).values)
                                     for d in inner_kdims]
                    with item_check(False):
                        selected = HoloMap(self.select(**dict(outer_vals+inner_vals)))
                        group = group_type(selected.reindex(inner_kdims))
                    groups.append((outer, group))
            return container_type(groups, kdims=outer_kdims)


    def grid(self, dimensions=None, **kwargs):
        return self.groupby(dimensions, container_type=GridSpace, **kwargs)


    def layout(self, dimensions=None, **kwargs):
        return self.groupby(dimensions, container_type=NdLayout, **kwargs)


    def overlay(self, dimensions=None, **kwargs):
        if dimensions is None:
            dimensions = self.kdims
        else:
            if not isinstance(dimensions, (list, tuple)):
                dimensions = [dimensions]
            dimensions = [self.get_dimension(d, strict=True)
                          for d in dimensions]
        dims = [d for d in self.kdims if d not in dimensions]
        return self.groupby(dims, group_type=NdOverlay)


    def hist(self, num_bins=20, bin_range=None, adjoin=True, individually=True, **kwargs):
        """
        Computes a histogram from the object and adjoins it by
        default.  By default the histogram is computed for the bottom
        layer, which can be overriden by supplying an ``index`` and
        for the first value dimension, which may be overridden by
        supplying an explicit ``dimension``.
        """
        def dynamic_hist(obj, **dynkwargs):
            if isinstance(obj, (NdOverlay, Overlay)):
                index = kwargs.get('index', 0)
                obj = obj.get(index)
            return obj.hist(num_bins=num_bins, bin_range=bin_range,
                            adjoin=False, **kwargs)

        from ..util import Dynamic
        hist = Dynamic(self, streams=self.streams, link_inputs=False,
                       operation=dynamic_hist)
        if adjoin:
            return self << hist
        else:
            return hist


    def reindex(self, kdims=[], force=False):
        """
        Reindexing a DynamicMap allows reordering the dimensions but
        not dropping an individual dimension. The force argument which
        usually allows dropping non-constant dimensions is therefore
        ignored and only for API consistency.
        """
        kdims = [self.get_dimension(kd, strict=True) for kd in kdims]
        dropped = [kd for kd in self.kdims if kd not in kdims]
        if dropped:
            raise ValueError("DynamicMap does not allow dropping dimensions, "
                             "reindex may only be used to reorder dimensions.")
        return super(DynamicMap, self).reindex(kdims, force)


    def drop_dimension(self, dimensions):
        raise NotImplementedError('Cannot drop dimensions from a DynamicMap, '
                                  'cast to a HoloMap first.')

    def add_dimension(self, dimension, dim_pos, dim_val, vdim=False, **kwargs):
        raise NotImplementedError('Cannot add dimensions to a DynamicMap, '
                                  'cast to a HoloMap first.')

    def next(self):
        if self.callback.noargs:
            return self[()]
        else:
            raise Exception('The next method can only be used for DynamicMaps using'
                            'generators (or callables without arguments)')

    # For Python 2 and 3 compatibility
    __next__ = next



class GridSpace(UniformNdMapping):
    """
    Grids are distinct from Layouts as they ensure all contained
    elements to be of the same type. Unlike Layouts, which have
    integer keys, Grids usually have floating point keys, which
    correspond to a grid sampling in some two-dimensional space. This
    two-dimensional space may have to arbitrary dimensions, e.g. for
    2D parameter spaces.
    """

    kdims = param.List(default=[Dimension("X"), Dimension("Y")], bounds=(1,2))

    def __init__(self, initial_items=None, kdims=None, **params):
        super(GridSpace, self).__init__(initial_items, kdims=kdims, **params)
        if self.ndims > 2:
            raise Exception('Grids can have no more than two dimensions.')


    def __mul__(self, other):
        if isinstance(other, GridSpace):
            if set(self.keys()) != set(other.keys()):
                raise KeyError("Can only overlay two ParameterGrids if their keys match")
            zipped = zip(self.keys(), self.values(), other.values())
            overlayed_items = [(k, el1 * el2) for (k, el1, el2) in zipped]
            return self.clone(overlayed_items)
        elif isinstance(other, UniformNdMapping) and len(other) == 1:
            view = other.last
        elif isinstance(other, UniformNdMapping) and len(other) != 1:
            raise Exception("Can only overlay with HoloMap of length 1")
        else:
            view = other

        overlayed_items = [(k, el * view) for k, el in self.items()]
        return self.clone(overlayed_items)


    def __lshift__(self, other):
        if isinstance(other, (ViewableElement, UniformNdMapping)):
            return AdjointLayout([self, other])
        elif isinstance(other, AdjointLayout):
            return AdjointLayout(other.data+[self])
        else:
            raise TypeError('Cannot append {0} to a AdjointLayout'.format(type(other).__name__))


    def _transform_indices(self, key):
        """
        Transforms indices by snapping to the closest value if
        values are numeric, otherwise applies no transformation.
        """
        ndims = self.ndims
        if all(not (isinstance(el, slice) or callable(el)) for el in key):
            dim_inds = []
            for dim in self.kdims:
                dim_type = self.get_dimension_type(dim)
                if isinstance(dim_type, type) and issubclass(dim_type, Number):
                    dim_inds.append(self.get_dimension_index(dim))
            str_keys = iter(key[i] for i in range(self.ndims)
                            if i not in dim_inds)
            num_keys = []
            if len(dim_inds):
                keys = list({tuple(k[i] if ndims > 1 else k for i in dim_inds)
                             for k in self.keys()})
                q = np.array([tuple(key[i] if ndims > 1 else key for i in dim_inds)])
                idx = np.argmin([np.inner(q - np.array(x), q - np.array(x))
                                 if len(dim_inds) == 2 else np.abs(q-x)
                                     for x in keys])
                num_keys = iter(keys[idx])
            key = tuple(next(num_keys) if i in dim_inds else next(str_keys)
                        for i in range(self.ndims))
        elif any(not (isinstance(el, slice) or callable(el)) for el in key):
            index_inds = [idx for idx, el in enumerate(key)
                         if not isinstance(el, (slice, str))]
            if len(index_inds):
                index_ind = index_inds[0]
                dim_keys = np.array([k[index_ind] for k in self.keys()])
                snapped_val = dim_keys[np.argmin(np.abs(dim_keys-key[index_ind]))]
                key = list(key)
                key[index_ind] = snapped_val
                key = tuple(key)
        return key


    def keys(self, full_grid=False):
        """
        Returns a complete set of keys on a GridSpace, even when GridSpace isn't fully
        populated. This makes it easier to identify missing elements in the
        GridSpace.
        """
        keys = super(GridSpace, self).keys()
        if self.ndims == 1 or not full_grid:
            return keys
        dim1_keys = sorted(set(k[0] for k in keys))
        dim2_keys = sorted(set(k[1] for k in keys))
        return [(d1, d2) for d1 in dim1_keys for d2 in dim2_keys]


    @property
    def last(self):
        """
        The last of a GridSpace is another GridSpace
        constituted of the last of the individual elements. To access
        the elements by their X,Y position, either index the position
        directly or use the items() method.
        """
        if self.type == HoloMap:
            last_items = [(k, v.last if isinstance(v, HoloMap) else v)
                          for (k, v) in self.data.items()]
        else:
            last_items = self.data
        return self.clone(last_items)


    def __len__(self):
        """
        The maximum depth of all the elements. Matches the semantics
        of __len__ used by Maps. For the total number of elements,
        count the full set of keys.
        """
        return max([(len(v) if hasattr(v, '__len__') else 1) for v in self.values()] + [0])


    def __add__(self, obj):
        return Layout.from_values([self, obj])


    @property
    def shape(self):
        keys = self.keys()
        if self.ndims == 1:
            return (len(keys), 1)
        return len(set(k[0] for k in keys)), len(set(k[1] for k in keys))



class GridMatrix(GridSpace):
    """
    GridMatrix is container type for heterogeneous Element types
    laid out in a grid. Unlike a GridSpace the axes of the Grid
    must not represent an actual coordinate space, but may be used
    to plot various dimensions against each other. The GridMatrix
    is usually constructed using the gridmatrix operation, which
    will generate a GridMatrix plotting each dimension in an
    Element against each other.
    """


    def _item_check(self, dim_vals, data):
        if not traversal.uniform(NdMapping([(0, self), (1, data)])):
            raise ValueError("HoloMaps dimensions must be consistent in %s." %
                             type(self).__name__)
        NdMapping._item_check(self, dim_vals, data)
