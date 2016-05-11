import itertools
import types
from numbers import Number
from itertools import groupby
from functools import partial

import numpy as np
import param

from . import traversal, util
from .dimension import OrderedDict, Dimension, ViewableElement
from .layout import Layout, AdjointLayout, NdLayout
from .ndmapping import UniformNdMapping, NdMapping, item_check
from .overlay import Overlay, CompositeOverlay, NdOverlay
from .options import Store, StoreOptions

class HoloMap(UniformNdMapping):
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
            mode = 'bounded'
            map_obj = None
        elif (isinstance(self, DynamicMap) and (other, DynamicMap) and
            self.mode != other.mode):
            raise ValueError("Cannot overlay DynamicMaps with mismatching mode.")
        else:
            map_obj = self if isinstance(self, DynamicMap) else other
            mode = map_obj.mode

        def dynamic_mul(*key):
            key = key[0] if mode == 'open' else key
            layers = []
            try:
                if isinstance(self, DynamicMap):
                    _, self_el = util.get_dynamic_item(self, dimensions, key)
                    if self_el is not None:
                        layers.append(self_el)
                else:
                    layers.append(self[key])
            except KeyError:
                pass
            try:
                if isinstance(other, DynamicMap):
                    _, other_el = util.get_dynamic_item(other, dimensions, key)
                    if other_el is not None:
                        layers.append(other_el)
                else:
                    layers.append(other[key])
            except KeyError:
                pass
            return Overlay(layers)
        if map_obj:
            return map_obj.clone(callback=dynamic_mul, shared_data=False,
                                 kdims=dimensions)
        else:
            return DynamicMap(callback=dynamic_mul, kdims=dimensions)


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
            dimensions = self.kdims

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
                from .operation import DynamicFunction
                def dynamic_mul(element):
                    return element * other
                return DynamicFunction(self, function=dynamic_mul)
            items = [(k, v * other) for (k, v) in self.data.items()]
            return self.clone(items, label=self._label, group=self._group)
        else:
            raise Exception("Can only overlay with {data} or {vmap}.".format(
                data=self.data_type, vmap=self.__class__.__name__))


    def __add__(self, obj):
        return Layout.from_values(self) + Layout.from_values(obj)


    def __lshift__(self, other):
        if isinstance(other, (ViewableElement, UniformNdMapping)):
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
        on the HoloMap. Homogenous Elements may be collapsed by
        supplying a function, inhomogenous elements are merged.
        """
        from .operation import MapOperation
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
            if isinstance(function, MapOperation):
                collapsed[key] = function(group, **kwargs)
            else:
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
        assumes homogenous and regularly sampled data.

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
                linsamples = zip(X.flat, Y.flat)
            else:
                raise NotImplementedError("Regular sampling not implemented "
                                          "for high-dimensional Views.")

            samples = list(util.unique_iterator(self.last.closest(linsamples)))

        sampled = self.clone([(k, view.sample(samples, **sample_values))
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



class DynamicMap(HoloMap):
    """
    A DynamicMap is a type of HoloMap where the elements are dynamically
    generated by a callback which may be either a callable or a
    generator. A DynamicMap supports two different modes depending on
    the type of callable supplied and the dimension declarations.

    The 'bounded' mode is used when the limits of the parameter space
    are known upon declaration (as specified by the ranges on the key
    dimensions) or 'open' which allows the continual generation of
    elements (e.g as data output by a simulator over an unbounded
    simulated time dimension).

    Generators always imply open mode but a callable that has any key
    dimension unbounded in any direction will also be in open
    mode. Bounded mode only applied to callables where all the key
    dimensions are fully bounded.
    """
    _sorted = False
    # Declare that callback is a positional parameter (used in clone)
    __pos_params = ['callback']

    callback = param.Parameter(doc="""
        The callable or generator used to generate the elements. In the
        simplest case where all key dimensions are bounded, this can be
        a callable that accepts the key dimension values as arguments
        (in the declared order) and returns the corresponding element.

        For open mode where there is an unbounded key dimension, the
        return type can specify a key as well as element as the tuple
        (key, element). If no key is supplied, a simple counter is used
        instead.

        If the callback is a generator, open mode is used and next() is
        simply called. If the callback is callable and in open mode, the
        element counter value will be supplied as the single
        argument. This can be used to avoid issues where multiple
        elements in a Layout each call next() leading to uncontrolled
        changes in simulator state (the counter can be used to indicate
        simulation time across the layout).
    """)

    cache_size = param.Integer(default=500, doc="""
       The number of entries to cache for fast access. This is an LRU
       cache where the least recently used item is overwritten once
       the cache is full.""")

    cache_interval = param.Integer(default=1, doc="""
       When the element counter modulo the cache_interval is zero, the
       element will be cached and therefore accessible when casting to a
       HoloMap.  Applicable in open mode only.""")

    sampled = param.Boolean(default=False, doc="""
       Allows defining a DynamicMap in bounded mode without defining the
       dimension bounds or values. The DynamicMap may then be explicitly
       sampled via getitem or the sampling is determined during plotting
       by a HoloMap with fixed sampling.
       """)

    def __init__(self, callback, initial_items=None, **params):
        super(DynamicMap, self).__init__(initial_items, callback=callback, **params)
        self.counter = 0
        if self.callback is None:
            raise Exception("A suitable callback must be "
                            "declared to create a DynamicMap")

        self.call_mode = self._validate_mode()
        self.mode = 'bounded' if self.call_mode == 'key' else 'open'


    def _initial_key(self):
        """
        Construct an initial key for bounded mode based on the lower
        range bounds or values on the key dimensions.
        """
        key = []
        for kdim in self.kdims:
            if kdim.values:
                key.append(kdim.values[0])
            elif kdim.range:
                key.append(kdim.range[0])
        return tuple(key)


    def _validate_mode(self):
        """
        Check the key dimensions and callback to determine the calling mode.
        """
        isgenerator = isinstance(self.callback, types.GeneratorType)
        if isgenerator:
            if self.sampled:
                raise ValueError("Cannot set DynamicMap containing generator "
                                 "to sampled")
            return 'generator'
        if self.sampled:
            return 'key'
        # Any unbounded kdim (any direction) implies open mode
        for kdim in self.kdims:
            if kdim.values:
                continue
            if None in kdim.range:
                return 'counter'
        return 'key'


    def _validate_key(self, key):
        """
        Make sure the supplied key values are within the bounds
        specified by the corresponding dimension range and soft_range.
        """
        key = util.wrap_tuple(key)
        assert len(key) == len(self.kdims)
        for ind, val in enumerate(key):
            kdim = self.kdims[ind]
            low, high = util.max_range([kdim.range, kdim.soft_range])
            if low is not np.NaN:
                if val < low:
                    raise StopIteration("Key value %s below lower bound %s"
                                        % (val, low))
            if high is not np.NaN:
                if val > high:
                    raise StopIteration("Key value %s above upper bound %s"
                                        % (val, high))


    def _style(self, retval):
        """
        Use any applicable OptionTree of the DynamicMap to apply options
        to the return values of the callback.
        """
        if self.id not in Store.custom_options():
            return retval
        spec = StoreOptions.tree_to_dict(Store.custom_options()[self.id])
        return retval(spec)


    def _execute_callback(self, *args):
        """
        Execute the callback, validating both the input key and output
        key where applicable.
        """
        if self.call_mode == 'key':
            self._validate_key(args)      # Validate input key

        if self.call_mode == 'generator':
            retval = next(self.callback)
        else:
            retval = self.callback(*args)

        if self.call_mode=='key':
            return self._style(retval)

        if isinstance(retval, tuple):
            self._validate_key(retval[0]) # Validated output key
            return (retval[0], self._style(retval[1]))
        else:
            self._validate_key((self.counter,))
            return (self.counter, self._style(retval))


    def clone(self, data=None, shared_data=True, new_type=None, *args, **overrides):
        """
        Clone method to adapt the slightly different signature of
        DynamicMap that also overrides Dimensioned clone to avoid
        checking items if data is unchanged.
        """
        if data is None and shared_data:
            data = self.data
        return super(UniformNdMapping, self).clone(overrides.pop('callback', self.callback),
                                                   shared_data, new_type,
                                                   *(data,) + args, **overrides)


    def reset(self):
        """
        Return a cleared dynamic map with a cleared cached
        and a reset counter.
        """
        if self.call_mode == 'generator':
            raise Exception("Cannot reset generators.")
        self.counter = 0
        self.data = OrderedDict()
        return self


    def _cross_product(self, tuple_key, cache):
        """
        Returns a new DynamicMap if the key (tuple form) expresses a
        cross product, otherwise returns None. The cache argument is a
        dictionary (key:element pairs) of all the data found in the
        cache for this key.

        Each key inside the cross product is looked up in the cache
        (self.data) to check if the appropriate element is
        available. Oherwise the element is computed accordingly.
        """
        if self.mode != 'bounded': return None
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
            data.append((key, val))
        return self.clone(data)


    def _slice_bounded(self, tuple_key):
        """
        Slices bounded DynamicMaps by setting the soft_ranges on key dimensions.
        """
        cloned = self.clone(self)
        for i, slc in enumerate(tuple_key):
            (start, stop) = slc.start, slc.stop
            if start is not None and start < cloned.kdims[i].range[0]:
                raise Exception("Requested slice below defined dimension range.")
            if stop is not None and stop > cloned.kdims[i].range[1]:
                raise Exception("Requested slice above defined dimension range.")
            cloned.kdims[i].soft_range = (start, stop)
        return cloned


    def __getitem__(self, key):
        """
        Return an element for any key chosen key (in'bounded mode') or
        for a previously generated key that is still in the cache
        (for one of the 'open' modes)
        """
        tuple_key = util.wrap_tuple(key)

        # Validation for bounded mode
        if self.mode == 'bounded':
            # DynamicMap(...)[:] returns a new DynamicMap with the same cache
            if key == slice(None, None, None):
                return self.clone(self)

            slices = [el for el in tuple_key if isinstance(el, slice)]
            if any(el.step for el in slices):
                raise Exception("Slices cannot have a step argument "
                                "in DynamicMap bounded mode ")
            if len(slices) not in [0, len(tuple_key)]:
                raise Exception("Slices must be used exclusively or not at all")
            if slices:
                return  self._slice_bounded(tuple_key)

        # Cache lookup
        try:
            cache = super(DynamicMap,self).__getitem__(key)
            # Return selected cache items in a new DynamicMap
            if isinstance(cache, DynamicMap) and self.mode=='open':
                cache = self.clone(cache)
        except KeyError as e:
            cache = None
            if self.mode == 'open' and len(self.data)>0:
                raise KeyError(str(e) + " Note: Cannot index outside "
                               "available cache in open interval mode.")

        # If the key expresses a cross product, compute the elements and return
        product = self._cross_product(tuple_key, cache.data if cache else {})
        if product is not None:
            return product

        # Not a cross product and nothing cached so compute element.
        if cache: return cache
        val = self._execute_callback(*tuple_key)
        if self.call_mode == 'counter':
            val = val[1]

        self._cache(tuple_key, val)
        return val


    def _cache(self, key, val):
        """
        Request that a key/value pair be considered for caching.
        """
        if self.mode == 'open' and (self.counter % self.cache_interval)!=0:
            return
        if len(self) >= self.cache_size:
            first_key = next(self.data.iterkeys())
            self.data.pop(first_key)
        self.data[key] = val


    def next(self):
        """
        Interface for 'open' mode. For generators, this simply calls the
        next() method. For callables callback, the counter is supplied
        as a single argument.
        """
        if self.mode == 'bounded':
            raise Exception("The next() method should only be called in "
                            "one of the open modes.")

        args = () if self.call_mode == 'generator' else (self.counter,)
        retval = self._execute_callback(*args)

        (key, val) = (retval if isinstance(retval, tuple)
                      else (self.counter, retval))

        key = util.wrap_tuple(key)
        if len(key) != len(self.key_dimensions):
            raise Exception("Generated key does not match the number of key dimensions")

        self._cache(key, val)
        self.counter += 1
        return val


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
            def outer_fn(*outer_key):
                if inner_dynamic:
                    def inner_fn(*inner_key):
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
                    return group_type(self.select(**dict(dim_vals))).reindex(inner_kdims)
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
                    def inner_fn(outer_vals, *key):
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
                    group = group_type(self.select(**dict(outer_vals+inner_vals)).reindex(inner_kdims))
                    groups.append((outer, group))
            return container_type(groups, kdims=outer_kdims)


    def grid(self, dimensions=None, **kwargs):
        return self.groupby(dimensions, container_type=GridSpace, **kwargs)


    def layout(self, dimensions=None, **kwargs):
        return self.groupby(dimensions, container_type=NdLayout, **kwargs)


    def overlay(self, dimensions=None, **kwargs):
        if dimensions is None:
            dimensions = self.kdims
        if not isinstance(dimensions, (list, tuple)):
            dimensions = [dimensions]
        dimensions = [self.get_dimension(d) for d in dimensions]
        dims = [d for d in self.kdims if d not in dimensions]
        return self.groupby(dims, group_type=NdOverlay)

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

    kdims = param.List(default=[Dimension(name="X"), Dimension(name="Y")],
                       bounds=(1,2))

    def __init__(self, initial_items=None, **params):
        super(GridSpace, self).__init__(initial_items, **params)
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
        return Layout.from_values(self) + Layout.from_values(obj)


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
