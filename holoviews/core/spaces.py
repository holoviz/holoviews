from numbers import Number
import numpy as np

import param

from .dimension import OrderedDict, Dimension, Dimensioned, ViewableElement
from .element import Element, NdElement, Tabular
from .layout import Layout, AdjointLayout
from .ndmapping import UniformNdMapping, NdMapping, item_check
from .overlay import Overlayable, Overlay, CompositeOverlay
from .tree import AttrTree


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
                return NdOverlay(self, **kwargs)
        else:
            dims = [d for d in self._cached_index_names
                    if d not in dimensions]
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
                return GridSpace(self, **kwargs)
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
                return NdLayout(self, **kwargs)
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
        return [tuple(zip(self._cached_index_names, [k] if self.ndims == 1 else k))
                for k in self.keys()]


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
        if isinstance(other, self.__class__):
            self_set = set(self._cached_index_names)
            other_set = set(other._cached_index_names)

            # Determine which is the subset, to generate list of keys and
            # dimension labels for the new view
            self_in_other = self_set.issubset(other_set)
            other_in_self = other_set.issubset(self_set)
            dimensions = self.kdims
            if self_in_other and other_in_self: # superset of each other
                super_keys = sorted(set(self._dimension_keys() + other._dimension_keys()))
            elif self_in_other: # self is superset
                dimensions = other.kdims
                super_keys = other._dimension_keys()
            elif other_in_self: # self is superset
                super_keys = self._dimension_keys()
            else: # neither is superset
                raise Exception('One set of keys needs to be a strict subset of the other.')

            items = []
            for dim_keys in super_keys:
                # Generate keys for both subset and superset and sort them by the dimension index.
                self_key = tuple(k for p, k in sorted(
                    [(self.get_dimension_index(dim), v) for dim, v in dim_keys
                     if dim in self._cached_index_names]))
                other_key = tuple(k for p, k in sorted(
                    [(other.get_dimension_index(dim), v) for dim, v in dim_keys
                     if dim in other._cached_index_names]))
                new_key = self_key if other_in_self else other_key
                # Append SheetOverlay of combined items
                if (self_key in self) and (other_key in other):
                    items.append((new_key, self[self_key] * other[other_key]))
                elif self_key in self:
                    items.append((new_key, self[self_key] * other.empty_element))
                else:
                    items.append((new_key, self.empty_element * other[other_key]))
            return self.clone(items, kdims=dimensions, label=self._label, group=self._group)
        elif isinstance(other, self.data_type):
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
        merge_type=merge_type if merge_type else self.__class__
        return Collator(self, merge_type=merge_type, drop=drop,
                        drop_constant=drop_constant)()


    def collapse(self, dimensions=None, function=None, **kwargs):
        """
        Allows collapsing one of any number of key dimensions
        on the HoloMap. Homogenous Elements may be collapsed by
        supplying a function, inhomogenous elements are merged.
        """
        from .operation import MapOperation
        if not dimensions:
            dimensions = self._cached_index_names
        if self.ndims > 1 and len(dimensions) != self.ndims:
            groups = self.groupby([dim for dim in self._cached_index_names
                                   if dim not in dimensions])
        else:
            [self.get_dimension(dim) for dim in dimensions]
            groups = HoloMap([(0, self)])
        collapsed = groups.clone(shared_data=False)
        for key, group in groups.items():
            if isinstance(function, MapOperation):
                collapsed[key] = function(group, **kwargs)
            else:
                data = group.type.collapse_data([el.data for el in group], function, **kwargs)
                collapsed[key] = group.last.clone(data)
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

                X,Y = np.meshgrid(xsamples, ysamples)
                linsamples = zip(X.flat, Y.flat)
            else:
                raise NotImplementedError("Regular sampling not implented"
                                          "for high-dimensional Views.")

            samples = set(self.last.closest(linsamples))

        sampled = self.clone([(k, view.sample(samples, **sample_values))
                              for k, view in self.data.items()])
        return sampled.table()


    def reduce(self, dimensions=None, function=None, **reduce_map):
        """
        Reduce each Element in the HoloMap using a function supplied
        via the kwargs, where the keyword has to match a particular
        dimension in the Elements.
        """
        reduced_items = [(k, v.reduce(dimensions, function, **reduce_map))
                         for k, v in self.items()]
        return self.clone(reduced_items).table()


    def relabel(self, label=None, group=None, depth=1):
        # Identical to standard relabel method except for default depth of 1
        return super(HoloMap, self).relabel(label=label, group=group, depth=depth)


    @property
    def empty_element(self):
        return self._type(None)


    def hist(self, num_bins=20, bin_range=None, adjoin=True, individually=True, **kwargs):
        histmap = self.clone(shared_data=False)

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
            histmap[k] = v.hist(adjoin=False, bin_range=bin_range,
                                individually=individually, num_bins=num_bins,
                                style_prefix=style_prefix, **kwargs)

        if adjoin and issubclass(self.type, (NdOverlay, Overlay)):
            layout = (self << histmap)
            layout.main_layer = kwargs['index']
            return layout

        return (self << histmap) if adjoin else histmap



class GridSpace(UniformNdMapping):
    """
    Grids are distinct from Layouts as they ensure all contained
    elements to be of the same type. Unlike Layouts, which have
    integer keys, Grids usually have floating point keys, which
    correspond to a grid sampling in some two-dimensional space. This
    two-dimensional space may have to arbitrary dimensions, e.g. for
    2D parameter spaces.
    """

    # NOTE: If further composite types supporting Overlaying and Layout these
    #       classes may be moved to core/composite.py

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
        if all(not isinstance(el, slice) for el in key):
            dim_inds = []
            for dim in self._cached_index_names:
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
        elif any(not isinstance(el, slice) for el in key):
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



class NdOverlay(UniformNdMapping, CompositeOverlay, Overlayable):
    """
    An NdOverlay allows a group of NdOverlay to be overlaid together. NdOverlay can
    be indexed out of an overlay and an overlay is an iterable that iterates
    over the contained layers.
    """

    kdims = param.List(default=[Dimension('Element')], constant=True, doc="""
        List of dimensions the NdOverlay can be indexed by.""")

    _deep_indexable = True

    def __init__(self, overlays=None, **params):
        super(NdOverlay, self).__init__(overlays, **params)


    def hist(self, num_bins=20, bin_range=None, adjoin=True, individually=True, **kwargs):
        from ..operation import histogram
        return histogram(self, num_bins=num_bins, bin_range=bin_range, adjoin=adjoin,
                         individually=individually, **kwargs)



class NdLayout(UniformNdMapping):
    """
    NdLayout is a UniformNdMapping providing an n-dimensional
    data structure to display the contained Elements and containers
    in a layout. Using the cols method the NdLayout can be rearranged
    with the desired number of columns.
    """

    data_type = (ViewableElement, AdjointLayout, UniformNdMapping)

    def __init__(self, initial_items=None, **params):
        self._max_cols = 4
        self._style = None
        super(NdLayout, self).__init__(initial_items=initial_items, **params)


    @property
    def uniform(self):
        return traversal.uniform(self)


    @property
    def shape(self):
        num = len(self.keys())
        if num <= self._max_cols:
            return (1, num)
        nrows = num // self._max_cols
        last_row_cols = num % self._max_cols
        return nrows+(1 if last_row_cols else 0), min(num, self._max_cols)


    def grid_items(self):
        """
        Compute a dict of {(row,column): (key, value)} elements from the
        current set of items and specified number of columns.
        """
        if list(self.keys()) == []:  return {}
        cols = self._max_cols
        return {(idx // cols, idx % cols): (key, item)
                for idx, (key, item) in enumerate(self.data.items())}


    def cols(self, n):
        self._max_cols = n
        return self


    def __add__(self, obj):
        return Layout.from_values(self) + Layout.from_values(obj)


    @property
    def last(self):
        """
        Returns another NdLayout constituted of the last views of the
        individual elements (if they are maps).
        """
        last_items = []
        for (k, v) in self.items():
            if isinstance(v, NdMapping):
                item = (k, v.clone((v.last_key, v.last)))
            elif isinstance(v, AdjointLayout):
                item = (k, v.last)
            else:
                item = (k, v)
            last_items.append(item)
        return self.clone(last_items)



class Collator(NdElement):
    """
    Collator is an NdMapping type which can merge any number
    of HoloViews components with whatever level of nesting
    by inserting the Collators key dimensions on the HoloMaps.
    If the items in the Collator do not contain HoloMaps
    they will be created. Collator also supports filtering
    of Tree structures and dropping of constant dimensions.
    """

    drop = param.List(default=[], doc="""
        List of dimensions to drop when collating data, specified
        as strings.""")

    drop_constant = param.Boolean(default=False, doc="""
        Whether to demote any non-varying key dimensions to
        constant dimensions.""")

    filters = param.List(default=[], doc="""
        List of paths to drop when collating data, specified
        as strings or tuples.""")

    group = param.String(default='Collator')


    progress_bar = param.Parameter(default=None, doc="""
         The progress bar instance used to report progress. Set to
         None to disable progress bars.""")

    merge_type = param.ClassSelector(class_=NdMapping, default=HoloMap,
                                     is_instance=False,instantiate=False)

    value_transform = param.Callable(default=None, doc="""
        If supplied the function will be applied on each Collator
        value during collation. This may be used to apply an operation
        to the data or load references from disk before they are collated
        into a displayable HoloViews object.""")

    vdims = param.List(default=[], doc="""
         Collator operates on HoloViews objects, if vdims are specified
         a value_transform function must also be supplied.""")

    _deep_indexable = False
    _auxiliary_component = False

    _nest_order = {HoloMap: ViewableElement,
                   GridSpace: (HoloMap, ViewableElement),
                   NdLayout: (GridSpace, HoloMap, ViewableElement),
                   NdOverlay: Element}

    def __call__(self):
        """
        Filter each Layout in the Collator with the supplied
        path_filters. If merge is set to True all Layouts are
        merged, otherwise an NdMapping containing all the
        Layouts is returned. Optionally a list of dimensions
        to be ignored can be supplied.
        """
        constant_dims = self.static_dimensions
        ndmapping = NdMapping(kdims=self.kdims)

        num_elements = len(self)
        for idx, (key, data) in enumerate(self.data.items()):
            if isinstance(data, AttrTree):
                data = data.filter(self.filters)
            if len(self.vdims):
                vargs = dict(zip(self.dimensions('value', label=True), data))
                data = self.value_transform(vargs)
            if not isinstance(data, Dimensioned):
                raise ValueError("Collator values must be Dimensioned objects "
                                 "before collation.")

            dim_keys = zip(self._cached_index_names, key)
            varying_keys = [(d, k) for d, k in dim_keys if not self.drop_constant or
                            (d not in constant_dims and d not in self.drop)]
            constant_keys = [(d if isinstance(d, Dimension) else Dimension(d), k)
                             for d, k in dim_keys if d in constant_dims
                             and d not in self.drop and self.drop_constant]
            if varying_keys or constant_keys:
                data = self._add_dimensions(data, varying_keys,
                                            dict(constant_keys))
            ndmapping[key] = data
            if self.progress_bar is not None:
                self.progress_bar(float(idx+1)/num_elements*100)

        components = ndmapping.values()
        accumulator = ndmapping.last.clone(components[0].data)
        for component in components:
            accumulator.update(component)
        return accumulator


    def _add_item(self, key, value, sort=True):
        Tabular._add_item(self, key, value, sort)


    @property
    def static_dimensions(self):
        """
        Return all constant dimensions.
        """
        dimensions = []
        for dim in self.kdims:
            if len(set(self.dimension_values(dim.name))) == 1:
                dimensions.append(dim)
        return dimensions


    def _add_dimensions(self, item, dims, constant_keys):
        """
        Recursively descend through an Layout and NdMapping objects
        in order to add the supplied dimension values to all contained
        HoloMaps.
        """
        if isinstance(item, Layout):
            item.fixed = False

        dim_vals = [(dim, val) for dim, val in dims[::-1]
                    if dim not in self.drop]
        if isinstance(item, self.merge_type):
            new_item = item.clone(cdims=constant_keys)
            for dim, val in dim_vals:
                dim = dim if isinstance(dim, Dimension) else Dimension(dim)
                if dim not in new_item.kdims:
                    new_item = new_item.add_dimension(dim, 0, val)
        elif isinstance(item, self._nest_order[self.merge_type]):
            if len(dim_vals):
                dimensions, key = zip(*dim_vals)
                new_item = self.merge_type({key: item}, kdims=dimensions,
                                           cdims=constant_keys)
            else:
                new_item = item
        else:
            new_item = item.clone(shared_data=False, cdims=constant_keys)
            for k, v in item.items():
                new_item[k] = self._add_dimensions(v, dims[::-1], constant_keys)
        if isinstance(new_item, Layout):
            new_item.fixed = True

        return new_item
