from numbers import Number
import numpy as np

import param

from .dimension import OrderedDict, Dimension, Dimensioned, ViewableElement
from .layout import Layout, AdjointLayout, NdLayout
from .ndmapping import UniformNdMapping, NdMapping, item_check
from .overlay import Overlayable, Overlay, CompositeOverlay, NdOverlay
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
                    items.append((new_key, Overlay([self[self_key]])))
                else:
                    items.append((new_key, Overlay([other[other_key]])))
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
        from .element import Collator
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
