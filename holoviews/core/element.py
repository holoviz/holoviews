from collections import OrderedDict
import itertools
from numbers import Number
import numpy as np

import param

from .dimension import Dimension, ViewableElement
from .layout import Composable, LayoutTree, AdjointLayout, NdLayout
from .ndmapping import UniformNdMapping
from .overlay import Overlayable, NdOverlay, Overlay, CompositeOverlay
from .util import find_minmax


class Element(ViewableElement, Composable, Overlayable):
    """
    Element is the baseclass for all ViewableElement types, with an x- and
    y-dimension. Subclasses should define the data storage in the
    constructor, as well as methods and properties, which define how
    the data maps onto the x- and y- and value dimensions.
    """

    value = param.String(default='Element')

    def hist(self, num_bins=20, bin_range=None, adjoin=True, individually=True, **kwargs):
        from ..operation import histogram
        return histogram(self, num_bins=num_bins, bin_range=bin_range, adjoin=adjoin,
                         individually=individually, **kwargs)


    ########################
    # Subclassable methods #
    ########################

    def __init__(self, data, **params):
        super(Element, self).__init__(data, **params)


    def __getitem__(self, key):
        if key is ():
            return self
        else:
            raise NotImplementedError("%s currently does not support getitem" %
                                      type(self).__name__)


    def closest(self, coords):
        """
        Class method that returns the exact keys for a given list of
        coordinates. The supplied bounds defines the extent within
        which the samples are drawn and the optional shape argument is
        the shape of the numpy array (typically the shape of the .data
        attribute) when applicable.
        """
        return coords


    def sample(self, **samples):
        """
        Base class signature to demonstrate API for sampling Views.
        To sample a ViewableElement kwargs, where the keyword matches a Dimension
        in the ViewableElement and the value matches a corresponding entry in the
        data.
        """
        raise NotImplementedError


    def reduce(self, label_prefix='', **reduce_map):
        """
        Base class signature to demonstrate API for reducing Views,
        using some reduce function, e.g. np.mean. Signature is the
        same as sample, however a label_prefix may be provided to
        describe the reduction operation.
        """
        raise NotImplementedError


    def table(self, **kwargs):
        """
        This method transforms any ViewableElement type into a Table
        as long as it implements a dimension_values method.
        """
        from ..element import Table
        keys = zip(*[self.dimension_values(dim.name)
                 for dim in self.key_dimensions])
        values = zip(*[self.dimension_values(dim.name)
                       for dim in self.value_dimensions])
        params = dict(key_dimensions=self.key_dimensions,
                      value_dimensions=self.value_dimensions,
                      label=self.label, value=self.value, **kwargs)
        return Table(zip(keys, values), **params)


    def dframe(self):
        import pandas
        column_names = self.dimensions(label=True)
        dim_vals = np.vstack([self.dimension_values(dim) for dim in column_names]).T
        return pandas.DataFrame(dim_vals, columns=column_names)


    def __repr__(self):
        params = ', '.join('%s=%r' % (k,v) for (k,v) in self.get_param_values())
        return "%s(%r, %s)" % (self.__class__.__name__, self.data, params)



class Element2D(Element):

    def __init__(self, data, extents=None, **params):
        self._xlim = None if extents is None else (extents[0], extents[2])
        self._ylim = None if extents is None else (extents[1], extents[3])
        super(Element2D, self).__init__(data, **params)

    @property
    def xlabel(self):
        return self.get_dimension(0).pprint_label

    @property
    def ylabel(self):
        return self.get_dimension(1).pprint_label

    @property
    def xlim(self):
        if self._xlim:
            return self._xlim
        else:
            return self.range(0)

    @xlim.setter
    def xlim(self, limits):
        if limits is None or (isinstance(limits, tuple) and len(limits) == 2):
            self._xlim = limits
        else:
            raise ValueError('xlim needs to be a length two tuple or None.')

    @property
    def ylim(self):
        if self._ylim:
            return self._ylim
        else:
            return self.range(1)

    @ylim.setter
    def ylim(self, limits):
        if limits is None or (isinstance(limits, tuple) and len(limits) == 2):
            self._ylim = limits
        else:
            raise ValueError('xlim needs to be a length two tuple or None.')

    @property
    def extents(self):
        """"
        For Element2D the extents is the 4-tuple (left, bottom, right, top).
        """
        l, r = self.xlim if self.xlim else (np.NaN, np.NaN)
        b, t = self.ylim if self.ylim else (np.NaN, np.NaN)
        return l, b, r, t


    @extents.setter
    def extents(self, extents):
        l, b, r, t = extents
        self.xlim, self.ylim = (l, r), (b, t)


class Element3D(Element2D):


    def __init__(self, data, extents=None, **params):
        if extents is not None:
            self._zlim = (extents[2], extents[5])
            extent = (extents[0], extents[1], extents[3], extents[4])
        else:
            self._zlim = None
        super(Element3D, self).__init__(data, extents=extents, **params)

    @property
    def extents(self):
        """"
        For Element3D the extents is the 6-tuple (left, bottom, -z, right, top, +z) using
        a right-handed Cartesian coordinate-system.
        """
        l, r = self.xlim if self.xlim else (np.NaN, np.NaN)
        b, t = self.ylim if self.ylim else (np.NaN, np.NaN)
        zminus, zplus = self.zlim if self.zlim else (np.NaN, np.NaN)
        return l, b, zminus, r, t, zplus

    @extents.setter
    def extents(self, extents):
        l, b, zminus, r, t, zplus = extents
        self.xlim, self.ylim, self.zlim = (l, r), (b, t), (zminus, zplus)

    @property
    def zlim(self):
        if self._zlim:
            return self._zlim
        else:
            return self.range(2)

    @zlim.setter
    def zlim(self, limits):
        if limits is None or (isinstance(limits, tuple) and len(limits) == 2):
            self._zlim = limits
        else:
            raise ValueError('zlim needs to be a length two tuple or None.')

    @property
    def zlabel(self):
        return self.get_dimension(2).pprint_label



class HoloMap(UniformNdMapping):
    """
    A HoloMap can hold any number of DataLayers indexed by a list of
    dimension values. It also has a number of properties, which can find
    the x- and y-dimension limits and labels.
    """

    value = param.String(default='HoloMap')

    data_type = (ViewableElement, UniformNdMapping)

    @property
    def layer_types(self):
        """
        The type of layers stored in the HoloMap.
        """
        if self.type == NdOverlay:
            return self.last.layer_types
        else:
            return (self.type)


    @property
    def xlabel(self):
        return self.last.xlabel


    @property
    def ylabel(self):
        return self.last.ylabel


    @property
    def xlim(self):
        xlim = self.last.xlim
        for data in self.values():
            xlim = find_minmax(xlim, data.xlim) if data.xlim and xlim else xlim
        return xlim


    @property
    def ylim(self):
        ylim = self.last.ylim
        for data in self.values():
            ylim = find_minmax(ylim, data.ylim) if data.ylim and ylim else ylim
        return ylim


    @property
    def extents(self):
        if self.xlim is None: return np.NaN, np.NaN, np.NaN, np.NaN
        l, r = self.xlim
        b, t = self.ylim
        return float(l), float(b), float(r), float(t)



    def overlay(self, dimensions):
        """
        Splits the UniformNdMapping along a specified number of dimensions and
        overlays items in the split out Maps.
        """
        if self.ndims == 1:
            return NdOverlay(self)
        else:
            dims = [d for d in self._cached_index_names
                    if d not in dimensions]
            return self.groupby(dims, group_type=NdOverlay)


    def grid(self, dimensions):
        """
        AxisLayout takes a list of one or two dimensions, and lays out the containing
        Views along these axes in a AxisLayout.
        """
        if self.ndims == 1:
            return AxisLayout(self)
        return self.groupby(dimensions, container_type=AxisLayout)


    def layout(self, dimensions):
        """
        AxisLayout takes a list of one or two dimensions, and lays out the containing
        Views along these axes in a AxisLayout.
        """
        return self.groupby(dimensions, container_type=NdLayout)


    def split_overlays(self):
        """
        Given a UniformNdMapping of Overlays of N layers, split out the layers into
        N separate Maps.
        """
        if not issubclass(self.type, CompositeOverlay):
            return None, self.clone(self.items())

        item_maps = OrderedDict()
        for k, overlay in self.items():
            for key, el in overlay.items():
                if key not in item_maps:
                    item_maps[key] = [(k, el)]
                else:
                    item_maps[key].append((k, el))

        maps, keys = [], []
        for k in item_maps.keys():
            maps.append(self.clone(item_maps[k]))
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
            dimensions = self.key_dimensions
            if self_in_other and other_in_self: # superset of each other
                super_keys = sorted(set(self._dimension_keys() + other._dimension_keys()))
            elif self_in_other: # self is superset
                dimensions = other.key_dimensions
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
            return self.clone(items, key_dimensions=dimensions)
        elif isinstance(other, self.data_type):
            items = [(k, v * other) for (k, v) in self.items()]
            return self.clone(items)
        else:
            raise Exception("Can only overlay with {data} or {vmap}.".format(
                data=self.data_type, vmap=self.__class__.__name__))


    def dframe(self):
        """
        Gets a dframe for each Element in the HoloMap, appends the
        dimensions of the HoloMap as series and concatenates the
        dframes.
        """
        import pandas
        dframes = []
        for key, view in self.data.items():
            view_frame = view.dframe()
            for val, dim in reversed(zip(key, self._cached_index_names)):
                dim = dim.replace(' ', '_')
                dimn = 1
                while dim in view_frame:
                    dim = dim+'_%d' % dimn
                    if dim in view_frame:
                        dimn += 1
                view_frame.insert(0, dim, val)
            dframes.append(view_frame)
        return pandas.concat(dframes)


    def __add__(self, obj):
        return LayoutTree.from_values(self) + LayoutTree.from_values(obj)


    def __lshift__(self, other):
        if isinstance(other, (ViewableElement, UniformNdMapping)):
            return AdjointLayout([self, other])
        elif isinstance(other, AdjointLayout):
            return AdjointLayout(other.data+[self])
        else:
            raise TypeError('Cannot append {0} to a AdjointLayout'.format(type(other).__name__))


    def sample(self, samples, bounds=None, **sample_values):
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
                lower, upper = (self.xlims[0],self.xlims[1]) if bounds is None else bounds
                edges = np.linspace(lower, upper, samples+1)
                linsamples = [(l+u)/2.0 for l,u in zip(edges[:-1], edges[1:])]
            elif dims == 2:
                (rows, cols) = samples
                (l,b,r,t) = self.last.extents if bounds is None else bounds

                xedges = np.linspace(l, r, cols+1)
                yedges = np.linspace(b, t, rows+1)
                xsamples = [(l+u)/2.0 for l,u in zip(xedges[:-1], xedges[1:])]
                ysamples = [(l+u)/2.0 for l,u in zip(yedges[:-1], yedges[1:])]

                X,Y = np.meshgrid(xsamples, ysamples)
                linsamples = zip(X.flat, Y.flat)
            else:
                raise NotImplementedError("Regular sampling not implented"
                                          "for high-dimensional Views.")

            samples = set(self.last.closest(linsamples))

        sampled_items = [(k, view.sample(samples, **sample_values))
                         for k, view in self.items()]
        return self.clone(sampled_items)


    def reduce(self, label_prefix='', **reduce_map):
        """
        Reduce each Element in the HoloMap using a function supplied
        via the kwargs, where the keyword has to match a particular
        dimension in the Elements.
        """
        reduced_items = [(k, v.reduce(label_prefix=label_prefix, **reduce_map))
                         for k, v in self.items()]
        return self.clone(reduced_items)


    def collate(self, collate_dim):
        """
        Collate splits out the specified dimension and joins the
        samples in each of the split out Maps into Curves. If there
        are multiple entries in the ItemTable it will lay them out
        into a AxisLayout.
        """
        from ..operation import table_collate
        return table_collate(self, collation_dim=collate_dim)


    @property
    def empty_element(self):
        return self._type(None)


    @property
    def N(self):
        return self.normalize()


    def hist(self, num_bins=20, bin_range=None, adjoin=True, individually=True, **kwargs):
        histmap = HoloMap(key_dimensions=self.key_dimensions)

        map_range = None if individually else self.range
        bin_range = map_range if bin_range is None else bin_range
        style_prefix = 'Custom[<' + self.name + '>]_'
        for k, v in self.items():
            histmap[k] = v.hist(adjoin=False, bin_range=bin_range,
                                individually=individually, num_bins=num_bins,
                                style_prefix=style_prefix, **kwargs)

        if adjoin and issubclass(self.type, (NdOverlay, Overlay)):
            layout = (self << histmap)
            layout.main_layer = kwargs['index']
            return layout

        return (self << histmap) if adjoin else histmap


    def normalize_elements(self, **kwargs):
        return self.map(lambda x, _: x.normalize(**kwargs))


    def normalize(self, min=0.0, max=1.0):
        data_max = np.max([el.data.max() for el in self.values()])
        data_min = np.min([el.data.min() for el in self.values()])
        norm_factor = data_max-data_min
        return self.map(lambda x, _: x.normalize(min=min, max=max,
                                                 norm_factor=norm_factor))


class AxisLayout(UniformNdMapping):
    """
    Grids are distinct from GridLayouts as they ensure all contained elements
    to be of the same type. Unlike GridLayouts, which have integer keys,
    Grids usually have floating point keys, which correspond to a grid
    sampling in some two-dimensional space. This two-dimensional space may
    have to arbitrary dimensions, e.g. for 2D parameter spaces.
    """

    # NOTE: If further composite types supporting Overlaying and Layout these
    #       classes may be moved to core/composite.py

    key_dimensions = param.List(default=[Dimension(name="X"), Dimension(name="Y")],
                                bounds=(1,2))

    label = param.String(constant=True, doc="""
      A short label used to indicate what kind of data is contained
      within the AxisLayout.""")

    value = param.String(default='AxisLayout')

    def __init__(self, initial_items=None, **params):
        super(AxisLayout, self).__init__(initial_items, **params)
        if self.ndims > 2:
            raise Exception('Grids can have no more than two dimensions.')
        self._style = None


    def __mul__(self, other):
        if isinstance(other, AxisLayout):
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


    def _transform_indices(self, key):
        """
        Transforms indices by snapping to the closest value if
        values are numeric, otherwise applies no transformation.
        """
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
                keys = list({tuple(k[i] for i in dim_inds)
                             for k in self.keys()})
                q = np.array([tuple(key[i] for i in dim_inds)])
                idx = np.argmin([np.inner(q - np.array(x), q - np.array(x))
                                 if len(dim_inds) == 2 else np.abs(q-x)
                                     for x in keys])
                num_keys = iter(keys[idx])
            key = tuple(next(num_keys) if i in dim_inds else next(str_keys)
                        for i in range(self.ndims))
        elif any(not isinstance(el, slice) for el in key):
            index_ind = [idx for idx, el in enumerate(key)
                         if not isinstance(el, (slice, str))][0]
            dim_keys = np.array([k[index_ind] for k in self.keys()])
            snapped_val = dim_keys[np.argmin(dim_keys-key[index_ind])]
            key = list(key)
            key[index_ind] = snapped_val
            key = tuple(key)
        return key


    def keys(self, full_grid=False):
        """
        Returns a complete set of keys on a AxisLayout, even when AxisLayout isn't fully
        populated. This makes it easier to identify missing elements in the
        AxisLayout.
        """
        keys = super(AxisLayout, self).keys()
        if self.ndims == 1 or not full_grid:
            return keys
        dim1_keys = sorted(set(k[0] for k in keys))
        dim2_keys = sorted(set(k[1] for k in keys))
        return [(d1, d2) for d1 in dim1_keys for d2 in dim2_keys]


    @property
    def last(self):
        """
        The last of a AxisLayout is another AxisLayout
        constituted of the last of the individual elements. To access
        the elements by their X,Y position, either index the position
        directly or use the items() method.
        """

        last_items = [(k, v.clone((list(v.keys())[-1], v.last)))
                      for (k, v) in self.items()]
        return self.clone(last_items)


    @property
    def layer_types(self):
        """
        The type of layers stored in the AxisLayout.
        """
        if self.type == NdOverlay:
            return self.values()[0].layer_types
        else:
            return (self.type,)


    def __len__(self):
        """
        The maximum depth of all the elements. Matches the semantics
        of __len__ used by Maps. For the total number of elements,
        count the full set of keys.
        """
        return max([(len(v) if hasattr(v, '__len__') else 1) for v in self.values()] + [0])


    def __add__(self, obj):
        return LayoutTree.from_values(self) + LayoutTree.from_values(obj)


    @property
    def all_keys(self):
        """
        Returns a list of all keys of the elements in the grid.
        """
        keys_list = []
        for v in self.values():
            if isinstance(v, AdjointLayout):
                v = v.main
            if isinstance(v, UniformNdMapping):
                keys_list.append(list(v.data.keys()))
        return sorted(set(itertools.chain(*keys_list)))


    @property
    def common_keys(self):
        """
        Returns a list of common keys. If all elements in the AxisLayout share
        keys it will return the full set common of keys, otherwise returns
        None.
        """
        keys_list = []
        for v in self.values():
            if isinstance(v, AdjointLayout):
                v = v.main
            if isinstance(v, UniformNdMapping):
                keys_list.append(list(v.data.keys()))
        if all(x == keys_list[0] for x in keys_list):
            return keys_list[0]
        else:
            return None

    @property
    def shape(self):
        keys = self.keys()
        if self.ndims == 1:
            return (len(keys), 1)
        return len(set(k[0] for k in keys)), len(set(k[1] for k in keys))


    @property
    def xlim(self):
        xlim = list(self.values())[-1].xlim
        for data in self.values():
            xlim = find_minmax(xlim, data.xlim)
        return xlim


    @property
    def ylim(self):
        ylim = list(self.values())[-1].ylim
        for data in self.values():
            ylim = find_minmax(ylim, data.ylim)
        if ylim[0] == ylim[1]: ylim = (ylim[0], ylim[0]+1.)
        return ylim

    @property
    def extents(self):
        if self.xlim is None: return np.NaN, np.NaN, np.NaN, np.NaN
        l, r = self.xlim
        b, t = self.ylim
        return float(l), float(b), float(r), float(t)

    @property
    def grid_lbrt(self):
        grid_dimensions = []
        for dim in self._cached_index_names:
            grid_dimensions.append(self.range(dim))
        if self.ndims == 1:
            grid_dimensions.append((0, 1))
        xdim, ydim = grid_dimensions
        return (xdim[0], ydim[0], xdim[1], ydim[1])


    def dframe(self):
        """
        Gets a Pandas dframe from each of the items in the AxisLayout, appends the
        AxisLayout coordinates and concatenates all the dframes.
        """
        import pandas
        dframes = []
        for coords, vmap in self.items():
            map_frame = vmap.dframe()
            for coord, dim in zip(coords, self._cached_index_names)[::-1]:
                map_frame.insert(0, dim.replace(' ','_'), coord)
            dframes.append(map_frame)
        return pandas.concat(dframes)
