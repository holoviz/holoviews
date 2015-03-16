from itertools import groupby
from numbers import Number
import numpy as np

import param

from .dimension import Dimension, Dimensioned, ViewableElement
from .layout import Composable, Layout, AdjointLayout, NdLayout
from .ndmapping import OrderedDict, UniformNdMapping, NdMapping
from .overlay import Overlayable, NdOverlay, Overlay, CompositeOverlay
from .tree import AttrTree
from .util import find_minmax


class Element(ViewableElement, Composable, Overlayable):
    """
    Element is the baseclass for all ViewableElement types, with an x- and
    y-dimension. Subclasses should define the data storage in the
    constructor, as well as methods and properties, which define how
    the data maps onto the x- and y- and value dimensions.
    """

    group = param.String(default='Element')

    def hist(self, dimension=None, num_bins=20, bin_range=None,
             adjoin=True, individually=True, **kwargs):
        """
        The hist method generates a histogram to be adjoined to the
        Element in an AdjointLayout. By default the histogram is
        computed along the first value dimension of the Element,
        however any dimension may be selected. The number of bins and
        the bin_ranges and any kwargs to be passed to the histogram
        operation may also be supplied.
        """
        from ..operation import histogram
        return histogram(self, num_bins=num_bins, bin_range=bin_range, adjoin=adjoin,
                         individually=individually, dimension=dimension, **kwargs)


    #======================#
    # Subclassable methods #
    #======================#

    def __init__(self, data, **params):
        super(Element, self).__init__(data, **params)


    def __getitem__(self, key):
        if key is ():
            return self
        else:
            raise NotImplementedError("%s currently does not support getitem" %
                                      type(self).__name__)


    @classmethod
    def collapse_data(cls, data, function=None, **kwargs):
        """
        Class method to collapse a list of data matching the
        data format of the Element type. By implementing this
        method HoloMap can collapse multiple Elements of the
        same type. The kwargs are passed to the collapse
        function. The collapse function must support the numpy
        style axis selection. Valid function include:
        np.mean, np.sum, np.product, np.std,
        scipy.stats.kurtosis etc.
        """
        raise NotImplementedError("Collapsing not implemented for %s." % cls.__name__)


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


    def reduce(self, **reduce_map):
        """
        Base class signature to demonstrate API for reducing Elements,
        using some reduce function, e.g. np.mean, which is applied
        along a particular Dimension. The dimensions and reduce functions
        should be passed as keyword arguments.
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
                      label=self.label, group=self.group, **kwargs)
        return Table(zip(keys, values), **params)


    def dframe(self):
        import pandas
        column_names = self.dimensions(label=True)
        dim_vals = np.vstack([self.dimension_values(dim) for dim in column_names]).T
        return pandas.DataFrame(dim_vals, columns=column_names)


    def __call__(self, **kwargs):
        """
        Sets the options in the same way as applicable to all
        Dimensioned objects except there is no need to specify the
        match key at the level of an individual Element (there is only
        one possible match).
        """
        identifier = '%s.%s' % (self.__class__.__name__, self.group)
        identifier += ('.%s' % self.label) if self.label else ''
        options = {identifier:{}}
        for k,d in kwargs.items():
            options[identifier][k] = d
        return super(Element, self).__call__(options)



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
        if self._xlim and None not in self._xlim:
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
        if self._ylim and None not in self._ylim:
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
        The extent on Element2D types and its subclasses are a tuple
        of all the dimension lower bounds followed by the upper bounds.
        In case of a simple Element2D this takes the form (l, b, r, t).
        """
        lbound, ubound = [], []
        for kd in self._cached_index_names:
            drange = self.range(kd)
            lower, upper = drange if drange else (np.NaN, np.NaN)
            lbound.append(lower)
            ubound.append(upper)
        return tuple(lbound+ubound)


    @extents.setter
    def extents(self, extents):
        l, b, r, t = extents
        self.xlim, self.ylim = (l, r), (b, t)


class NdElement(Element, NdMapping):
    """
    An NdElement is an Element that stores the contained data as
    an NdMapping. In addition to the usual multi-dimensional keys
    of NdMappings, NdElements also support multi-dimensional
    values. The values held in a multi-valued NdElement are tuples,
    where each component of the tuple maps to a column as described
    by the value_dimensions parameter.

    In other words, the data of a NdElement are partitioned into two
    groups: the columns based on the key and the value columns that
    contain the components of the value tuple.

    One feature of NdElements is that they support an additional level of
    index over NdMappings: the last index may be a column name or a
    slice over the column names (using alphanumeric ordering).
    """

    group = param.String(default='NdElement', doc="""
         The group is used to describe the NdElement.""")

    value_dimensions = param.List(default=[Dimension('Data')], doc="""
        The dimension description(s) of the values held in data tuples
        that map to the value columns of the table.

        Note: String values may be supplied in the constructor which
        will then be promoted to Dimension objects.""")

    _deep_indexable = False

    def __init__(self, data=None, **params):
        NdMapping.__init__(self, data, **dict(params, group=params.get('group',self.group)))
        for k, v in self.data.items():
            self[k] = v # Validates input


    def __setitem__(self, key, value):
        value = (value,) if np.isscalar(value) else tuple(value)
        key = key if isinstance(key, tuple) else (key,)
        self.data[key] = value


    def _filter_columns(self, index, col_names):
        "Returns the column names specified by index (which may be a slice)"
        if isinstance(index, slice):
            cols  = [col for col in sorted(col_names)]
            if index.start:
                cols = [col for col in cols if col > index.start]
            if index.stop:
                cols = [col for col in cols if col < index.stop]
            cols = cols[::index.step] if index.step else cols
        elif isinstance(index, (set, list)):
            nomatch = [val for val in index if val not in col_names]
            if nomatch:
                raise KeyError("No columns with dimension labels %r" % nomatch)
            cols = [col for col in col_names if col in index]
        elif index not in col_names:
            raise KeyError("No column with dimension label %r" % index)
        else:
            cols= [index]
        if cols==[]:
            raise KeyError("No columns selected in the given slice")
        return cols


    def _filter_data(self, subtable, value_dimensions):
        col_names = self.dimensions('value', label=True)
        cols = self._filter_columns(value_dimensions, col_names)
        indices = [col_names.index(col) for col in cols]
        value_dimensions = [self.value_dimensions[i] for i in indices]
        items = [(k, tuple(v[i] for i in indices)) for (k,v) in subtable.items()]
        return subtable.clone(items, value_dimensions=value_dimensions)


    def __getitem__(self, args):
        """
        In addition to usual NdMapping indexing, NdElements can be indexed
        by column name (or a slice over column names)
        """
        ndmap_index = args[:self.ndims] if isinstance(args, tuple) else args
        subtable = NdMapping.__getitem__(self, ndmap_index)

        if len(self.value_dimensions) > 1 and not isinstance(subtable, NdElement):
            # If a value tuple, turn into an ItemTable
            subtable = self.__class__([((), subtable)], label=self.label,
                                      key_dimensions=[],
                                      value_dimensions=self.value_dimensions)

        if not isinstance(args, tuple) or len(args) <= self.ndims:
            return subtable

        return self._filter_data(subtable, args[-1])


    def select(self, **selection):
        val_selection = selection.pop('value', None)
        selection = NdMapping.select(self, **selection)
        if val_selection:
            return self._filter_data(selection, val_selection)
        else:
            return selection


    def sample(self, samples=[]):
        """
        Allows sampling of the Table with a list of samples.
        """
        sample_data = OrderedDict()
        for sample in samples:
            sample_data[sample] = self[sample]
        return self.__class__(sample_data, **dict(self.get_param_values(onlychanged=True)))


    def reduce(self, dimensions=None, function=None, **reduce_map):
        """
        Allows collapsing the Table down by dimension by passing
        the dimension name and reduce_fn as kwargs. Reduces
        dimensionality of Table until only an ItemTable is left.
        """
        dimensions = self._valid_dimensions(dimensions)
        if dimensions and reduce_map:
            raise Exception("Pass reduced dimensions either as an argument"
                            "or as part of the kwargs not both.")
        elif dimensions:
            reduce_map = {d: function for d in dimensions}
        dim_labels = self._cached_index_names
        reduced_table = self
        for reduce_fn, group in groupby(reduce_map.items(), lambda x: x[1]):
            dims = [dim for dim, _ in group]
            split_dims = [self.get_dimension(d) for d in dim_labels if d not in dims]
            if len(split_dims) and reduced_table.ndims > 1:
                split_map = reduced_table.groupby([d.name for d in split_dims], container_type=HoloMap,
                                                  group_type=self.__class__)
                reduced_table = self.clone(shared_data=False, key_dimensions=split_dims)
                for k, table in split_map.items():
                    reduced = []
                    for vdim in self.value_dimensions:
                        valtable = table.select(value=vdim.name) if len(self.value_dimensions) > 1 else table
                        reduced.append(reduce_fn(valtable.data.values()))
                    reduced_table[k] = reduced
            else:
                reduced = tuple(reduce_fn(self.dimension_values(vdim.name))
                                for vdim in self.value_dimensions)
                params = dict(group=self.group) if self.value != type(self).__name__ else {}
                reduced_table = self.__class__([((), reduced)], label=self.label,
                                               value_dimensions=self.value_dimensions, **params)
        return reduced_table


    def _item_check(self, dim_vals, data):
        if isinstance(data, tuple):
            for el in data:
                self._item_check(dim_vals, el)
            return
        super(NdElement, self)._item_check(dim_vals, data)


    @classmethod
    def collapse_data(cls, data, function, **kwargs):
        groups = zip(*[(np.array(values) for values in odict.values()) for odict in data])
        return OrderedDict((key, np.squeeze(function(np.dstack(group), axis=-1, **kwargs), 0)
                                  if group[0].shape[0] > 1 else
                                  function(np.concatenate(group), **kwargs))
                             for key, group in zip(data[0].keys(), groups))


    def dimension_values(self, dim):
        if isinstance(dim, Dimension):
            raise Exception('Dimension to be specified by name')
        value_dims = self.dimensions('value', label=True)
        if dim in value_dims:
            index = value_dims.index(dim)
            return [v[index] for v in self.values()]
        else:
            return NdMapping.dimension_values(self, dim)


    def dframe(self, value_label='data'):
        try:
            import pandas
        except ImportError:
            raise Exception("Cannot build a DataFrame without the pandas library.")
        labels = [d.name for d in self.dimensions()]
        return pandas.DataFrame(
            [dict(zip(labels, np.concatenate([np.array(k),v])))
             for (k, v) in self.data.items()])



class Element3D(Element2D):


    def __init__(self, data, extents=None, **params):
        if extents is not None:
            self._zlim = (extents[2], extents[5])
            extents = (extents[0], extents[1], extents[3], extents[4])
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
        if self._zlim and None not in self._zlim:
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

    data_type = (ViewableElement, UniformNdMapping)

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
    def zlim(self):
        if not isinstance(self.last, Element3D):
            return (None, None)
        zlim = self.last.zlim
        for data in self.values():
            zlim = find_minmax(zlim, data.zlim) if data.zlim and zlim else zlim
        return zlim

    @property
    def extents(self):
        if self.xlim is None: return np.NaN, np.NaN, np.NaN, np.NaN
        l, r = self.xlim
        b, t = self.ylim
        if isinstance(self.last, Element3D):
            zmin, zmax = self.zlim
            return float(l), float(b), float(zmin), float(r), float(t), float(zmax)
        else:
            return float(l), float(b), float(r), float(t)


    def overlay(self, dimensions, **kwargs):
        """
        Splits the UniformNdMapping along a specified number of dimensions and
        overlays items in the split out Maps.
        """
        dimensions = self._valid_dimensions(dimensions)
        if len(dimensions) == self.ndims:
            return NdOverlay(self, **kwargs)
        else:
            dims = [d for d in self._cached_index_names
                    if d not in dimensions]
            return self.groupby(dims, group_type=NdOverlay, **kwargs)


    def grid(self, dimensions, **kwargs):
        """
        GridSpace takes a list of one or two dimensions, and lays out the containing
        Views along these axes in a GridSpace.
        """
        dimensions = self._valid_dimensions(dimensions)
        if len(dimensions) == self.ndims:
            return GridSpace(self, **kwargs)
        return self.groupby(dimensions, container_type=GridSpace, **kwargs)


    def layout(self, dimensions, **kwargs):
        """
        GridSpace takes a list of one or two dimensions, and lays out the containing
        Views along these axes in a GridSpace.
        """
        dimensions = self._valid_dimensions(dimensions)
        if len(dimensions) == self.ndims:
            return NdLayout(self, **kwargs)
        return self.groupby(dimensions, container_type=NdLayout, **kwargs)


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
            return self.clone(items, key_dimensions=dimensions, label=self._label, group=self._group)
        elif isinstance(other, self.data_type):
            items = [(k, v * other) for (k, v) in self.items()]
            return self.clone(items, label=self._label, group=self._group)
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
        return Layout.from_values(self) + Layout.from_values(obj)


    def __lshift__(self, other):
        if isinstance(other, (ViewableElement, UniformNdMapping)):
            return AdjointLayout([self, other])
        elif isinstance(other, AdjointLayout):
            return AdjointLayout(other.data+[self])
        else:
            raise TypeError('Cannot append {0} to a AdjointLayout'.format(type(other).__name__))


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
        from ..element import Table, ItemTable
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
                xsamples = [(lx+ux)/2.0 for lx,ux in zip(xedges[:-1], xedges[1:])]
                ysamples = [(ly+uy)/2.0 for ly,uy in zip(yedges[:-1], yedges[1:])]

                X,Y = np.meshgrid(xsamples, ysamples)
                linsamples = zip(X.flat, Y.flat)
            else:
                raise NotImplementedError("Regular sampling not implented"
                                          "for high-dimensional Views.")

            samples = set(self.last.closest(linsamples))

        sampled = self.clone([(k, view.sample(samples, **sample_values))
                              for k, view in self.items()])
        return sampled.table().reindex() if sampled.type in [ItemTable, Table] else sampled.table()


    def reduce(self, dimensions=None, function=None, **reduce_map):
        """
        Reduce each Element in the HoloMap using a function supplied
        via the kwargs, where the keyword has to match a particular
        dimension in the Elements.
        """
        reduced_items = [(k, v.reduce(dimensions, function, **reduce_map))
                         for k, v in self.items()]
        return self.clone(reduced_items).table()


    @property
    def empty_element(self):
        return self._type(None)


    def hist(self, num_bins=20, bin_range=None, adjoin=True, individually=True, **kwargs):
        histmap = self.clone(shared_data=False)

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



class Collator(NdMapping):
    """
    Collator is an NdMapping type which can merge any number
    of HoloViews components with whatever level of nesting
    by inserting the Collators key_dimensions on the HoloMaps.
    If the items in the Collator do not contain HoloMaps
    they will be created. Collator also supports filtering
    of Tree structures and dropping of constant_dimensions.

    Collator can also be subclassed to dynamically load data
    from different locations. This only requires subclassing
    the _process_data method, which should return the data
    to be merged, e.g. the Collator may contain a number of
    filenames as values, which the _process_data can
    dynamically load (and then merge) during the call.
    """

    drop = param.List(default=[], doc="""
        List of dimensions to drop when collating data, specified
        as strings.""")

    progress_bar = param.Parameter(default=None, doc="""
         The progress bar instance used to report progress. Set to
         None to disable progress bars.""")

    _deep_indexable = False

    def __call__(self, path_filters=[], merge=True):
        """
        Filter each Layout in the Collator with the supplied
        path_filters. If merge is set to True all Layouts are
        merged, otherwise an NdMapping containing all the
        Layouts is returned. Optionally a list of dimensions
        to be ignored can be supplied.
        """
        constant_dims = self.static_dimensions
        ndmapping = NdMapping(key_dimensions=self.key_dimensions)

        num_elements = len(self)
        for idx, (key, data) in enumerate(self.data.items()):
            if isinstance(data, AttrTree):
                data = data.filter(path_filters)
            data = self._process_data(data)

            if merge:
                dim_keys = zip(self._cached_index_names, key)
                varying_keys = [(d, k) for d, k in dim_keys
                                if d not in constant_dims]
                constant_keys = [(d, k) for d, k in dim_keys
                                 if d in constant_dims]
                data = self._add_dimensions(data, varying_keys,
                                            dict(constant_keys))
            ndmapping[key] = data
            if self.progress_bar is not None:
                self.progress_bar(float(idx+1)/num_elements*100)

        if merge:
            components = ndmapping.values()
            accumulator = ndmapping.last.clone(components[0].data)
            for component in components:
                accumulator.update(component)
            return accumulator
        return ndmapping


    @property
    def static_dimensions(self):
        """
        Return all constant dimensions.
        """
        dimensions = []
        for dim in self.key_dimensions:
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
        dimensions, key = zip(*dim_vals)
        if isinstance(item, HoloMap):
            new_item = item.clone(constant_dimensions=constant_keys)
            for dim, val in dim_vals:
                if dim not in [d.name for d in new_item.key_dimensions]:
                    new_item = new_item.add_dimension(dim, 0, val)
                else:
                    raise ValueError("Items already contain dimensions %s "
                                     "and cannot be collated.")
        elif isinstance(item, ViewableElement):
            new_item = HoloMap({key: item}, key_dimensions=dimensions,
                               constant_dimensions=constant_keys)
        else:
            new_item = item.clone(shared_data=False, constant_dimensions=constant_keys)
            for k, v in item.items():
                new_item[k] = self._add_dimensions(v, dims, constant_keys)
        if isinstance(new_item, Layout):
            new_item.fixed = True

        return new_item


    def _process_data(self, data):
        """"
        Subclassable to apply some processing to the data elements
        before filtering and merging them.
        """
        return data



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

    key_dimensions = param.List(default=[Dimension(name="X"), Dimension(name="Y")],
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
            index_ind = [idx for idx, el in enumerate(key)
                         if not isinstance(el, (slice, str))][0]
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
                          for (k, v) in self.items()]
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
        Gets a Pandas dframe from each of the items in the GridSpace, appends the
        GridSpace coordinates and concatenates all the dframes.
        """
        import pandas
        dframes = []
        for coords, vmap in self.items():
            map_frame = vmap.dframe()
            for coord, dim in zip(coords, self._cached_index_names)[::-1]:
                map_frame.insert(0, dim.replace(' ','_'), coord)
            dframes.append(map_frame)
        return pandas.concat(dframes)

__all__ = list(set([_k for _k, _v in locals().items()
                    if isinstance(_v, type) and issubclass(_v, Dimensioned)]))
