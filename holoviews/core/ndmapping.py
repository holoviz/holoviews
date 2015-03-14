"""
Supplies MultiDimensionalMapping and NdMapping which are multi-dimensional
map types. The former class only allows indexing whereas the latter
also enables slicing over multiple dimension ranges.
"""

from operator import itemgetter
import numpy as np

import param

from . import traversal
from .dimension import OrderedDict, Dimension, Dimensioned, ViewableElement
from .util import unique_iterator, allowable, dimension_sort


class MultiDimensionalMapping(Dimensioned):
    """
    An MultiDimensionalMapping is a Dimensioned mapping (like a
    dictionary or array) that uses fixed-length multidimensional
    keys. This behaves like a sparse N-dimensional array that does not
    require a dense sampling over the multidimensional space.

    If the underlying value for each (key,value) pair also supports
    indexing (such as a dictionary, array, or list), fully qualified
    (deep) indexing may be used from the top level, with the first N
    dimensions of the index selecting a particular Dimensioned object
    and the remaining dimensions indexing into that object.

    For instance, for a MultiDimensionalMapping with dimensions "Year"
    and "Month" and underlying values that are 2D floating-point
    arrays indexed by (r,c), a 2D array may be indexed with x[2000,3]
    and a single floating-point number may be indexed as
    x[2000,3,1,9].

    In practice, this class is typically only used as an abstract base
    class, because the NdMapping subclass extends it with a range of
    useful slicing methods for selecting subsets of the data. Even so,
    keeping the slicing support separate from the indexing and data
    storage methods helps make both classes easier to understand.
    """

    group = param.String(default='MultiDimensionalMapping')

    key_dimensions = param.List(default=[Dimension("Default")], constant=True)

    data_type = None          # Optional type checking of elements
    _deep_indexable = False
    _sorted = True

    def __init__(self, initial_items=None, **params):
        if isinstance(initial_items, NdMapping):
            map_type = type(initial_items)
            own_params = self.params()
            new_params = dict(initial_items.get_param_values(onlychanged=True))
            if new_params.get('group') == map_type.__name__:
                new_params.pop('group')
            params = dict({name: value for name, value in new_params.items()
                           if name in own_params}, **params)
        super(MultiDimensionalMapping, self).__init__(OrderedDict(), **params)

        self._next_ind = 0
        self._check_key_type = True
        self._cached_index_types = [d.type for d in self.key_dimensions]
        self._cached_index_values = {d.name:d.values for d in self.key_dimensions}
        self._cached_categorical = any(d.values for d in self.key_dimensions)

        self._instantiated = False
        if isinstance(initial_items, tuple):
            self._add_item(initial_items[0], initial_items[1])
        elif initial_items is not None:
            self.update(OrderedDict(initial_items))
        self._instantiated = True


    def _item_check(self, dim_vals, data):
        """
        Applies optional checks to individual data elements before
        they are inserted ensuring that they are of a certain
        type. Subclassed may implement further element restrictions.
        """
        if self.data_type is not None and not isinstance(data, self.data_type):
            if isinstance(self.data_type, tuple):
                data_type = tuple(dt.__name__ for dt in self.data_type)
            else:
                data_type = self.data_type.__name__
            raise TypeError('{slf} does not accept {data} type, data elements have '
                            'to be a {restr}.'.format(slf=type(self).__name__,
                                                      data=type(data).__name__,
                                                      restr=data_type))
        elif not len(dim_vals) == self.ndims:
            raise KeyError('Key has to match number of dimensions.')


    def _add_item(self, dim_vals, data, sort=True):
        """
        Adds item to the data, applying dimension types and ensuring
        key conforms to Dimension type and values.
        """
        if not isinstance(dim_vals, tuple):
            dim_vals = (dim_vals,)

        self._item_check(dim_vals, data)

        # Apply dimension types
        dim_types = zip(self._cached_index_types, dim_vals)
        dim_vals = tuple(v if t is None else t(v) for t, v in dim_types)

        # Check and validate for categorical dimensions
        if self._cached_categorical:
            valid_vals = zip(self._cached_index_names, dim_vals)
        else:
            valid_vals = []

        for dim, val in valid_vals:
            vals = self._cached_index_values[dim]
            if vals == 'initial': self._cached_index_values[dim] = []
            if not self._instantiated and self.get_dimension(dim).values == 'initial':
                if val not in vals:
                    self._cached_index_values[dim].append(val)
            elif vals and val not in vals:
                raise KeyError('%s Dimension value %s not in'
                               ' specified Dimension values.' % (dim, repr(val)))

        # Updates nested data structures rather than simply overriding them.
        if ((dim_vals in self.data)
            and isinstance(self.data[dim_vals], (NdMapping, OrderedDict))):
            self.data[dim_vals].update(data)
        else:
            self.data[dim_vals] = data

        if sort:
            self._resort()


    def _apply_key_type(self, keys):
        """
        If a type is specified by the corresponding key dimension,
        this method applies the type to the supplied key.
        """
        typed_key = ()
        for dim, key in zip(self.key_dimensions, keys):
            key_type = dim.type
            if key_type is None:
                typed_key += (key,)
            elif isinstance(key, slice):
                sl_vals = [key.start, key.stop, key.step]
                typed_key += (slice(*[key_type(el) if el is not None else None
                                      for el in sl_vals]),)
            elif key is Ellipsis:
                typed_key += (key,)
            elif isinstance(key, list):
                typed_key += ([key_type(k) for k in key],)
            else:
                typed_key += (key_type(key),)
        return typed_key


    def _split_index(self, key):
        """
        Partitions key into key and deep dimension groups. If only key
        indices are supplied, the data is indexed with an empty tuple.
        """
        if not isinstance(key, tuple):
            key = (key,)
        map_slice = key[:self.ndims]
        if self._check_key_type:
            map_slice = self._apply_key_type(map_slice)
        if len(key) == self.ndims:
            return map_slice, ()
        else:
            return map_slice, key[self.ndims:]


    def _dataslice(self, data, indices):
        """
        Returns slice of data element if the item is deep
        indexable. Warns if attempting to slice an object that has not
        been declared deep indexable.
        """
        if isinstance(data, Dimensioned):
            return data[indices]
        elif len(indices) > 0:
            self.warning('Cannot index into data element, extra data'
                         ' indices ignored.')
        return data


    def _resort(self):
        resorted = dimension_sort(self.data, self.key_dimensions,
                                  self._cached_categorical,
                                  self._cached_index_values)
        self.data = OrderedDict(resorted)


    def groupby(self, dimensions, container_type=None, group_type=None, **kwargs):
        """
        Splits the mapping into groups by key dimension which are then
        returned together in a mapping of class container_type. The
        individual groups are of the same type as the original map.
        """
        if self.ndims == 1:
            self.warning('Cannot split Map with only one dimension.')
            return self

        container_type = container_type if container_type else type(self)
        group_type = group_type if group_type else type(self)
        dims, inds = zip(*((self.get_dimension(dim), self.get_dimension_index(dim))
                         for dim in dimensions))
        inames, idims = zip(*((dim.name, dim) for dim in self.key_dimensions
                              if not dim.name in dimensions))
        selects = unique_iterator(itemgetter(*inds)(key) if len(inds) > 1 else (key[inds[0]],)
                                  for key in self.data.keys())
        groups = [(sel, group_type(self.select(**dict(zip(dimensions, sel))).reindex(inames), **kwargs))
                  for sel in selects]
        return container_type(groups, key_dimensions=dims)


    def add_dimension(self, dimension, dim_pos, dim_val, **kwargs):
        """
        Create a new object with an additional key dimensions along
        which items are indexed. Requires the dimension name, the
        desired position in the key_dimensions and a key value that
        will be used across the dimension. This is particularly useful
        for merging several mappings together.
        """
        if isinstance(dimension, str):
            dimension = Dimension(dimension)

        if dimension.name in self._cached_index_names:
            raise Exception('{dim} dimension already defined'.format(dim=dimension.name))

        dimensions = self.key_dimensions[:]
        dimensions.insert(dim_pos, dimension)

        items = OrderedDict()
        for key, val in self.data.items():
            new_key = list(key)
            new_key.insert(dim_pos, dim_val)
            items[tuple(new_key)] = val

        return self.clone(items, key_dimensions=dimensions, **kwargs)


    def drop_dimension(self, dim):
        """
        Returns a new mapping with the named dimension
        removed. Ensures that the dropped dimension is constant (owns
        only a single key value) before dropping it.
        """
        dim_labels = [d for d in self._cached_index_names if d != dim]
        return self.reindex(dim_labels)


    def dimension_values(self, dimension):
        "Returns the values along the specified dimension."
        all_dims = [d.name for d in self.dimensions()]
        if isinstance(dimension, int):
            dimension = all_dims[dimension]

        if dimension in self._cached_index_names:
            return [k[self.get_dimension_index(dimension)] for k in self.data.keys()]
        elif dimension in all_dims:
            values = [el.dimension_values(dimension) for el in self
                      if dimension in el.dimensions()]
            return np.concatenate(values)
        else:
            return super(MultiDimensionalMapping, self).dimension_values(dimension)


    def reindex(self, dimension_labels=[], force=False):
        """
        Create a new object with a re-ordered or reduced set of key
        dimensions.

        Reducing the number of key dimensions will discard information
        from the keys. All data values are accessible in the newly
        created object as the new labels must be sufficient to address
        each value uniquely.
        """
        if not len(dimension_labels):
            dimension_labels = [d for d in self._cached_index_names
                                if not len(set(self.dimension_values(d))) == 1]

        indices = [self.get_dimension_index(el) for el in dimension_labels]

        keys = [tuple(k[i] for i in indices) for k in self.data.keys()]
        reindexed_items = OrderedDict(
            (k, v) for (k, v) in zip(keys, self.data.values()))
        reduced_dims = set(self._cached_index_names).difference(dimension_labels)
        dimensions = [self.get_dimension(d) for d in dimension_labels if d not in reduced_dims]

        if len(set(keys)) != len(keys) and not force:
            raise Exception("Given dimension labels not sufficient to address all values uniquely")

        if len(keys):
            constant_dimensions = {self.get_dimension(d): self.dimension_values(d)[0] for d in reduced_dims}
        else:
            constant_dimensions = {}
        return self.clone(reindexed_items, key_dimensions=dimensions,
                          constant_dimensions=constant_dimensions)


    @property
    def last(self):
        "Returns the item highest data item along the map dimensions."
        return list(self.data.values())[-1] if len(self) else None


    @property
    def last_key(self):
        "Returns the last key value."
        return list(self.keys())[-1] if len(self) else None


    @property
    def info(self):
        """
        Prints information about the Dimensioned object, including the
        number and type of objects contained within it and information
        about its dimensions.
        """
        info_str = self.__class__.__name__ +\
                   " containing %d items of type %s\n" % (len(self.keys()),
                                                          type(self.values()[0]).__name__)
        info_str += ('-' * (len(info_str)-1)) + "\n\n"
        for group in self._dim_groups:
            dimensions = getattr(self, group)
            if dimensions:
                info_str += '%s Dimensions: \n' % group.capitalize()
            for d in dimensions:
                dmin, dmax = self.range(d.name)
                if d.formatter:
                    dmin, dmax = d.formatter(dmin), d.formatter(dmax)
                info_str += '\t %s: %s...%s \n' % (str(d), dmin, dmax)
        print(info_str)


    def table(self, **kwargs):
        "Creates a table from the stored keys and data."

        table = None
        for key, value in self.data.items():
            value = value.table(**kwargs)
            for idx, (dim, val) in enumerate(zip(self.key_dimensions, key)):
                value = value.add_dimension(dim, idx, val)
            if table is None:
                table = value
            else:
                table.update(value)
        return table


    def dframe(self):
        "Creates a pandas DataFrame from the stored keys and data."
        try:
            import pandas
        except ImportError:
            raise Exception("Cannot build a DataFrame without the pandas library.")
        labels = self._cached_index_names + [self.group]
        return pandas.DataFrame(
            [dict(zip(labels, k + (v,))) for (k, v) in self.data.items()])


    def update(self, other):
        """
        Updates the current mapping with some other mapping or
        OrderedDict instance, making sure that they are indexed along
        the same set of dimensions. The order of key_dimensions
        remains unchanged after the update.
        """
        if isinstance(other, NdMapping):
            if self.key_dimensions != other.key_dimensions:
                raise KeyError("Cannot update with NdMapping that has"
                               " a different set of key dimensions.")
        for key, data in other.items():
            self._add_item(key, data, sort=False)
        self._resort()


    def keys(self):
        " Returns the keys of all the elements."
        if self.ndims == 1:
            return [k[0] for k in self.data.keys()]
        else:
            return list(self.data.keys())


    def values(self):
        " Returns the values of all the elements."
        return list(self.data.values())


    def items(self):
        "Returns all elements as a list in (key,value) format."
        return list(zip(list(self.keys()), list(self.values())))


    def get(self, key, default=None):
        "Standard get semantics for all mapping types"
        try:
            if key is None:
                return None
            return self[key]
        except:
            return default


    def pop(self, key, default=None):
        "Standard pop semantics for all mapping types"
        if not isinstance(key, tuple): key = (key,)
        return self.data.pop(key, default)


    def __getitem__(self, key):
        """
        Allows multi-dimensional indexing in the order of the
        specified key dimensions, passing any additional indices to
        the data elements.
        """
        if key in [Ellipsis, ()]:
            return self
        map_slice, data_slice = self._split_index(key)
        return self._dataslice(self.data[map_slice], data_slice)


    def __setitem__(self, key, value):
        self._add_item(key, value)


    def __str__(self):
        return repr(self)


    def __iter__(self):
        return iter(self.values())


    def __contains__(self, key):
        if self.ndims == 1:
            return key in self.data.keys()
        else:
            return key in self.keys()

    def __len__(self):
        return len(self.data)




class NdMapping(MultiDimensionalMapping):
    """
    NdMapping supports the same indexing semantics as
    MultiDimensionalMapping but also supports slicing semantics.

    Slicing semantics on an NdMapping is dependent on the ordering
    semantics of the keys. As MultiDimensionalMapping sort the keys, a
    slice on an NdMapping is effectively a way of filtering out the
    keys that are outside the slice range.
    """

    group = param.String(default='NdMapping')

    def __getitem__(self, indexslice):
        """
        Allows slicing operations along the key and data
        dimensions. If no data slice is supplied it will return all
        data elements, otherwise it will return the requested slice of
        the data.
        """
        if indexslice in [Ellipsis, ()]:
            return self

        map_slice, data_slice = self._split_index(indexslice)
        map_slice = self._transform_indices(map_slice)
        map_slice = self._expand_slice(map_slice)

        if all(not isinstance(el, (slice, set, list, tuple)) for el in map_slice):
            return self._dataslice(self.data[map_slice], data_slice)
        else:
            conditions = self._generate_conditions(map_slice)
            items = self.data.items()
            for cidx, (condition, dim) in enumerate(zip(conditions, self.key_dimensions)):
                values = self._cached_index_values.get(dim.name, None)
                items = [(k, v) for k, v in items
                         if condition(values.index(k[cidx]) if values else k[cidx])]
            items = [(k, self._dataslice(v, data_slice)) for k, v in items]
            if self.ndims == 1:
                items = [(k[0], v) for (k, v) in items]
            if len(items) == 0:
                raise KeyError('No items within specified slice.')
            return self.clone(items)


    def _expand_slice(self, indices):
        """
        Expands slices containing steps into a list.
        """
        keys = list(self.data.keys())
        expanded = []
        for idx, ind in enumerate(indices):
            if isinstance(ind, slice) and ind.step is not None:
                dim_ind = slice(ind.start, ind.stop)
                if dim_ind == slice(None):
                    condition = self._all_condition()
                elif dim_ind.start is None:
                    condition = self._upto_condition(dim_ind)
                elif dim_ind.stop is None:
                    condition = self._from_condition(dim_ind)
                else:
                    condition = self._range_condition(dim_ind)
                dim_vals = unique_iterator(k[idx] for k in keys)
                expanded.append(set([k for k in dim_vals if condition(k)][::int(ind.step)]))
            else:
                expanded.append(ind)
        return tuple(expanded)


    def _transform_indices(self, indices):
        """
        Identity function here but subclasses can implement transforms
        of the dimension indices from one coordinate system to another.
        """
        return indices


    def _generate_conditions(self, map_slice):
        """
        Generates filter conditions used for slicing the data structure.
        """
        conditions = []
        for dim, dim_slice in zip(self.key_dimensions, map_slice):
            if isinstance(dim_slice, slice):
                start, stop = dim_slice.start, dim_slice.stop
                if dim.values:
                    values = self._cached_index_values[dim.name]
                    dim_slice = slice(None if start is None else values.index(start),
                                      None if stop is None else values.index(stop))
                if dim_slice == slice(None):
                    conditions.append(self._all_condition())
                elif start is None:
                    conditions.append(self._upto_condition(dim_slice))
                elif stop is None:
                    conditions.append(self._from_condition(dim_slice))
                else:
                    conditions.append(self._range_condition(dim_slice))
            elif isinstance(dim_slice, set):
                if dim.values:
                    dim_slice = [self._cached_index_values[dim.name].index(dim_val)
                                 for dim_val in dim_slice]
                conditions.append(self._values_condition(dim_slice))
            elif dim_slice is Ellipsis:
                conditions.append(self._all_condition())
            elif isinstance(dim_slice, (list, tuple)):
                raise ValueError("Keys may only be selected with sets, not lists or tuples.")
            else:
                if dim.values:
                    dim_slice = self._cached_index_values[dim.name].index(dim_slice)
                conditions.append(self._value_condition(dim_slice))
        return conditions


    def _value_condition(self, value):
        return lambda x: x == value


    def _values_condition(self, values):
        return lambda x: x in values


    def _range_condition(self, slice):
        if slice.step is None:
            lmbd = lambda x: slice.start <= x < slice.stop
        else:
            lmbd = lambda x: slice.start <= x < slice.stop and not (
                (x-slice.start) % slice.step)
        return lmbd


    def _upto_condition(self, slice):
        if slice.step is None:
            lmbd = lambda x: x < slice.stop
        else:
            lmbd = lambda x: x < slice.stop and not (x % slice.step)
        return lmbd


    def _from_condition(self, slice):
        if slice.step is None:
            lmbd = lambda x: x > slice.start
        else:
            lmbd = lambda x: x > slice.start and ((x-slice.start) % slice.step)
        return lmbd

    def _all_condition(self):
        return lambda x: True


class UniformNdMapping(NdMapping):
    """
    A UniformNdMapping is a map of Dimensioned objects and is itself
    indexed over a number of specified dimensions. The dimension may
    be a spatial dimension (i.e., a ZStack), time (specifying a frame
    sequence) or any other combination of Dimensions.

    UniformNdMapping objects can be sliced, sampled, reduced, overlaid
    and split along its and its containing Views
    dimensions. Subclasses should implement the appropriate slicing,
    sampling and reduction methods for their Dimensioned type.
    """

    data_type = (ViewableElement, NdMapping)

    _abstract = True
    _deep_indexable = True

    def __init__(self, initial_items=None, group=None, label=None, **params):
        self._type = None
        self._group_check, self.group = None, group
        self._label_check, self.label = None, label
        super(UniformNdMapping, self).__init__(initial_items, **params)


    def relabel(self, label=None, group=None):
        """
        Relabels the UniformNdMapping and all it's Elements
        with the supplied group and label.
        """
        return self.clone([(k, v.relabel(label, group)) for k, v in self.items()],
                          group=group if group else self.group,
                          label=self.label if label is None else label)

    @property
    def group(self):
        if self._group:
            return self._group
        elif self._group_check and self._group_check != self.type.__name__:
            return self._group_check
        else:
            return type(self).__name__

    @group.setter
    def group(self, group):
        if group is not None and not allowable(group):
            raise ValueError("Supplied group %s contains invalid "
                             "characters." % self.group)
        self._group = group

    @property
    def label(self):
        if self._label:
            return self._label
        elif self._label_check:
            return self._label_check
        else:
            return ''

    @label.setter
    def label(self, label):
        if label is not None and not allowable(label):
            raise ValueError("Supplied group %s contains invalid "
                             "characters." % self.group)
        self._label = label

    @property
    def type(self):
        """
        The type of elements stored in the map.
        """
        if self._type is None and len(self):
            self._type = self.values()[0].__class__
        return self._type

    @property
    def empty_element(self):
        return self._type(None)


    def _item_check(self, dim_vals, data):
        if self.type is not None and (type(data) != self.type):
            raise AssertionError("%s must only contain one type of ViewableElement." %
                                 self.__class__.__name__)
        if self._group is None:
            self._group_check = data.group
        if self._label is None:
            self._label_check = data.label
        elif self._group_check and data.group != self._group_check:
            raise ValueError("Elements in %s need to have uniform values.")
        elif self._label_check and data.label != self._label_check:
            raise ValueError("Elements in %s need to have uniform labels.")

        if not traversal.uniform(NdMapping([(0, self), (1, data)])):
            raise ValueError("HoloMaps dimensions must be consistent in %s." %
                             type(self).__name__)
        super(UniformNdMapping, self)._item_check(dim_vals, data)
