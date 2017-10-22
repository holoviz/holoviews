"""
Supplies MultiDimensionalMapping and NdMapping which are multi-dimensional
map types. The former class only allows indexing whereas the latter
also enables slicing over multiple dimension ranges.
"""

from itertools import cycle
from operator import itemgetter
import numpy as np

import param

from . import util
from .dimension import OrderedDict, Dimension, Dimensioned, ViewableElement
from .util import (unique_iterator, sanitize_identifier, dimension_sort,
                   basestring, wrap_tuple, process_ellipses, get_ndmapping_label, pd)


class item_check(object):
    """
    Context manager to allow creating NdMapping types without
    performing the usual item_checks, providing significant
    speedups when there are a lot of items. Should only be
    used when both keys and values are guaranteed to be the
    right type, as is the case for many internal operations.
    """

    def __init__(self, enabled):
        self.enabled = enabled

    def __enter__(self):
        self._enabled = MultiDimensionalMapping._check_items
        MultiDimensionalMapping._check_items = self.enabled

    def __exit__(self, exc_type, exc_val, exc_tb):
        MultiDimensionalMapping._check_items = self._enabled


class sorted_context(object):
    """
    Context manager to temporarily disable sorting on NdMapping
    types. Retains the current sort order, which can be useful as
    an optimization on NdMapping instances where sort=True but the
    items are already known to have been sorted.
    """

    def __init__(self, enabled):
        self.enabled = enabled

    def __enter__(self):
        self._enabled = MultiDimensionalMapping.sort
        MultiDimensionalMapping.sort = self.enabled

    def __exit__(self, exc_type, exc_val, exc_tb):
        MultiDimensionalMapping.sort = self._enabled



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

    group = param.String(default='MultiDimensionalMapping', constant=True)

    kdims = param.List(default=[Dimension("Default")], constant=True)

    vdims = param.List(default=[], bounds=(0, 0), constant=True)

    sort = param.Boolean(default=True, doc="""
        Whether the items should be sorted in the constructor.""")

    data_type = None          # Optional type checking of elements
    _deep_indexable = False
    _check_items = True

    def __init__(self, initial_items=None, kdims=None, **params):
        if isinstance(initial_items, MultiDimensionalMapping):
            params = dict(util.get_param_values(initial_items),
                          **dict({'sort': self.sort}, **params))
        if kdims is not None:
            params['kdims'] = kdims
        super(MultiDimensionalMapping, self).__init__(OrderedDict(), **dict(params))
        if type(initial_items) is dict and not self.sort:
            raise ValueError('If sort=False the data must define a fixed '
                             'ordering, please supply a list of items or '
                             'an OrderedDict, not a regular dictionary.')

        self._next_ind = 0
        self._check_key_type = True
        self._cached_index_types = [d.type for d in self.kdims]
        self._cached_index_values = {d.name:d.values for d in self.kdims}
        self._cached_categorical = any(d.values for d in self.kdims)

        if initial_items is None: initial_items = []
        if isinstance(initial_items, tuple):
            self._add_item(initial_items[0], initial_items[1])
        elif not self._check_items:
            if isinstance(initial_items, dict):
                initial_items = initial_items.items()
            elif isinstance(initial_items, MultiDimensionalMapping):
                initial_items = initial_items.data.items()
            self.data = OrderedDict((k if isinstance(k, tuple) else (k,), v)
                                    for k, v in initial_items)
            if self.sort:
                self._resort()
        elif initial_items is not None:
            self.update(OrderedDict(initial_items))


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


    def _add_item(self, dim_vals, data, sort=True, update=True):
        """
        Adds item to the data, applying dimension types and ensuring
        key conforms to Dimension type and values.
        """
        sort = sort and self.sort
        if not isinstance(dim_vals, tuple):
            dim_vals = (dim_vals,)

        self._item_check(dim_vals, data)

        # Apply dimension types
        dim_types = zip(self._cached_index_types, dim_vals)
        dim_vals = tuple(v if None in [t, v] else t(v) for t, v in dim_types)

        # Check and validate for categorical dimensions
        if self._cached_categorical:
            valid_vals = zip(self.kdims, dim_vals)
        else:
            valid_vals = []

        for dim, val in valid_vals:
            vals = self._cached_index_values[dim.name]
            if vals and val is not None and val not in vals:
                raise KeyError('%s dimension value %s not in'
                               ' specified dimension values.' % (dim, repr(val)))

        # Updates nested data structures rather than simply overriding them.
        if (update and (dim_vals in self.data)
            and isinstance(self.data[dim_vals], (MultiDimensionalMapping, OrderedDict))):
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
        for dim, key in zip(self.kdims, keys):
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
        Keys with indices than there are dimensions will be padded.
        """
        if not isinstance(key, tuple):
            key = (key,)
        elif key == ():
            return (), ()

        if key[0] is Ellipsis:
            num_pad = self.ndims - len(key) + 1
            key = (slice(None),) * num_pad + key[1:]
        elif len(key) < self.ndims:
            num_pad = self.ndims - len(key)
            key = key + (slice(None),) * num_pad

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
        if self._deep_indexable and isinstance(data, Dimensioned) and indices:
            return data[indices]
        elif len(indices) > 0:
            self.warning('Cannot index into data element, extra data'
                         ' indices ignored.')
        return data


    def _resort(self):
        resorted = dimension_sort(self.data, self.kdims, self.vdims,
                                  self._cached_categorical,
                                  range(self.ndims),
                                  self._cached_index_values)
        self.data = OrderedDict(resorted)


    def clone(self, data=None, shared_data=True, *args, **overrides):
        """
        Overrides Dimensioned clone to avoid checking items if data
        is unchanged.
        """
        with item_check(not shared_data and self._check_items):
            return super(MultiDimensionalMapping, self).clone(data, shared_data,
                                                              *args, **overrides)


    def groupby(self, dimensions, container_type=None, group_type=None, **kwargs):
        """
        Splits the mapping into groups by key dimension which are then
        returned together in a mapping of class container_type. The
        individual groups are of the same type as the original map.
        This operation will always sort the groups and the items in
        each group.
        """
        if self.ndims == 1:
            self.warning('Cannot split Map with only one dimension.')
            return self
        container_type = container_type if container_type else type(self)
        group_type = group_type if group_type else type(self)
        dimensions = [self.get_dimension(d, strict=True) for d in dimensions]
        with item_check(False):
            return util.ndmapping_groupby(self, dimensions, container_type,
                                          group_type, sort=True, **kwargs)


    def add_dimension(self, dimension, dim_pos, dim_val, vdim=False, **kwargs):
        """
        Create a new object with an additional key dimensions.
        Requires the dimension name or object, the desired position
        in the key dimensions and a key value scalar or sequence of
        the same length as the existing keys.
        """
        if not isinstance(dimension, Dimension):
            dimension = Dimension(dimension)

        if dimension in self.dimensions():
            raise Exception('{dim} dimension already defined'.format(dim=dimension.name))

        if vdim and self._deep_indexable:
            raise Exception('Cannot add value dimension to object that is deep indexable')

        if vdim:
            dims = self.vdims[:]
            dims.insert(dim_pos, dimension)
            dimensions = dict(vdims=dims)
            dim_pos += self.ndims
        else:
            dims = self.kdims[:]
            dims.insert(dim_pos, dimension)
            dimensions = dict(kdims=dims)

        if isinstance(dim_val, basestring) or not hasattr(dim_val, '__iter__'):
            dim_val = cycle([dim_val])
        else:
            if not len(dim_val) == len(self):
                raise ValueError("Added dimension values must be same length"
                                 "as existing keys.")

        items = OrderedDict()
        for dval, (key, val) in zip(dim_val, self.data.items()):
            if vdim:
                new_val = list(val)
                new_val.insert(dim_pos, dval)
                items[key] = tuple(new_val)
            else:
                new_key = list(key)
                new_key.insert(dim_pos, dval)
                items[tuple(new_key)] = val

        return self.clone(items, **dict(dimensions, **kwargs))


    def drop_dimension(self, dimensions):
        """
        Returns a new mapping with the named dimension(s) removed.
        """
        dimensions = [dimensions] if np.isscalar(dimensions) else dimensions
        dims = [d for d in self.kdims if d not in dimensions]
        dim_inds = [self.get_dimension_index(d) for d in dims]
        key_getter = itemgetter(*dim_inds)
        return self.clone([(key_getter(k), v) for k, v in self.data.items()],
                          kdims=dims)


    def dimension_values(self, dimension, expanded=True, flat=True):
        "Returns the values along the specified dimension."
        dimension = self.get_dimension(dimension, strict=True)
        if dimension in self.kdims:
            return np.array([k[self.get_dimension_index(dimension)] for k in self.data.keys()])
        if dimension in self.dimensions():
            values = [el.dimension_values(dimension) for el in self
                      if dimension in el.dimensions()]
            vals = np.concatenate(values)
            return vals if expanded else util.unique_array(vals)
        else:
            return super(MultiDimensionalMapping, self).dimension_values(dimension, expanded, flat)


    def reindex(self, kdims=[], force=False):
        """
        Create a new object with a re-ordered or reduced set of key
        dimensions.

        Reducing the number of key dimensions will discard information
        from the keys. All data values are accessible in the newly
        created object as the new labels must be sufficient to address
        each value uniquely.
        """
        old_kdims = [d.name for d in self.kdims]
        if not len(kdims):
            kdims = [d for d in old_kdims
                     if not len(set(self.dimension_values(d))) == 1]
        indices = [self.get_dimension_index(el) for el in kdims]

        keys = [tuple(k[i] for i in indices) for k in self.data.keys()]
        reindexed_items = OrderedDict(
            (k, v) for (k, v) in zip(keys, self.data.values()))
        reduced_dims = set([d.name for d in self.kdims]).difference(kdims)
        dimensions = [self.get_dimension(d) for d in kdims
                      if d not in reduced_dims]

        if len(set(keys)) != len(keys) and not force:
            raise Exception("Given dimension labels not sufficient"
                            "to address all values uniquely")

        if len(keys):
            cdims = {self.get_dimension(d): self.dimension_values(d)[0] for d in reduced_dims}
        else:
            cdims = {}
        with item_check(indices == sorted(indices)):
            return self.clone(reindexed_items, kdims=dimensions,
                              cdims=cdims)


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
        if (len(self.values()) > 0):
            info_str = self.__class__.__name__ +\
                       " containing %d items of type %s\n" % (len(self.keys()),
                                                              type(self.values()[0]).__name__)
        else:
            info_str = self.__class__.__name__ + " containing no items\n"
        info_str += ('-' * (len(info_str)-1)) + "\n\n"
        aliases = {v: k for k, v in self._dim_aliases.items()}
        for group in self._dim_groups:
            dimensions = getattr(self, group)
            if dimensions:
                group = aliases[group].split('_')[0]
                info_str += '%s Dimensions: \n' % group.capitalize()
            for d in dimensions:
                dmin, dmax = self.range(d.name)
                if d.value_format:
                    dmin, dmax = d.value_format(dmin), d.value_format(dmax)
                info_str += '\t %s: %s...%s \n' % (d.pprint_label, dmin, dmax)
        print(info_str)


    def table(self, datatype=None, **kwargs):
        "Creates a table from the stored keys and data."
        if datatype is None:
            datatype = ['dataframe' if pd else 'dictionary']

        tables = []
        for key, value in self.data.items():
            value = value.table(datatype=datatype, **kwargs)
            for idx, (dim, val) in enumerate(zip(self.kdims, key)):
                value = value.add_dimension(dim, idx, val)
            tables.append(value)
        return value.interface.concatenate(tables)


    def dframe(self):
        "Creates a pandas DataFrame from the stored keys and data."
        try:
            import pandas
        except ImportError:
            raise Exception("Cannot build a DataFrame without the pandas library.")
        labels = self.dimensions('key', True) + [self.group]
        return pandas.DataFrame(
            [dict(zip(labels, k + (v,))) for (k, v) in self.data.items()])


    def update(self, other):
        """
        Updates the current mapping with some other mapping or
        OrderedDict instance, making sure that they are indexed along
        the same set of dimensions. The order of key dimensions remains
        unchanged after the update.
        """
        if isinstance(other, NdMapping):
            dims = [d for d in other.kdims if d not in self.kdims]
            if len(dims) == other.ndims:
                raise KeyError("Cannot update with NdMapping that has"
                               " a different set of key dimensions.")
            elif dims:
                other = other.drop_dimension(dims)
            other = other.data
        for key, data in other.items():
            self._add_item(key, data, sort=False)
        if self.sort:
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
        except KeyError:
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
        self._add_item(key, value, update=False)


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

    group = param.String(default='NdMapping', constant=True)

    def __getitem__(self, indexslice):
        """
        Allows slicing operations along the key and data
        dimensions. If no data slice is supplied it will return all
        data elements, otherwise it will return the requested slice of
        the data.
        """
        if isinstance(indexslice, np.ndarray) and indexslice.dtype.kind == 'b':
            if not len(indexslice) == len(self):
                raise IndexError("Boolean index must match length of sliced object")
            selection = zip(indexslice, self.data.items())
            return self.clone([item for c, item in selection if c])
        elif indexslice == () and not self.kdims:
            return self.data[()]
        elif indexslice in [Ellipsis, ()]:
            return self
        elif Ellipsis in wrap_tuple(indexslice):
            indexslice = process_ellipses(self, indexslice)

        map_slice, data_slice = self._split_index(indexslice)
        map_slice = self._transform_indices(map_slice)
        map_slice = self._expand_slice(map_slice)

        if all(not (isinstance(el, (slice, set, list, tuple)) or callable(el))
               for el in map_slice):
            return self._dataslice(self.data[map_slice], data_slice)
        else:
            conditions = self._generate_conditions(map_slice)
            items = self.data.items()
            for cidx, (condition, dim) in enumerate(zip(conditions, self.kdims)):
                values = self._cached_index_values.get(dim.name, None)
                items = [(k, v) for k, v in items
                         if condition(values.index(k[cidx])
                                      if values else k[cidx])]
            sliced_items = []
            for k, v in items:
                val_slice = self._dataslice(v, data_slice)
                if val_slice or isinstance(val_slice, tuple):
                    sliced_items.append((k, val_slice))
            if len(sliced_items) == 0:
                raise KeyError('No items within specified slice.')
            with item_check(False):
                return self.clone(sliced_items)


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
        for dim, dim_slice in zip(self.kdims, map_slice):
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
            elif isinstance(dim_slice, (set, list)):
                if dim.values:
                    dim_slice = [self._cached_index_values[dim.name].index(dim_val)
                                 for dim_val in dim_slice]
                conditions.append(self._values_condition(dim_slice))
            elif dim_slice is Ellipsis:
                conditions.append(self._all_condition())
            elif callable(dim_slice):
                conditions.append(dim_slice)
            elif isinstance(dim_slice, (tuple)):
                raise IndexError("Keys may only be selected with sets or lists, not tuples.")
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
            lmbd = lambda x: x >= slice.start
        else:
            lmbd = lambda x: x >= slice.start and ((x-slice.start) % slice.step)
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
    _auxiliary_component = False

    def __init__(self, initial_items=None, kdims=None, group=None, label=None, **params):
        self._type = None
        self._group_check, self.group = None, group
        self._label_check, self.label = None, label
        super(UniformNdMapping, self).__init__(initial_items, kdims=kdims, **params)


    def clone(self, data=None, shared_data=True, new_type=None, *args, **overrides):
        """
        Returns a clone of the object with matching parameter values
        containing the specified args and kwargs.

        If shared_data is set to True and no data explicitly supplied,
        the clone will share data with the original.
        """
        settings = dict(self.get_param_values())
        if settings.get('group', None) != self._group:
            settings.pop('group')
        if settings.get('label', None) != self._label:
            settings.pop('label')
        if new_type is None:
            clone_type = self.__class__
        else:
            clone_type = new_type
            new_params = new_type.params()
            settings = {k: v for k, v in settings.items()
                      if k in new_params}
        settings = dict(settings, **overrides)
        if 'id' not in settings and new_type in [type(self), None]:
            settings['id'] = self.id

        if data is None and shared_data:
            data = self.data
            settings['plot_id'] = self._plot_id
        # Apply name mangling for __ attribute
        pos_args = getattr(self, '_' + type(self).__name__ + '__pos_params', [])
        with item_check(not shared_data and self._check_items):
            return clone_type(data, *args, **{k:v for k,v in settings.items()
                                              if k not in pos_args})


    @property
    def group(self):
        if self._group:
            return self._group
        group =  get_ndmapping_label(self, 'group') if len(self) else None
        if group is None:
            return type(self).__name__
        return group


    @group.setter
    def group(self, group):
        if group is not None and not sanitize_identifier.allowable(group):
            raise ValueError("Supplied group %s contains invalid "
                             "characters." % self.group)
        self._group = group


    @property
    def label(self):
        if self._label:
            return self._label
        else:
            if len(self):
                label = get_ndmapping_label(self, 'label')
                return '' if label is None else label
            else:
                return ''


    @label.setter
    def label(self, label):
        if label is not None and not sanitize_identifier.allowable(label):
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
        return self.type(None)


    def _item_check(self, dim_vals, data):
        if self.type is not None and (type(data) != self.type):
            raise AssertionError("%s must only contain one type of object, not both %s and %s." %
                                 (self.__class__.__name__, type(data).__name__, self.type.__name__))
        super(UniformNdMapping, self)._item_check(dim_vals, data)


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
            key_dims = reversed(list(zip(key, self.dimensions('key', True))))
            for val, dim in key_dims:
                dimn = 1
                while dim in view_frame:
                    dim = dim+'_%d' % dimn
                    if dim in view_frame:
                        dimn += 1
                view_frame.insert(0, dim, val)
            dframes.append(view_frame)
        return pandas.concat(dframes)
