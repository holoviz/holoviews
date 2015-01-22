"""
Supplies NdIndexableMapping and NdMapping which are multi-dimensional
map types. The former class only allows indexing whereas the latter
also enables slicing over multiple dimension ranges.
"""

from collections import OrderedDict
import numpy as np

import param

from .dimension import Dimension, DimensionedData, ViewableElement


class MultiDimensionalMapping(DimensionedData):
    """
    An MultiDimensionalMapping is a type of mapping (like a dictionary or
    array) that uses fixed-length multidimensional keys. This behaves
    like a sparse N-dimensional array that does not require a dense
    sampling over the multidimensional space.

    If the underlying type of data for each (key,value) pair also
    supports indexing (such as a dictionary, array, or list), fully
    qualified indexing can be used from the top level, with the first
    N dimensions of the index selecting a particular piece of data
    stored in the MultiDimensionalMapping object, and the remaining
    dimensions used to index into the underlying data.

    For instance, for an MultiDimensionalMapping x with dimensions "Year"
    and "Month" and an underlying data type that is a 2D
    floating-point array indexed by (r,c), a 2D array can be indexed
    with x[2000,3] and a single floating-point number may be indexed
    as x[2000,3,1,9].

    In practice, this class is typically only used as an abstract base
    class, because the NdMapping subclass extends it with a range of
    useful slicing methods for selecting subsets of the data. Even so,
    keeping the slicing support separate from the indexing and data
    storage methods helps make both classes easier to understand.
    """

    key_dimensions = param.List(default=[Dimension("Default")], constant=True)

    value = param.String(default='MultiDimensionalMapping')

    data_type = None

    _deep_indexable = False
    _sorted = True

    def __init__(self, initial_items=None, **params):
        super(MultiDimensionalMapping, self).__init__(OrderedDict(), **params)

        self._next_ind = 0
        self._check_key_type = True
        self._cached_index_types = [d.type for d in self.key_dimensions]
        self._cached_index_values = {d.name:d.values for d in self.key_dimensions}
        self._cached_categorical = any(d.values for d in self.key_dimensions)

        if isinstance(initial_items, tuple):
            self._add_item(initial_items[0], initial_items[1])
        elif initial_items is not None:
            self.update(OrderedDict(initial_items))


    def _item_check(self, dim_vals, data):
        """
        Applies checks to individual data elements before they are inserted
        ensuring that they are of a certain type. Can be subclassed to implement
        further element restrictions.
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


    def _resort(self):
        """
        Sorts data by key or index in pre-defined index in Dimension
        values.
        """
        sortkws = {}
        dimensions = self.key_dimensions
        if self._cached_categorical:
            sortkws['key'] = lambda (k, v): tuple(dimensions[i].values.index(k[i])
                                                  if dimensions[i].values else k[i]
                                                  for i in range(self.ndims))
        self.data = OrderedDict(sorted(self.data.items(), **sortkws))


    def _add_item(self, dim_vals, data, sort=True):
        """
        Adds item to the data, applying dimension types and
        ensuring key conforms to Dimension type and values.
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
            if vals and val not in vals:
                raise KeyError('%s Dimension value %s not in'
                               ' specified Dimension values.' % (dim, repr(val)))

        self._update_item(dim_vals, data)
        if sort:
            self._resort()


    def _update_item(self, dim_vals, data):
        """
        Subclasses default method to allow updating of nested data
        structures rather than simply overriding them.
        """
        if dim_vals in self.data and isinstance(self.data[dim_vals], (NdMapping, OrderedDict)):
            self.data[dim_vals].update(data)
        else:
            self.data[dim_vals] = data


    def update(self, other):
        """
        Updates the NdMapping with another NdMapping or OrderedDict
        instance, checking that they are indexed along the same number
        of dimensions.
        """
        for key, data in other.items():
            self._add_item(key, data, sort=False)
        self._resort()


    def reindex(self, dimension_labels):
        """
        Create a new object with a re-ordered or reduced set of index
        dimensions.

        Reducing the number of index dimensions will discard
        information from the keys. All data values are accessible in
        the newly created object as the new labels must be sufficient
        to address each value uniquely.
        """

        indices = [self.get_dimension_index(el) for el in dimension_labels]

        keys = [tuple(k[i] for i in indices) for k in self.data.keys()]
        reindexed_items = OrderedDict(
            (k, v) for (k, v) in zip(keys, self.data.values()))
        reduced_dims = set(self._cached_index_names).difference(dimension_labels)
        dimensions = [self.get_dimension(d) for d in dimension_labels if d not in reduced_dims]

        if len(set(keys)) != len(keys):
            raise Exception("Given dimension labels not sufficient to address all values uniquely")

        return self.clone(reindexed_items, key_dimensions=dimensions)


    def add_dimension(self, dimension, dim_pos, dim_val, **kwargs):
        """
        Create a new object with an additional dimension along which
        items are indexed. Requires the dimension name, the desired
        position in the key_dimensions and a dimension value that
        applies to all existing elements.
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


    def copy(self):
        return self.clone(list(self.items()))


    def dframe(self):
        try:
            import pandas
        except ImportError:
            raise Exception("Cannot build a DataFrame without the pandas library.")
        labels = self._cached_index_names + [self.value]
        return pandas.DataFrame(
            [dict(zip(labels, k + (v,))) for (k, v) in self.data.items()])


    def dimension_values(self, dimension):
        all_dims = [d.name for d in self.dimensions]
        if isinstance(dimension, int):
            dimension = all_dims[dimension]

        if dimension in self._cached_index_names:
            values = [k[self.get_dimension_index(dimension)] for k in self.data.keys()]
        elif dimension in all_dims:
            values = [el.dimension_values(dimension) for el in self
                      if dimension in el.dimensions]
            values = np.concatenate(values)
        else:
            raise Exception('Dimension %s not found.' % dimension)
        return values


    def _apply_key_type(self, keys):
        """
        If a key type is set in the dim_info dictionary, this method applies the
        type to the supplied key.
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
            else:
                typed_key += (key_type(key),)
        return typed_key


    def _split_index(self, key):
        """
        Splits key into map and data indices. If only map indices are
        supplied the data is passed an index of None.
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


    def __getitem__(self, key):
        """
        Allows indexing in the indexed dimensions, passing any
        additional indices to the data elements.
        """
        if key in [Ellipsis, ()]:
            return self
        map_slice, data_slice = self._split_index(key)
        return self._dataslice(self.data[map_slice], data_slice)


    def _dataslice(self, data, indices):
        """
        Returns slice of data element if the item is deep
        indexable. Warns if attempting to slice an object that has not
        been declared deep indexable.
        """
        if getattr(data, '_deep_indexable', False):
            return data[indices]
        elif len(indices) > 0:
            self.warning('Cannot index into data element, extra data'
                         ' indices ignored.')
        return data


    def __setitem__(self, key, value):
        self._add_item(key, value)


    def __str__(self):
        return repr(self)


    def key_items(self, key):
        """
        Returns a dictionary of dimension and key values.
        """
        if not isinstance(key, (tuple, list)): key = (key,)
        return dict(zip(self._cached_index_names, key))


    @property
    def last(self):
        """"
        Returns the item highest data item along the map dimensions.
        """
        return list(self.data.values())[-1] if len(self) else None


    @property
    def last_key(self):
        """"
        Returns the last key.
        """
        return list(self.keys())[-1] if len(self) else None


    def dimension_keys(self):
        """
        Returns the list of keys together with the dimension labels.
        """
        return [tuple(zip(self._cached_index_names, [k] if self.ndims == 1 else k))
                for k in self.keys()]


    def pprint_dimkey(self, key):
        """
        Takes a key of the right length as input and returns a
        formatted string of the dimension and value pairs.
        """
        key = key if isinstance(key, (tuple, list)) else (key,)
        return ', '.join(self.key_dimensions[i].pprint_value(v)
                         for i, v in enumerate(key))


    def drop_dimension(self, dim, val):
        """
        Drop dimension from the NdMapping using the supplied
        dimension name and value.
        """
        slices = [slice(None) for i in range(self.ndims)]
        slices[self.get_dimension_index(dim)] = val
        dim_labels = [d for d in self._cached_index_names if d != dim]
        return self[tuple(slices)].reindex(dim_labels)


    def keys(self):
        """
        Returns indices for all data elements.
        """
        if self.ndims == 1:
            return [k[0] for k in self.data.keys()]
        else:
            return list(self.data.keys())


    def values(self):
        return list(self.data.values())


    def items(self):
        return list(zip(list(self.keys()), list(self.values())))


    def get(self, key, default=None):
        try:
            if key is None:
                return None
            return self[key]
        except:
            return default


    def map(self, map_fn, **kwargs):
        """
        Map a function across the MultiDimensionalMapping.
        """
        mapped_items = [(k, map_fn(el, k)) for k, el in self.items()]
        if isinstance(mapped_items[0][1], tuple):
            split = [[(k, v) for v in val] for (k, val) in mapped_items]
            item_groups = [list(el) for el in zip(*split)]
        else:
            item_groups = [mapped_items]
        clones = tuple(self.clone(els, **kwargs)
                       for (i, els) in enumerate(item_groups))
        return clones if len(clones) > 1 else clones[0]

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
                info_str += '\t %s: %s...%s \n' % (str(d), dmin, dmax)
        print(info_str)


    def pop(self, *args):
        if len(args) > 0 and not isinstance(args[0], tuple):
            args[0] = (args[0],)
        return self.data.pop(*args)


    def __iter__(self):
        return iter(self.values())

    def __contains__(self, key):
        if self.ndims == 1:
            return key in self.data.keys()
        else:
            return key in self.keys()


    def __len__(self):
        return len(self.data)


    def sort_key(self, unordered):
        """
        Given an unordered list of (dimension, value) pairs returns
        the sorted key.
        """
        dim_orderfn = lambda k: self.get_dimension_index(k[0].name)
        return tuple([v for k, v in sorted(unordered, key=dim_orderfn)])


    def split_dimensions(self, dimensions, map_type=None):
        """
        Split the dimensions in the NdMapping across two NdMappings,
        where the inner mapping is of the same type as the original
        Map.
        """
        inner_dims = [d for d in dimensions if d in self._cached_index_names]
        deep_dims = [d for d in dimensions
                     if d in [d.name for d in self.deep_dimensions]]
        if self.ndims == 1:
            self.warning('Cannot split Map with only one dimension.')
            return self
        if len(deep_dims):
            raise Exception('NdMapping does not support splitting of deep dimensions.')
        first_dims, first_keys, second_dims, second_keys = self._split_dim_keys(inner_dims)
        self._check_key_type = False # Speed optimization
        own_keys = self.data.keys()

        map_type = map_type if map_type else NdMapping
        split_data = map_type(key_dimensions=first_dims)
        split_data._check_key_type = False # Speed optimization
        for fk in first_keys:  # The first groups keys
            split_data[fk] = self.clone(key_dimensions=second_dims)
            split_data[fk]._check_key_type = False # Speed optimization
            for sk in set(second_keys):  # The second groups keys
                # Generate a candidate expanded key
                unordered_dimkeys = list(zip(first_dims, fk)) + list(zip(second_dims, sk))
                sorted_key = self.sort_key(unordered_dimkeys)
                if sorted_key in own_keys:  # If the expanded key actually exists...
                    split_data[fk][sk] = self[sorted_key]
            split_data[fk]._check_key_type = True # Speed optimization
        split_data._check_key_type = True # Speed optimization

        self._check_key_type = True # Re-enable checks

        return split_data


    def _split_dim_keys(self, dimensions):
        """
        Split the NdMappings keys into two groups given a list of
        dimensions to split out.
        """

        # Find dimension indices
        first_dims = [d for d in self.key_dimensions if d.name not in dimensions]
        first_inds = [self.get_dimension_index(d.name) for d in first_dims]
        second_dims = [d for d in self.key_dimensions if d.name in dimensions]
        second_inds = [self.get_dimension_index(d.name) for d in second_dims]

        # Split the keys
        keys = list(self.data.keys())
        first_keys, second_keys = zip(*[(tuple(k[fi] for fi in first_inds),
                                        tuple(k[si] for si in second_inds))
                                        for k in keys])
        return first_dims, first_keys, second_dims, second_keys



class NdMapping(MultiDimensionalMapping):
    """
    NdMapping supports the same indexing semantics as
    MultiDimensionalMapping but also supports slicing semantics.

    Slicing semantics on an NdMapping is dependent on the ordering
    semantics of the keys. As MultiDimensionalMapping sort the keys, a
    slice on an NdMapping is effectively a way of filtering out the
    keys that are outside the slice range.
    """

    value = param.String(default='NdMapping')

    def __getitem__(self, indexslice):
        """
        Allows slicing operations along the map and data
        dimensions. If no data slice is supplied it will return all
        data elements, otherwise it will return the requested slice of
        the data.
        """
        if indexslice in [Ellipsis, ()]:
            return self

        map_slice, data_slice = self._split_index(indexslice)
        map_slice = self._transform_indices(map_slice)

        if all(not isinstance(el, slice) for el in map_slice):
            return self._dataslice(self.data[map_slice], data_slice)
        else:
            conditions = self._generate_conditions(map_slice)
            items = self.data.items()
            for cidx, condition in enumerate(conditions):
                items = [(k, v) for k, v in items if condition(k[cidx])]
            items = [(k, self._dataslice(v, data_slice)) for k, v in items]
            if self.ndims == 1:
                items = [(k[0], v) for (k, v) in items]
            if len(items) == 0:
                raise KeyError('No items within specified slice.')
            return self.clone(items)


    def select(self, **kwargs):
        """
        Allows selecting slices or indices into the NdMapping using
        keyword arguments matching the names of the dimensions.
        """
        deep_select = any([kw for kw in kwargs.keys() if (kw in self.deep_dimensions)
                           and (kw not in self._cached_index_names)])
        selection_depth = len(self.dimensions) if deep_select else self.ndims
        selection = [slice(None) for i in range(selection_depth)]
        for dim, val in kwargs.items():
            selection[self.get_dimension_index(dim)] = val
        return self.__getitem__(tuple(selection))


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
        for dim in map_slice:
            if isinstance(dim, slice):
                if dim == slice(None):
                    conditions.append(self._all_condition())
                elif dim.start is None:
                    conditions.append(self._upto_condition(dim))
                elif dim.stop is None:
                    conditions.append(self._from_condition(dim))
                else:
                    conditions.append(self._range_condition(dim))
            elif dim is Ellipsis:
                conditions.append(self._all_condition())
            else:
                conditions.append(self._value_condition(dim))
        return conditions


    def _value_condition(self, value):
        return lambda x: x == value


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
    A UniformNdMapping is a map of Views over a number of specified dimensions. The
    dimension may be a spatial dimension (i.e., a ZStack), time
    (specifying a frame sequence) or any other combination of
    Dimensions.  UniformNdMapping also adds handling of styles, appending the
    Dimension keys and values to titles and a number of methods to
    manipulate the Dimensions.

    UniformNdMapping objects can be sliced, sampled, reduced, overlaid and split
    along its and its containing Views dimensions. Subclasses should
    implement the appropriate slicing, sampling and reduction methods
    for their ViewableElement type.
    """

    title_suffix = param.String(default='\n {dims}', doc="""
       A string appended to the ViewableElement titles when they are added to the
       UniformNdMapping. Default adds a new line with the formatted dimensions
       of the UniformNdMapping inserted using the {dims} formatting keyword.""")

    value = param.String(default='UniformNdMapping')

    data_type = (ViewableElement, NdMapping)

    _abstract = True
    _deep_indexable = True
    _type = None
    _style = None

    @property
    def type(self):
        """
        The type of elements stored in the map.
        """
        if self._type is None:
            self._type = None if len(self) == 0 else self.last.__class__
        return self._type


    @property
    def style(self):
        """
        The style of elements stored in the map.
        """
        if self._style is None:
            self._style = None if len(self) == 0 else self.last.style
        return self._style


    @style.setter
    def style(self, style_name):
        self._style = style_name
        for val in self.values():
            val.style = style_name


    @property
    def empty_element(self):
        return self._type(None)


    def _item_check(self, dim_vals, data):
        if self.style is not None and self.style != data.style:
            data.style = self.style

        if self.type is not None and (type(data) != self.type):
            raise AssertionError("%s must only contain one type of ViewableElement." %
                                 self.__class__.__name__)
        super(UniformNdMapping, self)._item_check(dim_vals, data)


    def get_title(self, key, item, group_size=2):
        """
        Resolves the title string on the ViewableElement being added to the UniformNdMapping,
        adding the Maps title suffix.
        """
        if self.ndims == 1 and self.get_dimension('Default'):
            title_suffix = ''
        else:
            title_suffix = self.title_suffix
        dimension_labels = [dim.pprint_value(k) for dim, k in
                            zip(self.key_dimensions, key)]
        groups = [', '.join(dimension_labels[i*group_size:(i+1)*group_size])
                  for i in range(len(dimension_labels))]
        dims = '\n '.join(g for g in groups if g)
        title_suffix = title_suffix.format(dims=dims)
        return item.title + title_suffix


    def table(self, **kwargs):
        """
        Creates Table from all the elements in the UniformNdMapping.
        """

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