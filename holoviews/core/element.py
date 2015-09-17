import operator
from itertools import groupby
from numbers import Number
import numpy as np

import param

from .dimension import Dimension, Dimensioned, ViewableElement
from .layout import Composable, Layout, AdjointLayout
from .ndmapping import OrderedDict, UniformNdMapping, NdMapping, item_check
from .overlay import Overlayable, Overlay, CompositeOverlay
from .tree import AttrTree
from .util import sanitize_identifier

class Element(ViewableElement, Composable, Overlayable):
    """
    Element is the baseclass for all ViewableElement types, with an x- and
    y-dimension. Subclasses should define the data storage in the
    constructor, as well as methods and properties, which define how
    the data maps onto the x- and y- and value dimensions.
    """

    group = param.String(default='Element', constant=True)

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
        convert = isinstance(data, Element)
        if convert:
            params = dict(data.get_param_values(onlychanged=True),
                          **params)
            element = data
            data = []
        super(Element, self).__init__(data, **params)
        if convert:
            self.data = self._convert_element(element)


    def _convert_element(self, element):
        type_str = self.__class__.__name__
        type_name = type_str.lower()
        table = element.table()
        conversion = getattr(table.to, type_name)
        if conversion is None:
            return element
        try:
            converted = conversion(self._cached_index_names,
                                   self._cached_value_names)
        except:
            raise
        return converted.data


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


    def sample(self, samples=[], **sample_values):
        """
        Base class signature to demonstrate API for sampling Elements.
        To sample an Element supply either a list of sampels or keyword
        arguments, where the key should match an existing key dimension
        on the Element.
        """
        raise NotImplementedError


    def reduce(self, dimensions=[], function=None, **reduce_map):
        """
        Base class signature to demonstrate API for reducing Elements,
        using some reduce function, e.g. np.mean, which is applied
        along a particular Dimension. The dimensions and reduce functions
        should be passed as keyword arguments or as a list of dimensions
        and a single function.
        """
        raise NotImplementedError


    def _reduce_map(self, dimensions, function, reduce_map):
        dimensions = self._valid_dimensions(dimensions)
        if dimensions and reduce_map:
            raise Exception("Pass reduced dimensions either as an argument"
                            "or as part of the kwargs not both.")
        elif dimensions:
            reduce_map = {d: function for d in dimensions}
        elif not reduce_map:
            reduce_map = {d: function for d in self._cached_index_names}
        sanitized = {sanitize_identifier(kd): kd
                     for kd in self._cached_index_names}
        return {sanitized.get(d, d): fn for d, fn in reduce_map.items()}


    def table(self, **kwargs):
        """
        This method transforms any ViewableElement type into a Table
        as long as it implements a dimension_values method.
        """
        from ..element import Table
        keys = zip(*[self.dimension_values(dim.name)
                 for dim in self.kdims])
        values = zip(*[self.dimension_values(dim.name)
                       for dim in self.vdims])
        kwargs = {'label': self.label
                  for k, v in self.get_param_values(onlychanged=True)
                  if k in ['group', 'label']}
        params = dict(kdims=self.kdims,
                      vdims=self.vdims,
                      label=self.label)
        if not self.params()['group'].default == self.group:
            params['group'] = self.group
        if not keys: keys = [()]*len(values)
        if not values: [()]*len(keys)
        return Table(zip(keys, values), **dict(params, **kwargs))


    def dframe(self):
        import pandas
        column_names = self.dimensions(label=True)
        dim_vals = np.vstack([self.dimension_values(dim) for dim in column_names]).T
        return pandas.DataFrame(dim_vals, columns=column_names)



class Tabular(NdMapping):
    """
    Baseclass to give an NdMapping objects an API to generate a
    table representation.
    """

    __abstract = True

    @property
    def rows(self):
        return len(self.data) + 1

    @property
    def cols(self):
        return self.ndims + max([1, len(self.vdims)])


    def pprint_cell(self, row, col):
        """
        Get the formatted cell value for the given row and column indices.
        """
        ndims = self.ndims
        if col >= self.cols:
            raise Exception("Maximum column index is %d" % self.cols-1)
        elif row >= self.rows:
            raise Exception("Maximum row index is %d" % self.rows-1)
        elif row == 0:
            if col >= ndims:
                if self.vdims:
                    return str(self.vdims[col - ndims])
                else:
                    return ''
            return str(self.kdims[col])
        else:
            dim = self.get_dimension(col)
            if col >= ndims:
                row_values = self.values()[row-1]
                if self.vdims:
                    val = row_values[col - ndims]
                else:
                    val = row_values
            else:
                row_data = list(self.data.keys())[row-1]
                val = row_data[col]
            return dim.pprint_value(val)


    def cell_type(self, row, col):
        """
        Returns the cell type given a row and column index. The common
        basic cell types are 'data' and 'heading'.
        """
        return 'heading' if row == 0 else 'data'



class Element2D(Element):

    extents = param.Tuple(default=(None, None, None, None),
                          doc="""Allows overriding the extents
              of the Element in 2D space defined as four-tuple
              defining the (left, bottom, right and top) edges.""")


class NdElement(Element, Tabular):
    """
    An NdElement is an Element that stores the contained data as
    an NdMapping. In addition to the usual multi-dimensional keys
    of NdMappings, NdElements also support multi-dimensional
    values. The values held in a multi-valued NdElement are tuples,
    where each component of the tuple maps to a column as described
    by the value dimensions parameter.

    In other words, the data of a NdElement are partitioned into two
    groups: the columns based on the key and the value columns that
    contain the components of the value tuple.

    One feature of NdElements is that they support an additional level of
    index over NdMappings: the last index may be a column name or a
    slice over the column names (using alphanumeric ordering).
    """

    group = param.String(default='NdElement', constant=True, doc="""
         The group is used to describe the NdElement.""")

    vdims = param.List(default=[Dimension('Data')], doc="""
        The dimension description(s) of the values held in data tuples
        that map to the value columns of the table.

        Note: String values may be supplied in the constructor which
        will then be promoted to Dimension objects.""")

    _deep_indexable = False

    def __init__(self, data=None, **params):
        if isinstance(data, Element):
            data = data.table()
        elif isinstance(data, list) and all(np.isscalar(el) for el in data):
            data = OrderedDict(list(((k,), v) for k, v in enumerate(data)))
        super(NdElement, self).__init__(data, **params)


    def _convert_element(self, element):
        if isinstance(element, NdElement):
            return element.data
        if isinstance(element, Element):
            return element.table().data
        else: return element


    def reindex(self, kdims=None, vdims=None, force=False):
        """
        Create a new object with a re-ordered set of dimensions.
        Allows converting key dimensions to value dimensions
        and vice versa.
        """
        if vdims is None:
            if kdims is None:
                return super(NdElement, self).reindex(force=force)
            else:
                vdims = self._cached_value_names
        elif kdims is None:
            kdims = [d for d in (self._cached_index_names + self._cached_value_names)
                     if d not in vdims]
        key_dims = [self.get_dimension(k) for k in kdims]
        val_dims = [self.get_dimension(v) for v in vdims]
        kidxs = [(i, k in self._cached_index_names, self.get_dimension_index(k))
                  for i, k in enumerate(kdims)]
        vidxs = [(i, v in self._cached_index_names, self.get_dimension_index(v))
                  for i, v in enumerate(vdims)]
        getter = operator.itemgetter(0)
        items = []
        for k, v in self.data.items():
            _, key = zip(*sorted(((i, k[idx] if iskey else v[idx-self.ndims])
                                  for i, iskey, idx in kidxs), key=getter))
            _, val = zip(*sorted(((i, k[idx] if iskey else v[idx-self.ndims])
                                  for i, iskey, idx in vidxs), key=getter))
            items.append((key, val))
        reindexed = self.clone(items, kdims=key_dims, vdims=val_dims)
        if not force and len(reindexed) != len(items):
            raise KeyError("Cannot reindex as not all resulting keys are unique.")
        return reindexed


    def _add_item(self, key, value, sort=True):
        value = (value,) if np.isscalar(value) else tuple(value)
        if len(value) != len(self.vdims):
            raise ValueError("%s values must match value dimensions"
                             % type(self).__name__)
        super(NdElement, self)._add_item(key, value, sort)


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


    def _filter_data(self, subtable, vdims):
        """
        Filters value dimensions in the supplied NdElement data.
        """
        if isinstance(subtable, tuple): subtable = {(): subtable}
        col_names = self.dimensions('value', label=True)
        cols = self._filter_columns(vdims, col_names)
        indices = [col_names.index(col) for col in cols]
        vdims = [self.vdims[i] for i in indices]
        items = [(k, tuple(v[i] for i in indices))
                 for (k,v) in subtable.items()]
        if len(items) == 1:
            data = items[0][1]
            if len(vdims) == 1:
                return data[0]
            else:
                from ..element.tabular import ItemTable
                kwargs = {'label': self.label
                          for k, v in self.get_param_values(onlychanged=True)
                          if k in ['group', 'label']}
                data = list(zip(vdims, data))
                return ItemTable(data, **kwargs)
        else:
            return subtable.clone(items, vdims=vdims)


    def __getitem__(self, args):
        """
        In addition to usual NdMapping indexing, NdElements can be indexed
        by column name (or a slice over column names)
        """
        ndmap_index = args[:self.ndims] if isinstance(args, tuple) else args
        subtable = NdMapping.__getitem__(self, ndmap_index)

        if len(self.vdims) > 1 and not isinstance(subtable, NdElement):
            subtable = self.__class__([((), subtable)], label=self.label,
                                      kdims=[], vdims=self.vdims)

        # If subtable is not a slice return as reduced type
        if not isinstance(args, tuple): args = (args,)
        shallow = len(args) <= self.ndims
        slcs = any(isinstance(a, (slice, set)) for a in args[:self.ndims])
        if shallow and not (slcs or len(args) == 0):
            args = list(args) + [self.dimensions('value', True)]
        elif shallow:
            return subtable

        return self._filter_data(subtable, args[-1])


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
        reduce_map = self._reduce_map(dimensions, function, reduce_map)

        dim_labels = self._cached_index_names
        reduced_table = self
        for reduce_fn, group in groupby(reduce_map.items(), lambda x: x[1]):
            dims = [dim for dim, _ in group]
            split_dims = [self.get_dimension(d) for d in dim_labels if d not in dims]
            if len(split_dims) and reduced_table.ndims > 1:
                split_map = reduced_table.groupby([d.name for d in split_dims], group_type=self.__class__)
                reduced_table = self.clone(shared_data=False, kdims=split_dims)
                for k, table in split_map.items():
                    reduced = []
                    for vdim in self.vdims:
                        valtable = table.select(value=vdim.name) if len(self.vdims) > 1 else table
                        reduced.append(reduce_fn(valtable.data.values()))
                    reduced_table[k] = reduced
            else:
                reduced = tuple(reduce_fn(self.dimension_values(vdim.name))
                                for vdim in self.vdims)
                reduced_dims = [d for d in self.kdims if d.name not in reduce_map]
                params = dict(group=self.group) if self.group != type(self).__name__ else {}
                reduced_table = self.__class__([((), reduced)], label=self.label, kdims=reduced_dims,
                                               vdims=self.vdims, **params)
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
        dim = self.get_dimension(dim).name
        if dim in self._cached_value_names:
            index = self._cached_value_names.index(dim)
            return [v[index] for v in self.data.values()]
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

    extents = param.Tuple(default=(None, None, None,
                                   None, None, None),
        doc="""Allows overriding the extents of the Element
               in 3D space defined as (xmin, ymin, zmin,
               xmax, ymax, zmax).""")



__all__ = list(set([_k for _k, _v in locals().items()
                    if isinstance(_v, type) and issubclass(_v, Dimensioned)]))
