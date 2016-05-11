import operator
from itertools import groupby
import numpy as np

import param

from .dimension import Dimension, Dimensioned, ViewableElement
from .layout import Composable, Layout, NdLayout
from .ndmapping import OrderedDict, NdMapping
from .overlay import Overlayable, NdOverlay, CompositeOverlay
from .spaces import HoloMap, GridSpace
from .tree import AttrTree
from .util import (dimension_sort, get_param_values, dimension_sanitizer,
                   unique_array)


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
        if not isinstance(dimension, list): dimension = [dimension]
        hists = []
        for d in dimension[::-1]:
            hist = histogram(self, num_bins=num_bins, bin_range=bin_range,
                             adjoin=False, individually=individually,
                             dimension=d, **kwargs)
            hists.append(hist)
        if adjoin:
            layout = self
            for didx in range(len(dimension)):
                layout = layout << hists[didx]
        elif len(dimension) > 1:
            layout = Layout(hists)
        else:
            layout = hists[0]
        return layout

    #======================#
    # Subclassable methods #
    #======================#


    def __getitem__(self, key):
        if key is ():
            return self
        else:
            raise NotImplementedError("%s currently does not support getitem" %
                                      type(self).__name__)


    @classmethod
    def collapse_data(cls, data, function=None, kdims=None, **kwargs):
        """
        Class method to collapse a list of data matching the
        data format of the Element type. By implementing this
        method HoloMap can collapse multiple Elements of the
        same type. The kwargs are passed to the collapse
        function. The collapse function must support the numpy
        style axis selection. Valid function include:
        np.mean, np.sum, np.product, np.std, scipy.stats.kurtosis etc.
        Some data backends also require the key dimensions
        to aggregate over.
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
        if dimensions and reduce_map:
            raise Exception("Pass reduced dimensions either as an argument "
                            "or as part of the kwargs not both.")
        if len(set(reduce_map.values())) > 1:
            raise Exception("Cannot define reduce operations with more than "
                            "one function at a time.")
        sanitized_dict = {dimension_sanitizer(kd): kd
                          for kd in self.dimensions('key', True)}
        if reduce_map:
            reduce_map = reduce_map.items()
        if dimensions:
            reduce_map = [(d, function) for d in dimensions]
        elif not reduce_map:
            reduce_map = [(d, function) for d in self.kdims]
        reduced = [(d.name if isinstance(d, Dimension) else d, fn)
                   for d, fn in reduce_map]
        sanitized = [(sanitized_dict.get(d, d), fn) for d, fn in reduced]
        grouped = [(fn, [dim for dim, _ in grp]) for fn, grp in groupby(sanitized, lambda x: x[1])]
        return grouped[0]


    def table(self, datatype=None):
        """
        Converts the data Element to a Table, optionally may
        specify a supported data type. The default data types
        are 'numpy' (for homogeneous data), 'dataframe', and
        'dictionary'.
        """
        if datatype and not isinstance(datatype, list):
            datatype = [datatype]
        from ..element import Table
        return Table(self, **(dict(datatype=datatype) if datatype else {}))


    def dframe(self, dimensions=None):
        import pandas as pd
        column_names = dimensions if dimensions else self.dimensions(label=True)
        dim_vals = OrderedDict([(dim, self[dim]) for dim in column_names])
        return pd.DataFrame(dim_vals)


    def mapping(self, kdims=None, vdims=None, **kwargs):
        length = len(self)
        if not kdims: kdims = self.kdims
        if kdims:
            keys = zip(*[self.dimension_values(dim.name)
                         for dim in self.kdims])
        else:
            keys = [()]*length

        if not vdims: vdims = self.vdims
        if vdims:
            values = zip(*[self.dimension_values(dim.name)
                           for dim in vdims])
        else:
            values = [()]*length

        data = zip(keys, values)
        overrides = dict(kdims=kdims, vdims=vdims, **kwargs)
        return NdElement(data, **dict(get_param_values(self), **overrides))


    def array(self, dimensions=[]):
        if dimensions:
            dims = [self.get_dimension(d) for d in dimensions]
        else:
            dims = [d for d in self.kdims + self.vdims if d != 'Index']
        columns, types = [], []
        for dim in dims:
            column = self.dimension_values(dim)
            columns.append(column)
            types.append(column.dtype.kind)
        if len(set(types)) > 1:
            columns = [c.astype('object') for c in columns]
        return np.column_stack(columns)



class Tabular(Element):
    """
    Baseclass to give an NdMapping objects an API to generate a
    table representation.
    """

    __abstract = True

    @property
    def rows(self):
        return len(self) + 1

    @property
    def cols(self):
        return len(self.dimensions())


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
            values = self[dim.name]
            return dim.pprint_value(values[row-1])


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


class NdElement(NdMapping, Tabular):
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
    _sorted = False

    def __init__(self, data=None, **params):
        if isinstance(data, list) and all(np.isscalar(el) for el in data):
            data = (((k,), (v,)) for k, v in enumerate(data))

        if isinstance(data, Element):
            params = dict(get_param_values(data), **params)
            mapping = data if isinstance(data, NdElement) else data.mapping()
            data = mapping.data
            if 'kdims' not in params:
                params['kdims'] = mapping.kdims
            elif 'Index' not in params['kdims']:
                params['kdims'] = ['Index'] + params['kdims']
            if 'vdims' not in params:
                params['vdims'] = mapping.vdims

        kdims = params.get('kdims', self.kdims)
        if (data is not None and not isinstance(data, NdMapping)
            and 'Index' not in kdims):
            params['kdims'] = ['Index'] + list(kdims)
            data_items = data.items() if isinstance(data, dict) else data
            data = [((i,)+((k,) if np.isscalar(k) else k), v) for i, (k, v) in enumerate(data_items)]
        super(NdElement, self).__init__(data, **params)


    @property
    def shape(self):
        return (len(self), len(self.dimensions()))


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
                vdims = [d for d in self.vdims if d not in kdims]
        elif kdims is None:
            kdims = [d for d in self.dimensions() if d not in vdims]
        if 'Index' not in kdims: kdims = ['Index'] + kdims
        key_dims = [self.get_dimension(k) for k in kdims]
        val_dims = [self.get_dimension(v) for v in vdims]

        kidxs = [(i, k in self.kdims, self.get_dimension_index(k))
                  for i, k in enumerate(kdims)]
        vidxs = [(i, v in self.kdims, self.get_dimension_index(v))
                  for i, v in enumerate(vdims)]
        getter = operator.itemgetter(0)
        items = []
        for k, v in self.data.items():
            if key_dims:
                _, key = zip(*sorted(((i, k[idx] if iskey else v[idx-self.ndims])
                                      for i, iskey, idx in kidxs), key=getter))
            else:
                key = ()
            if val_dims:
                _, val = zip(*sorted(((i, k[idx] if iskey else v[idx-self.ndims])
                                      for i, iskey, idx in vidxs), key=getter))
            else:
                val = ()
            items.append((key, val))
        reindexed = self.clone(items, kdims=key_dims, vdims=val_dims)
        if not force and len(reindexed) != len(items):
            raise KeyError("Cannot reindex as not all resulting keys are unique.")
        return reindexed


    def _add_item(self, key, value, sort=True, update=True):
        if np.isscalar(value):
            value = (value,)
        elif not isinstance(value, NdElement):
            value = tuple(value)
        if len(value) != len(self.vdims) and not isinstance(value, NdElement):
            raise ValueError("%s values must match value dimensions"
                             % type(self).__name__)
        super(NdElement, self)._add_item(key, value, sort, update)


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
        return subtable.clone(items, vdims=vdims)


    def __getitem__(self, args):
        """
        In addition to usual NdMapping indexing, NdElements can be indexed
        by column name (or a slice over column names)
        """
        if isinstance(args, np.ndarray) and args.dtype.kind == 'b':
            return NdMapping.__getitem__(self, args)
        elif args in self.dimensions():
            return self.dimension_values(args)
        if not isinstance(args, tuple): args = (args,)
        ndmap_index = args[:self.ndims]
        val_index = args[self.ndims:]
        if val_index:
            if len(val_index) == 1 and val_index[0] in self.vdims:
                val_index = val_index[0]
            else:
                reindexed = self.reindex(self.kdims+list(self.vdims))
                subtable = reindexed[args]

        if not val_index or not isinstance(val_index, tuple):
            subtable = NdMapping.__getitem__(self, ndmap_index)

        if isinstance(subtable, NdElement) and all(np.isscalar(idx) for idx in ndmap_index[1:]):
            if len(subtable) == 1:
                subtable = list(subtable.data.values())[0]
        if not isinstance(subtable, NdElement):
            if len(self.vdims) > 1:
                subtable = self.__class__([(args[1:], subtable)], label=self.label,
                                          kdims=self.kdims[1:], vdims=self.vdims)
            else:
                if np.isscalar(subtable):
                    return subtable
                return subtable[0]

        if val_index and not isinstance(val_index, tuple):
            return self._filter_data(subtable, args[-1])
        else:
            return subtable


    def sort(self, by=[]):
        if not isinstance(by, list): by = [by]
        if not by: by = range(self.ndims)
        indexes = [self.get_dimension_index(d) for d in by]
        return self.clone(dimension_sort(self.data, self.kdims, self.vdims,
                                         False, indexes, self._cached_index_values))


    def sample(self, samples=[]):
        """
        Allows sampling of the Table with a list of samples.
        """
        sample_data = []
        offset = 0
        for i, sample in enumerate(samples):
            sample = (sample,) if np.isscalar(sample) else sample
            value = self[(slice(None),)+sample]
            if isinstance(value, NdElement):
                for idx, (k, v) in enumerate(value.data.items()):
                    sample_data.append(((i+offset+idx,)+k, v))
                offset += idx
            else:
                sample_data.append(((i+offset,)+sample, (value,)))
        return self.clone(sample_data)


    def aggregate(self, dimensions, function, **kwargs):
        """
        Allows aggregating.
        """
        rows = []
        grouped = self.groupby(dimensions) if len(dimensions) else HoloMap({(): self}, kdims=[])
        for k, group in grouped.data.items():
            reduced = []
            for vdim in self.vdims:
                data = group[vdim.name]
                if isinstance(function, np.ufunc):
                    reduced.append(function.reduce(data, **kwargs))
                else:
                    reduced.append(function(data, **kwargs))
            rows.append((k, tuple(reduced)))
        return self.clone(rows, kdims=grouped.kdims)


    def dimension_values(self, dim, expanded=True, flat=True):
        dim = self.get_dimension(dim, strict=True)
        value_dims = self.dimensions('value', label=True)
        if dim.name in value_dims:
            index = value_dims.index(dim.name)
            vals = np.array([v[index] for v in self.data.values()])
            return vals if expanded else unique_array(vals)
        else:
            return NdMapping.dimension_values(self, dim.name,
                                              expanded, flat)


    def values(self):
        " Returns the values of all the elements."
        values = self.data.values()
        if len(self.vdims) == 1:
            return  [v[0] for v in values]
        return list(values)



class Element3D(Element2D):

    extents = param.Tuple(default=(None, None, None,
                                   None, None, None),
        doc="""Allows overriding the extents of the Element
               in 3D space defined as (xmin, ymin, zmin,
               xmax, ymax, zmax).""")


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
                   GridSpace: (HoloMap, CompositeOverlay, ViewableElement),
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

            dim_keys = zip(self.kdims, key)
            varying_keys = [(d, k) for d, k in dim_keys if not self.drop_constant or
                            (d not in constant_dims and d not in self.drop)]
            constant_keys = [(d, k) for d, k in dim_keys if d in constant_dims
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


    def _add_item(self, key, value, sort=True, update=True):
        NdMapping._add_item(self, key, value, sort, update)


    @property
    def static_dimensions(self):
        """
        Return all constant dimensions.
        """
        dimensions = []
        for dim in self.kdims:
            if len(set(self[dim.name])) == 1:
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


__all__ = list(set([_k for _k, _v in locals().items()
                    if isinstance(_v, type) and issubclass(_v, Dimensioned)]))
