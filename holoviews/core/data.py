"""
The data module provides utility classes to interface with various
data backends.
"""

from collections import defaultdict
from itertools import groupby

try:
    import itertools.izip as zip
except ImportError:
    pass

import numpy as np
try:
    import pandas as pd
except ImportError:
    pd = None

import param

from .dimension import OrderedDict, Dimension
from .element import Element, NdElement
from .spaces import HoloMap
from . import util


class Columns(Element):

    def __init__(self, data, **kwargs):
        data, params = ColumnarData._process_data(data, self.params(), **kwargs)
        super(Columns, self).__init__(data, **params)
        self.data = self._validate_data(self.data)


    def _validate_data(self, data):
        if self.interface is None:
            return data
        else:
            return self.interface.validate_data(data)


    def __setstate__(self, state):
        """
        Restores OrderedDict based Columns objects, converting
        them to the up-to-date NdElement format.
        """
        self.__dict__ = state
        if isinstance(self.data, OrderedDict):
            self.data = OrderedDict(self.data, kdims=self.kdims,
                                    vdims=self.vdims, group=self.group,
                                    label=self.label)


    def closest(self, coords):
        if self.ndims > 1:
            NotImplementedError("Closest method currently only "
                                "implemented for 1D Elements")
        elif self.interface is None:
            return self.data.closest(coords)
        else:
            return self.interface.closest(coords)


    def add_dimension(self, dimension, dim_pos, dim_val, **kwargs):
        """
        Create a new object with an additional key dimensions.
        Requires the dimension name or object, the desired position
        in the key dimensions and a key value scalar or sequence of
        the same length as the existing keys.
        """
        if isinstance(dimension, str):
            dimension = Dimension(dimension)

        if dimension.name in self.kdims:
            raise Exception('{dim} dimension already defined'.format(dim=dimension.name))

        dimensions = self.kdims[:]
        dimensions.insert(dim_pos, dimension)

        if self.interface is None:
            data = self.data.add_dimension(dimension, dim_pos, dim_val, **kwargs)
        else:
            data = self.interface.add_dimension(self.data, dimension, dim_pos, dim_val)
        return self.clone(data, kdims=dimensions)


    def select(self, selection_specs=None, **selection):
        if selection_specs and not self.matches(selection_specs):
            return self

        if self.interface is None:
            data = self.data.select(**selection)
        else:
            data = self.interface.select(**selection)
        if np.isscalar(data):
            return data
        else:
            return self.clone(data)
        

    @property
    def interface(self):
        if util.is_dataframe(self.data):
            return ColumnarDataFrame(self)
        elif isinstance(self.data, np.ndarray):
            return ColumnarArray(self)


    def reindex(self, kdims=None, vdims=None):
        """
        Create a new object with a re-ordered set of dimensions.
        Allows converting key dimensions to value dimensions
        and vice versa.
        """
        if self.interface is None:
            return self.data.reindex(kdims, vdims)

        if vdims is None:
            val_dims = self.vdims
        else:
            val_dims = [self.get_dimension(v) for v in vdims]

        if kdims is None:
            key_dims = [d for d in self.dimensions()
                        if d not in vdims]
        else:
            key_dims = [self.get_dimension(k) for k in kdims]

        data = self.interface.reindex(key_dims, val_dims)
        return self.clone(data, kdims=key_dims, vdims=val_dims)


    def __getitem__(self, slices):
        """
        Implements slicing or indexing of the data by the data x-value.
        If a single element is indexed reduces the Element2D to a single
        Scatter object.
        """
        if slices is (): return self
        if not isinstance(slices, tuple): slices = (slices,)
        value_select = None
        if len(slices) == 1 and slices[0] in self.dimensions():
            return self.dimension_values(slices[0])
        elif len(slices) == self.ndims+1 and slices[self.ndims] in self.dimensions():
            selection = dict(zip(self.dimensions('key', label=True), slices))
            value_select = slices[self.ndims]
        else:
            selection = dict(zip(self.dimensions(label=True), slices))
        data = self.select(**selection)
        if value_select:
            values = data.dimension_values(value_select)
            if len(values) > 1:
                return values
            else:
                return values[0]
        return data


    def sample(self, samples=[]):
        """
        Allows sampling of Columns as an iterator of coordinates
        matching the key dimensions.
        """
        if self.interface is None:
            return self.clone(self.data.sample(samples))
        else:
            return self.clone(self.interface.sample(samples))


    def reduce(self, dimensions=[], function=None, **reduce_map):
        """
        Allows collapsing of Columns objects using the supplied map of
        dimensions and reduce functions.
        """
        reduce_dims, reduce_map = self._reduce_map(dimensions, function, reduce_map)
        reduced = self
        for reduce_fn, group in reduce_map:
            if self.interface is None:
                reduced = self.data.reduce(reduced, group, function)
            else:
                reduced = self.interface.reduce(reduced, group, function)

        if np.isscalar(reduced):
            return reduced
        else:
            kdims = [kdim for kdim in self.kdims if kdim not in reduce_dims]
            return self.clone(reduced, kdims=kdims)



    def aggregate(self, dimensions, function):
        """
        Groups over the supplied dimensions and aggregates.
        """
        if self.interface is None:
            aggregated = self.data.aggregate(dimensions, function)
        else:
            aggregated = self.interface.aggregate(dimensions, function)
        kdims = [self.get_dimension(d) for d in dimensions]
        return self.clone(aggregated, kdims=kdims)


    def groupby(self, dimensions, container_type=HoloMap, **kwargs):
        if self.interface is None:
            return self.data.groupby(dimensions, container_type, **kwargs)
        else:
            return self.interface.groupby(dimensions, container_type, **kwargs)

    @classmethod
    def collapse_data(cls, data, function=None, kdims=None, **kwargs):
        if isinstance(data[0], NdElement):
            return data[0].collapse_data(data, function, kdims, **kwargs)
        elif isinstance(data[0], np.ndarray):
            return ColumnarArray.collapse_data(data, function, kdims, **kwargs)
        elif util.is_dataframe(data[0]):
            return ColumnarDataFrame.collapse_data(data, function, kdims, **kwargs)


    def __len__(self):
        if self.interface is None:
            return len(self.data)
        else:
            return len(self.interface)


    @property
    def shape(self):
        if self.interface is None:
            return (len(self), len(self.dimensions()))
        else:
            return self.interface.shape


    def dimension_values(self, dim):
        if self.interface is None:
            return self.data.dimension_values(dim)
        else:
            dim = self.get_dimension(dim).name
            return self.interface.values(dim)


    def dframe(self, as_table=False):
        if self.interface is None:
            return self.data.dframe(as_table)
        else:
            return self.interface.dframe(as_table)


    def array(self, as_table=False):
        if self.interface is None:
            return super(Columns, self).array(as_table)
        array = self.interface.array()
        if as_table:
            from ..element import Table
            if array.dtype.kind in ['S', 'O', 'U']:
                raise ValueError("%s data contains non-numeric type, "
                                 "could not convert to array based "
                                 "Element" % type(self).__name__)
            return Table(array, **util.get_param_values(self, Table))
        return array



class ColumnarData(param.Parameterized):

    def __init__(self, element, **params):
        self.element = element


    def array(self):
        NotImplementedError


    @property
    def shape(self):
        return self.element.data.shape


    @classmethod
    def _process_data(cls, data, paramobjs, **kwargs):
        params = {}
        if isinstance(data, Element):
            params['kdims'] = data.kdims
            params['vdims'] = data.vdims
            params['label'] = data.label
            if data.group != data.params()['group'].default:
                params['group'] = data.group

        if isinstance(data, NdElement):
            pass
        elif isinstance(data, Columns):
            data = data.data
        elif isinstance(data, Element):
            dimensions = data.dimensions(label=True)
            columns = OrderedDict([(dim, data.dimension_values(dim))
                                   for dim in dimensions])
            if pd:
                data = pd.DataFrame(columns)
            else:
                data = OrderedDict([(row[:data.ndims], row[data.ndims:])
                                    for row in zip(*columns.values())])
        elif util.is_dataframe(data):
            kdims, vdims = cls._process_df_dims(data, paramobjs, **params)
            params['kdims'] = kdims
            params['vdims'] = vdims
        elif not isinstance(data, (np.ndarray, dict)):
            if isinstance(data, tuple):
                data = np.column_stack(data)
                array = data
            else:
                data = np.array() if data is None else list(data)
                array = np.array(data)
            # Check if data is of non-numeric type
            if array.dtype.kind in ['S', 'U', 'O'] or array.ndim > 2:
                # If data is in NdElement dictionary format or pandas
                # is not available convert to OrderedDict
                if ((not np.isscalar(data[0]) and len(data[0]) == 2 and
                    any(not np.isscalar(data[0][i]) for i in range(2)))
                    or not pd):
                    data = OrderedDict(data)
                else:
                    dimensions = (kwargs.get('kdims', ) +
                                  kwargs.get('vdims', paramobjs['vdims'].default))
                    columns = [d.name if isinstance(d, Dimension) else d
                               for d in dimensions]
                    data = pd.DataFrame(data, columns=columns)
            else:
                data = array
        params.update(kwargs)
        if 'kdims' not in params:
            params['kdims'] = paramobjs['kdims'].default
        if 'vdims' not in params:
            params['vdims'] = paramobjs['vdims'].default
        if isinstance(data, dict):
            data = NdElement(data, kdims=params['kdims'],
                             vdims=params['vdims'])
        elif util.is_dataframe(data):
            data = data.sort_values(by=[d.name if isinstance(d, Dimension) else d
                                        for dims in ['kdims', 'vdims'] for d in params[dims]])
        return data, params


    @classmethod
    def _process_df_dims(cls, data, paramobjs, **kwargs):
        if 'kdims' in kwargs or 'vdims' in kwargs:
            kdims = kwargs.get('kdims', [])
            vdims = kwargs.get('vdims', [])
            col_labels = [c.name if isinstance(c, Dimension) else c
                          for c in kdims+vdims]
            if not all(c in data.columns for c in col_labels):
                raise ValueError("Supplied dimensions don't match columns"
                                 "in the dataframe.")
        else:
            ndim = len(paramobjs['kdims'].default)
            kdims = list(data.columns[:ndim])
            vdims = list(data.columns[ndim:])
        return kdims, vdims


    @classmethod
    def _datarange(cls, data):
        """
        Should return minimum and maximum of data
        returned by values method.
        """
        raise NotImplementedError


    def range(self, dim, data_range=True):
        dim_idx = self.get_dimension_index(dim)
        if dim.range != (None, None):
            return dim.range
        elif dim_idx < len(self.dimensions()):
            if len(self):
                data = self.values(dim_idx)
                data_range = self._datarange(data)
            else:
                data_range = (np.NaN, np.NaN)
        if data_range:
            return util.max_range([data_range, dim.soft_range])
        else:
            return dim.soft_range


    def as_ndelement(self, **kwargs):
        """
        This method transforms any ViewableElement type into a Table
        as long as it implements a dimension_values method.
        """
        if self.kdims:
            keys = zip(*[self.values(dim.name)
                         for dim in self.kdims])
        else:
            keys = [()]*len(values)

        if self.vdims:
            values = zip(*[self.values(dim.name)
                           for dim in self.vdims])
        else:
            values = [()]*len(keys)

        data = zip(keys, values)
        params = dict(kdims=self.kdims, vdims=self.vdims, label=self.label)
        if not self.params()['group'].default == self.group:
            params['group'] = self.group
        el_type = type(self.element) 
        return el_type(data, **dict(params, **kwargs))


    def __len__(self):
        return len(self.element.data)


    @classmethod
    def validate_data(cls, data):
        return data


class ColumnarDataFrame(ColumnarData):

    def groupby(self, dimensions, container_type=HoloMap, **kwargs):
        invalid_dims = list(set(dimensions) - set(self.element.dimensions('key', True)))
        if invalid_dims:
            raise Exception('Following dimensions could not be found:\n%s.'
                            % invalid_dims)

        index_dims = [self.get_dimension(d) for d in dimensions]
        mapping = container_type(None, kdims=index_dims)
        for k, v in self.data.groupby(dimensions):
            data = v.drop(dimensions, axis=1)
            mapping[k] = self.clone(data,
                                    kdims=[self.get_dimension(d)
                                           for d in data.columns], **kwargs)
        return mapping


    @classmethod
    def reduce(cls, columns, reduce_dims, function=None):
        """
        The aggregate function accepts either a list of Dimensions
        and a function to apply to find the aggregate across
        those Dimensions or a list of dimension/function pairs
        to apply one by one.
        """
        reduced = columns.data
        kdims = [kdim.name for kdim in columns.kdims if kdim not in reduce_dims]
        vdims = columns.dimensions('value', True)
        if kdims:
            reduced = reduced.reindex(columns=kdims+vdims).groupby(kdims).aggregate(function).reset_index()
        else:
            if isinstance(function, np.ufunc):
                reduced = function.reduce(columns.data, axis=0)
            else:
                reduced = function(columns.data, axis=0)[vdims]
            if len(reduced) == 1:
                reduced = reduced[0]
            else:
                reduced = pd.DataFrame([reduced], columns=vdims)
        return reduced


    def array(self):
        return self.element.data.values


    def reindex(self, kdims=None, vdims=None):
        # DataFrame based tables don't need to be reindexed
        return self.element.data


    @classmethod
    def _datarange(cls, data): 
        return data.min(), data.max()


    @classmethod
    def collapse_data(cls, data, function, kdims, **kwargs):
        return pd.concat(data).groupby([d.name for d in kdims]).agg(function).reset_index()


    def select(self, selection_specs=None, **select):
        """
        Allows slice and select individual values along the DataFrameView
        dimensions. Supply the dimensions and values or slices as
        keyword arguments.
        """
        df = self.element.data
        selected_kdims = []
        slcs = []
        for dim, k in select.items():
            if isinstance(k, tuple):
                k = slice(*k)
            if isinstance(k, slice):
                if k.start is not None:
                    slcs.append(k.start < df[dim])
                if k.stop is not None:
                    slc.append(df[dim] < k.stop)
            elif isinstance(k, (set, list)):
                iter_slcs = []
                for ik in k:
                    iter_slcs.append(df[dim] == ik)
                slcs.append(np.logical_or.reduce(iter_slcs))
            else:
                if dim in self.element.kdims: selected_kdims.append(dim)
                slcs.append(df[dim] == k)
        df = df.iloc[np.logical_and.reduce(slcs)]
        if len(set(selected_kdims)) == self.element.ndims:
            if len(df) and len(self.element.vdims) == 1:
                df = df[self.element.vdims[0].name].iloc[0]
        return df


    def values(self, dim):
        data = self.element.data[dim]
        if util.dd and isinstance(data, util.dd.Series):
            data = data.compute()
        return np.array(data)


    def sample(self, samples=[]):
        """
        Sample the Element data with a list of samples.
        """
        data = self.element.data
        mask = np.zeros(len(self), dtype=bool)
        for sample in samples:
            if np.isscalar(sample): sample = [sample]
            for i, v in enumerate(sample):
                mask = np.logical_or(mask, data.iloc[:, i]==v)
        return data[mask]


    @classmethod
    def add_dimension(cls, data, dimension, dim_pos, values):
        data.insert(dim_pos, dimension.name, values)
        return data


    def dframe(self, as_table=False):
        if as_table:
            from ..element import Table
            params = self.element.get_param_values(onlychanged=True)
            return Table(self.element.data, **params)
        return self.element.data


class ColumnarArray(ColumnarData):

    @classmethod
    def validate_data(cls, data):
        if data.ndim == 1:
            data = np.column_stack([np.arange(len(data)), data])
        return data


    @classmethod
    def add_dimension(cls, data, dimension, dim_pos, values):
        return np.insert(data, dim_pos, values, axis=1)


    def array(self):
        return self.element.data

    def dframe(self, as_table=False):
        return Element.dframe(self.element, as_table)


    def closest(self, coords):
        """
        Given single or multiple x-values, returns the list
        of closest actual samples.
        """
        if not isinstance(coords, list): coords = [coords]
        xs = self.element.data[:, 0]
        idxs = [np.argmin(np.abs(xs-coord)) for coord in coords]
        return [xs[idx] for idx in idxs] if len(coords) > 1 else xs[idxs[0]]


    @classmethod
    def _datarange(cls, data): 
        return np.nanmin(data), np.nanmax(data)


    def values(self, dim):
        data = self.element.data
        dim_idx = self.element.get_dimension_index(dim)
        if data.ndim == 1:
            data = np.atleast_2d(data).T
        return data[:, dim_idx]


    def reindex(self, kdims=None, vdims=None):
        # DataFrame based tables don't need to be reindexed
        dims = kdims + vdims
        data = [self.element.dimension_values(d) for d in dims]
        return np.column_stack(data)


    def groupby(self, dimensions, container_type=HoloMap, **kwargs):
        data = self.element.data

        # Get dimension objects, labels, indexes and data
        dimensions = [self.element.get_dimension(d) for d in dimensions]
        dim_idxs = [self.element.get_dimension_index(d) for d in dimensions]
        dim_data = {d: self.element.dimension_values(d) for d in dimensions}
        ndims = len(dimensions)
        kwargs['kdims'] = [kdim for kdim in self.element.kdims
                           if kdim not in dimensions]

        # Find unique entries along supplied dimensions
        indices = data[:, dim_idxs]
        view = indices.view(np.dtype((np.void, indices.dtype.itemsize * indices.shape[1])))
        _, idx = np.unique(view, return_index=True)
        unique_indices = indices[idx]

        # Iterate over the unique entries building masks
        # to apply the group selection
        grouped_data = []
        for group in unique_indices:
            mask = np.zeros(len(data), dtype=bool)
            for d, v in zip(dimensions, group):
                mask = np.logical_or(mask, dim_data[d] == v)
            group_element = self.element.clone(data[mask, ndims:], **kwargs)
            grouped_data.append((tuple(group), group_element))
        return container_type(grouped_data, kdims=dimensions)


    def select(self, **selection):
        data = self.element.data
        selected_kdims = []
        value = selection.pop('value', None)
        for d, slc in selection.items():
            idx = self.element.get_dimension_index(d)
            if isinstance(slc, slice):
                start = -float("inf") if slc.start is None else slc.start
                stop = float("inf") if slc.stop is None else slc.stop
                clip_start = start <= data[:, idx]
                clip_stop = data[:, idx] < stop
                data = data[np.logical_and(clip_start, clip_stop), :]
            elif isinstance(slc, (set, list)):
                filt = np.in1d(data[:, idx], list(slc))
                data = data[filt, :]
            else:
                if d in self.element.kdims: selected_kdims.append(d)
                if self.element.ndims == 1:
                    data_index = np.argmin(np.abs(data[:, idx] - slc))
                else:
                    data_index = data[:, idx] == slc
                data = np.atleast_2d(data[data_index, :])
        if len(data) and len(set(selected_kdims)) == self.element.ndims:
            if len(data) == 1 and len(self.element.vdims) == 1:
                data = data[0, self.element.ndims]
        return data


    @classmethod
    def collapse_data(cls, data, function, kdims=None, **kwargs):
        """
        Applies a groupby operation along the supplied key dimensions
        then aggregates across the groups with the supplied function.
        """
        ndims = data[0].shape[1]
        nkdims = len(kdims)
        vdims = ['Value Dimension %s' % i for i in range(ndims-len(kdims))]
        joined_data = Columns(np.concatenate(data), kdims=kdims, vdims=vdims)

        rows = []
        for k, group in joined_data.groupby(kdims).items():
            row = np.zeros(ndims)
            row[:ndims] = np.array(k)
            for i, vdim in enumerate(group.vdims):
                group_data = group.dimension_values(vdim)
                if isinstance(function, np.ufunc):
                    collapsed = function.reduce(group_data)
                else:
                    collapsed = function(group_data, axis=0, **kwargs)
                row[nkdims+i] = collapsed
            rows.append(row)
        return np.array(rows)


    def sample(self, samples=[]):
        """
        Sample the Element data with a list of samples.
        """
        data = self.element.data
        mask = np.zeros(len(self), dtype=bool)
        for sample in samples:
            if np.isscalar(sample): sample = [sample]
            for i, v in enumerate(sample):
                mask = np.logical_or(mask, data[:, i]==v)
        return data[mask]


    @classmethod
    def reduce(cls, columns, reduce_dims, function):
        """
        This implementation allows reducing dimensions by aggregating
        over all the remaining key dimensions using the collapse_data
        method.
        """
        kdims = [kdim for kdim in columns.kdims if kdim not in reduce_dims]
        if len(kdims):
            reindexed = columns.reindex(kdims)
            reduced = reindexed.collapse_data([reindexed.data], function, kdims)
        else:
            if isinstance(function, np.ufunc):
                reduced = function.reduce(columns.data, axis=0)
            else:
                reduced = function(columns.data, axis=0)
            reduced = reduced[columns.ndims:]
        if reduced.ndim == 1:
            if len(reduced) == 1:
                return reduced[0]
            else:
                return np.atleast_2d(reduced)
        return reduced

