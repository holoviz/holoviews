"""
The data module provides utility classes to interface with various
data backends.
"""

from collections import defaultdict
from itertools import groupby

import numpy as np
import param

from .dimension import OrderedDict, Dimension
from .element import Element, NdElement
from .spaces import HoloMap
from . import util


class Columns(Element):

    def __init__(self, data, **kwargs):
        if 'kdims' not in kwargs:
            kwargs['kdims'] = self.kdims
        if 'vdims' not in kwargs:
            kwargs['vdims'] = self.vdims
        data, params = ColumnarData._process_data(data, **kwargs)
        super(Columns, self).__init__(data, **params)
        self.data = self._validate_data(self.data)


    def _validate_data(self, data):
        if self.interface is None:
            return data
        else:
            return self.interface.validate_data(data)


    def select(self, selection_specs=None, **selection):
        if self.interface is None:
            data = self.data.select(**selection)
        else:
            data = self.interface.select(selection) 
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
            vdims = self._cached_value_names
        elif kdims is None:
            dimensions = (self._cached_index_names +
                          self._cached_value_names)
            kdims = [d for d in dimensions if d not in vdims]
        key_dims = [self.get_dimension(k) for k in kdims]
        val_dims = [self.get_dimension(v) for v in vdims]
        data = self.interface.reindex(self.data, key_dims, val_dims)
        return self.clone(data, key_dims, val_dims)


    def __getitem__(self, slices):
        """
        Implements slicing or indexing of the data by the data x-value.
        If a single element is indexed reduces the Element2D to a single
        Scatter object.
        """
        if slices is (): return self
        if not isinstance(slices, tuple): slices = (slices,)
        selection = dict(zip(self.dimensions(label=True), slices))
        if self.interface is None:
            data = self.data.select(**selection)
        else:
            data = self.interface.select(**selection) 
        return self.clone(data)


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
        Allows collapsing of Chart objects using the supplied map of
        dimensions and reduce functions.
        """
        reduce_map = self._reduce_map(dimensions, function, reduce_map)

        if len(reduce_map) > 1:
            raise ValueError("Chart Elements may only be reduced to a point.")
        dim, reduce_fn = list(reduce_map.items())[0]
        if dim in self._cached_index_names:
            reduced_data = OrderedDict(zip(self.vdims, reduce_fn(self.data[:, self.ndims:], axis=0)))
        else:
            raise Exception("Dimension %s not found in %s" % (dim, type(self).__name__))
        return self.clone(reduced_data)


    def groupby(self, dimensions, container_type=HoloMap, **kwargs):
        if self.interface is None:
            return self.data.groupby(dimensions, container_type, **kwargs)
        else:
            return self.interface.groupby(dimensions, container_type, **kwargs)


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
            return self.interface.values(dim)


    def dframe(self):
        if self.interface is None:
            return self.data.dframe()
        else:
            return self.interface.dframe()


    def array(self):
        if self.interface is None:
            dims = self._cached_index_names + self._cached_value_names
            return np.column_stack([self.dimension_values(d) for d in dims])
        else:
            return self.interface.array()




class ColumnarData(param.Parameterized):

    def __init__(self, element, **params):
        self.element = element


    def array(self):
        NotImplementedError

    @property
    def ndims(self):
        self.element.ndims


    @property
    def shape(self):
        return self.element.data.shape


    @classmethod
    def _process_data(cls, data, **kwargs):
        params = {}
        if isinstance(data, NdElement):
            params['kdims'] = data.kdims
            params['vdims'] = data.vdims
            params['label'] = data.label
        elif isinstance(data, Element):
            params = dict(data.get_param_values(onlychanged=True))
            data = data.data
        elif util.is_dataframe(data):
            kdims, vdims = cls._process_df_dims(data, params)
            params['kdims'] = kdims
            params['vdims'] = vdims
        elif isinstance(data, tuple):
            data = np.column_stack(data)
        elif not isinstance(data, (np.ndarray, dict)):
            data = np.array() if data is None else list(data)
            if all(np.isscalar(d) for coord in data for d in coord):
                data = np.array(data)
            elif len(data):
                data = OrderedDict(data)
        params.update(kwargs)
        if isinstance(data, dict):
            data = NdElement(data, kdims=params['kdims'],
                             vdims=params['vdims'])
        return data, params


    @classmethod
    def _process_df_dims(cls, data, kwargs):
        if 'kdims' in kwargs or 'vdims' in kwargs:
            kdims = kwargs.get('kdims', [])
            vdims = kwargs.get('vdims', [])
            col_labels = [c.name if isinstance(c, Dimension) else c
                          for c in kdims+vdims]
            if not all(c in data.columns for c in col_labels):
                raise ValueError("Supplied dimensions don't match columns"
                                 "in the dataframe.")
        else:
            kdims = list(data.columns[:2])
            vdims = list(data.columns[2:])
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
        keys = zip(*[self.values(dim.name)
                     for dim in self.kdims])
        values = zip(*[self.values(dim.name)
                       for dim in self.vdims])
        if not keys: keys = [()]*len(values)
        if not values: [()]*len(keys)
        data = zip(keys, values)
        kwargs = {'label': self.label
                  for k, v in self.get_param_values(onlychanged=True)
                  if k in ['group', 'label']}
        params = dict(kdims=self.kdims,
                      vdims=self.vdims,
                      label=self.label)
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
        invalid_dims = list(set(dimensions) - set(self._cached_index_names))
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


    def reduce(self, dimensions=[], function=None, **reductions):
        """
        The aggregate function accepts either a list of Dimensions
        and a function to apply to find the aggregate across
        those Dimensions or a list of dimension/function pairs
        to apply one by one.
        """
        if not dimensions and not reductions:
            raise Exception("Supply either a list of Dimensions or"
                            "reductions as keyword arguments")
        reduced = self.element.data
        dfnumeric = reduced.applymap(np.isreal).all(axis=0)
        unreducable = list(dfnumeric[dfnumeric == False].index)
        if dimensions:
            if not function:
                raise Exception("Supply a function to reduce the Dimensions with.")
            reductions.update({d: function for d in dimensions})
        if reductions:
            reduce_ops = defaultdict(list)
            for d, fn in reductions.items(): reduce_ops[fn].append(fn)
            for fn, dims  in reduce_ops.items():
                reduced = reduced.groupby(dims, as_index=True).aggregate(fn)
                reduced_indexes = [reduced.index.names.index(d) for d in unreducable]
                reduced = reduced.reset_index(level=reduced_indexes)
        kdims = [self.element.get_dimension(d) for d in reduced.columns]
        return self.element.clone(reduced, kdims=kdims)


    def array(self):
        return self.element.data.iloc

    def reindex(self, kdims=None, vdims=None):
        # DataFrame based tables don't need to be reindexed
        return self.element.data


    @classmethod
    def _datarange(cls, data): 
        return data.min(), data.max()


    def select(self, selection_specs=None, **select):
        """
        Allows slice and select individual values along the DataFrameView
        dimensions. Supply the dimensions and values or slices as
        keyword arguments.
        """
        df = self.element.data
        for dim, k in select.items():
            if isinstance(k, tuple):
                k = slice(*k)
            if isinstance(k, slice):
                df = df[(k.start < df[dim]) & (df[dim] < k.stop)]
            else:
                df = df[df[dim] == k]
        return df


    def values(self, dim):
        data = self.element.data[dim]
        if util.dd and isinstance(data, util.dd.Series):
            data = data.compute()
        return data


    @classmethod
    def add_dimension(cls, data, dimension, values):
        data[dimension] = values
        return data


    def dframe(self):
        return self.element.data


class ColumnarArray(ColumnarData):

    @classmethod
    def validate_data(cls, data):
        if data.ndim == 1:
            data = np.column_stack([np.arange(len(data)), data])
        return data


    def array(self):
        return self.element.data

    @classmethod
    def add_dimension(cls, data, dimension, values):
        if np.isscalar(values):
            values = [values]*len(data)
        return np.column_stack([data, values])


    def closest(self, coords):
        """
        Given single or multiple x-values, returns the list
        of closest actual samples.
        """
        if not isinstance(coords, list): coords = [coords]
        xs = self.data[:, 0]
        idxs = [np.argmin(np.abs(xs-coord)) for coord in coords]
        return [xs[idx] for idx in idxs]


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
        dim_labels = [d.name for d in dimensions]
        dim_idxs = [self.element.get_dimension_index(d) for d in dimensions]
        dim_data = {d: self.element.dimension_values(d) for d in dim_labels}

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
            for d, v in zip(dim_labels, group):
                mask = np.logical_or(mask, dim_data[d] == v)
            group_element = self.element.clone(data[mask, :], **kwargs)
            grouped_data.append((tuple(group), group_element))
        return container_type(grouped_data, kdims=dimensions)


    def select(self, **selection):
        data = self.element.data
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
                if self.element.ndims == 1:
                    data_index = np.argmin(np.abs(data[:, idx] - slc))
                    data = data[data_index, :]
                else:
                    data = data[data[:, idx] == slc, :]
        return data


    @classmethod
    def collapse_data(cls, data, function, **kwargs):
        new_data = [arr[:, self.ndim:] for arr in data]
        if isinstance(function, np.ufunc):
            collapsed = function.reduce(new_data)
        else:
            collapsed = function(np.dstack(new_data), axis=-1, **kwargs)
        return np.hstack([data[0][:, self.ndims:, np.newaxis], collapsed])


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


    def reduce(self, dimensions=[], function=None, **reduce_map):
        """
        Allows collapsing of Chart objects using the supplied map of
        dimensions and reduce functions.
        """
        reduce_map = self._reduce_map(dimensions, function, reduce_map)

        dim, reduce_fn = list(reduce_map.items())[0]
        if dim in self._cached_index_names:
            reduced_data = OrderedDict(zip(self.vdims, reduce_fn(self.data[:, self.ndims:], axis=0)))
        else:
            raise Exception("Dimension %s not found in %s" % (dim, type(self).__name__))
        params = dict(self.get_param_values(onlychanged=True), vdims=self.vdims,
                      kdims=[])
        params.pop('extents', None)
        return ItemTable(reduced_data, **params)


    def dframe(self):
        import pandas as pd
        column_names = self.dimensions(label=True)
        dim_vals = np.vstack([self.dimension_values(dim) for dim in column_names]).T
        return pd.DataFrame(dim_vals, columns=column_names)
