import numpy as np

import param

from ..core import util
from ..core import OrderedDict, Dimension, UniformNdMapping, Element, Element2D, NdElement, HoloMap
from .tabular import ItemTable, Table
from .util import compute_edges

class Chart(Element2D):
    """
    The data held within Chart is a numpy array of shape (N, D),
    where N is the number of samples and D the number of dimensions.
    Chart Elements are sliceable along up to two key dimensions.
    The data may be supplied in one of three formats:

    1) As a numpy array of shape (N, D).
    2) As a list of length N containing tuples of length D.
    3) As a tuple of length D containing iterables of length N.
    """

    kdims = param.List(default=[Dimension('x')], bounds=(1,2), doc="""
        The key dimensions of the Chart, determining the number of
        indexable dimensions.""")

    group = param.String(default='Chart', constant=True)

    vdims = param.List(default=[Dimension('y')], bounds=(1,None), doc="""
        The value dimensions of the Chart, usually corresponding to a
        number of dependent variables.""")

    _null_value = np.array([[], []]).T # For when data is None

    def __init__(self, data, **kwargs):
        data, params = self._process_data(data)
        params.update(kwargs)
        super(Chart, self).__init__(data, **params)
        self.data = self._validate_data(self.data)


    def _convert_element(self, element):
        if isinstance(element, Chart):
            return element.data
        elif isinstance(element, NdElement):
            return np.vstack([np.concatenate([key, vals])
                              for key, vals in element.data.items()]).astype(np.float)
        else:
            return super(Chart, self)._convert_element(element)


    def _process_data(self, data):
        params = {}
        if isinstance(data, UniformNdMapping) or (isinstance(data, list) and data
                                                  and isinstance(data[0], Element2D)):
            params = dict([v for v in data][0].get_param_values(onlychanged=True))
            data = np.concatenate([v.data for v in data])
        elif isinstance(data, Element):
            pass
        elif isinstance(data, tuple):
            data = np.column_stack(data)
        elif not isinstance(data, np.ndarray):
            data = self._null_value if (data is None) else list(data)
            if len(data):
                data = np.array(data)

        return data, params


    def _validate_data(self, data):
        if data.ndim == 1:
            data = np.array(list(zip(range(len(data)), data)))
        if not data.shape[1] == len(self.dimensions()):
            raise ValueError("Data has to match number of key and value dimensions")
        return data


    def closest(self, coords):
        """
        Given single or multiple x-values, returns the list
        of closest actual samples.
        """
        if not isinstance(coords, list): coords = [coords]
        xs = self.data[:, 0]
        idxs = [np.argmin(np.abs(xs-coord)) for coord in coords]
        return [xs[idx] for idx in idxs]


    def __getitem__(self, slices):
        """
        Implements slicing or indexing of the data by the data x-value.
        If a single element is indexed reduces the Element2D to a single
        Scatter object.
        """
        if slices is ():
            return self
        if not isinstance(slices, tuple): slices = (slices,)
        if len(slices) > self.ndims:
            raise Exception("Slice must match number of key dimensions.")

        data = self.data
        lower_bounds, upper_bounds = [], []
        for idx, slc in enumerate(slices):
            if isinstance(slc, slice):
                start = -float("inf") if slc.start is None else slc.start
                stop = float("inf") if slc.stop is None else slc.stop

                clip_start = start <= data[:, idx]
                clip_stop = data[:, idx] < stop
                data = data[np.logical_and(clip_start, clip_stop), :]
                lbound = self.extents[idx]
                ubound = self.extents[self.ndims:][idx]
                lower_bounds.append(lbound if slc.start is None else slc.start)
                upper_bounds.append(ubound if slc.stop is None else slc.stop)
            else:
                if self.ndims == 1:
                    data_index = np.argmin(np.abs(data[:, idx] - slc))
                    data = data[data_index, :]
                else:
                    raise KeyError("Only 1D Chart types may be indexed.")
        if not any(isinstance(slc, slice) for slc in slices):
            if data.ndim == 1:
                data = data[self.ndims:]
                dims = data.shape[0]
            else:
                data = data[:, self.ndims:]
                dims = data.shape[1]
            return data[0] if dims == 1 else data
        if self.ndims == 1:
            lower_bounds.append(None)
            upper_bounds.append(None)

        return self.clone(data, extents=tuple(lower_bounds + upper_bounds))


    @classmethod
    def collapse_data(cls, data, function, **kwargs):
        new_data = [arr[:, 1:] for arr in data]
        if isinstance(function, np.ufunc):
            collapsed = function.reduce(new_data)
        else:
            collapsed = function(np.dstack(new_data), axis=-1, **kwargs)
        return np.hstack([data[0][:, 0, np.newaxis], collapsed])


    def sample(self, samples=[]):
        """
        Allows sampling of Chart Elements using the default
        syntax of providing a map of dimensions and sample pairs.
        """
        sample_data = OrderedDict()
        for sample in samples:
            data = self[sample]
            data = data if np.isscalar(data) else tuple(data)
            sample_data[sample] = data
        params = dict(self.get_param_values(onlychanged=True))
        params.pop('extents', None)
        return Table(sample_data, **dict(params, kdims=self.kdims,
                                         vdims=self.vdims))


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
        params = dict(self.get_param_values(onlychanged=True), vdims=self.vdims,
                      kdims=[])
        params.pop('extents', None)
        return ItemTable(reduced_data, **params)


    def __len__(self):
        return len(self.data)


    def dimension_values(self, dim):
        index = self.get_dimension_index(dim)
        if index < len(self.dimensions()):
            return self.data[:, index]
        else:
            return super(Chart, self).dimension_values(dim)


    def range(self, dim, data_range=True):
        dim_idx = dim if isinstance(dim, int) else self.get_dimension_index(dim)
        dim = self.get_dimension(dim_idx)
        if dim.range != (None, None):
            return dim.range
        elif dim_idx < len(self.dimensions()):
            if self.data.ndim == 1:
                data = np.atleast_2d(self.data).T
            else:
                data = self.data
            if len(data):
                data = data[:, dim_idx]
                data_range = np.nanmin(data), np.nanmax(data)
            else:
                data_range = (np.NaN, np.NaN)
        if data_range:
            return util.max_range([data_range, dim.soft_range])
        else:
            return dim.soft_range


    def dframe(self):
        import pandas as pd
        columns = [d.name for d in self.dimensions()]
        return pd.DataFrame(self.data, columns=columns)



class Scatter(Chart):
    """
    Scatter is a Element2D type which gets displayed as a number of
    disconnected points.
    """

    group = param.String(default='Scatter', constant=True)

    @classmethod
    def collapse_data(cls, data, function=None, **kwargs):
        if function:
            raise Exception("Scatter elements are inhomogenous and "
                            "cannot be collapsed with a function.")
        return np.concatenate(data)


class Curve(Chart):
    """
    Curve is a simple Chart Element providing 1D indexing along
    the x-axis.
    """

    group = param.String(default='Curve', constant=True)

    def progressive(self):
        """
        Create map indexed by Curve x-axis with progressively expanding number
        of curve samples.
        """
        vmap = HoloMap(None, kdims=self.kdims,
                       title=self.title+' {dims}')
        for idx in range(len(self.data)):
            x = self.data[0]
            if x in vmap:
                vmap[x].data.append(self.data[0:idx])
            else:
                vmap[x] = self.clone(self.data[0:idx])
        return vmap



class ErrorBars(Chart):
    """
    ErrorBars is a Chart Element type representing any number of
    errorbars situated in a 2D space. The errors must be supplied
    as an Nx3 or Nx4 array representing the x/y-positions and
    either the symmetric error or assymetric errors respectively.
    Internally the data is always held as an Nx4 array with the
    lower and upper bounds.
    """

    group = param.String(default='ErrorBars', constant=True, doc="""
        A string describing the quantitity measured by the ErrorBars
        object.""")

    kdims = param.List(default=[Dimension('x'), Dimension('y')],
                       bounds=(2, 2), constant=True, doc="""
        The Dimensions corresponding to the x- and y-positions of
        the error bars.""")

    vdims = param.List(default=[Dimension('lerror'), Dimension('uerror')],
                       bounds=(2,2), constant=True)

    def _validate_data(self, data):
        if data.shape[1] == 3:
            return np.column_stack([data, data[:, 2]])
        else:
            return data


    def range(self, dim, data_range=True):
        drange = super(ErrorBars, self).range(dim, data_range)
        didx = self.get_dimension_index(dim)
        if didx == 1 and data_range:
            lower = np.nanmin(self.data[:, 1] - self.data[:, 2])
            upper = np.nanmax(self.data[:, 1] + self.data[:, 3])
            return util.max_range([(lower, upper), drange])
        else:
            return drange



class Spread(ErrorBars):
    """
    Spread is a Chart Element type respresenting a spread of
    values as given by a mean and standard error or confidence
    intervals. Just like the ErrorBars Element type, mean and
    deviations from the mean should be supplied as either an
    Nx3 or Nx4 array representing the x-values, mean values
    and symmetric or assymetric errors respective. Internally
    the data is always expanded to an Nx4 array.
    """

    group = param.String(default='Spread', constant=True)



class Bars(NdElement):
    """
    Bars is an Element type, representing a number of stacked and
    grouped bars, depending the dimensionality of the key and value
    dimensions. Bars is useful for categorical data, which may be
    laid via groups, categories and stacks. Internally Bars is
    a NdElement with up to three key dimensions and a single value
    dimension.
    """

    group = param.String(default='Bars', constant=True)

    kdims = param.List(default=[Dimension('x')], bounds=(1,3))

    vdims = param.List(default=[Dimension('y')], bounds=(1,1))



class Histogram(Element2D):
    """
    Histogram contains a number of bins, which are defined by the
    upper and lower bounds of their edges and the computed bin values.
    """

    kdims = param.List(default=[Dimension('x')], bounds=(1,1), doc="""
        Dimensions on Element2Ds determine the number of indexable
        dimensions.""")

    group = param.String(default='Histogram', constant=True)

    vdims = param.List(default=[Dimension('Frequency')])

    def __init__(self, values, edges=None, extents=None, **params):
        self.values, self.edges, settings = self._process_data(values, edges)
        settings.update(params)
        super(Histogram, self).__init__((self.values, self.edges), **settings)
        self._width = None
        self._extents = (None, None, None, None) if extents is None else extents


    def __getitem__(self, key):
        """
        Implements slicing or indexing of the Histogram
        """
        if key is (): return self # May no longer be necessary
        if isinstance(key, tuple) and len(key) > self.ndims:
            raise Exception("Slice must match number of key dimensions.")

        centers = [(float(l)+r)/2 for (l,r) in zip(self.edges, self.edges[1:])]
        if isinstance(key, slice):
            start, stop = key.start, key.stop
            if [start, stop] == [None,None]: return self
            start_idx, stop_idx = None,None
            if start is not None:
                start_idx = np.digitize([start], centers, right=True)[0]
            if stop is not None:
                stop_idx = np.digitize([stop], centers, right=True)[0]

            slice_end = stop_idx+1 if stop_idx is not None else None
            slice_values = self.values[start_idx:stop_idx]
            slice_edges =  self.edges[start_idx: slice_end]

            extents = (min(slice_edges), self.extents[1],
                       max(slice_edges), self.extents[3])
            return self.clone((slice_values, slice_edges), extents=extents)
        else:
            if not (self.edges.min() <= key < self.edges.max()):
                raise Exception("Key value %s is out of the histogram bounds" % key)
            idx = np.digitize([key], self.edges)[0]
            return self.values[idx-1 if idx>0 else idx]



    def _process_data(self, values, edges):
        """
        Ensure that edges are specified as left and right edges of the
        histogram bins rather than bin centers.
        """
        settings = {}
        (values, edges) = values if isinstance(values, tuple) else (values, edges)
        if isinstance(values, Element2D):
            settings = dict(values.get_param_values(onlychanged=True))
            edges = values.data[:, 0].copy()
            values = values.data[:, 1].copy()
        elif isinstance(values, np.ndarray) and len(values.shape) == 2:
            edges = values[:, 0]
            values = values[:, 1]
        elif all(isinstance(el, tuple) for el in values):
            edges, values = zip(*values)
        else:
            values = np.array(values)
            if edges is None:
                edges = np.arange(len(values), dtype=np.float)
            else:
                edges = np.array(edges, dtype=np.float)

        if len(edges) == len(values):
            edges = compute_edges(edges)
        return values, edges, settings


    @property
    def extents(self):
        if any(lim is not None for lim in self._extents):
            return self._extents
        else:
            return (np.min(self.edges), None, np.max(self.edges), None)


    @extents.setter
    def extents(self, extents):
        self._extents = extents


    def dimension_values(self, dim):
        dim = self.get_dimension(dim).name
        if dim in self._cached_value_names:
            return self.values
        elif dim in self._cached_index_names:
            return np.convolve(self.edges, np.ones((2,))/2, mode='valid')
        else:
            return super(Histogram, self).dimension_values(dim)


    def sample(self, samples=[], **sample_values):
        raise NotImplementedError('Cannot sample a Histogram.')


    def reduce(self, dimensions=None, function=None, **reduce_map):
        raise NotImplementedError('Reduction of Histogram not implemented.')



class Points(Chart):
    """
    Allows sets of points to be positioned over a sheet coordinate
    system. Each points may optionally be associated with a chosen
    numeric value.

    The input data can be a Nx2 or Nx3 Numpy array where the first two
    columns corresponds to the X,Y coordinates in sheet coordinates,
    within the declared bounding region. For Nx3 arrays, the third
    column corresponds to the magnitude values of the points. Any
    additional columns will be ignored (use VectorFields instead).

    The input data may be also be passed as a tuple of elements that
    may be numpy arrays or values that can be cast to arrays. When
    such a tuple is supplied, the elements are joined column-wise into
    a single array, allowing the magnitudes to be easily supplied
    separately.

    Note that if magnitudes are to be rendered correctly by default,
    they should lie in the range [0,1].
    """

    kdims = param.List(default=[Dimension('x'), Dimension('y')],
                       bounds=(2, 2), constant=True, doc="""
        The label of the x- and y-dimension of the Points in form
        of a string or dimension object.""")

    group = param.String(default='Points', constant=True)

    vdims = param.List(default=[])


    _min_dims = 2                      # Minimum number of columns

    def __len__(self):
        return self.data.shape[0]

    def __iter__(self):
        i = 0
        while i < len(self):
            yield tuple(self.data[i, ...])
            i += 1

    @classmethod
    def collapse_data(cls, data, function, **kwargs):
        return Scatter.collapse_data(data, function, **kwargs)



class VectorField(Points):
    """
    A VectorField contains is a collection of vectors where each
    vector has an associated position in sheet coordinates.

    The constructor of VectorField is similar to the constructor of
    Points: the input data can be an NxM Numpy array where the first
    two columns corresponds to the X,Y coordinates in sheet
    coordinates, within the declared bounding region. As with Points,
    the input can be a tuple of array objects or of objects that can
    be cast to arrays (the tuple elements are joined column-wise).

    The third column maps to the vector angle which must be specified
    in radians. Note that it is possible to supply a collection which
    isn't a numpy array, whereby each element of the collection is
    assumed to be an iterable corresponding to a single column of the
    NxM array.

    The visualization of any additional columns is decided by the
    plotting code. For instance, the fourth and fifth columns could
    correspond to arrow length and colour map value. All that is
    assumed is that these additional dimension are normalized between
    0.0 and 1.0 for the default visualization to work well.

    The only restriction is that the final data array is NxM where
    M>3. In other words, the vector must have a dimensionality of 2 or
    higher.
    """

    group = param.String(default='VectorField', constant=True)

    vdims = param.List(default=[Dimension('Angle', cyclic=True, range=(0,2*np.pi)),
                                Dimension('Magnitude')], bounds=(1, 2))

    _null_value = np.array([[], [], [], []]).T # For when data is None
    _min_dims = 3                              # Minimum number of columns

    def __init__(self, data, **params):
        if not isinstance(data, np.ndarray):
            data = np.array([
                [el for el in (col.flat if isinstance(col,np.ndarray) else col)]
                for col in data]).T
        super(VectorField, self).__init__(data, **params)
