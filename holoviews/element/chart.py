import numpy as np

import param

from ..core import util
from ..core import Dimension, Dataset, Element2D
from .util import compute_edges

class Chart(Dataset, Element2D):
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

    def __getitem__(self, index):
        sliced = super(Chart, self).__getitem__(index)
        if not isinstance(sliced, Chart):
            return sliced

        if not isinstance(index, tuple): index = (index,)
        ndims = len(self.extents)//2
        lower_bounds, upper_bounds = [None]*ndims, [None]*ndims
        for i, slc in enumerate(index[:ndims]):
            if isinstance(slc, slice):
                lbound = self.extents[i]
                ubound = self.extents[ndims:][i]
                lower_bounds[i] = lbound if slc.start is None else slc.start
                upper_bounds[i] = ubound if slc.stop is None else slc.stop
        sliced.extents = tuple(lower_bounds+upper_bounds)
        return sliced


class Scatter(Chart):
    """
    Scatter is a Element2D type which gets displayed as a number of
    disconnected points.
    """

    group = param.String(default='Scatter', constant=True)



class Curve(Chart):
    """
    Curve is a simple Chart Element providing 1D indexing along
    the x-axis.
    """

    group = param.String(default='Curve', constant=True)



class ErrorBars(Chart):
    """
    ErrorBars is a Chart Element type representing any number of
    errorbars situated in a 2D space. The errors must be supplied
    as an Nx3 or Nx4 array representing the x/y-positions and
    either the symmetric error or asymmetric errors respectively.
    """

    group = param.String(default='ErrorBars', constant=True, doc="""
        A string describing the quantity measured by the ErrorBars
        object.""")

    kdims = param.List(default=[Dimension('x')],
                       bounds=(1, 2), constant=True, doc="""
        The Dimensions corresponding to the x- and y-positions of
        the error bars.""")

    vdims = param.List(default=[Dimension('y'), Dimension('yerror')],
                       bounds=(1, 3), constant=True)


    def range(self, dim, data_range=True):
        didx = self.get_dimension_index(dim)
        dim = self.get_dimension(dim)
        if didx == 1 and data_range and len(self):
            mean = self.dimension_values(1)
            neg_error = self.dimension_values(2)
            if len(self.dimensions()) > 3:
                pos_error = self.dimension_values(3)
            else:
                pos_error = neg_error
            lower = np.nanmin(mean-neg_error)
            upper = np.nanmax(mean+pos_error)
            return util.dimension_range(lower, upper, dim)
        return super(ErrorBars, self).range(dim, data_range)



class Spread(ErrorBars):
    """
    Spread is a Chart Element type representing a spread of
    values as given by a mean and standard error or confidence
    intervals. Just like the ErrorBars Element type, mean and
    deviations from the mean should be supplied as either an
    Nx3 or Nx4 array representing the x-values, mean values
    and symmetric or asymmetric errors respective. Internally
    the data is always expanded to an Nx4 array.
    """

    group = param.String(default='Spread', constant=True)



class Bars(Chart):
    """
    Bars is an Element type, representing a number of stacked and
    grouped bars, depending the dimensionality of the key and value
    dimensions. Bars is useful for categorical data, which may be
    laid via groups, categories and stacks.
    """

    group = param.String(default='Bars', constant=True)

    kdims = param.List(default=[Dimension('x')], bounds=(1,3))

    vdims = param.List(default=[Dimension('y')], bounds=(1, None))



class BoxWhisker(Chart):
    """
    BoxWhisker represent data as a distributions highlighting
    the median, mean and various percentiles.
    """

    group = param.String(default='BoxWhisker', constant=True)

    kdims = param.List(default=[], bounds=(0,None))

    vdims = param.List(default=[Dimension('y')], bounds=(1,1))

    _auto_indexable_1d = False


class Histogram(Element2D):
    """
    Histogram contains a number of bins, which are defined by the
    upper and lower bounds of their edges and the computed bin values.
    """

    kdims = param.List(default=[Dimension('x')], bounds=(1,1), doc="""
        Dimensions on Element2Ds determine the number of indexable
        dimensions.""")

    group = param.String(default='Histogram', constant=True)

    vdims = param.List(default=[Dimension('Frequency')], bounds=(1,1))

    def __init__(self, values, edges=None, **params):
        if edges is not None:
            self.warning("Histogram edges should be supplied as a tuple "
                         "along with the values, passing the edges will "
                         "be deprecated in holoviews 2.0.")
        self.values, self.edges, settings = self._process_data(values, edges)
        settings.update(params)
        super(Histogram, self).__init__((self.values, self.edges), **settings)

    def __getitem__(self, key):
        """
        Implements slicing or indexing of the Histogram
        """
        if key in self.dimensions(): return self.dimension_values(key)
        if key is () or key is Ellipsis: return self # May no longer be necessary
        key = util.process_ellipses(self, key)
        if not isinstance(key, tuple): pass
        elif len(key) == self.ndims + 1:
            if key[-1] != slice(None) and (key[-1] not in self.vdims):
                raise KeyError("%r is the only selectable value dimension" %
                                self.vdims[0].name)
            key = key[0]
        elif len(key) == self.ndims + 1: key = key[0]
        else:
            raise KeyError("Histogram cannot slice more than %d dimension."
                            % len(self.kdims)+1)

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
                raise KeyError("Key value %s is out of the histogram bounds" % key)
            idx = np.digitize([key], self.edges)[0]
            return self.values[idx-1 if idx>0 else idx]



    def _process_data(self, values, edges):
        """
        Ensure that edges are specified as left and right edges of the
        histogram bins rather than bin centers.
        """
        settings = {}
        (values, edges) = values if isinstance(values, tuple) else (values, edges)
        if isinstance(values, Chart):
            settings = dict(values.get_param_values(onlychanged=True))
            edges = values.dimension_values(0)
            values = values.dimension_values(1)
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


    def range(self, dimension, data_range=True):
        if self.get_dimension_index(dimension) == 0 and data_range:
            dim = self.get_dimension(dimension)
            lower, upper = np.min(self.edges), np.max(self.edges)
            return util.dimension_range(lower, upper, dim)
        else:
            return super(Histogram, self).range(dimension, data_range)


    def dimension_values(self, dim):
        dim = self.get_dimension(dim, strict=True).name
        if dim in self.vdims:
            return self.values
        elif dim in self.kdims:
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
                                Dimension('Magnitude')], bounds=(1, None))

    _null_value = np.array([[], [], [], []]).T # For when data is None
    _min_dims = 3                              # Minimum number of columns

    def __init__(self, data, kdims=None, vdims=None, **params):
        if isinstance(data, list) and data and all(isinstance(d, np.ndarray) for d in data):
            data = np.column_stack([d.flat if d.ndim > 1 else d for d in data])
        super(VectorField, self).__init__(data, kdims=kdims, vdims=vdims, **params)



class Spikes(Chart):
    """
    Spikes is a 1D or 2D Element, which represents a series of
    vertical or horizontal lines distributed along some dimension. If
    an additional dimension is supplied it will be used to specify the
    height of the lines. The Element may therefore be used to
    represent 1D distributions, spectrograms or spike trains in
    electrophysiology.
    """

    group = param.String(default='Spikes', constant=True)

    kdims = param.List(default=[Dimension('x')], bounds=(1, 1))

    vdims = param.List(default=[])

    _auto_indexable_1d = False


class Area(Curve):
    """
    An Area Element represents the area under a Curve
    and is specified in the same format as a regular
    Curve, with the key dimension corresponding to a
    column of x-values and the value dimension
    corresponding to a column of y-values. Optionally
    a second value dimension may be supplied to shade
    the region between the curves.
    """

    group = param.String(default='Area', constant=True)

    @classmethod
    def stack(cls, areas):
        """
        Stacks an (Nd)Overlay of Area or Curve Elements by offsetting
        their baselines. To stack a HoloMap or DynamicMap use the map
        method.
        """
        if not len(areas):
            return stacked
        baseline = np.zeros(len(areas.values()[0]))
        stacked = areas.clone(shared_data=False)
        vdims = [areas.values()[0].vdims[0], 'Baseline']
        for k, area in areas.items():
            x, y = (area.dimension_values(i) for i in range(2))
            stacked[k] = area.clone((x, y+baseline, baseline), vdims=vdims,
                                    new_type=Area)
            baseline += y
        return stacked
