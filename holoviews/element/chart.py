from collections import OrderedDict

import numpy as np

import param

from ..core import Dimension, NdMapping, Element2D, HoloMap
from .tabular import ItemTable, Table


class Chart(Element2D):
    """
    The data held within an Array is a numpy array of shape (n, m).
    Element2D objects are sliceable along the X dimension allowing easy
    selection of subsets of the data.
    """

    key_dimensions = param.List(default=[Dimension('x')], bounds=(1,2), doc="""
        Dimensions on Element2Ds determine the number of indexable
        dimensions.""")

    value = param.String(default='Chart')

    value_dimensions = param.List(default=[Dimension('y')], bounds=(1,3), doc="""
        Dimensions on Element2Ds determine the number of indexable
        dimensions.""")


    def __init__(self, data, **params):
        settings = {}
        if isinstance(data, Chart):
            settings = dict(data.get_param_values())
            data = data.data
        elif isinstance(data, NdMapping) or (isinstance(data, list) and data
                                           and isinstance(data[0], Element2D)):
            data, settings = self._process_map(data)
        data = list(data)
        if len(data) and not isinstance(data, np.ndarray):
            data = np.array(data)
        settings.update(params)
        super(Chart, self).__init__(data, **settings)


    def _process_map(self, ndmap):
        """
        Base class to process an NdMapping to be collapsed into a Chart.
        Should return the data and parameters of the new Chart.
        """
        if isinstance(ndmap, Table):
            data = [tuple(k for k in key) + tuple(v for v in vals)
                    for key, vals in ndmap.data.items()]
            settings = dict(ndmap.get_param_values())
        else:
            data = np.concatenate([v.data for v in ndmap])
            settings = dict([v for v in ndmap][0].get_param_values())
        return data, settings


    def closest(self, coords):
        """
        Given single or multiple x-values, returns the list
        of closest actual samples.
        """
        if not isinstance(coords, list): coords = [coords]
        xs = self.data[:, 0]
        idxs = [np.argmin(xs-coord) for coord in coords]
        return [xs[idx] for idx in idxs]


    def __getitem__(self, slc):
        """
        Implements slicing or indexing of the data by the data x-value.
        If a single element is indexed reduces the Element2D to a single
        Scatter object.
        """
        if slc is ():
            return self
        xvals = self.data[:, 0]
        if isinstance(slc, slice):
            start, stop = slc.start, slc.stop
            start_idx = np.abs((xvals - start)).argmin()
            start_idx += 0 if start <= xvals[start_idx] else 1
            stop_idx = np.abs((xvals - stop)).argmin()
            stop_idx -= 0 if stop > xvals[stop_idx] else 1
            return self.__class__(self.data[start_idx:stop_idx+1, :],
                                  **dict(self.get_param_values()))
        else:
            index = np.abs((xvals - slc)).argmin()
            data = {(self.data[index, 0],): self.data[index, 1]}
            return Table(data, **dict(self.get_param_values()))


    def sample(self, samples=[]):
        """
        Allows sampling of Element2D objects using the default
        syntax of providing a map of dimensions and sample pairs.
        """
        sample_data = OrderedDict()
        for sample in samples:
            sample_data[sample] = self[sample]
        return Table(sample_data, **dict(self.get_param_values()))


    def reduce(self, label_prefix='', **reduce_map):
        """
        Allows collapsing of Element2D objects using the supplied map of
        dimensions and reduce functions.
        """
        reduced_data = OrderedDict()
        value = ' '.join([self.value.name] + ([label_prefix] if label_prefix else []))
        for dimension, reduce_fn in reduce_map.items():
            reduced_data[value] = reduce_fn(self.data[:, 1])
        return ItemTable(reduced_data, label=self.label, title=self.title,
                         value=self.value(value))


    def __len__(self):
        return len(self.data)


    def dimension_values(self, dim):
        index = self.get_dimension_index(dim)
        return self.data[:, index]

    def dframe(self):
        import pandas as pd
        columns = [d.name for d in self.dimensions()]
        return pd.DataFrame(self.data, columns=columns)



class Scatter(Chart):
    """
    Scatter is a Element2D type which gets displayed as a number of
    disconnected points.
    """

    value = param.String(default='Scatter')



class Curve(Chart):
    """
    Curve is a simple Chart Element providing 1D indexing along
    the x-axis.
    """

    value = param.String(default='Curve')

    def progressive(self):
        """
        Create map indexed by Curve x-axis with progressively expanding number
        of curve samples.
        """
        vmap = HoloMap(None, key_dimensions=self.key_dimensions,
                       title=self.title+' {dims}')
        for idx in range(len(self.data)):
            x = self.data[0]
            if x in vmap:
                vmap[x].data.append(self.data[0:idx])
            else:
                vmap[x] = self.clone(self.data[0:idx])
        return vmap


class Bars(Chart):
    """
    A bar is a simple Chart element, which assumes that the data is
    sorted by x-value and there are no gaps in the bars.
    """

    value = param.String(default='Bars')

    def __init__(self, data, width=None, **params):
        super(Bars, self).__init__(data, **params)
        self._width = width

    @property
    def width(self):
        if self._width == None:
            try:
                return list(set(np.diff(self.data[:, 0])))[0]
            except:
                return None
        else:
            return self._width

    @property
    def edges(self):
        try:
            return list(self.data[:, 0] - self.width) + [self.data[-1, 0] + self.width]
        except:
            return range(len(self)+1)

    @property
    def values(self):
        return np.array(self.data[:, 1], dtype=np.float64)

    @width.setter
    def width(self, width):
        if np.isscalar(width) or len(width) == len(self):
            self._width = width
        else:
            raise ValueError('width should be either a scalar or '
                             'match the number of bars in length.')



class Histogram(Element2D):
    """
    Histogram contains a number of bins, which are defined by the
    upper and lower bounds of their edges and the computed bin values.
    """

    key_dimensions = param.List(default=[Dimension('x')], bounds=(1,1), doc="""
        Dimensions on Element2Ds determine the number of indexable
        dimensions.""")

    value = param.String(default='Histogram')

    value_dimensions = param.List(default=[Dimension('Frequency')])

    def __init__(self, values, edges=None, **params):
        self.values, self.edges, settings = self._process_data(values, edges)
        settings.update(params)
        super(Histogram, self).__init__((self.values, self.edges), **settings)
        self._width = None


    def _process_data(self, values, edges):
        """
        Ensure that edges are specified as left and right edges of the
        histogram bins rather than bin centers.
        """
        settings = {}
        (value, edges) = values if isinstance(values, tuple) else (values, edges)
        if isinstance(values, Element2D):
            values = values.data[:, 0]
            edges = values.data[:, 1]
            settings = dict(values.get_param_values())
        elif isinstance(values, np.ndarray) and len(values.shape) == 2:
            values = values[:, 0]
            edges = values[:, 1]
        else:
            values = np.array(values)
            edges = np.array(edges, dtype=np.float)

        if len(edges) == len(values):
            widths = list(set(np.diff(edges)))
            if len(widths) == 1:
                width = widths[0]
            else:
                raise Exception('Centered bins have to be of equal width.')
            edges -= width/2.
            edges = np.concatenate([edges, [edges[-1]+width]])
        return values, edges, settings


    def dimension_values(self, dim):
        if isinstance(dim, int):
            dim = self.get_dimension(dim).name
        if dim in self._cached_value_names:
            return self.values
        elif dim in self._cached_index_names:
            return self.edges
        else:
            raise Exception("Could not find dimension.")


    def sample(self, **samples):
        raise NotImplementedError('Cannot sample a Histogram.')


    def reduce(self, **dimreduce_map):
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

    key_dimensions = param.List(default=[Dimension('x'), Dimension('y')],
                                  bounds=(2, 2), constant=True, doc="""
        The label of the x- and y-dimension of the Matrix in form
        of a string or dimension object.""")

    value = param.String(default='Points')

    value_dimensions = param.List(default=[], bounds=(0, 2))

    _null_value = np.array([[], []]).T # For when data is None
    _min_dims = 2                      # Minimum number of columns

    def __init__(self, data, **params):
        if isinstance(data, tuple):
            arrays = [np.array(d) for d in data]
            if not all(len(arr)==len(arrays[0]) for arr in arrays):
                raise Exception("All input arrays must have the same length.")

            arr = np.hstack(tuple(arr.reshape(arr.shape if len(arr.shape)==2
                                              else (len(arr), 1)) for arr in arrays))
        elif isinstance(data, Table):
            table_dims = data.dimension_labels('all', True)
            arr = np.array(zip(*[data.dimension_values(dim) for dim in table_dims]))
            if 'key_dimensions' not in params:
                params['key_dimensions'] = data.key_dimensions
            if 'value_dimensions' not in params:
                params['value_dimensions'] = data.value_dimensions
        else:
            arr = np.array(data)

        data = self._null_value if (data is None) or (len(arr) == 0) else arr
        if data.shape[1] <self._min_dims:
            raise Exception("%s requires a minimum of %s columns."
                            % (self.__class__.__name__, self._min_dims))

        super(Points, self).__init__(data, **params)


    def __len__(self):
        return self.data.shape[0]


    def __iter__(self):
        i = 0
        while i < len(self):
            yield tuple(self.data[i, ...])
            i += 1


    def dimension_values(self, dim):
        if dim in [d.name for d in self.dimensions()]:
            dim_index = self.get_dimension_index(dim)
            if dim_index < self.data.shape[1]:
                return self.data[:, dim_index]
            else:
                return [np.NaN] * len(self)
        else:
            raise Exception("Dimension %s not found in %s." %
                            (dim, self.__class__.__name__))



class VectorField(Points):
    """
    A VectorField contains is a collection of vectors where each
    vector has an associated position in sheet coordinates.

    The constructor of VectorField is the same as the constructor of
    Points: the input data can be an NxM Numpy array where the first
    two columns corresponds to the X,Y coordinates in sheet
    coordinates, within the declared bounding region. As with Points,
    the input can be a tuple of array objects or of objects that can
    be cast to arrays (the tuple elements are joined column-wise).

    The third column maps to the vector angle which must be specified
    in radians.

    The visualization of any additional columns is decided by the
    plotting code. For instance, the fourth and fifth columns could
    correspond to arrow length and colour map value. All that is
    assumed is that these additional dimension are normalized between
    0.0 and 1.0 for the default visualization to work well.

    The only restriction is that the final data array is NxM where
    M>3. In other words, the vector must have a dimensionality of 2 or
    higher.
    """

    value = param.String(default='VectorField')

    value_dimensions = param.List(default=[Dimension('Angle', cyclic=True, range=(0,2*np.pi)),
                                           Dimension('Magnitude')], bounds=(2, 2))

    _null_value = np.array([[], [], [], []]).T # For when data is None
    _min_dims = 3                              # Minimum number of columns