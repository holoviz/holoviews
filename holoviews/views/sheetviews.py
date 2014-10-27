from collections import OrderedDict
import numpy as np

import param

from ..core import Dimension, NdMapping, Layer
from ..core.boundingregion import BoundingRegion, BoundingBox
from ..core.sheetcoords import SheetCoordinateSystem, Slice
from ..core.holoview import find_minmax
from ..core.options import options
from .dataviews import Histogram, Curve
from .tabular import Table


class Matrix(Layer):
    """
    Matrix is a basic 2D atomic View type.

    Arrays with a shape of (X,Y) or (X,Y,Z) are valid. In the case of
    3D arrays, each depth layer is interpreted as a channel of the 2D
    representation.
    """

    dimensions = param.List(default=[Dimension('X'), Dimension('Y')],
                            constant=True, doc="""
        The label of the x- and y-dimension of the Matrix in form
        of a string or dimension object.""")

    value = param.ClassSelector(class_=(str, Dimension),
                                default=Dimension('Z'), doc="""
        The dimension description of the data held in the data array.""")

    def __init__(self, data, lbrt, **kwargs):
        super(Matrix, self).__init__(data, **kwargs)
        self.xlim = lbrt[0], lbrt[2]
        self.ylim = lbrt[1], lbrt[3]


    def __getitem__(self, slc):
        raise NotImplementedError('Slicing Matrix Views currently'
                                  ' not implemented.')


    def normalize(self, min=0.0, max=1.0, norm_factor=None, div_by_zero='ignore'):
        norm_factor = self.cyclic_range if norm_factor is None else norm_factor
        if norm_factor is None:
            norm_factor = self.data.max() - self.data.min()
        else:
            min, max = (0.0, 1.0)

        if div_by_zero in ['ignore', 'warn']:
            if (norm_factor == 0.0) and div_by_zero == 'warn':
                self.warning("Ignoring divide by zero in normalization.")
            norm_factor = 1.0 if (norm_factor == 0.0) else norm_factor

        norm_data = (((self.data - self.data.min()) / norm_factor) * abs(
            (max - min))) + min
        return self.clone(norm_data)


    def hist(self, num_bins=20, bin_range=None, adjoin=True, individually=True, **kwargs):
        """
        Returns a Histogram of the Matrix data, binned into
        num_bins over the bin_range (if specified).

        If adjoin is True, the histogram will be returned adjoined to
        the Matrix as a side-plot.

        The 'individually' argument specifies whether the histogram
        will be rescaled for each Matrix in a HoloMap.
        """
        range = find_minmax(self.range, (0, -float('inf')))\
            if bin_range is None else bin_range

        # Avoids range issues including zero bin range and empty bins
        if range == (0, 0):
            range = (0.0, 0.1)
        try:
            data = self.data.flatten()
            data = data[np.invert(np.isnan(data))]
            hist, edges = np.histogram(data, normed=True,
                                       range=range, bins=num_bins)
        except:
            edges = np.linspace(range[0], range[1], num_bins + 1)
            hist = np.zeros(num_bins)
        hist[np.isnan(hist)] = 0

        hist_view = Histogram(hist, edges, dimensions=[self.value],
                              label=self.label, value='Frequency')

        # Set plot and style options
        style_prefix = kwargs.get('style_prefix',
                                  'Custom[<' + self.name + '>]_')
        opts_name = style_prefix + hist_view.label.replace(' ', '_')
        hist_view.style = opts_name
        options[opts_name] = options.plotting(self)(
            **dict(rescale_individually=individually))
        return (self << hist_view) if adjoin else hist_view


    def _coord2matrix(self, coord):
        xd, yd = self.data.shape
        l, b, r, t = self.lbrt
        xvals = np.linspace(l, r, xd)
        yvals = np.linspace(b, t, yd)
        xidx = np.argmin(np.abs(xvals-coord[0]))
        yidx = np.argmin(np.abs(yvals-coord[1]))
        return (xidx, yidx)


    def sample(self, samples=[], **sample_values):
        """
        Sample the Matrix along one or both of its dimensions,
        returning a reduced dimensionality type, which is either
        a ItemTable, Curve or Scatter. If two dimension samples
        and a new_xaxis is provided the sample will be the value
        of the sampled unit indexed by the value in the new_xaxis
        tuple.
        """
        if isinstance(samples, tuple):
            X, Y = samples
            samples = zip(X, Y)
        if len(sample_values) == self.ndims or len(samples):
            if not len(samples):
                samples = zip(*[c if isinstance(c, list) else [c] for didx, c in
                               sorted([(self.dim_index(k), v) for k, v in
                                       sample_values.items()])])
            table_data = OrderedDict()
            for c in samples:
                table_data[c] = self.data[self._coord2matrix(c)]
            return Table(table_data, dimensions=self.dimensions,
                             label=self.label, value=self.value)
        else:
            dimension, sample_coord = samples.items()[0]
            if isinstance(sample_coord, slice):
                raise ValueError(
                    'Matrix sampling requires coordinates not slices,'
                    'use regular slicing syntax.')
            other_dimension = [d for d in self.dimensions if
                               d.name != dimension]
            # Indices inverted for indexing
            sample_ind = self.dim_index(other_dimension[0].name)

            # Generate sample slice
            sample = [slice(None) for i in range(self.ndims)]
            coord_fn = (lambda v: (v, 0)) if sample_ind else (lambda v: (0, v))
            sample[sample_ind] = self._coord2matrix(coord_fn(sample_coord))[sample_ind]

            # Sample data
            x_vals = self.dimension_values(dimension)
            data = zip(x_vals, self.data[sample])
            return Curve(data, **dict(self.get_param_values(),
                                      dimensions=other_dimension))


    def reduce(self, label_prefix='', **dimreduce_map):
        """
        Reduces the Matrix using functions provided via the
        kwargs, where the keyword is the dimension to be reduced.
        Optionally a label_prefix can be provided to prepend to
        the result View label.
        """
        label = ' '.join([label_prefix, self.label])
        if len(dimreduce_map) == self.ndims:
            reduced_view = self
            for dim, reduce_fn in dimreduce_map.items():
                reduced_view = reduced_view.reduce(label_prefix=label_prefix,
                                                   **{dim: reduce_fn})
                label_prefix = ''
            return reduced_view
        else:
            dimension, reduce_fn = dimreduce_map.items()[0]
            other_dimension = [d for d in self.dimensions if d.name != dimension]
            x_vals = self.dimension_values(dimension)
            data = zip(x_vals, reduce_fn(self.data, axis=self.dim_index(dimension)))
            return Curve(data, dimensions=other_dimension, label=label,
                         title=self.title, value=self.value)


    @property
    def cyclic_range(self):
        """
        For a cyclic quantity, the range over which the values
        repeat. For instance, the orientation of a mirror-symmetric
        pattern in a plane is pi-periodic, with orientation x the same
        as orientation x+pi (and x+2pi, etc.). The property determines
        the cyclic_range from the value dimensions range parameter.
        """
        if isinstance(self.value, Dimension) and self.value.cyclic:
            return self.value.range[1]
        else:
            return None


    @property
    def range(self):
        if self.cyclic_range:
            return (0, self.cyclic_range)
        else:
            return (self.data.min(), self.data.max())


    @property
    def depth(self):
        return 1 if len(self.data.shape) == 2 else self.data.shape[2]


    @property
    def mode(self):
        """
        Mode specifying the color space for visualizing the array data
        and is a function of the depth. For a depth of one, a colormap
        is used as determined by the style. If the depth is 3 or 4,
        the mode is 'rgb' or 'rgba' respectively.
        """
        if   self.depth == 1:  return 'cmap'
        elif self.depth == 3:  return 'rgb'
        elif self.depth == 4:  return 'rgba'
        else:
            raise Exception("Mode cannot be determined from the depth")


    @property
    def N(self):
        return self.normalize()


class HeatMap(Matrix):
    """
    HeatMap is an atomic View element used to visualize two dimensional
    parameter spaces. It supports sparse or non-linear spaces, dynamically
    upsampling them to a dense representation, which can be visualized.

    A HeatMap can be initialized with any dict or NdMapping type with
    two-dimensional keys. Once instantiated the dense representation is
    available via the .data property.
    """

    _deep_indexable = True

    def __init__(self, data, **kwargs):
        dimensions = kwargs['dimensions'] if 'dimensions' in kwargs else self.dimensions
        if isinstance(data, NdMapping):
            self._data = data
            if 'dimensions' not in kwargs:
                kwargs['dimensions'] = data.dimensions
        elif isinstance(data, (dict, OrderedDict)):
            self._data = NdMapping(data, dimensions=dimensions)
        elif data is None:
            self._data = NdMapping(dimensions=dimensions)
        else:
            raise TypeError('HeatMap only accepts dict or NdMapping types.')

        self._style = None
        self._xlim = None
        self._ylim = None
        param.Parameterized.__init__(self, **kwargs)


    def __getitem__(self, coords):
        """
        Slice the underlying NdMapping.
        """
        return self.clone(self._data.select(**dict(zip(self._data.dimension_labels, coords))))


    def dense_keys(self):
        keys = self._data.keys()
        dim1_keys = sorted(set(k[0] for k in keys))
        dim2_keys = sorted(set(k[1] for k in keys))
        return dim1_keys, dim2_keys


    @property
    def data(self):
        dim1_keys, dim2_keys = self.dense_keys()
        grid_keys = [((i1, d1), (i2, d2)) for i1, d1 in enumerate(dim1_keys)
                     for i2, d2 in enumerate(dim2_keys)]

        array = np.zeros((len(dim2_keys), len(dim1_keys)))
        for (i1, d1), (i2, d2) in grid_keys:
            array[len(dim2_keys)-i2-1, i1] = self._data.get((d1, d2), np.NaN)

        return array


    @property
    def range(self):
        vals = self._data.values()
        return (min(vals), max(vals))


    @property
    def xlim(self):
        if self._xlim: return self._xlim
        dim1_keys, _ = self.dense_keys()
        return min(dim1_keys), max(dim1_keys)


    @property
    def ylim(self):
        if self._ylim: return self._ylim
        _, dim2_keys = self.dense_keys()
        return min(dim2_keys), max(dim2_keys)


class SheetMatrix(SheetCoordinateSystem, Matrix):
    """
    SheetMatrix is the atomic unit as which 2D data is stored, along with its
    bounds object. Allows slicing operations of the data in sheet coordinates or
    direct access to the data, via the .data attribute.

    Arrays with a shape of (X,Y) or (X,Y,Z) are valid. In the case of
    3D arrays, each depth layer is interpreted as a channel of the 2D
    representation.
    """

    bounds = param.ClassSelector(class_=BoundingRegion, default=BoundingBox(), doc="""
       The bounding region in sheet coordinates containing the data.""")

    dimensions = param.List(default=[Dimension('X'), Dimension('Y')],
                            constant=True, doc="""
        The label of the x- and y-dimension of the SheetMatrix in form
        of a string or dimension object.""")

    roi_bounds = param.ClassSelector(class_=BoundingRegion, default=None, doc="""
        The ROI can be specified to select only a sub-region of the bounds to
        be stored as data.""")

    value = param.ClassSelector(class_=(str, Dimension),
                                default=Dimension('Z'), doc="""
        The dimension description of the data held in the data array.""")

    _deep_indexable = True

    def __init__(self, data, bounds=None, xdensity=None, ydensity=None, **kwargs):
        bounds = bounds if bounds else BoundingBox()
        data = np.array([[0]]) if data is None else data
        l, b, r, t = bounds.lbrt()
        (dim1, dim2) = data.shape[0], data.shape[1]
        xdensity = xdensity if xdensity else dim1/(r-l)
        ydensity = ydensity if ydensity else dim2/(t-b)

        SheetCoordinateSystem.__init__(self, bounds, xdensity, ydensity)
        Layer.__init__(self, data, **kwargs)
        self._lbrt = self.bounds.lbrt()


    def __getitem__(self, coords):
        """
        Slice the underlying numpy array in sheet coordinates.
        """
        if coords is () or coords == slice(None, None):
            return self

        if not any([isinstance(el, slice) for el in coords]):
            return self.data[self.sheet2matrixidx(*coords)]
        if all([isinstance(c, slice) for c in coords]):
            l, b, r, t = self.bounds.lbrt()
            xcoords, ycoords = coords
            xstart = l if xcoords.start is None else max(l, xcoords.start)
            xend = r if xcoords.stop is None else min(r, xcoords.stop)
            ystart = b if ycoords.start is None else max(b, ycoords.start)
            yend = t if ycoords.stop is None else min(t, ycoords.stop)
            bounds = BoundingBox(points=((xstart, ystart), (xend, yend)))
        else:
            raise IndexError('Indexing requires x- and y-slice ranges.')

        return SheetMatrix(Slice(bounds, self).submatrix(self.data),
                         bounds, xdensity=self.xdensity, ydensity=self.ydensity,
                         label=self.label, style=self.style,
                         value=self.value)

    @property
    def xlim(self):
        l, _, r, _ = self.bounds.lbrt()
        return (l, r)


    @property
    def ylim(self):
        _, b, _, t = self.bounds.lbrt()
        return (b, t)


    @property
    def lbrt(self):
        if hasattr(self, '_lbrt'):
            return self._lbrt
        else:
            return self.bounds.lbrt()

    @lbrt.setter
    def lbrt(self, lbrt):
        self._lbrt = lbrt


    def _coord2matrix(self, coord):
        return self.sheet2matrixidx(*coord)


    @property
    def roi(self):
        bounds = self.roi_bounds if self.roi_bounds else self.bounds
        return self.get_roi(bounds)


    def get_roi(self, roi_bounds):
        if self.depth == 1:
            data = Slice(roi_bounds, self).submatrix(self.data)
        else:
            data = np.dstack([Slice(roi_bounds, self).submatrix(
                self.data[:, :, i]) for i in range(self.depth)])
        return SheetMatrix(data, roi_bounds, style=self.style, value=self.value)


    def dimension_values(self, dimension):
        """
        The set of samples available along a particular dimension.
        """
        dim_index = self.dim_index(dimension)
        l, b, r, t = self.lbrt
        dim_min, dim_max = [(l, r), (b, t)][dim_index]
        dim_len = self.data.shape[dim_index]
        half_unit = (dim_max - dim_min)/dim_len/2.
        coord_fn = (lambda v: (0, v)) if dim_index else (lambda v: (v, 0))
        return [self.closest_cell_center(*coord_fn(v))[dim_index]
                for v in np.linspace(dim_min+half_unit, dim_max-half_unit, dim_len)]


class Points(Layer):
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

    dimensions = param.List(default=[Dimension('X'), Dimension('Y')],
                            constant=True, doc="""
        The label of the x- and y-dimension of the SheetMatrix in form
        of a string or dimension object.""")


    _null_value = np.array([[], []]).T # For when data is None
    _min_dims = 2                      # Minimum number of columns
    _range_column = 3                 # Column used by range property

    value = param.ClassSelector(class_=(str, Dimension),
                                default=Dimension('Magnitude'))

    def __init__(self, data, **kwargs):
        if isinstance(data, tuple):
            arrays = [np.array(d) for d in data]
            if not all(len(arr)==len(arrays[0]) for arr in arrays):
                raise Exception("All input arrays must have the same length.")

            arr = np.hstack(tuple(arr.reshape(arr.shape if len(arr.shape)==2
                                              else (len(arr), 1)) for arr in arrays))
        else:
            arr = np.array(data)

        data = self._null_value if (data is None) or (len(arr) == 0) else arr
        if data.shape[1] <self._min_dims:
            raise Exception("%s requires a minimum of %s columns."
                            % (self.__class__.__name__, self._min_dims))

        super(Points, self).__init__(data, **kwargs)


    def __getitem__(self, keys):
        pass


    @property
    def range(self):
        """
        The range of magnitudes (if available) otherwise None.
        """
        col = self._range_column
        if self.data.shape[1] < col: return None
        return (self.data[:,col-1:col].min(),
                self.data[:,col-1:col].max())


    def __len__(self):
        return self.data.shape[0]


    def __iter__(self):
        i = 0
        while i < len(self):
            yield tuple(self.data[i, ...])
            i += 1


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

    _null_value = np.array([[], [], [], []]).T # For when data is None
    _min_dims = 3                              # Minimum number of columns
    _range_column = 4                          # Column used by range property

    value = param.ClassSelector(class_=(str, Dimension),
                                default=Dimension('PolarVector', cyclic=True,
                                                  range=(0,2*np.pi)))


class Contours(Layer):
    """
    Allows sets of contour lines to be defined over a
    SheetCoordinateSystem.

    The input data is a list of Nx2 numpy arrays where each array
    corresponds to a contour in the group. Each point in the numpy
    array corresponds to an X,Y coordinate.
    """

    dimensions = param.List(default=[Dimension('X'), Dimension('Y')],
                            constant=True, doc="""
        The label of the x- and y-dimension of the SheetMatrix in form
        of a string or dimension object.""")

    def __init__(self, data, **kwargs):
        data = [] if data is None else data
        super(Contours, self).__init__(data, **kwargs)


    def resize(self, bounds):
        return Contours(self.contours, bounds, style=self.style)


    def __len__(self):
        return len(self.data)

    @property
    def xlim(self):
        if self._xlim: return self._xlim
        xmin = min(min(c[:, 0]) for c in self.data)
        xmax = max(max(c[:, 0]) for c in self.data)
        return xmin, xmax

    @property
    def ylim(self):
        if self._ylim: return self._ylim
        ymin = min(min(c[:, 0]) for c in self.data)
        ymax = max(max(c[:, 0]) for c in self.data)
        return ymin, ymax