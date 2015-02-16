from itertools import product
from collections import OrderedDict
import numpy as np

import param

from ..core import Dimension, NdMapping, Element2D
from ..core.boundingregion import BoundingRegion, BoundingBox
from ..core.sheetcoords import SheetCoordinateSystem, Slice
from .chart import Curve
from .tabular import Table


class Raster(Element2D):
    """
    Raster is a basic 2D atomic Element type.

    Arrays with a shape of (N,M) are valid inputs for Raster wheras
    subclasses of Raster (e.g. RGB) may also accept 3D arrays
    containing channel information.
    """

    key_dimensions = param.List(default=[Dimension('x'), Dimension('y')],
                                  bounds=(2, 2), constant=True, doc="""
        The label of the x- and y-dimension of the Raster in form
        of a string or dimension object.""")

    value = param.String(default='Raster')

    value_dimensions = param.List(default=[Dimension('z')], bounds=(1, 1), doc="""
        The dimension description of the data held in the data array.""")

    def __init__(self, data, extents=(0, 0, 1, 1), **params):
        super(Raster, self).__init__(data, extents=extents, **params)


    def normalize(self, min=0.0, max=1.0, norm_factor=None, div_by_zero='ignore'):
        norm_factor = self.range(2)[1] if norm_factor is None else norm_factor
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


    def _coord2matrix(self, coord):
        xd, yd = self.data.shape
        l, b, r, t = self.extents
        xvals = np.linspace(l, r, xd)
        yvals = np.linspace(b, t, yd)
        xidx = np.argmin(np.abs(xvals-coord[0]))
        yidx = np.argmin(np.abs(yvals-coord[1]))
        return (xidx, yidx)


    @classmethod
    def collapse_data(cls, data_list, function, **kwargs):
        if not function:
            raise Exception("Must provide function to collapse %s data." % cls.__name__)
        return function(np.dstack(data_list), axis=-1, **kwargs)


    def sample(self, samples=[], **sample_values):
        """
        Sample the Raster along one or both of its dimensions,
        returning a reduced dimensionality type, which is either
        a ItemTable, Curve or Scatter. If two dimension samples
        and a new_xaxis is provided the sample will be the value
        of the sampled unit indexed by the value in the new_xaxis
        tuple.
        """
        if isinstance(samples, tuple):
            X, Y = samples
            samples = zip(X, Y)
        params = dict(self.get_param_values(onlychanged=True))
        if len(sample_values) == self.ndims or len(samples):
            if not len(samples):
                samples = zip(*[c if isinstance(c, list) else [c] for didx, c in
                               sorted([(self.get_dimension_index(k), v) for k, v in
                                       sample_values.items()])])
            table_data = OrderedDict()
            for c in samples:
                table_data[c] = self.data[self._coord2matrix(c)]
            params['key_dimensions'] = self.key_dimensions
            return Table(table_data, **params)
        else:
            dimension, sample_coord = sample_values.items()[0]
            if isinstance(sample_coord, slice):
                raise ValueError(
                    'Raster sampling requires coordinates not slices,'
                    'use regular slicing syntax.')
            other_dimension = [d for d in self.key_dimensions if
                               d.name != dimension]
            # Indices inverted for indexing
            sample_ind = self.get_dimension_index(other_dimension[0].name)

            # Generate sample slice
            sample = [slice(None) for i in range(self.ndims)]
            coord_fn = (lambda v: (v, 0)) if sample_ind else (lambda v: (0, v))
            sample[sample_ind] = self._coord2matrix(coord_fn(sample_coord))[sample_ind]

            # Sample data
            x_vals = sorted(set(self.dimension_values(dimension)))
            data = zip(x_vals, self.data[sample])
            params['key_dimensions'] = other_dimension
            return Curve(data, **params)


    def reduce(self, dimensions=None, function=None, **reduce_map):
        """
        Reduces the Raster using functions provided via the
        kwargs, where the keyword is the dimension to be reduced.
        Optionally a label_prefix can be provided to prepend to
        the result Element label.
        """
        dimensions = self._valid_dimensions(dimensions)
        if dimensions and reduce_map:
            raise Exception("Pass reduced dimensions either as an argument"
                            "or as part of the kwargs not both.")
        elif dimensions:
            reduce_map = {d: function for d in dimensions}
        elif not reduce_map:
            reduce_map = {d: function for d in self._cached_index_names}

        if len(reduce_map) == self.ndims:
            reduced_view = self
            for dim, reduce_fn in reduce_map.items():
                reduced_view = reduced_view.reduce(**{dim: reduce_fn})
            return reduced_view
        else:
            dimension, reduce_fn = reduce_map.items()[0]
            other_dimension = [d for d in self.key_dimensions if d.name != dimension]
            x_vals = sorted(set(self.dimension_values(dimension)))
            data = zip(x_vals, reduce_fn(self.data, axis=self.get_dimension_index(dimension)))
            params = dict(dict(self.get_param_values(onlychanged=True)),
                          key_dimensions=other_dimension)
            return Table(data, **params)

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


    def dimension_values(self, dim):
        """
        The set of samples available along a particular dimension.
        """
        dim_idx = self.get_dimension_index(dim)
        if dim_idx in [0, 1]:
            (l, r), (b, t) = self.xlim, self.ylim
            shape = self.data.shape[abs(dim_idx-1)]
            dim_min, dim_max = [(l, r), (b, t)][dim_idx]
            dim_len = self.data.shape[dim_idx]
            half_unit = (dim_max - dim_min)/dim_len/2.
            coord_fn = (lambda v: (0, v)) if dim_idx else (lambda v: (v, 0))
            linspace = np.linspace(dim_min+half_unit, dim_max-half_unit, dim_len)
            coords = [self.closest(coord_fn(v))[dim_idx]
                      for v in linspace] * shape
            return coords if dim_idx else sorted(coords)
        elif dim_idx == 2:
            return np.flipud(self.data).T.flatten()
        else:
            raise Exception("Dimension not found.")


class HeatMap(Raster):
    """
    HeatMap is an atomic Element used to visualize two dimensional
    parameter spaces. It supports sparse or non-linear spaces, dynamically
    upsampling them to a dense representation, which can be visualized.

    A HeatMap can be initialized with any dict or NdMapping type with
    two-dimensional keys. Once instantiated the dense representation is
    available via the .data property.
    """

    value = param.String(default='HeatMap')

    def __init__(self, data, **params):
        self._data, array, dimensions = self._process_data(data, params)
        super(HeatMap, self).__init__(array, **dict(params, **dimensions))


    def _process_data(self, data, params):
        dimensions = {group: params.get(group, getattr(self, group))
                      for group in self._dim_groups[:2]}
        if isinstance(data, NdMapping):
            if 'key_dimensions' not in params:
                dimensions['key_dimensions'] = data.key_dimensions
            if 'value_dimensions' not in params:
                dimensions['value_dimensions'] = data.value_dimensions
        elif isinstance(data, (dict, OrderedDict, type(None))):
            data = NdMapping(data, **dimensions)
        else:
            raise TypeError('HeatMap only accepts dict or NdMapping types.')

        keys = data.keys()
        dim1_keys = sorted(set(k[0] for k in keys))
        dim2_keys = sorted(set(k[1] for k in keys))
        grid_keys = [((i1, d1), (i2, d2)) for i1, d1 in enumerate(dim1_keys)
                     for i2, d2 in enumerate(dim2_keys)]

        array = np.zeros((len(dim2_keys), len(dim1_keys)))
        for (i1, d1), (i2, d2) in grid_keys:
            val = data.get((d1, d2), np.NaN)
            array[len(dim2_keys)-i2-1, i1] = val[0] if isinstance(val, tuple) else val

        return data, array, dimensions


    def __getitem__(self, coords):
        """
        Slice the underlying NdMapping.
        """
        return self.clone(self._data.select(**dict(zip(self._data._cached_index_names, coords))))


    def dense_keys(self):
        keys = list(self._data.keys())
        dim1_keys = sorted(set(k[0] for k in keys))
        dim2_keys = sorted(set(k[1] for k in keys))
        return dim1_keys, dim2_keys


    def dimension_values(self, dim):
        if isinstance(dim, int):
            dim = self.get_dimension(dim)

        if dim in self._cached_index_names:
            idx = self.get_dimension_index(dim)
            return [k[idx] for k in self._data.keys()]
        elif dim in self._cached_value_names:
            return [v if isinstance(v, tuple) else v
                    for v in self._data.values()]
        else:
            raise Exception("Dimension %s not found." % dim)


    def dframe(self, dense=False):
        if dense:
            keys1, keys2 = self.dense_keys()
            dense_map = HeatMap({(k1, k2): self._data.get((k1, k2), np.NaN)
                                 for k1, k2 in product(keys1, keys2)})
            return dense_map.dframe()
        return super(HeatMap, self).dframe()


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


class Matrix(SheetCoordinateSystem, Raster):
    """
    Matrix is the atomic unit as which 2D data is stored, along with
    its bounds object. The input data may be a numpy.matrix object or
    a two-dimensional numpy array.

    Allows slicing operations of the data in sheet coordinates or direct
    access to the data, via the .data attribute.
    """

    bounds = param.ClassSelector(class_=BoundingRegion, default=BoundingBox(), doc="""
       The bounding region in sheet coordinates containing the data.""")

    value = param.String(default='Matrix')

    value_dimensions = param.List(default=[Dimension('Luminance')],
                                  bounds=(1, 1), doc="""
        The dimension description of the data held in the matrix.""")


    def __init__(self, data, bounds=None, xdensity=None, ydensity=None, **params):
        bounds = bounds if bounds is not None else BoundingBox()
        if isinstance(bounds, tuple):
            l, b, r, t = bounds
            bounds = BoundingBox(points=((l, b), (r, t)))
        elif np.isscalar(bounds):
            bounds = BoundingBox(radius=bounds)
        data = np.array([[0]]) if data is None else data
        l, b, r, t = bounds.lbrt()
        (dim1, dim2) = data.shape[0], data.shape[1]
        xdensity = xdensity if xdensity else dim1/(r-l)
        ydensity = ydensity if ydensity else dim2/(t-b)

        SheetCoordinateSystem.__init__(self, bounds, xdensity, ydensity)
        Element2D.__init__(self, data, extents=self.lbrt, bounds=bounds, **params)

        if len(self.data.shape) == 3:
            if self.data.shape[2] != len(self.value_dimensions):
                raise ValueError("Input array has shape %r but %d value dimensions defined"
                                 % (self.data.shape, len(self.value_dimensions)))



    def closest(self, coords):
        """
        Given a single coordinate tuple (or list of coordinates)
        return the coordinate (or coordinatess) needed to address the
        corresponding Matrix exactly.
        """
        if isinstance(coords, tuple):
            return self.closest_cell_center(*coords)
        else:
            return [self.closest_cell_center(*el) for el in coords]


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

        return self.clone(Slice(bounds, self).submatrix(self.data),
                          bounds=bounds)


    @property
    def xlim(self):
        l, _, r, _ = self.bounds.lbrt()
        return (l, r)


    @property
    def ylim(self):
        _, b, _, t = self.bounds.lbrt()
        return (b, t)


    def range(self, dim, data_range=True):
        dim_idx = dim if isinstance(dim, int) else self.get_dimension_index(dim)
        if dim_idx in [0, 1]:
            if dim_idx:
                return self.ylim
            return self.xlim
        return super(Matrix, self).range(dim, data_range=data_range)


    def _coord2matrix(self, coord):
        return self.sheet2matrixidx(*coord)


    def get_roi(self, bounds):
        if self.depth == 1:
            data = Slice(bounds, self).submatrix(self.data)
        else:
            data = np.dstack([Slice(bounds, self).submatrix(
                self.data[:, :, i]) for i in range(self.depth)])
        return Matrix(data, bounds, style=self.style, value=self.value)



class RGB(Matrix):
    """
    An RGB element is a Matrix containing channel data for the the
    red, green, blue and (optionally) the alpha channels. The values
    of each channel must be in the range 0.0 to 1.0.

    In input array may have a shape of NxMx4 or NxMx3. In the latter
    case, the defined alpha dimension parameter is appended to the
    list of value dimensions.
    """

    value = param.String(default='RGB')

    alpha_dimension = param.ClassSelector(default=Dimension('A',range=(0,1)),
                                          class_=Dimension, doc="""
        The alpha dimension definition to add the value_dimensions if
        an alpha channel is supplied.""")

    value_dimensions = param.List(
        default=[Dimension('R', range=(0,1)), Dimension('G',range=(0,1)),
                 Dimension('B', range=(0,1))], bounds=(3, 4), doc="""
        The dimension description of the data held in the matrix.

        If an alpha channel is supplied, the defined alpha_dimension
        is automatically appended to this list.""")

    def __init__(self, data, **params):
        sliced = None
        if len(data.shape) != 3:
            raise ValueError("Three dimensional matrices or arrays required")
        elif data.shape[2] == 4:
            sliced = data[:,:,:-1]

        if len(params.get('value_dimensions',[])) == 4:
            alpha_dim = params['value_dimensions'].pop(3)
            params['alpha_dimension'] = alpha_dim

        super(RGB, self).__init__(data if sliced is None else sliced, **params)
        if sliced is not None:
            self.value_dimensions.append(self.alpha_dimension)
            self.data = data


    def __getitem__(self, coords):
        """
        Slice the underlying numpy array in sheet coordinates.
        """
        if len(coords) > self.ndims:
            value = coords[self.ndims:]
            if len(value) > 1:
                raise KeyError()
            sliced = super(RGB, self).__getitem__(coords[:self.ndims])
            vidx = self.get_dimension_index(value[0])
            data = sliced.data[:,:, vidx]
            return Matrix(data, **dict(self.get_param_values(onlychanged=True),
                                       value_dimensions=[self.value_dimensions[vidx-2]]))
        else:
            return super(RGB, self).__getitem__(coords)


