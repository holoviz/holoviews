from operator import itemgetter
from itertools import product
import numpy as np
import colorsys
import param

from ..core import util
from ..core.data import DFColumns # FIXME: Waiting for new interface
from ..core import (OrderedDict, Dimension, NdMapping, Element2D,
                    Overlay, Element, Columns)
from ..core.boundingregion import BoundingRegion, BoundingBox
from ..core.sheetcoords import SheetCoordinateSystem, Slice
from .chart import Curve
from .tabular import Table
from .util import compute_edges, toarray

class Raster(Element2D):
    """
    Raster is a basic 2D element type for presenting either numpy or
    dask arrays as two dimensional raster images.

    Arrays with a shape of (N,M) are valid inputs for Raster wheras
    subclasses of Raster (e.g. RGB) may also accept 3D arrays
    containing channel information.

    Raster does not support slicing like the Image or RGB subclasses
    and the extents are in matrix coordinates if not explicitly
    specified.
    """

    group = param.String(default='Raster', constant=True)

    kdims = param.List(default=[Dimension('x'), Dimension('y')],
                       bounds=(2, 2), constant=True, doc="""
        The label of the x- and y-dimension of the Raster in form
        of a string or dimension object.""")

    vdims = param.List(default=[Dimension('z')], bounds=(1, 1), doc="""
        The dimension description of the data held in the data array.""")

    def __init__(self, data, extents=None, **params):
        if extents is None:
            (d1, d2) = data.shape[:2]
            extents = (0, 0, d2, d1)
        super(Raster, self).__init__(data, extents=extents, **params)


    @property
    def _zdata(self):
        return self.data


    def __getitem__(self, slices):
        if slices in self.dimensions(): return self.dimension_values(slices)
        if not isinstance(slices, tuple): slices = (slices, slice(None))
        slc_types = [isinstance(sl, slice) for sl in slices]
        data = self.data.__getitem__(slices[::-1])
        if all(slc_types):
            return self.clone(data, extents=None)
        elif not any(slc_types):
            return toarray(data, index_value=True)
        else:
            return self.clone(np.expand_dims(data, axis=slc_types.index(True)),
                              extents=None)


    def _coord2matrix(self, coord):
        return int(round(coord[1])), int(round(coord[0]))


    @classmethod
    def collapse_data(cls, data_list, function, kdims=None, **kwargs):
        if isinstance(function, np.ufunc):
            return function.reduce(data_list)
        else:
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
        params = dict(self.get_param_values(onlychanged=True),
                      vdims=self.vdims)
        params.pop('extents', None)
        params.pop('bounds', None)
        if len(sample_values) == self.ndims or len(samples):
            if not len(samples):
                samples = zip(*[c if isinstance(c, list) else [c] for didx, c in
                               sorted([(self.get_dimension_index(k), v) for k, v in
                                       sample_values.items()])])
            table_data = [c+(self._zdata[self._coord2matrix(c)],)
                          for c in samples]
            params['kdims'] = self.kdims
            return Table(table_data, **params)
        else:
            dimension, sample_coord = list(sample_values.items())[0]
            if isinstance(sample_coord, slice):
                raise ValueError(
                    'Raster sampling requires coordinates not slices,'
                    'use regular slicing syntax.')
            # Indices inverted for indexing
            sample_ind = self.get_dimension_index(dimension)
            if sample_ind is None:
                raise Exception("Dimension %s not found during sampling" % dimension)
            other_dimension = [d for i, d in enumerate(self.kdims) if
                               i != sample_ind]

            # Generate sample slice
            sample = [slice(None) for i in range(self.ndims)]
            coord_fn = (lambda v: (v, 0)) if not sample_ind else (lambda v: (0, v))
            sample[sample_ind] = self._coord2matrix(coord_fn(sample_coord))[abs(sample_ind-1)]

            # Sample data
            x_vals = self.dimension_values(other_dimension[0].name, unique=True)
            ydata = self._zdata[sample[::-1]]
            if hasattr(self, 'bounds') and sample_ind == 0: ydata = ydata[::-1]
            data = list(zip(x_vals, ydata))
            params['kdims'] = other_dimension
            return Curve(data, **params)


    def reduce(self, dimensions=None, function=None, **reduce_map):
        """
        Reduces the Raster using functions provided via the
        kwargs, where the keyword is the dimension to be reduced.
        Optionally a label_prefix can be provided to prepend to
        the result Element label.
        """
        dims, reduce_map = self._reduce_map(dimensions, function, reduce_map)
        if len(dims) == self.ndims:
            function = reduce_map[0][0]
            if isinstance(function, np.ufunc):
                return function.reduce(self.data, axis=None)
            else:
                return function(self.data)
        else:
            reduce_fn, dimensions = reduce_map[0]
            dimension = dimensions[0]
            other_dimension = [d for d in self.kdims if d.name != dimension]
            oidx = self.get_dimension_index(other_dimension[0])
            x_vals = self.dimension_values(other_dimension[0].name, unique=True)
            reduced = reduce_fn(self._zdata, axis=oidx)
            data = zip(x_vals, reduced if not oidx else reduced[::-1])
            params = dict(dict(self.get_param_values(onlychanged=True)),
                          kdims=other_dimension, vdims=self.vdims)
            params.pop('bounds', None)
            params.pop('extents', None)
            return Table(data, **params)


    def dimension_values(self, dim, unique=False):
        """
        The set of samples available along a particular dimension.
        """
        dim_idx = self.get_dimension_index(dim)
        if unique and dim_idx == 0:
            return np.array(range(self.data.shape[1]))
        elif unique and dim_idx == 1:
            return np.array(range(self.data.shape[0]))
        elif dim_idx in [0, 1]:
            D1, D2 = np.mgrid[0:self.data.shape[1], 0:self.data.shape[0]]
            return D1.flatten() if dim_idx == 0 else D2.flatten()
        elif dim_idx == 2:
            return toarray(self.data.T).flatten()
        else:
            return super(Raster, self).dimension_values(dim)


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



class QuadMesh(Raster):
    """
    QuadMesh is a Raster type to hold x- and y- bin values
    with associated values. The x- and y-values of the QuadMesh
    may be supplied either as the edges of each bin allowing
    uneven sampling or as the bin centers, which will be converted
    to evenly sampled edges.

    As a secondary but less supported mode QuadMesh can contain
    a mesh of quadrilateral coordinates that is not laid out in
    a grid. The data should then be supplied as three separate
    2D arrays for the x-/y-coordinates and grid values.
    """

    group = param.String(default="QuadMesh", constant=True)

    kdims = param.List(default=[Dimension('x'), Dimension('y')])

    vdims = param.List(default=[Dimension('z')])

    def __init__(self, data, **params):
        data = self._process_data(data)
        Element2D.__init__(self, data, **params)
        self.data = self._validate_data(self.data)
        self._grid = self.data[0].ndim == 1


    def _process_data(self, data):
        data = tuple(np.array(el) for el in data)
        x, y, zarray = data
        ys, xs = zarray.shape
        if x.ndim == 1 and len(x) == xs:
            x = compute_edges(x)
        if y.ndim == 1 and len(y) == ys:
            y = compute_edges(y)
        return (x, y, zarray)


    @property
    def _zdata(self):
        return self.data[2]


    def _validate_data(self, data):
        x, y, z = data
        if not z.ndim == 2:
            raise ValueError("Z-values must be 2D array")

        ys, xs = z.shape
        shape_errors = []
        if x.ndim == 1 and xs+1 != len(x):
            shape_errors.append('x')
        if x.ndim == 1 and ys+1 != len(y):
            shape_errors.append('y')
        if shape_errors:
            raise ValueError("%s-edges must match shape of z-array." %
                             '/'.join(shape_errors))
        return data


    def __getitem__(self, slices):
        if slices in self.dimensions(): return self.dimension_values(key)
        if not self._grid:
            raise IndexError("Indexing of non-grid based QuadMesh"
                             "currently not supported")
        if not isinstance(slices, tuple): slices = (slices, slice(None))
        slc_types = [isinstance(sl, slice) for sl in slices]
        if not any(slc_types):
            indices = []
            for idx, data in zip(slices, self.data[:self.ndims]):
                indices.append(np.digitize([idx], data)-1)
            return self.data[2][tuple(indices[::-1])]
        else:
            sliced_data, indices = [], []
            for slc, data in zip(slices, self.data[:self.ndims]):
                if isinstance(slc, slice):
                    low, high = slc.start, slc.stop
                    lidx = ([None] if low is None else
                            max((np.digitize([low], data)-1, 0)))[0]
                    hidx = ([None] if high is None else
                            np.digitize([high], data))[0]
                    sliced_data.append(data[lidx:hidx])
                    indices.append(slice(lidx, (hidx if hidx is None else hidx-1)))
                else:
                    index = (np.digitize([slc], data)-1)[0]
                    sliced_data.append(data[index:index+2])
                    indices.append(index)
            z = np.atleast_2d(self.data[2][tuple(indices[::-1])])
            if not all(slc_types) and not slc_types[0]:
                z = z.T
            return self.clone(tuple(sliced_data+[z]))


    @classmethod
    def collapse_data(cls, data_list, function, kdims=None, **kwargs):
        """
        Allows collapsing the data of a number of QuadMesh
        Elements with a function.
        """
        if not self._grid:
            raise Exception("Collapsing of non-grid based QuadMesh"
                            "currently not supported")
        xs, ys, zs = zip(data_list)
        if isinstance(function, np.ufunc):
            z = function.reduce(zs)
        else:
            z = function(np.dstack(zs), axis=-1, **kwargs)
        return xs[0], ys[0], z


    def _coord2matrix(self, coord):
        return tuple((np.digitize([coord[i]], self.data[i])-1)[0]
                     for i in [1, 0])


    def range(self, dimension):
        idx = self.get_dimension_index(dimension)
        if idx in [0, 1]:
            data = self.data[idx]
            return np.min(data), np.max(data)
        elif idx == 2:
            data = self.data[idx]
            return np.nanmin(data), np.nanmax(data)
        super(QuadMesh, self).range(dimension)


    def dimension_values(self, dimension, unique=False):
        idx = self.get_dimension_index(dimension)
        data = self.data[idx]
        if idx in [0, 1]:
            if not self._grid:
                return data.flatten()
            odim = 1 if unique else self.data[2].shape[idx]
            vals = np.tile(np.convolve(data, np.ones((2,))/2, mode='valid'), odim)
            if idx:
                return np.sort(vals)
            else:
                return vals
        elif idx == 2:
            return data.flatten()
        else:
            return super(QuadMesh, self).dimension_values(idx)



class HeatMap(Raster):
    """
    HeatMap is an atomic Element used to visualize two dimensional
    parameter spaces. It supports sparse or non-linear spaces, dynamically
    upsampling them to a dense representation, which can be visualized.

    A HeatMap can be initialized with any dict or NdMapping type with
    two-dimensional keys. Once instantiated the dense representation is
    available via the .data property.
    """

    group = param.String(default='HeatMap', constant=True)

    def __init__(self, data, extents=None, **params):
        self._data, array, dimensions = self._process_data(data, params)
        super(HeatMap, self).__init__(array, **dict(params, **dimensions))


    def _process_data(self, data, params):
        dimensions = {group: params.get(group, getattr(self, group))
                      for group in self._dim_groups[:2]}
        if isinstance(data, Columns):
            if 'kdims' not in params:
                dimensions['kdims'] = data.kdims
            if 'vdims' not in params:
                dimensions['vdims'] = data.vdims
        elif isinstance(data, (dict, OrderedDict, type(None))):
            data = Columns(data, **dimensions)
        elif isinstance(data, Element):
            data = data.table()
            if not data.ndims == 2:
                raise TypeError('HeatMap conversion requires 2 key dimensions')
        else:
            raise TypeError('HeatMap only accepts Columns or dict types.')

        if len(dimensions['vdims']) > 1:
            raise ValueError("HeatMap data may only have one value dimension")

        d1keys = data.dimension_values(0, True)
        d2keys = data.dimension_values(1, True)
        coords = [(d1, d2, np.NaN) for d1 in d1keys for d2 in d2keys]
        dense_data = data.clone(coords)
        concat_data = DFColumns.concat([data, dense_data])
        data = data.clone(concat_data).aggregate(data.kdims, np.nanmean).sort(data.kdims)
        array = data.dimension_values(2).reshape(len(d1keys), len(d2keys))
        return data, np.flipud(array.T), dimensions


    def clone(self, data=None, shared_data=True, *args, **overrides):
        if (data is None) and shared_data:
            data = self._data
        return super(HeatMap, self).clone(data, shared_data)


    def __getitem__(self, coords):
        """
        Slice the underlying NdMapping.
        """
        if coords in self.dimensions(): return self.dimension_values(coords)
        return self.clone(self._data.select(**dict(zip(self._data.kdims, coords))))


    def dense_keys(self):
        d1keys = np.unique(self._data.dimension_values(0))
        d2keys = np.unique(self._data.dimension_values(1))
        return list(zip(*[(d1, d2) for d1 in d1keys for d2 in d2keys]))


    def dimension_values(self, dim, unique=False):
        dim = self.get_dimension(dim).name
        if dim in self.kdims:
            if unique:
                return np.unique(self._data.dimension_values(dim))
            else:
                idx = self.get_dimension_index(dim)
                return self.dense_keys()[idx]
        elif dim in self.vdims:
            if unique:
                return self._data.dimension_values(dim)
            else:
                return np.rot90(self.data, 3).flatten()
        else:
            return super(HeatMap, self).dimension_values(dim)


    def dframe(self, dense=False):
        if dense:
            keys1, keys2 = self.dense_keys()
            dense_map = self.clone({(k1, k2): self._data.get((k1, k2), np.NaN)
                                 for k1, k2 in product(keys1, keys2)})
            return dense_map.dframe()
        return super(HeatMap, self).dframe()



class Image(SheetCoordinateSystem, Raster):
    """
    Image is the atomic unit as which 2D data is stored, along with
    its bounds object. The input data may be a numpy.matrix object or
    a two-dimensional numpy array.

    Allows slicing operations of the data in sheet coordinates or direct
    access to the data, via the .data attribute.
    """

    bounds = param.ClassSelector(class_=BoundingRegion, default=BoundingBox(), doc="""
       The bounding region in sheet coordinates containing the data.""")

    group = param.String(default='Image', constant=True)

    vdims = param.List(default=[Dimension('z')],
                       bounds=(1, 1), doc="""
        The dimension description of the data held in the matrix.""")


    def __init__(self, data, bounds=None, extents=None, xdensity=None, ydensity=None, **params):
        bounds = bounds if bounds is not None else BoundingBox()
        if np.isscalar(bounds):
            bounds = BoundingBox(radius=bounds)
        elif isinstance(bounds, (tuple, list, np.ndarray)):
            l, b, r, t = bounds
            bounds = BoundingBox(points=((l, b), (r, t)))
        if data is None: data = np.array([[0]])
        l, b, r, t = bounds.lbrt()
        extents = extents if extents else (None, None, None, None)
        Element2D.__init__(self, data, extents=extents, bounds=bounds,
                           **params)

        (dim1, dim2) = self.data.shape[1], self.data.shape[0]
        xdensity = xdensity if xdensity else dim1/float(r-l)
        ydensity = ydensity if ydensity else dim2/float(t-b)
        SheetCoordinateSystem.__init__(self, bounds, xdensity, ydensity)

        if len(self.data.shape) == 3:
            if self.data.shape[2] != len(self.vdims):
                raise ValueError("Input array has shape %r but %d value dimensions defined"
                                 % (self.data.shape, len(self.vdims)))


    def _convert_element(self, data):
        if isinstance(data, (Raster, HeatMap)):
            return data.data
        else:
            return super(Image, self)._convert_element(data)


    def closest(self, coords=[], **kwargs):
        """
        Given a single coordinate or multiple coordinates as
        a tuple or list of tuples or keyword arguments matching
        the dimension closest will find the closest actual x/y
        coordinates.
        """
        if kwargs and coords:
            raise ValueError("Specify coordinate using as either a list "
                             "keyword arguments not both")
        if kwargs:
            coords = []
            getter = []
            for k, v in kwargs.items():
                idx = self.get_dimension_index(k)
                if np.isscalar(v):
                    coords.append((0, v) if idx else (v, 0))
                else:
                    if isinstance(coords, tuple):
                        coords = [(0, c) if idx else (c, 0) for c in v]
                    if len(coords) not in [0, len(v)]:
                        raise ValueError("Length of samples must match")
                    elif len(coords):
                        coords = [(t[abs(idx-1)], c) if idx else (c, t[abs(idx-1)])
                                  for c, t in zip(v, coords)]
                getter.append(idx)
        else:
            getter = [0, 1]
        getter = itemgetter(*sorted(getter))
        coords = list(coords)
        if len(coords) == 1:
            coords = coords[0]
        if isinstance(coords, tuple):
            return getter(self.closest_cell_center(*coords))
        else:
            return [getter(self.closest_cell_center(*el)) for el in coords]


    def __getitem__(self, coords):
        """
        Slice the underlying numpy array in sheet coordinates.
        """
        if coords in self.dimensions(): return self.dimension_values(coords)
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


    def range(self, dim, data_range=True):
        dim_idx = dim if isinstance(dim, int) else self.get_dimension_index(dim)
        dim = self.get_dimension(dim_idx)
        if dim.range != (None, None):
            return dim.range
        elif dim_idx in [0, 1]:
            l, b, r, t = self.bounds.lbrt()
            if dim_idx:
                drange = (b, t)
            else:
                drange = (l, r)
        elif dim_idx < len(self.vdims) + 2:
            dim_idx -= 2
            data = np.atleast_3d(self.data)[:, :, dim_idx]
            drange = (np.nanmin(data), np.nanmax(data))
        if data_range:
            soft_range = [r for r in dim.soft_range if r is not None]
            if soft_range:
                return util.max_range([drange, soft_range])
            else:
                return drange
        else:
            return dim.soft_range


    def _coord2matrix(self, coord):
        return self.sheet2matrixidx(*coord)


    def dimension_values(self, dim, unique=False):
        """
        The set of samples available along a particular dimension.
        """
        dim_idx = self.get_dimension_index(dim)
        if dim_idx in [0, 1]:
            l, b, r, t = self.bounds.lbrt()
            dim2, dim1 = self.data.shape[:2]
            d1_half_unit = (r - l)/dim1/2.
            d2_half_unit = (t - b)/dim2/2.
            d1lin = np.linspace(l+d1_half_unit, r-d1_half_unit, dim1)
            d2lin = np.linspace(b+d2_half_unit, t-d2_half_unit, dim2)
            if unique:
                return d2lin if dim_idx else d1lin
            else:
                X, Y = np.meshgrid(d1lin, d2lin)
                return Y.flatten() if dim_idx else X.flatten()
        elif dim_idx == 2:
            return np.flipud(self.data).T.flatten()
        else:
            super(Image, self).dimension_values(dim)



class RGB(Image):
    """
    An RGB element is a Image containing channel data for the the
    red, green, blue and (optionally) the alpha channels. The values
    of each channel must be in the range 0.0 to 1.0.

    In input array may have a shape of NxMx4 or NxMx3. In the latter
    case, the defined alpha dimension parameter is appended to the
    list of value dimensions.
    """

    group = param.String(default='RGB', constant=True)

    alpha_dimension = param.ClassSelector(default=Dimension('A',range=(0,1)),
                                          class_=Dimension, instantiate=False,  doc="""
        The alpha dimension definition to add the value dimensions if
        an alpha channel is supplied.""")

    vdims = param.List(
        default=[Dimension('R', range=(0,1)), Dimension('G',range=(0,1)),
                 Dimension('B', range=(0,1))], bounds=(3, 4), doc="""
        The dimension description of the data held in the matrix.

        If an alpha channel is supplied, the defined alpha_dimension
        is automatically appended to this list.""")

    @property
    def rgb(self):
        """
        Returns the corresponding RGB element.

        Other than the updating parameter definitions, this is the
        only change needed to implemented an arbitrary colorspace as a
        subclass of RGB.
        """
        return self


    @classmethod
    def load_image(cls, filename, height=1, array=False, bounds=None, bare=False, **kwargs):
        """
        Returns an raster element or raw numpy array from a PNG image
        file, using matplotlib.

        The specified height determines the bounds of the raster
        object in sheet coordinates: by default the height is 1 unit
        with the width scaled appropriately by the image aspect ratio.

        Note that as PNG images are encoded as RGBA, the red component
        maps to the first channel, the green component maps to the
        second component etc. For RGB elements, this mapping is
        trivial but may be important for subclasses e.g. for HSV
        elements.

        Setting bare=True will apply options disabling axis labels
        displaying just the bare image. Any additional keyword
        arguments will be passed to the Image object.
        """
        try:
            from matplotlib import pyplot as plt
        except:
            raise ImportError("RGB.load_image requires matplotlib.")

        data = plt.imread(filename)
        if array:  return data

        (h, w, channels) = data.shape
        if bounds is None:
            f = float(height) / h
            xoffset, yoffset = w*f/2, h*f/2
            bounds=(-xoffset, -yoffset, xoffset, yoffset)
        rgb = cls(data, bounds=bounds, **kwargs)
        if bare: rgb = rgb(plot=dict(xaxis=None, yaxis=None))
        return rgb


    def dimension_values(self, dim, unique=False):
        """
        The set of samples available along a particular dimension.
        """
        dim_idx = self.get_dimension_index(dim)
        if self.ndims <= dim_idx < len(self.dimensions()):
            return np.flipud(self.data[:,:,dim_idx-self.ndims]).T.flatten()
        return super(RGB, self).dimension_values(dim, unique=True)


    def __init__(self, data, **params):
        sliced = None
        if isinstance(data, Overlay):
            images = data.values()
            if not all(isinstance(im, Image) for im in images):
                raise ValueError("Input overlay must only contain Image elements")
            shapes = [im.data.shape for im in images]
            if not all(shape==shapes[0] for shape in shapes):
                raise ValueError("Images in the input overlays must contain data of the consistent shape")
            ranges = [im.vdims[0].range for im in images]
            if any(None in r for r in ranges):
                raise ValueError("Ranges must be defined on all the value dimensions of all the Images")
            arrays = [(im.data - r[0]) / (r[1] - r[0]) for r,im in zip(ranges, images)]
            data = np.dstack(arrays)

        if not isinstance(data, Element):
            if len(data.shape) != 3:
                raise ValueError("Three dimensional matrices or arrays required")
            elif data.shape[2] == 4:
                sliced = data[:,:,:-1]

        if len(params.get('vdims',[])) == 4:
            alpha_dim = params['vdims'].pop(3)
            params['alpha_dimension'] = alpha_dim

        super(RGB, self).__init__(data if sliced is None else sliced, **params)
        if sliced is not None:
            self.vdims.append(self.alpha_dimension)
            self.data = data


    def __getitem__(self, coords):
        """
        Slice the underlying numpy array in sheet coordinates.
        """
        if coords in self.dimensions(): return self.dimension_values(coords)
        if not isinstance(coords, slice) and len(coords) > self.ndims:
            value = coords[self.ndims:]
            if len(value) > 1:
                raise KeyError("Only one value dimension may be indexed at a time")

            sliced = super(RGB, self).__getitem__(coords[:self.ndims])
            vidx = self.get_dimension_index(value[0])
            val_index = vidx - self.ndims
            data = sliced.data[:,:, val_index]
            return Image(data, **dict(self.get_param_values(onlychanged=True),
                                       vdims=[self.vdims[val_index]]))
        else:
            return super(RGB, self).__getitem__(coords)


class HSV(RGB):
    """
    Example of a commonly used color space subclassed from RGB used
    for working in a HSV (hue, saturation and value) color space.
    """

    group = param.String(default='HSV', constant=True)

    alpha_dimension = param.ClassSelector(default=Dimension('A',range=(0,1)),
                                          class_=Dimension, instantiate=False,  doc="""
        The alpha dimension definition to add the value dimensions if
        an alpha channel is supplied.""")

    vdims = param.List(
        default=[Dimension('H', range=(0,1), cyclic=True),
                 Dimension('S',range=(0,1)),
                 Dimension('V', range=(0,1))], bounds=(3, 4), doc="""
        The dimension description of the data held in the array.

        If an alpha channel is supplied, the defined alpha_dimension
        is automatically appended to this list.""")

    hsv_to_rgb = np.vectorize(colorsys.hsv_to_rgb)

    @property
    def rgb(self):
        """
        Conversion from HSV to RGB.
        """
        hsv = self.hsv_to_rgb(self.data[:,:,0],
                              self.data[:,:,1],
                              self.data[:,:,2])
        if len(self.vdims) == 4:
            hsv += (self.data[:,:,3],)

        return RGB(np.dstack(hsv), bounds=self.bounds,
                   group=self.group,
                   label=self.label)
