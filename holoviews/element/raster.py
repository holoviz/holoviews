from operator import itemgetter

import numpy as np
import colorsys
import param

from ..core import util, config
from ..core.data import ImageInterface, GridInterface
from ..core.data.interface import DataError
from ..core import Dimension, Element2D, Overlay, Dataset
from ..core.boundingregion import BoundingRegion, BoundingBox
from ..core.sheetcoords import SheetCoordinateSystem, Slice
from .chart import Curve
from .graphs import TriMesh
from .tabular import Table
from .util import compute_slice_bounds, categorical_aggregate2d


class Raster(Element2D):
    """
    Raster is a basic 2D element type for presenting either numpy or
    dask arrays as two dimensional raster images.

    Arrays with a shape of (N,M) are valid inputs for Raster whereas
    subclasses of Raster (e.g. RGB) may also accept 3D arrays
    containing channel information.

    Raster does not support slicing like the Image or RGB subclasses
    and the extents are in matrix coordinates if not explicitly
    specified.
    """

    kdims = param.List(default=[Dimension('x'), Dimension('y')],
                       bounds=(2, 2), constant=True, doc="""
        The label of the x- and y-dimension of the Raster in form
        of a string or dimension object.""")

    group = param.String(default='Raster', constant=True)

    vdims = param.List(default=[Dimension('z')],
                       bounds=(1, 1), doc="""
        The dimension description of the data held in the matrix.""")

    def __init__(self, data, kdims=None, vdims=None, extents=None, **params):
        if extents is None:
            (d1, d2) = data.shape[:2]
            extents = (0, 0, d2, d1)
        super(Raster, self).__init__(data, kdims=kdims, vdims=vdims, extents=extents, **params)


    def __getitem__(self, slices):
        if slices in self.dimensions(): return self.dimension_values(slices)
        slices = util.process_ellipses(self,slices)
        if not isinstance(slices, tuple):
            slices = (slices, slice(None))
        elif len(slices) > (2 + self.depth):
            raise KeyError("Can only slice %d dimensions" % 2 + self.depth)
        elif len(slices) == 3 and slices[-1] not in [self.vdims[0].name, slice(None)]:
            raise KeyError("%r is the only selectable value dimension" % self.vdims[0].name)

        slc_types = [isinstance(sl, slice) for sl in slices[:2]]
        data = self.data.__getitem__(slices[:2][::-1])
        if all(slc_types):
            return self.clone(data, extents=None)
        elif not any(slc_types):
            return data
        else:
            return self.clone(np.expand_dims(data, axis=slc_types.index(True)),
                              extents=None)


    def range(self, dim, data_range=True, dimension_range=True):
        idx = self.get_dimension_index(dim)
        if data_range and idx == 2:
            dimension = self.get_dimension(dim)
            lower, upper = np.nanmin(self.data), np.nanmax(self.data)
            if not dimension_range:
                return lower, upper
            return util.dimension_range(lower, upper, dimension.range, dimension.soft_range)
        return super(Raster, self).range(dim, data_range, dimension_range)


    def dimension_values(self, dim, expanded=True, flat=True):
        """
        The set of samples available along a particular dimension.
        """
        dim_idx = self.get_dimension_index(dim)
        if not expanded and dim_idx == 0:
            return np.array(range(self.data.shape[1]))
        elif not expanded and dim_idx == 1:
            return np.array(range(self.data.shape[0]))
        elif dim_idx in [0, 1]:
            values = np.mgrid[0:self.data.shape[1], 0:self.data.shape[0]][dim_idx]
            return values.flatten() if flat else values
        elif dim_idx == 2:
            arr = self.data.T
            return arr.flatten() if flat else arr
        else:
            return super(Raster, self).dimension_values(dim)


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
                samples = zip(*[c if isinstance(c, list) else [c] for _, c in
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
            x_vals = self.dimension_values(other_dimension[0].name, False)
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
        function, dims = self._reduce_map(dimensions, function, reduce_map)
        if len(dims) == self.ndims:
            if isinstance(function, np.ufunc):
                return function.reduce(self.data, axis=None)
            else:
                return function(self.data)
        else:
            dimension = dims[0]
            other_dimension = [d for d in self.kdims if d.name != dimension]
            oidx = self.get_dimension_index(other_dimension[0])
            x_vals = self.dimension_values(other_dimension[0].name, False)
            reduced = function(self._zdata, axis=oidx)
            if oidx and hasattr(self, 'bounds'):
                reduced = reduced[::-1]
            data = zip(x_vals, reduced)
            params = dict(dict(self.get_param_values(onlychanged=True)),
                          kdims=other_dimension, vdims=self.vdims)
            params.pop('bounds', None)
            params.pop('extents', None)
            return Table(data, **params)


    @property
    def depth(self):
        return len(self.vdims)


    @property
    def _zdata(self):
        return self.data


    def _coord2matrix(self, coord):
        return int(round(coord[1])), int(round(coord[0]))




class Image(Dataset, Raster, SheetCoordinateSystem):
    """
    Image is the atomic unit as which 2D data is stored, along with
    its bounds object. The input data may be a numpy.matrix object or
    a two-dimensional numpy array.

    Allows slicing operations of the data in sheet coordinates or direct
    access to the data, via the .data attribute.
    """

    bounds = param.ClassSelector(class_=BoundingRegion, default=BoundingBox(), doc="""
        The bounding region in sheet coordinates containing the data.""")

    datatype = param.List(default=['image', 'grid', 'xarray', 'cube', 'dataframe', 'dictionary'])

    group = param.String(default='Image', constant=True)

    kdims = param.List(default=[Dimension('x'), Dimension('y')],
                       bounds=(2, 2), constant=True, doc="""
        The label of the x- and y-dimension of the Raster in the form
        of a string or dimension object.""")

    vdims = param.List(default=[Dimension('z')],
                       bounds=(1, 1), doc="""
        The dimension description of the data held in the matrix.""")

    rtol = param.Number(default=None, doc="""
        The tolerance used to enforce regular sampling for regular, gridded
        data where regular sampling is expected. Expressed as the maximal 
        allowable sampling difference between sample locations.""")

    def __init__(self, data, kdims=None, vdims=None, bounds=None, extents=None,
                 xdensity=None, ydensity=None, rtol=None, **params):
        supplied_bounds = bounds
        if isinstance(data, Image):
            bounds = bounds or data.bounds
            xdensity = xdensity or data.xdensity
            ydensity = ydensity or data.ydensity
            if rtol is None: rtol = data.rtol
        extents = extents if extents else (None, None, None, None)
        if (data is None
            or (isinstance(data, (list, tuple)) and not data)
            or (isinstance(data, np.ndarray) and data.size == 0)):
            data = np.zeros((2, 2))
        if rtol is not None:
            params['rtol'] = rtol
        else:
            params['rtol'] = config.image_rtol

        Dataset.__init__(self, data, kdims=kdims, vdims=vdims, extents=extents, **params)
        if not self.interface.gridded:
            raise DataError("%s type expects gridded data, %s is columnar."
                            "To display columnar data as gridded use the HeatMap "
                            "element or aggregate the data." %
                            (type(self).__name__, self.interface.__name__))

        dim2, dim1 = self.interface.shape(self, gridded=True)[:2]
        if bounds is None:
            xvals = self.dimension_values(0, False)
            l, r, xdensity, _ = util.bound_range(xvals, xdensity, self._time_unit)
            yvals = self.dimension_values(1, False)
            b, t, ydensity, _ = util.bound_range(yvals, ydensity, self._time_unit)
            bounds = BoundingBox(points=((l, b), (r, t)))
        elif np.isscalar(bounds):
            bounds = BoundingBox(radius=bounds)
        elif isinstance(bounds, (tuple, list, np.ndarray)):
            l, b, r, t = bounds
            bounds = BoundingBox(points=((l, b), (r, t)))

        data_bounds = None
        if self.interface is ImageInterface and not isinstance(data, np.ndarray):
            data_bounds = self.bounds.lbrt()
        l, b, r, t = bounds.lbrt()
        xdensity = xdensity if xdensity else util.compute_density(l, r, dim1, self._time_unit)
        ydensity = ydensity if ydensity else util.compute_density(b, t, dim2, self._time_unit)
        if not np.isfinite(xdensity) or not np.isfinite(ydensity):
            raise ValueError('Density along Image axes could not be determined. '
                             'If the data contains only one coordinate along the '
                             'x- or y-axis ensure you declare the bounds and/or '
                             'density.')
        SheetCoordinateSystem.__init__(self, bounds, xdensity, ydensity)
        self._validate(data_bounds, supplied_bounds)


    def _validate(self, data_bounds, supplied_bounds):
        if len(self.shape) == 3:
            if self.shape[2] != len(self.vdims):
                raise ValueError("Input array has shape %r but %d value dimensions defined"
                                 % (self.shape, len(self.vdims)))

        # Ensure coordinates are regularly sampled
        xdim, ydim = self.kdims
        xvals, yvals = (self.dimension_values(d, expanded=False) for d in self.kdims)
        xvalid = util.validate_regular_sampling(xvals, self.rtol)
        yvalid = util.validate_regular_sampling(yvals, self.rtol)
        msg = ("{clsname} dimension{dims} not evenly sampled to relative "
               "tolerance of {rtol}. Please use the QuadMesh element for "
               "irregularly sampled data or set a higher tolerance on "
               "hv.config.image_rtol or the rtol parameter in the "
               "{clsname} constructor.")
        dims = None
        if not xvalid:
            dims = ' %s is ' % xdim if yvalid else '(s) %s and %s are' % (xdim, ydim)
        elif not yvalid:
            dims = ' %s is' % ydim
        if dims:
            self.warning(msg.format(clsname=type(self).__name__, dims=dims, rtol=self.rtol))

        if not supplied_bounds:
            return

        if data_bounds is None:
            (x0, x1), (y0, y1) = (self.interface.range(self, kd.name) for kd in self.kdims)
            xstep = (1./self.xdensity)/2.
            ystep = (1./self.ydensity)/2.
            if not isinstance(x0, util.datetime_types):
                x0, x1 = (x0-xstep, x1+xstep)
            if not isinstance(y0, util.datetime_types):
                y0, y1 = (y0-ystep, y1+ystep)
            bounds = (x0, y0, x1, y1)
        else:
            bounds = data_bounds

        not_close = False
        for r, c in zip(bounds, self.bounds.lbrt()):
            if isinstance(r, util.datetime_types):
                r = util.dt_to_int(r)
            if isinstance(c, util.datetime_types):
                c = util.dt_to_int(c)
            if util.isfinite(r) and not np.isclose(r, c, rtol=self.rtol):
                not_close = True
        if not_close:
            raise ValueError('Supplied Image bounds do not match the coordinates defined '
                             'in the data. Bounds only have to be declared if no coordinates '
                             'are supplied, otherwise they must match the data. To change '
                             'the displayed extents set the range on the x- and y-dimensions.')


    def __setstate__(self, state):
        """
        Ensures old-style unpickled Image types without an interface
        use the ImageInterface.

        Note: Deprecate as part of 2.0
        """
        self.__dict__ = state
        if isinstance(self.data, np.ndarray):
            self.interface = ImageInterface
        super(Dataset, self).__setstate__(state)


    def clone(self, data=None, shared_data=True, new_type=None, *args, **overrides):
        """
        Returns a clone of the object with matching parameter values
        containing the specified args and kwargs.

        If shared_data is set to True and no data explicitly supplied,
        the clone will share data with the original. May also supply
        a new_type, which will inherit all shared parameters.
        """
        if data is None and (new_type is None or issubclass(new_type, Image)):
            sheet_params = dict(bounds=self.bounds, xdensity=self.xdensity,
                                ydensity=self.ydensity)
            overrides = dict(sheet_params, **overrides)
        return super(Image, self).clone(data, shared_data, new_type, *args, **overrides)


    def aggregate(self, dimensions=None, function=None, spreadfn=None, **kwargs):
        agg = super(Image, self).aggregate(dimensions, function, spreadfn, **kwargs)
        return Curve(agg) if isinstance(agg, Dataset) and len(self.vdims) == 1 else agg


    def select(self, selection_specs=None, **selection):
        """
        Allows selecting data by the slices, sets and scalar values
        along a particular dimension. The indices should be supplied as
        keywords mapping between the selected dimension and
        value. Additionally selection_specs (taking the form of a list
        of type.group.label strings, types or functions) may be
        supplied, which will ensure the selection is only applied if the
        specs match the selected object.
        """
        if selection_specs and not any(self.matches(sp) for sp in selection_specs):
            return self

        selection = {self.get_dimension(k).name: slice(*sel) if isinstance(sel, tuple) else sel
                     for k, sel in selection.items() if k in self.kdims}
        coords = tuple(selection[kd.name] if kd.name in selection else slice(None)
                       for kd in self.kdims)

        shape = self.interface.shape(self, gridded=True)
        if any([isinstance(el, slice) for el in coords]):
            bounds = compute_slice_bounds(coords, self, shape[:2])

            xdim, ydim = self.kdims
            l, b, r, t = bounds.lbrt()

            # Situate resampled region into overall slice
            y0, y1, x0, x1 = Slice(bounds, self)
            y0, y1 = shape[0]-y1, shape[0]-y0
            selection = (slice(y0, y1), slice(x0, x1))
            sliced = True
        else:
            y, x = self.sheet2matrixidx(coords[0], coords[1])
            y = shape[0]-y-1
            selection = (y, x)
            sliced = False

        datatype = list(util.unique_iterator([self.interface.datatype]+self.datatype))
        data = self.interface.ndloc(self, selection)
        if not sliced:
            if np.isscalar(data):
                return data
            elif isinstance(data, tuple):
                data = data[self.ndims:]
            return self.clone(data, kdims=[], new_type=Dataset,
                              datatype=datatype)
        else:
            return self.clone(data, xdensity=self.xdensity, datatype=datatype,
                              ydensity=self.ydensity, bounds=bounds)


    def sample(self, samples=[], **kwargs):
        """
        Allows sampling of an Image as an iterator of coordinates
        matching the key dimensions, returning a new object containing
        just the selected samples. Alternatively may supply kwargs to
        sample a coordinate on an object. On an Image the coordinates
        are continuously indexed and will always snap to the nearest
        coordinate.
        """
        kwargs = {k: v for k, v in kwargs.items() if k != 'closest'}
        if kwargs and samples:
            raise Exception('Supply explicit list of samples or kwargs, not both.')
        elif kwargs:
            sample = [slice(None) for _ in range(self.ndims)]
            for dim, val in kwargs.items():
                sample[self.get_dimension_index(dim)] = val
            samples = [tuple(sample)]

        # If a 1D cross-section of 2D space return Curve
        shape = self.interface.shape(self, gridded=True)
        if len(samples) == 1:
            dims = [kd for kd, v in zip(self.kdims, samples[0])
                    if not (np.isscalar(v) or isinstance(v, util.datetime_types))]
            if len(dims) == 1:
                kdims = [self.get_dimension(kd) for kd in dims]
                sample = tuple(np.datetime64(s) if isinstance(s, util.datetime_types) else s
                               for s in samples[0])
                sel = {kd.name: s for kd, s in zip(self.kdims, sample)}
                dims = [kd for kd, v in sel.items() if not np.isscalar(v)]
                selection = self.select(**sel)
                selection = tuple(selection.columns(kdims+self.vdims).values())
                datatype = list(util.unique_iterator(self.datatype+['dataframe', 'dict']))
                return self.clone(selection, kdims=kdims, new_type=Curve,
                                  datatype=datatype)
            else:
                kdims = self.kdims
        else:
            kdims = self.kdims

        xs, ys = zip(*samples)
        if isinstance(xs[0], util.datetime_types):
            xs = np.array(xs).astype(np.datetime64)
        if isinstance(ys[0], util.datetime_types):
            ys = np.array(ys).astype(np.datetime64)
        yidx, xidx = self.sheet2matrixidx(np.array(xs), np.array(ys))
        yidx = shape[0]-yidx-1

        # Detect out-of-bounds indices
        out_of_bounds= (yidx<0) | (xidx<0) | (yidx>=shape[0]) | (xidx>=shape[1])
        if out_of_bounds.any():
            coords = [samples[idx] for idx in np.where(out_of_bounds)[0]]
            raise IndexError('Coordinate(s) %s out of bounds for %s with bounds %s' %
                             (coords, type(self).__name__, self.bounds.lbrt()))

        data = self.interface.ndloc(self, (yidx, xidx))
        return self.clone(data, new_type=Table, datatype=['dataframe', 'dict'])


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
                    if isinstance(v, list):
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
        if len(coords) == 1:
            coords = coords[0]
        if isinstance(coords, tuple):
            return getter(self.closest_cell_center(*coords))
        else:
            return [getter(self.closest_cell_center(*el)) for el in coords]


    def range(self, dim, data_range=True, dimension_range=True):
        idx = self.get_dimension_index(dim)
        dimension = self.get_dimension(dim)
        if idx in [0, 1] and data_range and dimension.range == (None, None):
            if self.interface.datatype == 'image':
                l, b, r, t = self.bounds.lbrt()
                return (b, t) if idx else (l, r)
            low, high = super(Image, self).range(dim, data_range)
            density = self.ydensity if idx else self.xdensity
            halfd = (1./density)/2.
            if isinstance(low, util.datetime_types):
                halfd = np.timedelta64(int(round(halfd)), self._time_unit)
            return (low-halfd, high+halfd)
        else:
            return super(Image, self).range(dim, data_range, dimension_range)


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
        return self.clone(self.columns(), new_type=Table,
                          **(dict(datatype=datatype) if datatype else {}))


    def _coord2matrix(self, coord):
        return self.sheet2matrixidx(*coord)


class GridImage(Image):
    def __init__(self, *args, **kwargs):
        self.warning('GridImage is now deprecated. Please use Image element instead.')
        super(GridImage, self).__init__(*args, **kwargs)


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

    _vdim_reductions = {1: Image}

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

        (h, w, _) = data.shape
        if bounds is None:
            f = float(height) / h
            xoffset, yoffset = w*f/2, h*f/2
            bounds=(-xoffset, -yoffset, xoffset, yoffset)
        rgb = cls(data, bounds=bounds, **kwargs)
        if bare: rgb = rgb(plot=dict(xaxis=None, yaxis=None))
        return rgb


    def __init__(self, data, kdims=None, vdims=None, **params):
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
        if vdims is None:
            vdims = list(self.vdims)
        else:
            vdims = list(vdims) if isinstance(vdims, list) else [vdims]
        if isinstance(data, np.ndarray):
            if data.shape[-1] == 4 and len(vdims) == 3:
                vdims.append(self.alpha_dimension)
        super(RGB, self).__init__(data, kdims=kdims, vdims=vdims, **params)



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
        coords = tuple(self.dimension_values(d, expanded=False)
                       for d in self.kdims)
        data = [self.dimension_values(d, flat=False)
                for d in self.vdims]

        hsv = self.hsv_to_rgb(*data[:3])
        if len(self.vdims) == 4:
            hsv += (data[3],)

        params = util.get_param_values(self)
        del params['vdims']
        return RGB(coords+hsv, bounds=self.bounds,
                   xdensity=self.xdensity, ydensity=self.ydensity,
                   **params)


class QuadMesh(Dataset, Element2D):
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

    kdims = param.List(default=[Dimension('x'), Dimension('y')],
                       bounds=(2, 2), constant=True)

    vdims = param.List(default=[Dimension('z')], bounds=(1, None))

    _binned = True

    def __init__(self, data, kdims=None, vdims=None, **params):
        super(QuadMesh, self).__init__(data, kdims, vdims, **params)
        if not self.interface.gridded:
            raise DataError("%s type expects gridded data, %s is columnar."
                            "To display columnar data as gridded use the HeatMap "
                            "element or aggregate the data." %
                            (type(self).__name__, self.interface.__name__))


    def __setstate__(self, state):
        """
        Ensures old-style QuadMesh types without an interface can be unpickled.

        Note: Deprecate as part of 2.0
        """
        if 'interface' not in state:
            self.interface = GridInterface
            x, y = state['_kdims_param_value']
            z = state['_vdims_param_value'][0]
            data = state['data']
            state['data'] = {x.name: data[0], y.name: data[1], z.name: data[2]}
        super(Dataset, self).__setstate__(state)


    def trimesh(self):
        """
        Converts a QuadMesh into a TriMesh.
        """
        # Generate vertices
        xs = self.interface.coords(self, 0, edges=True)
        ys = self.interface.coords(self, 1, edges=True)
        if xs.ndim == 1:
            xs, ys = (np.tile(xs[:, np.newaxis], len(ys)).T,
                      np.tile(ys[:, np.newaxis], len(xs)))
        vertices = (xs.T.flatten(), ys.T.flatten())

        # Generate triangle simplexes
        shape = self.dimension_values(2, flat=False).shape
        s0 = shape[0]
        t1 = np.arange(np.product(shape))
        js = (t1//s0)
        t1s = js*(s0+1)+t1%s0
        t2s = t1s+1
        t3s = (js+1)*(s0+1)+t1%s0
        t4s = t2s
        t5s = t3s
        t6s = t3s+1
        t1 = np.concatenate([t1s, t6s])
        t2 = np.concatenate([t2s, t5s])
        t3 = np.concatenate([t3s, t4s])
        ts = (t1, t2, t3)
        for vd in self.vdims:
            zs = self.dimension_values(vd)
            ts = ts + (np.concatenate([zs, zs]),)

        # Construct TriMesh
        params = util.get_param_values(self)
        params['kdims'] = params['kdims'] + TriMesh.node_type.kdims[2:]
        nodes = TriMesh.node_type(vertices+(np.arange(len(vertices[0])),),
                                  **{k: v for k, v in params.items()
                                     if k != 'vdims'})
        return TriMesh(((ts,), nodes), **{k: v for k, v in params.items()
                                          if k != 'kdims'})



class HeatMap(Dataset, Element2D):
    """
    HeatMap is an atomic Element used to visualize two dimensional
    parameter spaces. It supports sparse or non-linear spaces, dynamically
    upsampling them to a dense representation, which can be visualized.

    A HeatMap can be initialized with any dict or NdMapping type with
    two-dimensional keys.
    """

    group = param.String(default='HeatMap', constant=True)

    kdims = param.List(default=[Dimension('x'), Dimension('y')],
                       bounds=(2, 2), constant=True)

    vdims = param.List(default=[Dimension('z')], constant=True)

    def __init__(self, data, kdims=None, vdims=None, **params):
        super(HeatMap, self).__init__(data, kdims=kdims, vdims=vdims, **params)
        self.gridded = categorical_aggregate2d(self)

    @property
    def raster(self):
        self.warning("The .raster attribute on HeatMap is deprecated, "
                     "the 2D aggregate is now computed dynamically "
                     "during plotting.")
        return self.gridded.dimension_values(2, flat=False)
