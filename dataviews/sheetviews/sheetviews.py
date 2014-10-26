import numpy as np

import param

from .boundingregion import BoundingBox, BoundingRegion
from .sheetcoords import SheetCoordinateSystem, Slice

from ..dataviews import Matrix, LayerMap, Layer
from ..ndmapping import NdMapping, Dimension
from ..options import channels
from ..views import View, Overlay, Annotation, Grid


class SheetView(SheetCoordinateSystem, Matrix):
    """
    SheetView is the atomic unit as which 2D data is stored, along with its
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
        The label of the x- and y-dimension of the SheetView in form
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

        return SheetView(Slice(bounds, self).submatrix(self.data),
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
        return SheetView(data, roi_bounds, style=self.style, value=self.value)


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
        The label of the x- and y-dimension of the SheetView in form
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
        The label of the x- and y-dimension of the SheetView in form
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


__all__ = list(set([_k for _k,_v in locals().items() if isinstance(_v,type) and
                    (issubclass(_v, NdMapping) or issubclass(_v, View))]))
