import numpy as np

import param

from .boundingregion import BoundingBox, BoundingRegion
from .sheetcoords import SheetCoordinateSystem, Slice

from ..dataviews import Matrix, DataStack, TableStack
from ..ndmapping import NdMapping, Dimension
from ..options import channels
from ..views import View, Overlay, Annotation, Grid, find_minmax


class SheetLayer(View):
    """
    A SheetLayer is a data structure for holding one or more numpy
    arrays embedded within a two-dimensional space. The array(s) may
    correspond to a discretisation of an image (i.e. a rasterisation)
    or vector elements such as points or lines. Lines may be linearly
    interpolated or correspond to control nodes of a smooth vector
    representation such as Bezier splines.
    """

    dimensions = param.List(default=[Dimension('X'), Dimension('Y')])

    bounds = param.ClassSelector(class_=BoundingRegion, default=BoundingBox(), doc="""
       The bounding region in sheet coordinates containing the data.""")

    roi_bounds = param.ClassSelector(class_=BoundingRegion, default=None, doc="""
        The ROI can be specified to select only a sub-region of the bounds to
        be stored as data.""")

    value = param.ClassSelector(class_=(str, Dimension),
                                default=Dimension('Z'), doc="""
        The default dimension for SheetLayers is the Z-axis.""")

    _abstract = True


    @property
    def stack_type(self):
        return SheetStack


    def __init__(self, data, bounds, **kwargs):
        View.__init__(self, data, bounds=bounds, **kwargs)


    def __mul__(self, other):

        if isinstance(other, SheetStack):
            items = [(k, self * v) for (k, v) in other.items()]
            return other.clone(items=items)

        self_layers = self.data if isinstance(self, SheetOverlay) else [self]
        other_layers = other.data if isinstance(other, SheetOverlay) else [other]
        combined_layers = self_layers + other_layers

        if isinstance(other, Annotation):
            return SheetOverlay(combined_layers, self.bounds,
                                roi_bounds=self.roi_bounds)

        if self.bounds is None:
            self.bounds = other.bounds
        elif other.bounds is None:
            other.bounds = self.bounds

        roi_bounds = self.roi_bounds if self.roi_bounds else other.roi_bounds
        roi_bounds = self.bounds if roi_bounds is None else roi_bounds
        return SheetOverlay(combined_layers, self.bounds, roi_bounds=roi_bounds)


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


    @property
    def xlabel(self):
        return str(self.dimensions[0])


    @property
    def ylabel(self):
        return str(self.dimensions[1])


    @property
    def xlim(self):
        l, _, r, _ = self.bounds.lbrt()
        return (l, r)


    @property
    def ylim(self):
        _, b, _, t = self.bounds.lbrt()
        return (b, t)



class SheetOverlay(SheetLayer, Overlay):
    """
    SheetOverlay extends a regular Overlay with bounds checking and an
    ROI property, which applies the roi_bounds to all SheetLayer
    objects it contains. When adding SheetLayers to an Overlay, a
    common ROI bounds is enforced.

    A SheetOverlay may be used to overlay lines or points over a
    SheetView. In addition, if an overlay consists of three or four
    SheetViews of depth 1, the overlay may be converted to an RGB(A)
    SheetView via the rgb property.
    """

    channels = channels

    @property
    def roi(self):
        """
        Apply the roi_bounds to all elements in the SheetOverlay
        """
        return SheetOverlay([el.get_roi(self.roi_bounds) for el in self.data],
                            bounds=self.roi_bounds if self.roi_bounds else self.bounds)


    @property
    def range(self):
        range = self[0].range
        cyclic = self[0].cyclic_range is not None
        for view in self:
            if isinstance(view, SheetView):
                if cyclic != (self[0].cyclic_range is not None):
                    raise Exception("Overlay contains cyclic and non-cyclic "
                                    "SheetViews, cannot compute range.")
                range = find_minmax(range, view.range)
        return range


    def add(self, layer):
        """
        Overlay a single layer on top of the existing overlay.
        """
        if isinstance(layer, Annotation):
            self.data.append(layer)
            return
        elif layer.bounds.lbrt() != self.bounds.lbrt():
            if layer.bounds is None:
                layer.bounds = self.bounds
            else:
                raise Exception("Layer must have same bounds as SheetOverlay")
        self.data.append(layer)


    def hist(self, index=None, adjoin=True, **kwargs):

        valid_ind = isinstance(index, int) and (0 <= index < len(self))
        valid_label = index in [el.label for el in self.data]
        if index is None or not any([valid_ind, valid_label]):
            raise TypeError("Please supply a suitable index for the histogram data")

        hist = self[index].hist(adjoin=False, **kwargs)
        if adjoin:
            layout = self << hist
            layout.main_layer = index
            return layout
        else:
            return hist


    def __getstate__(self):
        """
        When pickling, make sure to save the relevant channel
        definitions.
        """
        obj_dict = self.__dict__.copy()
        channels = dict((k, self.channels[k]) for k in self.channels.keys())
        obj_dict['channel_definitions'] = channels
        return obj_dict


    def __setstate__(self, d):
        """
        When unpickled, restore the saved channel definitions.
        """

        if 'channel_definitions' not in d:
            self.__dict__.update(d)
            return

        unpickled_channels = d.pop('channel_definitions')
        for key, defs in unpickled_channels.items():
            self.channels[key] = defs
        self.__dict__.update(d)


    def __len__(self):
        return len(self.data)



class SheetView(SheetCoordinateSystem, SheetLayer, Matrix):
    """
    SheetView is the atomic unit as which 2D data is stored, along with its
    bounds object. Allows slicing operations of the data in sheet coordinates or
    direct access to the data, via the .data attribute.

    Arrays with a shape of (X,Y) or (X,Y,Z) are valid. In the case of
    3D arrays, each depth layer is interpreted as a channel of the 2D
    representation.
    """

    dimensions = param.List(default=[Dimension('X'), Dimension('Y')],
                            constant=True, doc="""
        The label of the x- and y-dimension of the SheetView in form
        of a string or dimension object.""")


    value = param.ClassSelector(class_=(str, Dimension),
                                default=Dimension('Z'), doc="""
        The dimension description of the data held in the data array.""")

    _deep_indexable = True

    def __init__(self, data, bounds=None, **kwargs):
        bounds = bounds if bounds else BoundingBox()
        data = np.array([[0]]) if data is None else data
        self.lbrt = bounds.lbrt()
        l, b, r, t = bounds.lbrt()
        (dim1, dim2) = data.shape[0], data.shape[1]
        xdensity = dim1/(r-l)
        ydensity = dim2/(t-b)

        SheetCoordinateSystem.__init__(self, bounds, xdensity, ydensity)
        SheetLayer.__init__(self, data, bounds, **kwargs)


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
                         bounds, label=self.label, style=self.style,
                         value=self.value)


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



class Points(SheetLayer):
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

    null_value = np.array([[], []]).T # For when data is None
    min_dims = 2                      # Minimum number of columns

    def __init__(self, data, bounds=None, **kwargs):
        bounds = bounds if bounds else BoundingBox()

        self._range_column = 3

        if isinstance(data, tuple):
            arrays = [np.array(d) for d in data]
            if not all(len(arr)==len(arrays[0]) for arr in arrays):
                raise Exception("All input arrays must have the same length.")

            arr = np.hstack(tuple(arr.reshape(arr.shape if len(arr.shape)==2
                                              else (len(arr), 1)) for arr in arrays))
        else:
            arr = np.array(data)

        if arr.shape[1] <self.min_dims:
            raise Exception("%s requires a minimum of %s columns."
                            % (self.__class__.__name__, self.min_dims))

        data = self.null_value if data is None else arr
        super(Points, self).__init__(data, bounds, **kwargs)

    def resize(self, bounds):
        return Points(self.data, bounds, style=self.style)

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


    @property
    def roi(self):
        if self.roi_bounds is None: return self
        (N,_) = self.data.shape
        roi_data = self.data[[n for n in range(N)
                              if self.data[n,...][:2] in self.roi_bounds]]
        roi_bounds = self.roi_bounds if self.roi_bounds else self.bounds
        return Points(roi_data, roi_bounds, style=self.style)


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

    null_value = np.array([[], [], [], []]).T # For when data is None
    min_dims = 3                              # Minimum number of columns


    @property
    def range_column(self):
        return self._range_column


    @range_column.setter
    def range_column(self, val):
        self._range_column = val



class Contours(SheetLayer):
    """
    Allows sets of contour lines to be defined over a
    SheetCoordinateSystem.

    The input data is a list of Nx2 numpy arrays where each array
    corresponds to a contour in the group. Each point in the numpy
    array corresponds to an X,Y coordinate.
    """

    def __init__(self, data, bounds=None, **kwargs):
        bounds = bounds if bounds else BoundingBox()
        data = [] if data is None else data
        super(Contours, self).__init__(data, bounds, **kwargs)


    def resize(self, bounds):
        return Contours(self.contours, bounds, style=self.style)


    def __len__(self):
        return len(self.data)


    @property
    def roi(self):
        # Note: Data returned is not sliced to ROI because vertices
        # outside the bounds need to be snapped to the bounding box
        # edges.
        bounds = self.roi_bounds if self.roi_bounds else self.bounds
        return Contours(self.data, bounds, style=self.style, label=self.label)



class SheetStack(DataStack):
    """
    A SheetStack is a stack of SheetLayers over some dimensions. The
    dimension may be a spatial dimension (i.e., a ZStack), time
    (specifying a frame sequence) or any other dimensions along
    which SheetLayers may vary.
    """

    bounds = None

    data_type = (SheetLayer, Annotation)

    overlay_type = SheetOverlay


    def sample(self, coords=[], **samples):
        """
        Sample each SheetView in the Stack by passing either a list
        of coords or the dimension name and the corresponding sample
        values as kwargs.
        """
        if len(samples) == 1: stack_type = DataStack
        else: stack_type = TableStack
        return stack_type([(k, v.sample(coords=coords, **samples)) for k, v in
                           self.items()], **dict(self.get_param_values()))


    def reduce(self, label_prefix='', **reduce_map):
        """
        Reduce each SheetView in the Stack using a function supplied via
        the kwargs, where the keyword has to match a particular dimension
        in the View.
        """
        if len(reduce_map) == 1: stack_type = DataStack
        else: stack_type = TableStack
        return stack_type([(k, v.reduce(label_prefix=label_prefix, **reduce_map))
                           for k, v in self.items()], **dict(self.get_param_values()))


    def grid_sample(self, rows, cols, collate=None, lbrt=None):
        """
        Creates a CoordinateGrid of curves according sampled according to
        the supplied rows and cols. A sub-region to be sampled can be specified
        using the lbrt argument, which expresses the subsampling in sheet
        coordinates. The usual sampling semantics apply.
        """
        xdensity, ydensity = self.last.xdensity, self.last.ydensity
        l, b, r, t = self.last.bounds.lbrt()
        half_x_unit = ((r-l) / xdensity) / 2.
        half_y_unit = ((t-b) / ydensity) / 2.
        if lbrt is None:
            bounds = self.last.bounds
            l, b, r, t = (l+half_x_unit, b+half_y_unit, r-half_x_unit, t-half_y_unit)
        else:
            l, b = self.last.closest_cell_center(lbrt[0], lbrt[1])
            r, t = self.last.closest_cell_center(lbrt[2], lbrt[3])
            bounds = BoundingBox(points=[(l-half_x_unit, b-half_y_unit),
                                         (r+half_x_unit, t+half_y_unit)])
        x, y = np.meshgrid(np.linspace(l, r, cols),
                           np.linspace(b, t, rows))
        coords = list(set([self.last.closest_cell_center(*c) for c in zip(x.flat, y.flat)]))

        grid = self.sample(coords=coords)
        if not collate:
            return grid

        coords = grid.last.data.keys()
        grid = grid.collate(collate)
        grid_data = list(zip(coords, grid.values()))
        return DataGrid(bounds, None, xdensity=self.last.xdensity,
                        ydensity=self.last.ydensity, initial_items=grid_data)


    def map(self, map_fn, **kwargs):
        """
        Map a function across the stack, using the bounds of first
        mapped item.
        """
        mapped_items = [(k, map_fn(el, k)) for k, el in self.items()]
        return self.clone(mapped_items, **kwargs)


    @property
    def empty_element(self):
        return self._type(None, self.bounds)


    @property
    def N(self):
        return self.normalize()


    @property
    def roi(self):
        return self.map(lambda x, _: x.roi)


    def hist(self, num_bins=20, bin_range=None, adjoin=True, individually=True, **kwargs):
        histstack = DataStack(dimensions=self.dimensions, title_suffix=self.title_suffix)

        stack_range = None if individually else self.range
        bin_range = stack_range if bin_range is None else bin_range
        for k, v in self.items():
            histstack[k] = v.hist(num_bins=num_bins, bin_range=bin_range,
                                  individually=individually,
                                  style_prefix='Custom[<' + self.name + '>]_',
                                  adjoin=False,
                                  **kwargs)

        if adjoin and issubclass(self.type, Overlay):
            layout = (self << histstack)
            layout.main_layer = kwargs['index']
            return layout

        return (self << histstack) if adjoin else histstack



    def _item_check(self, dim_vals, data):

        if isinstance(data, Annotation): pass
        elif self.bounds is None:
            self.bounds = data.bounds
        elif not data.bounds.lbrt() == self.bounds.lbrt():
            raise AssertionError("All SheetLayer elements must have matching bounds.")
        super(SheetStack, self)._item_check(dim_vals, data)


    def normalize_elements(self, **kwargs):
        return self.map(lambda x, _: x.normalize(**kwargs))


    def normalize(self, min=0.0, max=1.0):
        data_max = np.max([el.data.max() for el in self.values()])
        data_min = np.min([el.data.min() for el in self.values()])
        norm_factor = data_max-data_min
        return self.map(lambda x, _: x.normalize(min=min, max=max,
                                                 norm_factor=norm_factor))



class CoordinateGrid(Grid, SheetCoordinateSystem):
    """
    CoordinateGrid indexes other NdMapping objects, containing projections
    onto coordinate systems. The X and Y dimensions are mapped onto the bounds
    object, allowing for bounds checking and grid-snapping.
    """

    def __init__(self, bounds, shape, xdensity=None, ydensity=None, initial_items=None, **kwargs):
        (l, b, r, t) = bounds.lbrt()
        if not (xdensity and ydensity):
            (dim1, dim2) = shape
            xdensity = dim1 / (r-l) if (r-l) else 1
            ydensity = dim2 / (t-b) if (t-b) else 1

        SheetCoordinateSystem.__init__(self, bounds, xdensity, ydensity)
        super(CoordinateGrid, self).__init__(initial_items, **kwargs)


    def __getitem__(self, key):
        map_key, _ = self._split_index(key)
        transformed_key = self._transform_indices(map_key)

        ret = super(CoordinateGrid, self).__getitem__(transformed_key)
        if not isinstance(ret, CoordinateGrid):
            return ret

        # Adjust bounds to new slice
        x, y = [ret.dim_range(d) for d in ret.dimension_labels]
        l, b, r, t = self.lbrt
        half_unit_x = ((l-r) / float(ret.xdensity)) / 2.
        half_unit_y = ((t-b) / float(ret.ydensity)) / 2.

        new_bbox = BoundingBox(points=[(x[0]+half_unit_x, y[0]-half_unit_y),
                                       (x[1]-half_unit_x, y[1]+half_unit_y)])
        return self.clone(ret.items(), bounds=new_bbox)


    def _add_item(self, coords, data, sort=True):
        """
        Subclassed to provide bounds checking.
        """
        self._item_check(coords, data)
        coords = self._transform_indices(coords)
        super(CoordinateGrid, self)._add_item(coords, data, sort=sort)


    def _transform_indices(self, coords):
        return tuple([self._transform_index(i, coord)
                      for (i, coord) in enumerate(coords)])


    def _transform_index(self, dim, index):
        if isinstance(index, slice):
            return index
        else:
            return self._transform_value(index, dim)


    def _transform_value(self, val, dim):
        """
        Subclassed to discretize grid spacing.
        """
        if val is None: return None
        transformed = self.closest_cell_center(*((0, val) if dim else (val, 0)))[dim]
        return transformed


    def update(self, other):
        """
        Adds bounds checking to the default update behavior.
        """
        if hasattr(other, 'bounds') and (self.bounds.lbrt() != other.bounds.lbrt()):
            l1, b1, r1, t1 = self.bounds.lbrt()
            l2, b2, r2, t2 = other.bounds.lbrt()
            self.bounds = BoundingBox(points=((min(l1, l2), min(b1, b2)),
                                              (max(r1, r2), max(t1, t2))))
        super(CoordinateGrid, self).update(other)


    def clone(self, items=None, **kwargs):
        """
        Returns an empty duplicate of itself with all parameter values
        copied across.
        """
        settings = dict(self.get_param_values(), **kwargs)
        bounds = settings.pop('bounds') if 'bounds' in settings else self.bounds
        xdensity = settings.pop('xdensity') if 'xdensity' in settings else self.xdensity
        ydensity = settings.pop('ydensity') if 'ydensity' in settings else self.ydensity
        return self.__class__(bounds, None, initial_items=items, xdensity=xdensity,
                              ydensity=ydensity, **settings)



class DataGrid(CoordinateGrid):
    """
    DataGrid is mostly the same as CoordinateGrid, however it contains
    DataLayers or DataStacks as elements and can therefore not be overlaid
    with SheetLayer elements.
    """

    def __add__(self, obj):
        raise NotImplementedError


__all__ = list(set([_k for _k,_v in locals().items() if isinstance(_v,type) and
                    (issubclass(_v, NdMapping) or issubclass(_v, View))]))
