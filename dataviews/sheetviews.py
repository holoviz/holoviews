from itertools import groupby
from collections import OrderedDict
import numpy as np

import param

from boundingregion import BoundingBox, BoundingRegion
from dataviews import DataStack, DataOverlay, DataCurves
from ndmapping import NdMapping, Dimension
from sheetcoords import SheetCoordinateSystem, Slice
from views import View, Overlay, Stack, GridLayout


class SheetLayer(View):
    """
    A SheetLayer is a data structure for holding one or more numpy
    arrays embedded within a two-dimensional space. The array(s) may
    correspond to a discretisation of an image (i.e. a rasterisation)
    or vector elements such as points or lines. Lines may be linearly
    interpolated or correspond to control nodes of a smooth vector
    representation such as Bezier splines.
    """

    bounds = param.ClassSelector(class_=BoundingRegion, default=None, doc="""
       The bounding region in sheet coordinates containing the data.""")

    roi_bounds = param.ClassSelector(class_=BoundingRegion, default=None, doc="""
        The ROI can be specified to select only a sub-region of the bounds to
        be stored as data.""")

    _abstract = True

    def __init__(self, data, bounds, **kwargs):
        super(SheetLayer, self).__init__(data, bounds=bounds, **kwargs)


    def __mul__(self, other):
        if isinstance(other, SheetStack):
            items = [(k, self * v) for (k, v) in other.items()]
            return other.clone(items=items)
        elif isinstance(self, SheetOverlay):
            if isinstance(other, SheetOverlay):
                overlays = self.data + other.data
            else:
                overlays = self.data + [other]
        elif isinstance(other, SheetOverlay):
            overlays = [self] + other.data
        elif isinstance(other, SheetLayer):
            overlays = [self, other]
        else:
            raise TypeError('Can only create an overlay of SheetLayers.')

        if self.bounds is None:
            self.bounds = other.bounds
        elif other.bounds is None:
            other.bounds = self.bounds

        roi_bounds = self.roi_bounds if self.roi_bounds else other.roi_bounds
        roi_bounds = self.bounds if roi_bounds is None else roi_bounds
        return SheetOverlay(overlays, self.bounds,
                            style=self.style, metadata=self.metadata,
                            roi_bounds=roi_bounds)



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

    def add(self, layer):
        """
        Overlay a single layer on top of the existing overlay.
        """
        if layer.bounds.lbrt() != self.bounds.lbrt():
            if layer.bounds is None:
                layer.bounds = self.bounds
            else:
                raise Exception("Layer must have same bounds as SheetOverlay")
        self.data.append(layer)


    @property
    def rgb(self):
        """
        Convert an overlay of three or four SheetViews into a
        SheetView in RGB(A) mode.
        """
        if len(self) not in [3, 4]:
            raise Exception("Requires 3 or 4 layers to convert to RGB(A)")
        if not all(isinstance(el, SheetView) for el in self.data):
            raise Exception("All layers must be SheetViews to convert"
                            " to RGB(A) format")
        if not all(el.depth == 1 for el in self.data):
            raise Exception("All SheetViews must have a depth of one for"
                            " conversion to RGB(A) format")
        mode = 'rgb' if len(self) == 3 else 'rgba'
        return SheetView(np.dstack([el.data for el in self.data]), self.bounds,
                         roi_bounds=self.roi_bounds, mode=mode)


    @property
    def roi(self):
        """
        Apply the roi_bounds to all elements in the SheetOverlay
        """
        return SheetOverlay([el.get_roi(self.roi_bounds) for el in self.data],
                            bounds=self.roi_bounds if self.roi_bounds else self.bounds,
                            style=self.style, metadata=self.metadata)


    def __len__(self):
        return len(self.data)



class SheetView(SheetLayer, SheetCoordinateSystem):
    """
    SheetView is the atomic unit as which 2D data is stored, along with its
    bounds object. Allows slicing operations of the data in sheet coordinates or
    direct access to the data, via the .data attribute.

    Arrays with a shape of (X,Y) or (X,Y,Z) are valid. In the case of
    3D arrays, each depth layer is interpreted as a channel of the 2D
    representation.
    """

    cyclic_range = param.Number(default=None, bounds=(0, None), allow_None=True, doc="""
        For a cyclic quantity, the range over which the values repeat. For
        instance, the orientation of a mirror-symmetric pattern in a plane is
        pi-periodic, with orientation x the same as orientation x+pi (and
        x+2pi, etc.) A cyclic_range of None declares that the data are not
        cyclic. This parameter is metadata, declaring properties of the data
        that can be useful for automatic plotting and/or normalization, and is
        not used within this class itself.""")

    _deep_indexable = True

    def __init__(self, data, bounds, **kwargs):
        data = np.array([[0]]) if data is None else data
        (l, b, r, t) = bounds.lbrt()
        (dim1, dim2) = data.shape[0], data.shape[1]
        xdensity = dim1/(r-l)
        ydensity = dim2/(t-b)

        self._mode = kwargs.pop('mode', None)
        SheetLayer.__init__(self, data, bounds, **kwargs)
        SheetCoordinateSystem.__init__(self, bounds, xdensity, ydensity)


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
                         bounds, cyclic_range=self.cyclic_range,
                         style=self.style, metadata=self.metadata)


    def normalize(self, min=0.0, max=1.0, norm_factor=None):
        norm_factor = self.cyclic_range if norm_factor is None else norm_factor
        if norm_factor is None:
            norm_factor = self.data.max() - self.data.min()
        else:
            min, max = (0.0, 1.0)
        norm_data = (((self.data - self.data.min())/norm_factor) * abs((max-min))) + min
        return SheetView(norm_data, self.bounds, cyclic_range=self.cyclic_range,
                         metadata=self.metadata, roi_bounds=self.roi_bounds,
                         style=self.style)


    @property
    def depth(self):
        return 1 if len(self.data.shape) == 2 else self.data.shape[2]


    @property
    def mode(self):
        """
        Mode specifying the color space for visualizing the array
        data. The string returned corresponds to the matplotlib colour
        map name unless depth is 3 or 4 with modes 'rgb' or 'rgba'
        respectively.

        If not explicitly specified, the mode defaults to 'gray'
        unless the cyclic_range is set, in which case 'hsv' is
        returned.
        """
        if self._mode is not None:
            return self._mode
        return 'gray' if (self.cyclic_range is None) else 'hsv'


    @mode.setter
    def mode(self, val):
        self._mode = val


    @property
    def N(self):
        return self.normalize()


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
        return SheetView(data, roi_bounds, cyclic_range=self.cyclic_range,
                         style=self.style, metadata=self.metadata)



class SheetPoints(SheetLayer):
    """
    Allows sets of points to be positioned over a sheet coordinate
    system.

    The input data is an Nx2 Numpy array where each point in the numpy
    array corresponds to an X,Y coordinate in sheet coordinates,
    within the declared bounding region.
    """


    def __init__(self, data, bounds, **kwargs):
        data = np.array([[], []]).T if data is None else data
        super(SheetPoints, self).__init__(data, bounds, **kwargs)


    def resize(self, bounds):
        return SheetPoints(self.points, bounds, style=self.style, metadata=self.metadata)


    def __len__(self):
        return self.data.shape[0]


    @property
    def roi(self):
        (N,_) = self.data.shape
        roi_data = self.data[[n for n in range(N)
                              if self.data[n, :] in self.roi_bounds]]
        roi_bounds = self.roi_bounds if self.roi_bounds else self.bounds
        return SheetPoints(roi_data, roi_bounds, style=self.style,
                           metadata=self.metadata)


    def __iter__(self):
        i = 0
        while i < len(self):
            yield tuple(self.data[i, :])
            i += 1



class SheetLines(SheetLayer):
    """
    Allows sets of contour lines to be defined over a
    SheetCoordinateSystem.

    The input data is a list of Nx2 numpy arrays where each array
    corresponds to a contour in the group. Each point in the numpy
    array corresponds to an X,Y coordinate.
    """

    def __init__(self, data, bounds, **kwargs):
        data = [] if data is None else data
        super(SheetLines, self).__init__(data, bounds, **kwargs)


    def resize(self, bounds):
        return SheetLines(self.contours, bounds, style=self.style)


    def __len__(self):
        return self.data.shape[0]


    @property
    def roi(self):
        # Note: Data returned is not sliced to ROI because vertices
        # outside the bounds need to be snapped to the bounding box
        # edges.
        bounds = self.roi_bounds if self.roi_bounds else self.bounds
        return SheetLines(self.data, bounds, style=self.style,
                          metadata=self.metadata)



class SheetStack(Stack):
    """
    A SheetStack is a stack of SheetLayers over some dimensions. The
    dimension may be a spatial dimension (i.e., a ZStack), time
    (specifying a frame sequence) or any other dimensions along
    which SheetLayers may vary.
    """

    bounds = param.ClassSelector(class_=BoundingRegion, default=None, doc="""
       The bounding region in sheet coordinates containing the data""")

    cyclic_ranges = param.Dict(default={}, doc="""
        Determines the periodicity of dimensions, if they are cyclic.""")

    data_type = SheetLayer

    overlay_type = SheetOverlay

    def drop_dimension(self, dim, val):
        """
        Drop dimension from the NdMapping using the supplied
        dimension name and value.
        """
        slices = [slice(None) for i in range(self.ndims)]
        slices[self.dim_index(dim)] = val
        dim_labels = [d for d in self.dimension_labels if d != dim]
        return self[tuple(slices)].reindex(dim_labels)


    def sample(self, coords=[], x_axis=None, group_by=[]):
        """
        The sampling method provides an easy way of sampling contained SheetViews
        across various dimensions by providing coordinates. It returns DataStacks
        of DataCurves, sampled across the x_axis dimensions and grouped into
        DataOverlays depending on the provided group_by dimensions.
        """
        if self.type == SheetView:
            x_axis = getattr(self.metadata, 'x_axis', self.dimension_labels[-1])\
                if x_axis is None else x_axis
            x_dim = self.dim_dict[x_axis]
            x_ndim = self.dim_index(x_axis)

            # Get non-x_dim dimension labels
            dimensions = self._dimensions[:]
            del dimensions[x_ndim]
            dim_labels = [d.name for d in dimensions]

            # Get x_axis and non-x_axis dimension values
            keys = self.keys()
            dim_func = lambda x: [x[i] for i in range(self.ndims) if i != x_ndim]
            x_func = lambda x: x[x_ndim]
            dim_values = [tuple(dim) for dim, _ in groupby(keys, dim_func)]
            x_vals = [dim for dim, _ in groupby(keys, x_func)]

            # Sample all the coords
            data_stack = OrderedDict()
            self._check_key_type = False
            for k in dim_values:
                for x in x_vals:
                    if k not in data_stack:
                        data_stack[k] = OrderedDict()
                    key = list(k)
                    key.insert(x_ndim, x)
                    data_stack[k][x] = self[tuple(key)].data
            self._check_key_type = True

            cyclic_range = x_dim.range if x_dim.cyclic else None

            # Create stack and overlay dimension indices and titles
            stack_info = [(d, i) for i, d in enumerate(dimensions)
                          if d.name not in group_by]
            stack_dims, stack_inds = ([None], None)
            title = ""
            if len(stack_info):
                stack_dims, stack_inds = zip(*stack_info)
                title += ', '.join('{{label{0}}}={{value{0}}}'.format(i)
                                   for i in range(len(stack_dims)))
                stack_dims = list(stack_dims)
            overlay_inds = [dim_labels.index(od) for od in group_by]

            coord_stacks = []
            matrix_indices = [(coord, tuple(self.top.sheet2matrixidx(*coord)))
                              for coord in coords]
            for coord, idx in matrix_indices:
                curve_stack = DataStack(dimensions=stack_dims, title=title,
                                        metadata=self.metadata)
                for key, data in data_stack.items():
                    xy_values = [(x, d[idx]) for x, d in data.items()]
                    data = [np.vstack(zip(*xy_values)).T]
                    overlay_vals = [key[i] for i in overlay_inds]

                    label = ', '.join(self.dim_dict[dim].pprint_value(val)
                                      for dim, val in zip(group_by, overlay_vals))

                    curve = DataCurves(data, cyclic_range=cyclic_range,
                                       metadata=self.metadata, xlabel=x_axis.capitalize(),
                                       label=label)
                    # Overlay curves if stack keys overlap
                    stack_key = tuple([key[i] for i in stack_inds])\
                        if stack_inds is not None else (0,)
                    if stack_key not in curve_stack:
                        curve_stack[stack_key] = DataOverlay([curve])
                    else:
                        curve_stack[stack_key] *= curve
                coord_stacks.append((coord, curve_stack))
        elif self.type == self.overlay_type:
            raise NotImplementedError

        return coord_stacks


    def unit_sample(self, coord, **kwargs):
        """
        Returns a single DataStack for a particular coordinate, containing
        curves along the specified x_axis and grouped according to the groupby
        argument.
        """
        return self.sample([coord], **kwargs)[0][1]


    def grid_sample(self, rows, cols, lbrt=None, **kwargs):
        """
        Creates a CoordinateGrid of curves according sampled according to
        the supplied rows and cols. A sub-region to be sampled can be specified
        using the lbrt argument, which expresses the subsampling in sheet
        coordinates. The usual sampling semantics apply.
        """
        dim1, dim2 = self.top.shape
        if lbrt is None:
            l, t = self.top.matrixidx2sheet(0, 0)
            r, b = self.top.matrixidx2sheet(dim1-1, dim2-1)
        else:
            l, b, r, t = lbrt
        x, y = np.meshgrid(np.linspace(l, r, cols),
                           np.linspace(b, t, rows))
        coords = zip(x.flat, y.flat)
        shape = (rows, cols)
        bounds = BoundingBox(points=[(l, b), (r, t)])

        items = self.sample(coords, **kwargs)

        return DataGrid(bounds, shape, initial_items=items)


    def map(self, map_fn, **kwargs):
        """
        Map a function across the stack, using the bounds of first
        mapped item.
        """
        mapped_items = [(k, map_fn(el, k)) for k, el in self.items()]
        if isinstance(mapped_items[0][1], tuple):
            split = [[(k, v) for v in val] for (k, val) in mapped_items]
            item_groups = [list(el) for el in zip(*split)]
        else:
            item_groups = [mapped_items]
        clones = tuple(self.clone(els, bounds=els[0][1].bounds, **kwargs)
                       for (i, els) in enumerate(item_groups))
        return clones if len(clones) > 1 else clones[0]


    @property
    def empty_element(self):
        return self._type(None, self.bounds)


    @property
    def rgb(self):
        if self.type == self.overlay_type:
            return self.map(lambda x, _: x.rgb)
        else:
            raise Exception("Can only convert %s of overlays to RGB(A)" % self.__class__.__name__)


    @property
    def N(self):
        return self.normalize()


    @property
    def roi(self):
        return self.map(lambda x, _: x.roi)


    def _item_check(self, dim_vals, data):
        if self.bounds is None:
            self.bounds = data.bounds
        if not data.bounds.lbrt() == self.bounds.lbrt():
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



class CoordinateGrid(NdMapping, SheetCoordinateSystem):
    """
    CoordinateGrid indexes other NdMapping objects, containing projections
    onto coordinate systems. The X and Y dimensions are mapped onto the bounds
    object, allowing for bounds checking and grid-snapping.
    """

    dimensions = param.List(default=[Dimension(name="X"),
                                     Dimension(name="Y")])

    def __init__(self, bounds, shape, initial_items=None, **kwargs):
        (l, b, r, t) = bounds.lbrt()
        (dim1, dim2) = shape
        xdensity = dim1 / (r-l) if (r-l) else 1
        ydensity = dim2 / (t-b) if (t-b) else 1

        SheetCoordinateSystem.__init__(self, bounds, xdensity, ydensity)
        super(CoordinateGrid, self).__init__(initial_items, **kwargs)


    def _add_item(self, coords, data, sort=True):
        """
        Subclassed to provide bounds checking.
        """
        if not self.bounds.contains(*coords):
            self.warning('Specified coordinate is outside grid bounds,'
                         ' data could not be added')
        self._item_check(coords, data)
        coords = self._transform_indices(coords)
        super(CoordinateGrid, self)._add_item(coords, data, sort=sort)


    def _transform_indices(self, coords):
        return tuple([self._transform_index(i, coord)
                      for (i, coord) in enumerate(coords)])


    def _transform_index(self, dim, index):
        if isinstance(index, slice):
            [start, stop] = [self._transform_value(el, dim)
                             for el in (index.start, index.stop)]
            return slice(start, stop)
        else:
            return self._transform_value(index, dim)


    def _transform_value(self, val, dim):
        """
        Subclassed to discretize grid spacing.
        """
        if val is None: return None
        return self.closest_cell_center(*((0, val) if dim else (val, 0)))[dim]


    def update(self, other):
        """
        Adds bounds checking to the default update behavior.
        """
        if hasattr(other, 'bounds') and (self.bounds.lbrt() != other.bounds.lbrt()):
            raise Exception('Cannot combine %ss with different'
                            ' bounds.' % self.__class__)
        super(CoordinateGrid, self).update(other)


    def clone(self, items=None, **kwargs):
        """
        Returns an empty duplicate of itself with all parameter values and
        metadata copied across.
        """
        settings = dict(self.get_param_values(), **kwargs)
        settings.pop('metadata', None)
        return CoordinateGrid(bounds=self.bounds, shape=self.shape,
                              initial_items=items,
                              metadata=self.metadata, **settings)


    def __mul__(self, other):
        if isinstance(other, SheetStack) and len(other) == 1:
            other = other.top
        overlayed_items = [(k, el * other) for k, el in self.items()]
        return self.clone(overlayed_items)


    @property
    def top(self):
        """
        The top of a ProjectionGrid is another ProjectionGrid
        constituted of the top of the individual elements. To access
        the elements by their X,Y position, either index the position
        directly or use the items() method.
        """

        top_items = [(k, v.clone(items=(v.keys()[-1], v.top)))
                     for (k, v) in self.items()]
        return self.clone(top_items)


    def __len__(self):
        """
        The maximum depth of all the elements. Matches the semantics
        of __len__ used by SheetStack. For the total number of
        elements, count the full set of keys.
        """
        return max(len(v) for v in self.values())


    def __add__(self, obj):
        if not isinstance(obj, GridLayout):
            return GridLayout(initial_items=[[self, obj]])


    def map(self, map_fn, **kwargs):
        """
        Map a function across the stack, using the bounds of first
        mapped item.
        """
        mapped_items = [(k, map_fn(el, k)) for k, el in self.items()]
        if isinstance(mapped_items[0][1], tuple):
            split = [[(k, v) for v in val] for (k, val) in mapped_items]
            item_groups = [list(el) for el in zip(*split)]
        else:
            item_groups = [mapped_items]
        clones = tuple(self.clone(els, **kwargs)
                       for (i, els) in enumerate(item_groups))
        return clones if len(clones) > 1 else clones[0]


class DataGrid(CoordinateGrid):
    """
    DataGrid is mostly the same as CoordinateGrid, however it contains
    DataLayers or DataStacks as elements and can therefore not be overlaid
    with SheetLayer elements.
    """

    def __add__(self, obj):
        raise NotImplementedError

    def __mul__(self, other):
        raise NotImplementedError


__all__ = list(set([_k for _k,_v in locals().items() if isinstance(_v,type) and
                    (issubclass(_v, NdMapping) or issubclass(_v, View))]))