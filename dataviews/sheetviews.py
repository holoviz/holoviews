import numpy as np

import param

from boundingregion import BoundingBox, BoundingRegion
from dataviews import Stack, Histogram, DataStack, find_minmax
from ndmapping import NdMapping, Dimension
from options import options
from sheetcoords import SheetCoordinateSystem, Slice
from views import View, Overlay, Annotation, GridLayout


class SheetLayer(View):
    """
    A SheetLayer is a data structure for holding one or more numpy
    arrays embedded within a two-dimensional space. The array(s) may
    correspond to a discretisation of an image (i.e. a rasterisation)
    or vector elements such as points or lines. Lines may be linearly
    interpolated or correspond to control nodes of a smooth vector
    representation such as Bezier splines.
    """

    bounds = param.ClassSelector(class_=BoundingRegion, default=BoundingBox(), doc="""
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

        self_layers = self.data if isinstance(self, SheetOverlay) else [self]
        other_layers = other.data if isinstance(other, SheetOverlay) else [other]
        combined_layers = self_layers + other_layers

        if isinstance(other, Annotation):
            return SheetOverlay(combined_layers, self.bounds,
                                roi_bounds=self.roi_bounds,
                                metadata=self.metadata)

        if self.bounds is None:
            self.bounds = other.bounds
        elif other.bounds is None:
            other.bounds = self.bounds

        roi_bounds = self.roi_bounds if self.roi_bounds else other.roi_bounds
        roi_bounds = self.bounds if roi_bounds is None else roi_bounds
        return SheetOverlay(combined_layers, self.bounds,
                            metadata=self.metadata,
                            roi_bounds=roi_bounds)


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


    @property
    def roi(self):
        """
        Apply the roi_bounds to all elements in the SheetOverlay
        """
        return SheetOverlay([el.get_roi(self.roi_bounds) for el in self.data],
                            bounds=self.roi_bounds if self.roi_bounds else self.bounds,
                            metadata=self.metadata)


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

    def __init__(self, data, bounds=None, **kwargs):
        bounds = bounds if bounds else BoundingBox()
        data = np.array([[0]]) if data is None else data
        (l, b, r, t) = bounds.lbrt()
        (dim1, dim2) = data.shape[0], data.shape[1]
        xdensity = dim1/(r-l)
        ydensity = dim2/(t-b)

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
                         label=self.label,  style=self.style, metadata=self.metadata)


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


    def hist(self, num_bins=20, bin_range=None, individually=True, style_prefix=None):
        """
        Returns a Layout of the SheetView with an attached histogram.
        num_bins allows customizing the bin number. The container_name
        can additionally be specified to set a common cmap when viewing
        a Stack or Overlay.
        """
        range = find_minmax(self.range, (0, None)) if bin_range is None else bin_range

        # Avoids range issues including zero bin range and empty bins
        if range == (0, 0):
            range = (0.0, 0.1)
        try:
            hist, edges = np.histogram(self.data.flatten(), normed=True,
                                       range=range, bins=num_bins)
        except:
            edges = np.linspace(range[0], range[1], num_bins+1)
            hist = np.zeros(num_bins)
        hist[np.isnan(hist)] = 0

        hist_view = Histogram(hist, edges, cyclic_range=self.cyclic_range,
                              label=self.label + " Histogram",
                              metadata=self.metadata)

        # Set plot and style options
        style_prefix = 'Custom[<' + self.name + '>]_' if style_prefix is None else style_prefix
        opts_name = style_prefix + hist_view.label.replace(' ', '_')
        hist_view.style = opts_name
        options[opts_name] = options.plotting(self)(**dict(rescale_individually=individually))

        return hist_view


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



class Points(SheetLayer):
    """
    Allows sets of points to be positioned over a sheet coordinate
    system.

    The input data is an Nx2 Numpy array where each point in the numpy
    array corresponds to an X,Y coordinate in sheet coordinates,
    within the declared bounding region.
    """


    def __init__(self, data, bounds=None, **kwargs):
        bounds = bounds if bounds else BoundingBox()
        data = np.array([[], []]).T if data is None else data
        super(Points, self).__init__(data, bounds, **kwargs)


    def resize(self, bounds):
        return Points(self.points, bounds, style=self.style, metadata=self.metadata)


    def __len__(self):
        return self.data.shape[0]


    @property
    def roi(self):
        (N,_) = self.data.shape
        roi_data = self.data[[n for n in range(N)
                              if self.data[n, :] in self.roi_bounds]]
        roi_bounds = self.roi_bounds if self.roi_bounds else self.bounds
        return Points(roi_data, roi_bounds, style=self.style,
                           metadata=self.metadata)


    def __iter__(self):
        i = 0
        while i < len(self):
            yield tuple(self.data[i, :])
            i += 1



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
        return Contours(self.data, bounds, style=self.style,
                          metadata=self.metadata)



class SheetStack(Stack):
    """
    A SheetStack is a stack of SheetLayers over some dimensions. The
    dimension may be a spatial dimension (i.e., a ZStack), time
    (specifying a frame sequence) or any other dimensions along
    which SheetLayers may vary.
    """

    bounds = None

    data_type = (SheetLayer, Annotation)

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


    def _compute_samples(self, samples):
        """
        Transform samples as specified to a format suitable for _get_sample.

        May be overridden to compute transformation from sheetcoordinates to matrix
        coordinates in single pass as an optimization.
        """
        return [tuple(self.last.sheet2matrixidx(*s)) for s in samples]


    def _get_sample(self, view, sample):
        """
        Given a sample as processed by _compute_sample to extract a scalar sample
        value from the view. Uses __getitem__ by default but can operate on the view's
        data attribute if this helps optimize performance.
        """
        return view.data[sample]


    def _curve_labels(self, x_axis, sample, ylabel):
        """
        Subclasses _curve_labels in regular Stack to correctly label curves
        sampled from a SheetStack.
        """
        curve_label = " ".join(["Coord:", str(sample), x_axis.capitalize(), ylabel])
        return curve_label, x_axis.capitalize(), ylabel


    def grid_sample(self, rows, cols, lbrt=None, **kwargs):
        """
        Creates a CoordinateGrid of curves according sampled according to
        the supplied rows and cols. A sub-region to be sampled can be specified
        using the lbrt argument, which expresses the subsampling in sheet
        coordinates. The usual sampling semantics apply.
        """
        dim1, dim2 = self.last.shape
        if lbrt is None:
            l, t = self.last.matrixidx2sheet(0, 0)
            r, b = self.last.matrixidx2sheet(dim1-1, dim2-1)
        else:
            l, b, r, t = lbrt
        x, y = np.meshgrid(np.linspace(l, r, cols),
                           np.linspace(b, t, rows))
        coords = zip(x.flat, y.flat)
        shape = (rows, cols)
        bounds = BoundingBox(points=[(l, b), (r, t)])

        grid = self.sample(coords, **kwargs)

        return DataGrid(bounds, shape, initial_items=zip(coords, grid.values()))


    def map(self, map_fn, **kwargs):
        """
        Map a function across the stack, using the bounds of first
        mapped item.
        """
        mapped_items = [(k, map_fn(el, k)) for k, el in self.items()]
        return self.clone(mapped_items, bounds=mapped_items[0][1].bounds, **kwargs)


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


    def hist(self, num_bins=20, individually=False, bin_range=None):
        histstack = DataStack(dimensions=self.dimensions, title=self.title,
                              metadata=self.metadata)

        stack_range = None if individually else self.range
        bin_range = stack_range if bin_range is None else bin_range
        for k, v in self.items():
            histstack[k] = v.hist(num_bins=num_bins, bin_range=bin_range,
                                  individually=individually,
                                  style_prefix='Custom[<' + self.name + '>]_')

        return histstack


    @property
    def range(self):
        range = self.last.range
        for view in self._data.values():
            range = find_minmax(range, view.range)
        return range


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
        self._style = None

        SheetCoordinateSystem.__init__(self, bounds, xdensity, ydensity)
        super(CoordinateGrid, self).__init__(initial_items, **kwargs)


    def _add_item(self, coords, data, sort=True):
        """
        Subclassed to provide bounds checking.
        """
        if not self.bounds.contains(*coords):
            self.warning('Specified coordinate %s is outside grid bounds %s' % (coords, self.lbrt))
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
            other = other.last
        overlayed_items = [(k, el * other) for k, el in self.items()]
        return self.clone(overlayed_items)


    @property
    def last(self):
        """
        The last of a ProjectionGrid is another ProjectionGrid
        constituted of the last of the individual elements. To access
        the elements by their X,Y position, either index the position
        directly or use the items() method.
        """

        last_items = [(k, v.clone(items=(v.keys()[-1], v.last)))
                     for (k, v) in self.items()]
        return self.clone(last_items)


    def __len__(self):
        """
        The maximum depth of all the elements. Matches the semantics
        of __len__ used by SheetStack. For the total number of
        elements, count the full set of keys.
        """
        return max([len(v) for v in self.values()] + [0])


    def __add__(self, obj):
        if not isinstance(obj, GridLayout):
            return GridLayout(initial_items=[self, obj])


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


    @property
    def xlim(self):
        xlim = self.values()[-1].xlim
        for data in self.values():
            xlim = find_minmax(xlim, data.xlim)
        return xlim


    @property
    def ylim(self):
        ylim = self.values()[-1].ylim
        for data in self.values():
            ylim = find_minmax(ylim, data.ylim)
        if ylim[0] == ylim[1]: ylim = (ylim[0], ylim[0]+1.)
        return ylim


    @property
    def style(self):
        """
        The name of the style that may be used to control display of
        this view. If a style name is not set and but a label is
        assigned, then the closest existing style name is returned.
        """
        if self._style:
            return self._style

        class_name = self.__class__.__name__
        matches = options.fuzzy_match_keys(class_name)
        return matches[0] if matches else class_name


    @style.setter
    def style(self, val):
        self._style = val



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
