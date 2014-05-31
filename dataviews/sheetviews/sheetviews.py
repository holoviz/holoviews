import numpy as np

import param

from .boundingregion import BoundingBox, BoundingRegion
from .sheetcoords import SheetCoordinateSystem, Slice

from ..dataviews import Table, Curve, Scatter, Histogram, DataStack, find_minmax
from ..ndmapping import NdMapping, Dimension
from ..options import options, channels
from ..views import View, Overlay, Annotation, GridLayout


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

    _abstract = True


    @property
    def stack_type(self):
        return SheetStack


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


    def dimension_values(self, dim_index):
        """
        Return the coordinates of the dimension corresponding to the
        supplied index.
        """
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



class SheetView(SheetLayer, SheetCoordinateSystem):
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
                         bounds, label=self.label,  style=self.style,
                         metadata=self.metadata)


    def normalize(self, min=0.0, max=1.0, norm_factor=None, div_by_zero='ignore'):
        norm_factor = self.cyclic_range if norm_factor is None else norm_factor
        if norm_factor is None:
            norm_factor = self.data.max() - self.data.min()
        else:
            min, max = (0.0, 1.0)

        if div_by_zero in ['ignore', 'warn']:
            if (norm_factor==0.0) and div_by_zero=='warn':
                self.warning("Ignoring divide by zero in normalization.")
            norm_factor = 1.0 if (norm_factor==0.0) else norm_factor

        norm_data = (((self.data - self.data.min())/norm_factor) * abs((max-min))) + min
        return SheetView(norm_data, self.bounds, metadata=self.metadata,
                         roi_bounds=self.roi_bounds, style=self.style, label=self.label,
                         value=self.value)


    def hist(self, num_bins=20, bin_range=None, adjoin=True, individually=True, **kwargs):
        """
        Returns a Histogram of the SheetView data, binned into
        num_bins over the bin_range (if specified).

        If adjoin is True, the histogram will be returned adjoined to
        the SheetView as a side-plot.

        The 'individually' argument specifies whether the histogram
        will be rescaled for each for SheetViews in a SheetStack
        """
        range = find_minmax(self.range, (0, -float('inf'))) if bin_range is None else bin_range

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

        hist_view = Histogram(hist, edges, dimensions=[self.value], label=self.label,
                              value='Frequency', metadata=self.metadata)

        # Set plot and style options
        style_prefix = kwargs.get('style_prefix', 'Custom[<' + self.name + '>]_')
        opts_name = style_prefix + hist_view.label.replace(' ', '_')
        hist_view.style = opts_name
        options[opts_name] = options.plotting(self)(**dict(rescale_individually=individually))
        return (self << hist_view) if adjoin else hist_view


    def sample(self, dimension_samples, new_xaxis=None):
        """
        Sample the SheetView along one or both of its dimensions,
        returning a reduced dimensionality type, which is either
        a Table, Curve or Scatter. If two dimension samples
        and a new_xaxis is provided the sample will be the value
        of the sampled unit indexed by the value in the new_xaxis
        tuple.
        """
        if len(dimension_samples) == self.ndims:
            coords = [c for didx, c in sorted([(self.dim_index(k), v)
                                               for k, v in dimension_samples.items()])]
            data = self.data[self.sheet2matrixidx(*coords)]
            if new_xaxis:
                dim, val = new_xaxis
                title = "Coord: %s " % str(tuple(coords)) + self.title
                return Scatter(zip([val], [data]), dimensions=[dim],
                                     label=self.label, title=title, value=self.value)
            else:
                return Table({tuple(coords): data}, label=self.label, title=self.title)
        elif not len(dimension_samples):
            return self
        else:
            dimension, sample_coord = dimension_samples.items()[0]
            if isinstance(sample_coord, slice):
                raise ValueError('SheetView sampling requires coordinates not slices,'
                                 'use regular slicing syntax.')
            other_dimension = [d for d in self._dimensions if d.name != dimension]
            # Indices inverted for indexing
            other_ind = self.dim_index(dimension)
            sample_ind = self.dim_index(other_dimension[0].name)

            # Generate sample slice
            sample = [slice(None) for i in range(self.ndims)]
            coord_fn = (lambda v: (v, 0)) if sample_ind else (lambda v: (0, v))
            sample[sample_ind] = self.sheet2matrixidx(*coord_fn(sample_coord))[sample_ind]

            # Sample data
            x_vals = self.dimension_values(other_ind)
            data = zip(x_vals, self.data[sample])
            return Curve(data, dimensions=other_dimension, label=self.label,
                         title=self.title, value=self.value)


    def reduce(self, dimreduce_map):
        """
        Reduces a SheetView to a lower dimensional View type using the provided
        dimreduce_map, which maps reduce function to a dimension.
        """
        if len(dimreduce_map) == self.ndims:
            reduced_view = self
            for dim, reduce_fn in dimreduce_map:
                reduced_view = reduced_view.collapse({dim: reduce_fn})
            return reduced_view
        elif not len(dimreduce_map):
            return self
        else:
            dimension, reduce_fn = dimreduce_map.items()[0]
            other_dimension = [d for d in self._dimensions if d.name != dimension]
            other_ind = self.dim_index(other_dimension[0].name)
            x_vals = self.dimension_values(other_ind)
            data = zip(x_vals, reduce_fn(self.data, axis=self.dim_index(dimension)))
            return Curve(data, dimensions=other_dimension, label=self.label,
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
        return SheetView(data, roi_bounds, style=self.style,
                         metadata=self.metadata, value=self.value)



class Points(SheetLayer):
    """
    Allows sets of points to be positioned over a sheet coordinate
    system.

    The input data is an Nx2 Numpy array where each point in the numpy
    array corresponds to an X,Y coordinate in sheet coordinates,
    within the declared bounding region. Otherwise, the input can be a
    list that can be cast to a suitable numpy array.
    """

    def __init__(self, data, bounds=None, **kwargs):
        bounds = bounds if bounds else BoundingBox()
        data = np.array([[], []]).T if data is None else np.array(data)
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
        return Points(roi_data, roi_bounds, style=self.style, metadata=self.metadata)


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
        return Contours(self.data, bounds, style=self.style, label=self.label,
                        metadata=self.metadata)



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


    def sample(self, view_samples, x_dimension=None):
        if x_dimension:
            stack = self.split_dimensions([x_dimension])
            new_dimensions = [d for d in self._dimensions if d.name != x_dimension]
        else:
            stack = self
            new_dimensions = self._dimensions
        new_stack = DataStack(dimensions=new_dimensions)
        for k, v in stack.items():
            if x_dimension is None:
                key = k if isinstance(k, tuple) else (k,)
                new_stack[k] = v.sample(view_samples, dict(zip(stack.dimensions, key)))
            else:
                sampled = v.sample(view_samples)
                new_stack[k] = Curve(sampled, dimensions=sampled.last._dimensions,
                                     label=v.last.label, value=self.last.value)
        return new_stack


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
        coords = list(zip(x.flat, y.flat))
        shape = (rows, cols)
        bounds = BoundingBox(points=[(l, b), (r, t)])

        grid = self.sample(coords, **kwargs)

        return DataGrid(bounds, shape, initial_items=list(zip(coords, grid.values())))


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
    def N(self):
        return self.normalize()


    @property
    def roi(self):
        return self.map(lambda x, _: x.roi)


    def hist(self, num_bins=20, bin_range=None, adjoin=True, individually=True, **kwargs):
        histstack = DataStack(dimensions=self.dimensions,
                              title_suffix=self.title_suffix,
                              metadata=self.metadata)

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

    dimensions = param.List(default=[Dimension(name="X"), Dimension(name="Y")])

    label = param.String(constant=True, doc="""
      A short label used to indicate what kind of data is contained
      within the CoordinateGrid.""")

    title = param.String(default='{label}', doc="""
       The title formatting string allows the title to be composed
       from the label and type.""")

    def __init__(self, bounds, shape, xdensity=None, ydensity=None, initial_items=None, **kwargs):
        (l, b, r, t) = bounds.lbrt()
        if not (xdensity and ydensity):
            (dim1, dim2) = shape
            xdensity = dim1 / (r-l) if (r-l) else 1
            ydensity = dim2 / (t-b) if (t-b) else 1
        self._style = None

        SheetCoordinateSystem.__init__(self, bounds, xdensity, ydensity)
        super(CoordinateGrid, self).__init__(initial_items, **kwargs)


    def __getitem__(self, key):
        ret = super(CoordinateGrid, self).__getitem__(key)
        if not isinstance(ret, CoordinateGrid):
            return ret

        # Adjust bounds to new slice
        map_key, _ = self._split_index(key)
        x, y = [ret.dim_range(d) for d in ret.dimension_labels]
        l, b, r, t = ret.lbrt
        half_unit_x = ((l-r) / ret.xdensity) / 2
        half_unit_y = ((t-b) / ret.ydensity) / 2

        new_bbox = BoundingBox(points=[(x[0]+half_unit_x, y[0]-half_unit_y),
                                       (x[1]-half_unit_x, y[1]+half_unit_y)])
        return self.clone(ret.items(), bounds=new_bbox)


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
        settings = dict(self.get_param_values(), metadata=self.metadata, **kwargs)
        bounds = settings.pop('bounds') if 'bounds' in settings else self.bounds
        xdensity = settings.pop('xdensity') if 'xdensity' in settings else self.xdensity
        ydensity = settings.pop('ydensity') if 'ydensity' in settings else self.ydensity
        return CoordinateGrid(bounds, None, initial_items=items, xdensity=xdensity,
                              ydensity=ydensity, **settings)


    def __mul__(self, other):

        if isinstance(other, CoordinateGrid):
            if set(self.keys()) != set(other.keys()):
                raise KeyError("Can only overlay two CoordinateGrids if their keys match")
            zipped = zip(self.keys(), self.values(), other.values())
            overlayed_items = [(k, el1 * el2) for (k, el1, el2) in zipped]
            return self.clone(overlayed_items)
        elif isinstance(other, SheetStack) and len(other) == 1:
            sheetview = other.last
        elif isinstance(other, SheetStack) and len(other) != 1:
            raise Exception("Can only overlay with SheetStack of length 1")
        else:
            sheetview = other

        overlayed_items = [(k, el * sheetview) for k, el in self.items()]
        return self.clone(overlayed_items)


    @property
    def last(self):
        """
        The last of a ProjectionGrid is another ProjectionGrid
        constituted of the last of the individual elements. To access
        the elements by their X,Y position, either index the position
        directly or use the items() method.
        """

        last_items = [(k, v.clone(items=(list(v.keys())[-1], v.last)))
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


    @property
    def xlim(self):
        xlim = list(self.values())[-1].xlim
        for data in self.values():
            xlim = find_minmax(xlim, data.xlim)
        return xlim


    @property
    def ylim(self):
        ylim = list(self.values())[-1].ylim
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
