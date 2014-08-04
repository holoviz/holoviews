import numpy as np
from collections import OrderedDict, defaultdict

import param

from .ndmapping import Dimension, NdMapping
from .options import options
from .views import View, Overlay, Annotation, Stack, find_minmax



class DataLayer(View):
    """
    DataLayer is a 2D View type used to hold data indexed
    by an x-dimension and y-dimension. The data held within
    the DataLayer is a numpy array of shape (n, 2).

    DataLayer objects are sliceable along the X dimension
    allowing easy selection of subsets of the data.
    """

    dimensions = param.List(default=[Dimension('X')])

    legend_label = param.String(default="", doc="Legend labels")

    value = param.ClassSelector(class_=(str, Dimension), default='Y')

    def __init__(self, data, **kwargs):
        settings = {}
        if isinstance(data, DataLayer):
            settings = dict(data.get_param_values())
            data = data.data
        elif isinstance(data, Stack) or (isinstance(data, list) and data
                                         and isinstance(data[0], DataLayer)):
            data, settings = self._process_stack(data)

        data = list(data)
        if len(data) and not isinstance(data, np.ndarray):
            data = np.array(data)

        self._xlim = None
        self._ylim = None
        settings.update(kwargs)
        super(DataLayer, self).__init__(data, **settings)


    def _process_stack(self, stack):
        """
        Base class to process a DataStack to be collapsed into a DataLayer.
        Should return the data and parameters of reduced View.
        """
        data = []
        for v in stack:
            data.append(v.data)
        return np.concatenate(data), dict(v.get_param_values())


    def sample(self, **samples):
        """
        Allows sampling of DataLayer objects using the default
        syntax of providing a map of dimensions and sample pairs.
        """
        sample_data = {}
        for sample_dim, samples in samples.items():
            if not isinstance(samples, list): samples = [samples]
            for sample in samples:
                if sample_dim in self.dimension_labels:
                    sample_data[sample] = self[sample]
                else:
                    self.warning('Sample dimension %s invalid on %s'
                                 % (sample_dim, type(self).__name__))
        return Table(sample_data, **dict(self.get_param_values()))


    def reduce(self, label_prefix='', **reduce_map):
        """
        Allows collapsing of DataLayer objects using the supplied map of
        dimensions and reduce functions.
        """
        reduced_data = {}
        value = self.value(' '.join([label_prefix, self.value.name]))
        for dimension, reduce_fn in reduce_map.items():
            data = reduce_fn(self.data[:, 1])
            reduced_data[value] = data
        return Table(reduced_data, label=self.label, title=self.title)


    def __getitem__(self, slc):
        """
        Implements slicing or indexing of the data by the data x-value.
        If a single element is indexed reduces the DataLayer to a single
        Scatter object.
        """
        if slc is ():
            return self
        if isinstance(slc, slice):
            start, stop = slc.start, slc.stop
            xvals = self.data[:, 0]
            start_idx = np.abs((xvals - start)).argmin()
            stop_idx = np.abs((xvals - stop)).argmin()
            return self.__class__(self.data[start_idx:stop_idx, :],
                                  **dict(self.get_param_values()))
        else:
            slc = np.where(self.data[:, 0] == slc)
            sample = self.data[slc, :]
            return Scatter(sample, **dict(self.get_param_values()))


    def __mul__(self, other):
        if isinstance(other, DataStack):
            items = [(k, self * v) for (k, v) in other.items()]
            return other.clone(items=items)
        elif isinstance(self, DataOverlay):
            if isinstance(other, DataOverlay):
                overlays = self.data + other.data
            else:
                overlays = self.data + [other]
        elif isinstance(other, DataOverlay):
            overlays = [self] + other.data
        elif isinstance(other, DataLayer):
            overlays = [self, other]
        else:
            raise TypeError('Can only create an overlay of DataViews.')

        return DataOverlay(overlays)


    @property
    def stack_type(self):
        return DataStack


    @property
    def cyclic_range(self):
        if self.dimensions[0].cyclic:
            return self.dimensions[0].range[1]
        else:
            return None


    @property
    def xlabel(self):
        return self.dimensions[0].pprint_label


    @property
    def ylabel(self):
        return str(self.value)


    @property
    def xlim(self):
        if self._xlim:
            return self._xlim
        elif isinstance(self, Overlay):
            return None
        elif self.cyclic_range is not None:
            return (0, self.cyclic_range)
        else:
            x_vals = self.data[:, 0]
            return (float(min(x_vals)), float(max(x_vals)))


    @xlim.setter
    def xlim(self, limits):
        xmin, xmax = limits
        xlim = self.xlim
        if self.cyclic_range and not isinstance(self, Overlay):
            self.warning('Cannot override the limits of a cyclic dimension')
        elif xlim is None or (xmin <= xlim[0] and xmax >= xlim[1]):
            self._xlim = (xmin, xmax)
        elif not isinstance(self, Overlay):
            self.warning('Applied x-limits need to be inclusive '
                         'of all data.')


    @property
    def ylim(self):
        if self._ylim:
            return self._ylim
        elif isinstance(self, Overlay):
            return None
        y_vals = self.data[:, 1]
        return (float(min(y_vals)), float(max(y_vals)))


    @ylim.setter
    def ylim(self, limits):
        ymin, ymax = limits
        ylim = self.ylim
        if ylim is None or (ymin <= ylim[0] and ymax >= ylim[1]):
            self._ylim = (ymin, ymax)
        elif not isinstance(self, Overlay):
            self.warning('Applied y-limits need to be inclusive '
                         'of all data.')


    @property
    def lbrt(self):
        if self.xlim is None: return None, None, None, None
        l, r = self.xlim
        b, t = self.ylim
        return l, b, r, t



class Scatter(DataLayer):
    """
    Scatter is a simple 1D View, which gets displayed as a number of
    disconnected points.
    """


class Curve(DataLayer):
    """
    Curve is a simple 1D View of points and therefore assumes the data is
    ordered.
    """

    def __init__(self, data, **kwargs):
        super(Curve, self).__init__(data, **kwargs)


    def dframe(self):
        import pandas as pd
        df = pd.DataFrame(self.data, columns=[self.dimension_labels[0], self.value.name])
        return df


    def stack(self):
        stack = DataStack(None, dimensions=[self.xlabel], title=self.title+' {dims}')
        for idx in range(len(self.data)):
            x = self.data[0]
            if x in stack:
                stack[x].data.append(self.data[0:idx])
            else:
                stack[x] = Curve(self.data[0:idx])
        return stack


class Bars(DataLayer):
    """
    A bar is a simple 1D View of bars, which assumes that the data is sorted by
    x-value and there are no gaps in the bars.
    """

    def __init__(self, data, width=None, **kwargs):
        super(Bars, self).__init__(data, **kwargs)
        self._width = width

    @property
    def width(self):
        if self._width == None:
            return set(np.diff(self.data[:, 1]))[0]
        else:
            return self._width



class Histogram(DataLayer):
    """
    Histogram contains a number of bins, which are defined by the upper
    and lower bounds of their edges and the computed bin values.
    """

    title = param.String(default='{label} {type}')

    value = param.ClassSelector(class_=(str, Dimension), default='Frequency')

    def __init__(self, values, edges=None, **kwargs):
        self.values, self.edges, settings = self._process_data(values, edges)
        settings.update(kwargs)
        super(Histogram, self).__init__([], **settings)
        self.data = (self.values, self.edges)


    def _process_data(self, values, edges):
        """
        Ensure that edges are specified as left and right edges of the
        histogram bins rather than bin centers.
        """
        settings = {}
        if isinstance(values, DataLayer):
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


    def __getitem__(self, slc):
        raise NotImplementedError('Slicing and indexing of histograms currently not implemented.')


    def sample(self, **samples):
        raise NotImplementedError('Cannot sample a Histogram.')


    def reduce(self, **dimreduce_map):
        raise NotImplementedError('Reduction of Histogram not implemented.')


    @property
    def xlim(self):
        if self.cyclic_range is not None:
            return (0, self.cyclic_range)
        else:
            return (min(self.edges), max(self.edges))


    @property
    def ylim(self):
        return (min(self.values), max(self.values))



class DataOverlay(DataLayer, Overlay):
    """
    A DataOverlay can contain a number of DataLayer objects, which are to be
    overlayed on one axis. When adding new DataLayers to the DataOverlay
    it ensures the DataLayers have the same x- and y-label and recomputes the
    axis limits.
    """

    def __init__(self, overlays, **kwargs):
        Overlay.__init__(self, [], **kwargs)
        self._xlim = None
        self._ylim = None
        self.set(overlays)


    def __getitem__(self, ind):
        return Overlay.__getitem__(self, ind)


    def add(self, layer):
        if isinstance(layer, Annotation): pass
        elif not len(self):
            self.xlim = layer.xlim
            self.ylim = layer.ylim
            self.dimensions = layer.dimensions
            self.value = layer.value
            self.label = layer.label
        else:
            self.xlim = layer.xlim if self.xlim is None else find_minmax(self.xlim, layer.xlim)
            self.ylim = layer.ylim if self.xlim is None else find_minmax(self.ylim, layer.ylim)
            if layer.dimension_labels != self.dimension_labels:
                raise Exception("DataLayers must share common dimensions.")
        self.data.append(layer)


    @property
    def cyclic_range(self):
        return self[0].cyclic_range if len(self) else None



class Matrix(View):
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
        self.lbrt = lbrt
        super(Matrix, self).__init__(data, **kwargs)


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
        will be rescaled for each Matrix in a Stack.
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


    def sample(self, coords=[], **samples):
        """
        Sample the Matrixalong one or both of its dimensions,
        returning a reduced dimensionality type, which is either
        a Table, Curve or Scatter. If two dimension samples
        and a new_xaxis is provided the sample will be the value
        of the sampled unit indexed by the value in the new_xaxis
        tuple.
        """
        if len(samples) == self.ndims or len(coords):
            if not len(coords):
                coords = zip(*[c if isinstance(c, list) else [c] for didx, c in
                               sorted([(self.dim_index(k), v) for k, v in
                                       samples.items()])])
            table_data = OrderedDict()
            for c in coords:
                table_data[c] = self.data[self._coord2matrix(c)]
            return Table(table_data, dimensions=self.dimensions,
                         label=self.label,
                         value=self.value)
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



class HeatMap(Matrix, DataLayer):
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
    def lbrt(self):
        dim1_keys, dim2_keys = self.dense_keys()
        return min(dim1_keys), min(dim2_keys), max(dim1_keys), max(dim2_keys)



class DataStack(Stack):
    """
    A DataStack can hold any number of DataLayers indexed by a list of
    dimension values. It also has a number of properties, which can find
    the x- and y-dimension limits and labels.
    """

    data_type = (DataLayer, Annotation, Matrix)

    overlay_type = DataOverlay

    @property
    def range(self):
        if not hasattr(self.last, 'range'):
            raise Exception('View type %s does not implement range.' % type(self.last))
        range = self.last.range
        for view in self._data.values():
            range = find_minmax(range, view.range)
        return range


    @property
    def xlabel(self):
        return self.last.xlabel


    @property
    def ylabel(self):
        return self.last.ylabel


    @property
    def xlim(self):
        xlim = self.last.xlim
        for data in self.values():
            xlim = find_minmax(xlim, data.xlim)
        return xlim


    @property
    def ylim(self):
        ylim = self.last.ylim
        for data in self.values():
            ylim = find_minmax(ylim, data.ylim)
        return ylim


    @property
    def lbrt(self):
        l, r = self.xlim
        b, t = self.ylim
        return float(l), float(b), float(r), float(t)



class Table(View):
    """
    A tabular view type to allow convenient visualization of either a
    standard Python dictionary or an OrderedDict. If an OrderedDict is
    used, the headings will be kept in the correct order.
    """

    @property
    def stack_type(self):
        return TableStack

    def __init__(self, data, **kwargs):
        super(Table, self).__init__(data=data, **kwargs)

        # Assume OrderedDict if not a vanilla Python dict
        headings = self.data.keys()
        if type(self.data) == dict:
            headings = sorted(headings)
            self.data = OrderedDict([(h, self.data[h]) for h in headings])
        self.heading_map = OrderedDict([(el, str(el)) for el in headings])


    def sample(self, samples=None):
        if callable(samples):
            sampled_data = OrderedDict([item for item in self.data.items()
                                        if samples(item)])
        else:
            sampled_data = OrderedDict([(s, self.data[s]) for s in samples])
        return self.clone(sampled_data)


    def reduce(self, **reduce_map):
        reduced_data = {}
        for reduce_label, reduce_fn in reduce_map.items():
            data = reduce_fn(self.data.values())
            reduced_data[reduce_label] = data
        return self.clone(reduced_data)


    @property
    def rows(self):
        return len(self.heading_map)


    @property
    def cols(self):
        return 2


    def __getitem__(self, heading):
        """
        Get the value associated with the given heading (key).
        """
        if heading is ():
            return self
        if heading not in self.heading_map:
            raise IndexError("%r not in available headings." % heading)
        return self.data[heading]


    def cell_value(self, row, col):
        """
        Get the stored value for a given row and column indices.
        """
        if col > 1:
            raise Exception("Only two columns available in a Table.")
        elif row >= self.rows:
            raise Exception("Maximum row index is %d" % len(self.headings)-1)
        elif col == 0:
            return list(self.heading_map.values())[row]
        else:
            heading = list(self.heading_map.keys())[row]
            return self.data[heading]


    def heading_values(self):
        return list(self.heading_map.keys())


    def heading_names(self):
        return list(self.heading_map.values())


    def cell_type(self, row, col):
        """
        Returns the cell type given a row and column index. The common
        basic cell types are 'data' and 'heading'.
        """
        if col == 0:  return 'heading'
        else:         return 'data'

    def dframe(self):
        """
        Generates a Pandas dframe from the Table.
        """
        from pandas import DataFrame
        df_dict = defaultdict(list)
        for key, val in self.data.items():
            if self.dimensions:
                for key_val, dim in zip(key, self.dimension_labels):
                    df_dict[dim.replace(' ','_')].append(key_val)
                value_label = str(self.value).replace(' ','_')
                df_dict[value_label].append(val)
            else:
                df_dict[key.replace(' ','_')].append(val)
        return DataFrame(dict(df_dict))



class TableStack(Stack):
    """
    A TableStack may hold any number of TableViews indexed by a list
    of dimension values. It also allows the values of a particular
    cell to be sampled by name across any valid dimension.
    """
    _type = Table

    _type_map = None


    def sample(self, samples):
        """
        Samples the Table elements in the Stack by the provided samples.
        If multiple samples are provided the samples are laid out side
        by side in a GridLayout. By providing an x_dimension the individual
        samples are joined up into a Curve.
        """
        return self.clone([(k, view.sample(samples)) for k, view in self.items()])



    def reduce(self, **reduce_map):
        """
        Reduces the Tables in the Stack using the provided the function
        provided in the reduce_tuple (reduced_label, reduce_fn).

        If an x_dimension is provided the reduced values are joined up
        to a Curve. By default reduces all values in a Table but using
        a match_fn a subset of elements in the Tables can be selected.
        """
        return self.clone([(k, view.reduce(reduce_map)) for k, view in self.items()])


    def collate(self, collate_dim):
        """
        Collate splits out the specified dimension and joins the samples
        in each of the split out Stacks into Curves. If there are multiple
        entries in the Table it will lay them out into a Grid.
        """
        if self.ndims == 1:
            nested_stack = {1: self}
            new_dimensions = ['Temp']
        else:
            nested_stack = self.split_dimensions([collate_dim])
            new_dimensions = [d for d in self.dimensions if d.name != collate_dim]
        collate_dim = self.dim_dict[collate_dim]

        # Generate a DataStack for every entry in the table
        stack_fn = lambda: DataStack(**dict(self.get_param_values(), dimensions=new_dimensions))
        entry_dims = OrderedDict([(str(k), k) for k in self.last.data.keys()])
        stacks = OrderedDict([(entry, stack_fn()) for entry in entry_dims])
        for new_key, collate_stack in nested_stack.items():
            curve_data = OrderedDict([(k, []) for k in entry_dims.keys()])
            # Get the x- and y-values for each entry in the Table
            xvalues = [float(k) for k in collate_stack.keys()]
            for x, table in collate_stack.items():
                for label, value in table.data.items():
                    curve_data[str(label)].append(float(value))

            # Get data from table
            table = collate_stack.last
            table_dimensions = table.dimensions
            table_title = ' ' + table.title
            table_label = table.label

            # Generate curves with correct dimensions
            for label, yvalues in curve_data.items():
                settings = dict(dimensions=[collate_dim])
                label = entry_dims[label]
                if len(table_dimensions):
                    if not isinstance(label, tuple): label = (label,)
                    title = ', '.join([d.pprint_value(label[idx]) for idx, d in
                                      enumerate(table_dimensions)]) + table_title
                    settings.update(value=table.value, label=table_label, title=title)
                else:
                    settings.update(value=label, label=table_label,
                                    title='{label} - {value}')
                stacks[str(label)][new_key] = Curve(zip(xvalues, yvalues), **settings)

        # If there are multiple table entries, generate grid
        stack_data = list(stacks.values())
        if self.ndims == 1: stack_data = [stack.last for stack in stack_data]
        stack_grid = stack_data[0]
        for stack in stack_data[1:]:
            stack_grid += stack
        return stack_grid


    def heading_values(self):
        return self.last.heading_values() if len(self) else []


    def heading_names(self):
        return self.last.heading_names() if len(self) else []


    def _item_check(self, dim_vals, data):

        if self._type_map is None:
            self._type_map = dict((str(k), type(v)) for (k,v) in data.data.items())

        if set(self._type_map.keys()) != set([str(k) for k in data.data.keys()]):
            raise AssertionError("All TableViews in a TableStack must have"
                                 " a common set of headings.")

        for k, v in data.data.items():
            key = str(k) # Cast dimension to string
            if key not in self._type_map:
                self._type_map[key] = None
            elif type(v) != self._type_map[key]:
                self._type_map[key] = None

        super(TableStack, self)._item_check(dim_vals, data)



__all__ = list(set([_k for _k,_v in locals().items() if isinstance(_v, type) and
                    (issubclass(_v, Stack) or issubclass(_v, View))]))
