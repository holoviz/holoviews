import numpy as np
from collections import OrderedDict, defaultdict

import param

from .ndmapping import Dimension, NdMapping
from .options import options
from .views import View, Overlay, Annotation, HoloMap, find_minmax



class Layer(View):
    """
    Layer is the baseclass for all 2D View types, with an x- and
    y-dimension. Subclasses should define the data storage in the
    constructor, as well as methods and properties, which define how
    the data maps onto the x- and y- and value dimensions.
    """

    legend_label = param.String(default="", doc="Legend labels")


    def _process_stack(self, stack):
        """
        Base class to process a LayerMap to be collapsed into a Layer.
        Should return the data and parameters of reduced View.
        """
        data = []
        for v in stack:
            data.append(v.data)
        return np.concatenate(data), dict(v.get_param_values())


    def __mul__(self, other):
        if isinstance(other, LayerMap):
            items = [(k, self * v) for (k, v) in other.items()]
            return other.clone(items=items)

        self_layers = self.data if isinstance(self, Overlay) else [self]
        other_layers = other.data if isinstance(other, Overlay) else [other]
        combined_layers = self_layers + other_layers

        return Overlay(combined_layers)


    def __mul__(self, other):
        if isinstance(other, LayerMap):
            items = [(k, self * v) for (k, v) in other.items()]
            return other.clone(items=items)
        elif isinstance(other, Overlay):
            overlays = [self] + other.data
        elif isinstance(other, (Layer, Annotation)):
            overlays = [self, other]
        else:
            raise TypeError('Can only create an overlay of DataViews.')

        return Overlay(overlays)


    ########################
    # Subclassable methods #
    ########################


    def __init__(self, data, **kwargs):
        self._xlim = None
        self._ylim = None
        super(Layer, self).__init__(data, **kwargs)


    @property
    def cyclic_range(self):
        if self.dimensions[0].cyclic:
            return self.dimensions[0].range[1]
        else:
            return None

    @property
    def range(self):
        if self.cyclic_range:
            return self.cyclic_range
        y_vals = self.data[:, 1]
        return (float(min(y_vals)), float(max(y_vals)))


    @property
    def xlabel(self):
        return self.dimensions[0].pprint_label


    @property
    def ylabel(self):
        if len(self.dimensions) == 1:
            return self.value.pprint_label
        else:
            return self.dimensions[1].pprint_label

    @property
    def xlim(self):
        if self._xlim:
            return self._xlim
        elif self.cyclic_range is not None:
            return (0, self.cyclic_range)
        else:
            x_vals = self.data[:, 0]
            return (float(min(x_vals)), float(max(x_vals)))

    @xlim.setter
    def xlim(self, limits):
        if self.cyclic_range:
            self.warning('Cannot override the limits of a '
                         'cyclic dimension.')
        elif limits is None or (isinstance(limits, tuple) and len(limits) == 2):
            self._xlim = limits
        else:
            raise ValueError('xlim needs to be a length two tuple or None.')


    @property
    def ylim(self):
        if self._ylim:
            return self._ylim
        else:
            y_vals = self.data[:, 1]
            return (float(min(y_vals)), float(max(y_vals)))


    @ylim.setter
    def ylim(self, limits):
        if limits is None or (isinstance(limits, tuple) and len(limits) == 2):
            self._ylim = limits
        else:
            raise ValueError('xlim needs to be a length two tuple or None.')


    @property
    def lbrt(self):
        l, r = self.xlim if self.xlim else (None, None)
        b, t = self.ylim if self.ylim else (None, None)
        return l, b, r, t


    @lbrt.setter
    def lbrt(self, lbrt):
        l, b, r, t = lbrt
        self.xlim, self.ylim = (l, r), (b, t)



class DataView(Layer):
    """
    The data held within an Array is a numpy array of shape (n, 2).
    Layer objects are sliceable along the X dimension allowing easy
    selection of subsets of the data.
    """

    dimensions = param.List(default=[Dimension('X')], doc="""
        Dimensions on Layers determine the number of indexable
        dimensions.""")

    value = param.ClassSelector(class_=Dimension, default=Dimension('Y'))

    def __init__(self, data, **kwargs):
        settings = {}
        if isinstance(data, DataView):
            settings = dict(data.get_param_values())
            data = data.data
        elif isinstance(data, HoloMap) or (isinstance(data, list) and data
                                           and isinstance(data[0], Layer)):
            data, settings = self._process_stack(data)

        if len(data) and not isinstance(data, np.ndarray):
            data = np.array(data)
        settings.update(kwargs)
        super(DataView, self).__init__(data, **settings)


    def __getitem__(self, slc):
        """
        Implements slicing or indexing of the data by the data x-value.
        If a single element is indexed reduces the Layer to a single
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


    def sample(self, samples=[]):
        """
        Allows sampling of Layer objects using the default
        syntax of providing a map of dimensions and sample pairs.
        """
        sample_data = OrderedDict()
        for sample in samples:
            sample_data[sample] = self[sample]
        return Table(sample_data, **dict(self.get_param_values()))


    def reduce(self, label_prefix='', **reduce_map):
        """
        Allows collapsing of Layer objects using the supplied map of
        dimensions and reduce functions.
        """
        reduced_data = OrderedDict()
        value = self.value(' '.join([label_prefix, self.value.name]))
        for dimension, reduce_fn in reduce_map.items():
            reduced_data[value] = reduce_fn(self.data[:, 1])
        return Items(reduced_data, label=self.label, title=self.title,
                     value=self.value(value))


    def dframe(self):
        import pandas as pd
        columns = [self.dimension_labels[0], self.value.name]
        return pd.DataFrame(self.data, columns=columns)



class Scatter(DataView):
    """
    Scatter is a simple 1D View, which gets displayed as a number of
    disconnected points.
    """
    
    pass



class Curve(DataView):
    """
    Curve is a simple 1D View of points and therefore assumes the data is
    ordered.
    """

    def progressive(self):
        """
        Create map indexed by Curve x-axis with progressively expanding number
        of curve samples.
        """
        stack = LayerMap(None, dimensions=[self.xlabel], title=self.title+' {dims}')
        for idx in range(len(self.data)):
            x = self.data[0]
            if x in stack:
                stack[x].data.append(self.data[0:idx])
            else:
                stack[x] = Curve(self.data[0:idx])
        return stack



class Bars(DataView):
    """
    A bar is a simple 1D View of bars, which assumes that the data is
    sorted by x-value and there are no gaps in the bars.
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

    @width.setter
    def width(self, width):
        if np.isscalar(width) or len(width) == len(self):
            self._width = width
        else:
            raise ValueError('width should be either a scalar or '
                             'match the number of bars in length.')


class Histogram(Layer):
    """
    Histogram contains a number of bins, which are defined by the
    upper and lower bounds of their edges and the computed bin values.
    """

    dimensions = param.List(default=[Dimension('X')], doc="""
        Dimensions on Layers determine the number of indexable
        dimensions.""")

    title = param.String(default='{label} {type}')

    value = param.ClassSelector(class_=Dimension, default=Dimension('Frequency'))

    def __init__(self, values, edges=None, **kwargs):
        self.values, self.edges, settings = self._process_data(values, edges)
        settings.update(kwargs)
        super(Histogram, self).__init__((self.values, self.edges), **settings)


    def _process_data(self, values, edges):
        """
        Ensure that edges are specified as left and right edges of the
        histogram bins rather than bin centers.
        """
        settings = {}
        if isinstance(values, Layer):
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
        a Items, Curve or Scatter. If two dimension samples
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



class LayerMap(HoloMap):
    """
    A LayerMap can hold any number of DataLayers indexed by a list of
    dimension values. It also has a number of properties, which can find
    the x- and y-dimension limits and labels.
    """

    data_type = (Layer, Overlay, Annotation)

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
            xlim = find_minmax(xlim, data.xlim) if data.xlim and xlim else xlim
        return xlim


    @property
    def ylim(self):
        ylim = self.last.ylim
        for data in self.values():
            ylim = find_minmax(ylim, data.ylim) if data.ylim and ylim else ulim
        return ylim


    @property
    def lbrt(self):
        l, r = self.xlim
        b, t = self.ylim
        return float(l), float(b), float(r), float(t)


    def sample(self, samples=[], **sample_values):
        """
        Sample each Layer in the HoloMap by passing either a list
        of samples or request a single sample using dimension-value
        pairs.
        """
        sampled_items = [(k, view.sample(samples, **sample_values))
                         for k, view in self.items()]
        return self.clone(sampled_items)


    def reduce(self, label_prefix='', **reduce_map):
        """
        Reduce each SheetView in the HoloMap using a function supplied
        via the kwargs, where the keyword has to match a particular
        dimension in the View.
        """
        reduced_items = [(k, v.reduce(label_prefix=label_prefix, **reduce_map))
                         for k, v in self.items()]
        return self.clone(reduced_items)


    def collate(self, collate_dim):
        """
        Collate splits out the specified dimension and joins the samples
        in each of the split out Stacks into Curves. If there are multiple
        entries in the Items it will lay them out into a Grid.
        """
        from .operation import TableCollate
        return TableCollate(self, collation_dim=collate_dim)


    def map(self, map_fn, **kwargs):
        """
        Map a function across the stack, using the bounds of first
        mapped item.
        """
        mapped_items = [(k, map_fn(el, k)) for k, el in self.items()]
        return self.clone(mapped_items, **kwargs)


    @property
    def empty_element(self):
        return self._type(None)


    @property
    def N(self):
        return self.normalize()


    def hist(self, num_bins=20, bin_range=None, adjoin=True, individually=True, **kwargs):
        histstack = LayerMap(dimensions=self.dimensions, title_suffix=self.title_suffix)

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


    def normalize_elements(self, **kwargs):
        return self.map(lambda x, _: x.normalize(**kwargs))


    def normalize(self, min=0.0, max=1.0):
        data_max = np.max([el.data.max() for el in self.values()])
        data_min = np.min([el.data.min() for el in self.values()])
        norm_factor = data_max-data_min
        return self.map(lambda x, _: x.normalize(min=min, max=max,
                                                 norm_factor=norm_factor))



class Items(Layer):
    """
    A tabular view type to allow convenient visualization of either a
    standard Python dictionary or an OrderedDict. If an OrderedDict is
    used, the headings will be kept in the correct order. Tables store
    heterogeneous data with different labels. Optionally a list of
    dimensions corresponding to the labels can be supplied.
    """

    xlabel, ylabel = None, None
    xlim, ylim = None, None
    lbrt = None, None, None, None

    @property
    def rows(self):
        return self.ndims


    @property
    def cols(self):
        return 2


    def __init__(self, data, **kwargs):
        # Assume OrderedDict if not a vanilla Python dict
        headings = data.keys()
        if type(data) == dict:
            headings = sorted(headings)
            data = OrderedDict([(h, data[h]) for h in headings])
        if 'dimensions' not in kwargs:
            kwargs['dimensions'] = headings
        super(Items, self).__init__(data=data, **kwargs)


    def __getitem__(self, heading):
        """
        Get the value associated with the given heading (key).
        """
        if heading is ():
            return self
        if heading not in self.dim_dict:
            raise IndexError("%r not in available headings." % heading)
        return self.data[heading]


    def sample(self, samples=None):
        if callable(samples):
            sampled_data = OrderedDict([item for item in self.data.items()
                                        if samples(item)])
        else:
            sampled_data = OrderedDict([(s, self.data[s]) for s in samples])
        return self.clone(sampled_data)


    def reduce(self, **reduce_map):
        raise NotImplementedError('Tables are for heterogeneous data, which'
                                  'cannot be reduced.')


    def cell_value(self, row, col):
        """
        Get the stored value for a given row and column indices.
        """
        if col > 2:
            raise Exception("Only two columns available in a Items.")
        elif row >= self.rows:
            raise Exception("Maximum row index is %d" % self.rows-1)
        elif col == 0:
            return list(self.dim_dict.values())[row]
        else:
            heading = list(self.dim_dict.keys())[row]
            return self.data[heading]


    def cell_type(self, row, col):
        """
        Returns the cell type given a row and column index. The common
        basic cell types are 'data' and 'heading'.
        """
        if col == 0:  return 'heading'
        else:         return 'data'


    def dframe(self):
        """
        Generates a Pandas dframe from the Items.
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
                key = key.name if isinstance(key, Dimension) else key
                df_dict[key.replace(' ','_')].append(val)
        return DataFrame(dict(df_dict))



class Table(Items, NdMapping):

    value = param.ClassSelector(class_=Dimension,
                                default=Dimension('Value'), doc="""
        The dimension description of the data held in the data array.""")

    def __init__(self, data, **params):
        super(Table, self).__init__(data, **params)
        self._data = self.data

    @property
    def rows(self):
        return len(self) + 1

    @property
    def cols(self):
        return self.ndims + 1


    def cell_value(self, row, col):
        """
        Get the stored value for a given row and column indices.
        """
        if col >= self.cols:
            raise Exception("Maximum column index is %d" % self.cols-1)
        elif row >= self.rows:
            raise Exception("Maximum row index is %d" % self.rows-1)
        elif row == 0:
            if col == self.ndims:
                return str(self.value)
            return str(self.dimensions[col])
        else:
            if col == self.ndims:
                return self.values()[row-1]
            return self.keys()[row-1][col]
            heading = list(self.dim_dict.keys())[row]
            return self.data[heading]


    def cell_type(self, row, col):
        """
        Returns the cell type given a row and column index. The common
        basic cell types are 'data' and 'heading'.
        """
        if col == self.ndims and row > 0:  return 'heading'
        else:         return 'data'


    def sample(self, samples=[]):
        """
        Allows sampling of the Table with a list of samples.
        """
        sample_data = OrderedDict()
        for sample in samples:
            sample_data[sample] = self[sample]
        return Table(sample_data, **dict(self.get_param_values()))


    def reduce(self, label_prefix='', **reduce_map):
        """
        Allows collapsing the Table down to an Items View
        with a single entry.
        """
        reduced_data = OrderedDict()
        value = self.value(' '.join([label_prefix, self.value.name]))
        for dimension, reduce_fn in reduce_map.items():
            reduced_data[value] = reduce_fn(self.values())
        return Items(reduced_data, title=self.title, value=self.value(value))


    def _item_check(self, dim_vals, data):
        if not np.isscalar(data):
            raise TypeError('Table only accepts scalar values.')
        super(Table, self)._item_check(dim_vals, data)


    def dframe(self):
        return NdMapping.dframe(self)


__all__ = list(set([_k for _k,_v in locals().items() if isinstance(_v, type) and
                    (issubclass(_v, HoloMap) or issubclass(_v, View))]))
