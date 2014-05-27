import numpy as np
from collections import OrderedDict

import param

from .ndmapping import Dimension
from .views import View, Overlay, Annotation, Stack

def find_minmax(lims, olims):
    """
    Takes (a1, a2) and (b1, b2) as input and returns
    (np.min(a1, b1), np.max(a2, b2)).
    """

    limzip = zip(list(lims), list(olims), [np.min, np.max])
    return tuple([float(fn([l, ol])) for l, ol, fn in limzip])



class DataLayer(View):
    """
    General purpose DataLayer for holding data to be plotted along some
    axes. Subclasses can implement specialized containers for data such as
    curves, points, bars or surfaces.
    """

    dimensions = param.List(default=[Dimension('X')])

    def __init__(self, data, **kwargs):
        settings = {}
        if isinstance(data, DataLayer):
            settings = dict(data.get_param_values())
            data = data.data
        elif isinstance(data, Stack) or (isinstance(data, list) and data
                                         and isinstance(data[0], DataLayer)):
            data, settings = self._process_stack(data)
        elif data and not isinstance(data, np.ndarray):
            data = np.array(list(data))

        self._xlim = None
        self._ylim = None
        settings.update(kwargs)
        super(DataLayer, self).__init__(data, **settings)


    def _process_stack(self, stack):
        """
        Base class to process a DataStack to be collapsed into a DataLayer.
        Should return the data and parameters of reduced View.
        """
        if isinstance(stack, DataStack):
            list = stack.values()
        data = []
        for v in list:
            data.append(v.data)
        return np.concatenate(data), dict(v.get_param_values())


    @property
    def cyclic_range(self):
        if self._dimensions[0].cyclic:
            return self._dimensions[0].range[1]
        else:
            return None


    def sample(self, dimension_samples):
        """
        Allows sampling of DataLayer objects using the default
        syntax of providing a map of dimensions and sample pairs.
        """
        dims, samples = zip(dimension_samples.items())
        if len(dims) > self.ndims:
            raise ValueError('%d sample dimensions provided, %s is %d-dimensional.'
                             % (len(dims), type(self).__name__, self.ndims))
        elif dimension_samples.keys()[0] in self.dimension_labels:
            return self[samples[0]]
        else:
            raise ValueError('Dimension %s could not be found.' % dimension_samples.keys()[0])


    def collapse(self, dimension_collapsefns, add_dimension={}):
        """

        """

        dims, collapsefns = zip(dimension_collapsefns.items())
        if len(dims) > self.ndims:
            raise ValueError('%d collapse dimensions provided, %s is %d-dimensional.'
                             % (len(dims), type(self).__name__, self.ndims))
        return Table({self.label: collapsefns[0](self.data[:, 0])},
                     **dict(self.get_param_values()))



    def __getitem__(self, slc):
        """
        Implements slicing or indexing of the data by the data x-value.
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
            return ScatterPoints(sample, **dict(self.get_param_values()))


    @property
    def stack_type(self):
        return DataStack


    @property
    def xlabel(self):
        return self._dimensions[0].pprint_label


    @property
    def ylabel(self):
        return self._label_dim.pprint_label


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

        return DataOverlay(overlays, metadata=self.metadata)


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
        l, r = (None, None) if self.xlim is None else self.xlim
        b, t = (None, None) if self.ylim is None else self.ylim
        return float(l), float(b), float(r), float(t)



class ScatterPoints(DataLayer):
    """
    ScatterPoints is a simple 1D View, which gets displayed as a number of
    disconnected points.
    """



class Curve(DataLayer):
    """
    Curve is a simple 1D View of disconnected points and therefore isn't
    necessarily ordered.
    """

    def stack(self):
        stack = DataStack(None, dimensions=[self.xlabel],
                          title=self.title+' {dims}', **self.metadata)
        for idx in range(len(self.data)):
            x = self.data[0]
            if x in stack:
                stack[x].data.append(self.data[0:idx])
            else:
                stack[x] = Curve(self.data[0:idx])
        return stack



class Histogram(DataLayer):
    """
    Histogram contains a number of bins, which are defined by the upper
    and lower bounds of their edges and the computed bin values.
    """

    def __init__(self, values, edges, **kwargs):
        self.values, self.edges = self._process_data(values, edges)
        super(Histogram, self).__init__((self.values, self.edges), **kwargs)


    def _process_data(self, values, edges):
        """
        Ensure that edges are specified as left and right edges of the
        histogram bins rather than bin centers.
        """
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
        return values, edges


    @property
    def ndims(self):
        return len(self.edges)-1


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
            self.label = layer._label_dim.name
        else:
            self.xlim = layer.xlim if self.xlim is None else find_minmax(self.xlim, layer.xlim)
            self.ylim = layer.ylim if self.xlim is None else find_minmax(self.ylim, layer.ylim)
            if layer._label_dim.name != self.label:
                raise Exception("DataLayers must share common dimensions.")
        self.data.append(layer)


    def cyclic_range(self):
        return self[0].cyclic_range



class DataStack(Stack):
    """
    A DataStack can hold any number of DataLayers indexed by a list of
    dimension values. It also has a number of properties, which can find
    the x- and y-dimension limits and labels.
    """

    data_type = (DataLayer, Annotation)

    overlay_type = DataOverlay

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
        if type(self.data) == dict: headings = sorted(headings)
        self.heading_map = OrderedDict([(el, str(el)) for el in headings])

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



class TableStack(Stack):
    """
    A TableStack may hold any number of TableViews indexed by a list
    of dimension values. It also allows the values of a particular
    cell to be sampled by name across any valid dimension.
    """
    _type = Table

    _type_map = None

    def heading_values(self):
        return self.last.heading_values() if len(self) else []


    def heading_names(self):
        return self.last.heading_names() if len(self) else []


    def _item_check(self, dim_vals, data):

        if self._type_map is None:
            self._type_map = dict((k,type(v)) for (k,v) in data.data.items())

        if set(self._type_map.keys()) != set(data.data.keys()):
            raise AssertionError("All TableViews in a TableStack must have"
                                 " a common set of headings.")

        for k, v in data.data.items():
            if k not in self._type_map:
                self._type_map[k] = None
            elif type(v) != self._type_map[k]:
                self._type_map[k] = None

        super(TableStack, self)._item_check(dim_vals, data)


    def sample(self, **kwargs):
        from .operation import sample_curve
        return sample_curve(self, **kwargs)




__all__ = list(set([_k for _k,_v in locals().items() if isinstance(_v, type) and
                    (issubclass(_v, Stack) or issubclass(_v, View))]))
