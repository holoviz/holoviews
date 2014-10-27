from collections import OrderedDict

import numpy as np

import param

from ..core import Dimension, HoloMap, Layer, ViewMap
from .tabular import ItemTable, Table


class DataView(Layer):
    """
    The data held within an Array is a numpy array of shape (n, 2).
    Layer objects are sliceable along the X dimension allowing easy
    selection of subsets of the data.
    """

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
        return ItemTable(reduced_data, label=self.label, title=self.title,
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
        stack = ViewMap(None, dimensions=[self.xlabel], title=self.title+' {dims}')
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