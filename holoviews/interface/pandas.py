"""
The interface subpackage provides View and Plot types to wrap external
objects with. Currently only a Pandas compatibility wrapper is
provided, which allows integrating Pandas DataFrames within the
holoviews compositioning and animation framework. Additionally, it
provides methods to apply operations to the underlying data and
convert it to standard holoviews View types.
"""

from __future__ import absolute_import

from collections import OrderedDict

import numpy as np

try:
    import pandas as pd
except:
    pd = None

import param

from ..core import Dimension, NdMapping, View, Layer, Overlay, ViewMap, GridLayout, Grid
from ..core.options import options, PlotOpts
from ..view import HeatMap, Table, Curve, Scatter, Bars, Points, VectorField


class DataFrameView(Layer):
    """
    DataFrameView provides a convenient compatibility wrapper around
    Pandas DataFrames. It provides several core functions:

        * Allows integrating several Pandas plot types with the
          holoviews plotting system (includes plot, boxplot, histogram
          and scatter_matrix).

        * Provides several convenient wrapper methods to apply
          DataFrame methods and slice data. This includes:

              1) The apply method, which takes the DataFrame method to
                 be applied as the first argument and passes any
                 supplied args or kwargs along.

              2) The select and __getitem__ method which allow for
                 selecting and slicing the data using NdMapping.
    """

    plot_type = param.ObjectSelector(default=None, 
                                     objects=['plot', 'boxplot',
                                              'hist', 'scatter_matrix',
                                              'autocorrelation_plot',
                                              None],
                                     doc="""Selects which Pandas plot type to use,
                                            when visualizing the View.""")

    x = param.String(doc="""Dimension to visualize along the x-axis.""")

    y = param.String(doc="""Dimension to visualize along the y-axis.""")

    value = param.ClassSelector(class_=(str, Dimension), precedence=-1,
                                doc="DataFrameView has no value dimension.")

    def __init__(self, data, dimensions=None, **params):
        if pd is None:
            raise Exception("Pandas is required for the Pandas interface.")
        if not isinstance(data, pd.DataFrame):
            raise Exception('DataFrame View type requires Pandas dataframe as data.')
        if dimensions is None:
            dims = list(data.columns)
        else:
            dims = ['' for i in range(len(data.columns))]
            for dim in dimensions:
                dim_name = dim.name if isinstance(dim, Dimension) else dim
                if dim_name in data.columns:
                    dims[list(data.columns).index(dim_name)] = dim

        self._xlim = None
        self._ylim = None
        View.__init__(self, data, dimensions=dims, **params)
        self.data.columns = self.dimension_labels


    def __getitem__(self, key):
        """
        Allows slicing and selecting along the DataFrameView dimensions.
        """
        if key is ():
            return self
        else:
            if len(key) <= self.ndims:
                return self.select(**dict(zip(self.dimension_labels, key)))
            else:
                raise Exception('Selection contains %d dimensions, DataFrameView '
                                'only has %d dimensions.' % (self.ndims, len(key)))


    def select(self, **select):
        """
        Allows slice and select individual values along the DataFrameView
        dimensions. Supply the dimensions and values or slices as
        keyword arguments.
        """
        df = self.data
        for dim, k in select.items():
            if isinstance(k, slice):
                df = df[(k.start < df[dim]) & (df[dim] < k.stop)]
            else:
                df = df[df[dim] == k]
        return self.clone(df)


    def apply(self, name, *args, **kwargs):
        """
        Applies the Pandas dframe method corresponding to the supplied
        name with the supplied args and kwargs.
        """
        return self.clone(getattr(self.data, name)(*args, **kwargs))


    def dframe(self):
        """
        Returns a copy of the internal dframe.
        """
        return self.data.copy()


    def _split_dimensions(self, dimensions, ndmapping_type=NdMapping):
        invalid_dims = list(set(dimensions) - set(self.dimension_labels))
        if invalid_dims:
            raise Exception('Following dimensions could not be found %s.'
                            % invalid_dims)

        ndmapping = ndmapping_type(None, dimensions=[self.dim_dict[d] for d in dimensions])
        view_dims = set(self.dimension_labels) - set(dimensions)
        view_dims = [self.dim_dict[d] for d in view_dims]
        for k, v in self.data.groupby(dimensions):
            ndmapping[k] = self.clone(v.drop(dimensions, axis=1),
                                      dimensions=view_dims)
        return ndmapping


    def overlay_dimensions(self, dimensions):
        return self._split_dimensions(dimensions, Overlay)


    def grid(self, dimensions=[], layout=False, cols=4):
        """
        Splits the supplied the dimensions out into a Grid.
        """
        if len(dimensions) > 2:
            raise Exception('Grids hold a maximum of two dimensions.')
        if layout:
            ndmapping = self._split_dimensions(dimensions, NdMapping)
            for keys, vmap in ndmapping._data.items():
                label = ', '.join([d.pprint_value(k) for d, k in
                                   zip(ndmapping.dimensions, keys)])
                vmap.title = ' '.join([label, vmap.title])
            return GridLayout(ndmapping).cols(cols)
        return self._split_dimensions(dimensions, Grid)


    def viewmap(self, dimensions=[]):
        """
        Splits the supplied dimensions out into a ViewMap.
        """
        return self._split_dimensions(dimensions, ViewMap)

    @property
    def xlabel(self):
        return self.x

    @property
    def ylabel(self):
        return self.y

    @property
    def xlim(self):
        if self._xlim:
            return self._xlim
        if self.x:
            xdata = self.data[self.x]
            return min(xdata), max(xdata)
        else:
            return None

    @property
    def ylim(self):
        if self._ylim:
            return self._ylim
        elif self.y:
            ydata = self.data[self.y]
            return min(ydata), max(ydata)
        else:
            return None



class DFrame(DataFrameView):
    """
    DFrame is a DataFrameView type, which additionally provides
    methods to convert Pandas DataFrames to different View types,
    currently including Tables and HeatMaps.

    The View conversion methods all share a common signature:

      * The value dimension (string).
      * The index dimensions (list of strings).
      * An optional reduce_fn.
      * Optional map_dims (list of strings).
    """

    def bars(self, *args, **kwargs):
        return self.table(*args, **dict(view_type=Bars, **kwargs))

    def curve(self, *args, **kwargs):
        return self.table(*args, **dict(view_type=Curve, **kwargs))

    def heatmap(self, *args, **kwargs):
        return self.table(*args, **dict(view_type=HeatMap, **kwargs))

    def points(self, *args, **kwargs):
        return self.table(*args, **dict(view_type=Points, **kwargs))

    def scatter(self, *args, **kwargs):
        return self.table(*args, **dict(view_type=Scatter, **kwargs))

    def vectorfield(self, *args, **kwargs):
        return self.table(*args, **dict(view_type=VectorField, **kwargs))

    def table(self, value_dim, dimensions, reduce_fn=None, map_dims=[], view_type=None):
        if map_dims:
            map_groups = self.data.groupby(map_dims)
            vm_dims = map_dims
        else:
            map_groups = [(0, self.data)]
            vm_dims = ['None']

        vmap = ViewMap(dimensions=vm_dims)
        vdims = [self.dim_dict[dim] for dim in dimensions]
        for map_key, group in map_groups:
            table_data = OrderedDict()
            for k, v in group.groupby(dimensions):
                data = np.array(v[value_dim])
                table_data[k] = reduce_fn(data) if reduce_fn else data[0]
                view = Table(table_data, dimensions=vdims,
                             value=self.dim_dict[value_dim])
            vmap[map_key] = view_type(view) if view_type else view

        return vmap if map_dims else vmap.last


options.DFrameView = PlotOpts()
