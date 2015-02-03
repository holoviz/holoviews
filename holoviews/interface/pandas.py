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

from ..core import Dimension, ViewableElement, NdMapping, NdOverlay,\
 NdLayout, AxisLayout, Element, HoloMap
from ..element import Table, Curve, Scatter, Bars, Points, VectorField, HeatMap, Scatter3D, Surface


class DataFrameView(Element):
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
                                            when visualizing the ViewableElement.""")

    x = param.String(doc="""Dimension to visualize along the x-axis.""")

    x2 = param.String(doc="""Dimension to visualize along a second
                             dependent axis.""")

    y = param.String(doc="""Dimension to visualize along the y-axis.""")

    value = param.String(default='DFrame')

    value_dimensions = param.List(doc="DataFrameView has no value dimension.")

    def __init__(self, data, dimensions={}, key_dimensions=None, **params):
        if pd is None:
            raise Exception("Pandas is required for the Pandas interface.")
        if not isinstance(data, pd.DataFrame):
            raise Exception('DataFrame ViewableElement type requires Pandas dataframe as data.')
        if key_dimensions:
            if len(key_dimensions) != len(data.columns):
                raise ValueError("Supplied key dimensions do not match data columns")
            dims = key_dimensions
        else:
            dims = list(data.columns)
        for name, dim in dimensions.items():
            if name in data.columns:
                dims[list(data.columns).index(name)] = dim

        self._xlim = None
        self._ylim = None
        ViewableElement.__init__(self, data, key_dimensions=dims, **params)
        self.data.columns = self._cached_index_names


    def __getitem__(self, key):
        """
        Allows slicing and selecting along the DataFrameView dimensions.
        """
        if key is ():
            return self
        else:
            if len(key) <= self.ndims:
                return self.select(**dict(zip(self._cached_index_names, key)))
            else:
                raise Exception('Selection contains %d dimensions, DataFrameView '
                                'only has %d index dimensions.' % (self.ndims, len(key)))


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


    def dimension_values(self, dim):
        return np.array(self.data[dim])


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


    def groupby(self, dimensions, container_type=NdMapping):
        invalid_dims = list(set(dimensions) - set(self._cached_index_names))
        if invalid_dims:
            raise Exception('Following dimensions could not be found %s.'
                            % invalid_dims)

        index_dims = [self.get_dimension(d) for d in dimensions]
        mapping = container_type(None, key_dimensions=index_dims)
        view_dims = set(self._cached_index_names) - set(dimensions)
        view_dims = [self.get_dimension(d) for d in view_dims]
        for k, v in self.data.groupby(dimensions):
            mapping[k] = self.clone(v.drop(dimensions, axis=1),
                                    key_dimensions=view_dims)
        return mapping


    def overlay(self, dimensions):
        return self.groupby(dimensions, NdOverlay)


    def layout(self, dimensions=[], cols=4):
        return self.groupby(dimensions, NdLayout).cols(4)


    def grid(self, dimensions):
        """
        Splits the supplied the dimensions out into a AxisLayout.
        """
        if len(dimensions) > 2:
            raise Exception('Grids hold a maximum of two dimensions.')
        return self.groupby(dimensions, AxisLayout)


    def viewmap(self, key_dimensions=[]):
        """
        Splits the supplied dimensions out into a HoloMap.
        """
        return self.groupby(key_dimensions, HoloMap)

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
    methods to convert Pandas DataFrames to different ViewableElement types,
    currently including Tables and HeatMaps.

    The ViewableElement conversion methods all share a common signature:

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

    def points3d(self, *args, **kwargs):
        return self.table(*args, **dict(view_type=Scatter3D, **kwargs))

    def scatter(self, *args, **kwargs):
        return self.table(*args, **dict(view_type=Scatter, **kwargs))

    def surface(self, *args, **kwargs):
        heatmap = self.table(*args, **dict(view_type=HeatMap, **kwargs))
        return Surface(heatmap.data, **kwargs)

    def vectorfield(self, *args, **kwargs):
        return self.table(*args, **dict(view_type=VectorField, **kwargs))

    def table(self, value_dims, view_dims, reduce_fn=None, map_dims=[], view_type=None, **kwargs):
        if map_dims:
            map_groups = self.data.groupby(map_dims)
            vm_dims = [self.get_dimension(d) for d in map_dims]
        else:
            map_groups = [(0, self.data)]
            vm_dims = ['None']

        vmap = HoloMap(key_dimensions=vm_dims)
        value = self.value if self.value != type(self).__name__ else 'Table'
        vdims = [self.get_dimension(d) for d in view_dims]
        valdims = [self.get_dimension(d) for d in value_dims]
        for map_key, group in map_groups:
            table_data = OrderedDict()
            for k, v in group.groupby(view_dims):
                data = np.vstack(np.array(v[d]) for d in value_dims)
                data = reduce_fn(data, axis=1) if reduce_fn else data[:, 0]
                table_data[k] = data
            view = Table(table_data, key_dimensions=vdims,
                         value_dimensions=valdims, label=self.label,
                         value=value)
            vmap[map_key] = view_type(view, **kwargs) if view_type else view
        return vmap if map_dims else vmap.last
