"""
The interface subpackage provides View and Plot types to wrap external
objects with. Currently only a Pandas compatibility wrapper is
provided, which allows integrating Pandas DataFrames within the
HoloViews compositioning and animation framework. Additionally, it
provides methods to apply operations to the underlying data and
convert it to standard HoloViews View types.
"""

from __future__ import absolute_import

import numpy as np

try:
    import pandas as pd
except:
    pd = None

import param

from ..core import ViewableElement, NdMapping, NdOverlay,\
    NdLayout, GridSpace, Element, HoloMap
from ..element import Chart, Table, Curve, Scatter, Bars, Points, VectorField, HeatMap, Scatter3D, Surface


class DataFrameView(Element):
    """
    DataFrameView provides a convenient compatibility wrapper around
    Pandas DataFrames. It provides several core functions:

        * Allows integrating several Pandas plot types with the
          HoloViews plotting system (includes plot, boxplot, histogram
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

    group = param.String(default='DFrame')

    value_dimensions = param.List(doc="DataFrameView has no value dimension.")

    def __init__(self, data, dimensions={}, key_dimensions=None, clone_override=False,
                 index=None, columns=None, dtype=None, copy=True, **params):
        if pd is None:
            raise Exception("Pandas is required for the Pandas interface.")
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data, index=index, columns=columns, dtype=dtype)
        elif copy:
            data = pd.DataFrame(data, copy=True)
        if clone_override:
            dim_dict = {d.name: d for d in key_dimensions}
            dims = [dim_dict.get(k, k) for k in data.columns]
        elif key_dimensions:
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
        if dim in self.data.columns:
            return np.array(self.data[dim])
        else:
            return super(DataFrameView, self).dimension_values(dim)


    def apply(self, name, *args, **kwargs):
        """
        Applies the Pandas dframe method corresponding to the supplied
        name with the supplied args and kwargs.
        """
        return self.clone(getattr(self.data, name)(*args, **kwargs),
                          clone_override=True)


    def dframe(self):
        """
        Returns a copy of the internal dframe.
        """
        return self.data.copy()


    def aggregate(self, dimensions=[], function=None, **reductions):
        """
        The aggregate function accepts either a list of Dimensions
        and a function to apply to find the aggregate across
        those Dimensions or a list of dimension/function pairs
        to apply one by one.
        """
        if not dimensions and not reductions:
            raise Exception("Supply either a list of Dimensions or"
                            "reductions as keyword arguments")
        reduced = self.data
        dfnumeric = reduced.applymap(np.isreal).all(axis=0)
        unreducable = list(dfnumeric[dfnumeric == False].index)
        if dimensions:
            if not function:
                raise Exception("Supply a function to reduce the Dimensions with")
            reduced = reduced.groupby(dimensions+unreducable, as_index=True).aggregate(function)
            reduced_indexes = [reduced.index.names.index(d) for d in unreducable if d not in dimensions]
            reduced = reduced.reset_index(level=reduced_indexes)
        if reductions:
            for dim, fn in reductions.items():
                reduced = reduced.groupby(dim, as_index=True).aggregate(fn)
                reduced_indexes = [reduced.index.names.index(d) for d in unreducable]
                reduced = reduced.reset_index(level=reduced_indexes)
        key_dimensions = [self.get_dimension(d) for d in reduced.columns]
        return self.clone(reduced, key_dimensions=key_dimensions)


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
            data = v.drop(dimensions, axis=1)
            mapping[k] = self.clone(data,
                                    key_dimensions=[self.get_dimension(d)
                                                    for d in data.columns])
        return mapping


    def overlay(self, dimensions):
        return self.groupby(dimensions, NdOverlay)


    def layout(self, dimensions=[], cols=4):
        return self.groupby(dimensions, NdLayout).cols(4)


    def grid(self, dimensions):
        """
        Splits the supplied the dimensions out into a GridSpace.
        """
        if len(dimensions) > 2:
            raise Exception('Grids hold a maximum of two dimensions.')
        return self.groupby(dimensions, GridSpace)


    def holomap(self, key_dimensions=[]):
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

    def _convert(self, kdims=[], vdims=[], mdims=[], reduce_fn=None,
                 view_type=None, dropna=False, **kwargs):
        """
        Conversion method to generate HoloViews objects from a
        DFrame. Accepts key, value and HoloMap dimensions.
        If no HoloMap dimensions are supplied then non-numeric
        dimensions are used. If a reduce_fn such as np.mean is
        supplied the data is aggregated for each group along the
        key_dimensions. Also supports a dropna option.
        """
        if not isinstance(kdims, list): kdims = [kdims]
        if not isinstance(vdims, list): vdims = [vdims]

        # Process dimensions
        all_dims = self.dimensions(label=True)
        sel_dims = kdims + vdims + mdims
        el_dims = kdims + vdims
        if not mdims and not reduce_fn:
            mdims = [dim for dim in self.dimensions(label=True)
                     if dim not in sel_dims]
        # Find leftover dimensions to reduce
        if reduce_fn:
            reduce_dims = kdims
        else:
            reduce_dims = []

        key_dims = [self.get_dimension(d) for d in kdims]
        val_dims = [self.get_dimension(d) for d in vdims]
        if mdims:
            groups = self.groupby(mdims, HoloMap)
            mdims = [self.get_dimension(d) for d in mdims]
        else:
            groups = NdMapping({0: self})
            mdims = ['Default']
        create_kwargs = dict(key_dimensions=key_dims,
                             value_dimensions=val_dims,
                             view_type=view_type)
        create_kwargs.update(kwargs)

        # Convert each element in the HoloMap
        hmap = HoloMap(key_dimensions=mdims)
        for k, v in groups.items():
            if reduce_dims:
                v = v.aggregate(reduce_dims, function=reduce_fn)
                v_indexes = [v.data.index.names.index(d) for d in kdims
                             if d in v.data.index.names]
                v = v.apply('reset_index', level=v_indexes)

            vdata = v.data.filter(el_dims)
            vdata = vdata.dropna() if dropna else vdata
            if issubclass(view_type, Chart):
                data = [np.array(vdata[d]) for d in el_dims]
                hmap[k] = self._create_chart(data, **create_kwargs)
            else:
                data = [np.array(vdata[d]) for d in el_dims]
                hmap[k] = self._create_table(data, **create_kwargs)
        return hmap if mdims != ['Default'] else hmap.last


    def _create_chart(self, data, key_dimensions=None, value_dimensions=None,
                      view_type=None, **kwargs):
        inherited = dict(key_dimensions=key_dimensions,
                         value_dimensions=value_dimensions, label=self.label)
        return view_type(np.vstack(data).T, **dict(inherited, **kwargs))


    def _create_table(self, data, key_dimensions=None, value_dimensions=None,
                      view_type=None, **kwargs):
        ndims = len(key_dimensions)
        key_data, value_data = data[:ndims], data[ndims:]
        keys = zip(*key_data)
        if ndims == 1:
            keys = [(k,) for k in keys]
        values = zip(*value_data)
        inherited = dict(key_dimensions=key_dimensions,
                         value_dimensions=value_dimensions, label=self.label)
        return view_type(zip(keys, values), **dict(inherited, **kwargs))


    def curve(self, kdims, vdims, mdims=[], reduce_fn=None, **kwargs):
        return self._convert(kdims, vdims, mdims, reduce_fn,
                             view_type=Curve, **kwargs)

    def points(self, kdims, vdims, mdims=[], reduce_fn=None, **kwargs):
        return self._convert(kdims, vdims, mdims, reduce_fn,
                             view_type=Points, **kwargs)

    def scatter3d(self, kdims, vdims, mdims=[], reduce_fn=None, **kwargs):
        return self._convert(kdims, vdims, mdims, reduce_fn,
                             view_type=Scatter3D, **kwargs)

    def scatter(self, kdims, vdims, mdims=[], reduce_fn=None, **kwargs):
        return self._convert(kdims, vdims, mdims, reduce_fn,
                             view_type=Scatter, **kwargs)

    def vectorfield(self, kdims, vdims, mdims=[], reduce_fn=None, **kwargs):
        return self._convert(kdims, vdims, mdims, reduce_fn,
                             view_type=VectorField, **kwargs)

    def bars(self, kdims, vdims, mdims=[], reduce_fn=None, **kwargs):
        return self._convert(kdims, vdims, mdims, reduce_fn,
                             view_type=Bars, **kwargs)

    def table(self, kdims, vdims, mdims=[], reduce_fn=None, **kwargs):
        return self._convert(kdims, vdims, mdims, reduce_fn,
                             view_type=Table, **kwargs)
    
    def heatmap(self, kdims, vdims, mdims=[], reduce_fn=None, **kwargs):
        tables = self.table(kdims, vdims, mdims, reduce_fn, **kwargs)

        if isinstance(tables, HoloMap):
            kwargs = dict(tables.last.get_param_values(onlychanged=True),
                          **kwargs)
            return tables.map(lambda x: HeatMap(x, **kwargs), ['Table'])
        else:
            kwargs = dict(tables.get_param_values(onlychanged=True),
                          **kwargs)
            return HeatMap(tables, **kwargs)
    
    def surface(self, kdims, vdims, mdims=[], reduce_fn=None, **kwargs):
        if not isinstance(kdims, list): kdims = [kdims]
        if not isinstance(vdims, list): vdims = [vdims]
        heatmap = self.heatmap(kdims, vdims, mdims, reduce_fn, **kwargs)
        key_dims = [self.get_dimension(d) for d in kdims]
        val_dims = [self.get_dimension(d) for d in vdims]
        kwargs = dict(kwargs, key_dimensions=key_dims, value_dimensions=val_dims,
                      label=self.label)
        if isinstance(heatmap, HoloMap):
            return heatmap.map(lambda x: Surface(x.data, **kwargs), ['HeatMap'])
        else:
            return Surface(heatmap.data, **kwargs)
