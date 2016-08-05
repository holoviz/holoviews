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
    from ..core.data import PandasInterface
except:
    pd = None
    PandasInterface = None

import param

from ..core import ViewableElement, NdMapping, Dataset, NdOverlay,\
    NdLayout, GridSpace, HoloMap
from ..element import (Chart, Table, Curve, Scatter, Bars, Points,
                       VectorField, HeatMap, Scatter3D, Surface)


class DataFrameView(Dataset):
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

    group = param.String(default='DFrame', constant=True)

    vdims = param.List(doc="DataFrameView has no value dimension.")

    def __init__(self, data, dimensions={}, kdims=None, clone_override=False,
                 index=None, columns=None, dtype=None, copy=True, **params):
        if pd is None:
            raise Exception("Pandas is required for the Pandas interface.")
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data, index=index, columns=columns, dtype=dtype)
        elif copy:
            data = pd.DataFrame(data, copy=True)
        if clone_override:
            dim_dict = {d.name: d for d in kdims}
            dims = [dim_dict.get(k, k) for k in data.columns]
        elif kdims:
            if len(kdims) != len(data.columns):
                raise ValueError("Supplied key dimensions do not match data columns")
            dims = kdims
        else:
            dims = list(data.columns)
        for name, dim in dimensions.items():
            if name in data.columns:
                dims[list(data.columns).index(name)] = dim

        ViewableElement.__init__(self, data, kdims=dims, **params)
        self.interface = PandasInterface
        self.data.columns = self.dimensions('key', True)


    def groupby(self, dimensions, container_type=NdMapping):
        invalid_dims = [d for d in dimensions if d not in self.dimensions()]
        if invalid_dims:
            raise Exception('Following dimensions could not be found %s.'
                            % invalid_dims)

        index_dims = [self.get_dimension(d) for d in dimensions]
        mapping_data = []
        for k, v in self.data.groupby([self.get_dimension(d).name for d in dimensions]):
            data = v.drop(dimensions, axis=1)
            mapping_data.append((k, self.clone(data, kdims=[self.get_dimension(d)
                                                            for d in data.columns])))
        return container_type(mapping_data, kdims=index_dims)


    def apply(self, name, *args, **kwargs):
        """
        Applies the Pandas dframe method corresponding to the supplied
        name with the supplied args and kwargs.
        """
        return self.clone(getattr(self.data, name)(*args, **kwargs),
                          clone_override=True)

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


    def holomap(self, kdims=[]):
        """
        Splits the supplied dimensions out into a HoloMap.
        """
        return self.groupby(kdims, HoloMap)


def is_type(df, baseType):
    test = [issubclass(np.dtype(d).type, baseType) for d in df.dtypes]
    return pd.DataFrame(data=test, index=df.columns, columns=["numeric"])


def is_number(df):
    try:
        return is_type(df, np.number)
    except:
        return False


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

        # Deprecation warning
        self.warning("The DFrame conversion interface is deprecated "
                     "and has been superseded by a real integration "
                     "with pandas.")

        if not isinstance(kdims, list): kdims = [kdims]
        if not isinstance(vdims, list): vdims = [vdims]

        # Process dimensions
        sel_dims = kdims + vdims + mdims
        el_dims = kdims + vdims
        if not mdims and not reduce_fn:
            numeric = is_number(self.data)
            mdims = [dim for dim in self.dimensions(label=True)
                     if dim not in sel_dims and not numeric.ix[dim][0]]
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
        create_kwargs = dict(kdims=key_dims, vdims=val_dims,
                            view_type=view_type)
        create_kwargs.update(kwargs)

        # Convert each element in the HoloMap
        hmap = HoloMap(kdims=mdims)
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


    def _create_chart(self, data, kdims=None, vdims=None,
                      view_type=None, **kwargs):
        inherited = dict(kdims=kdims,
                         vdims=vdims, label=self.label)
        return view_type(np.vstack(data).T, **dict(inherited, **kwargs))


    def _create_table(self, data, kdims=None, vdims=None,
                      view_type=None, **kwargs):
        ndims = len(kdims)
        key_data, value_data = data[:ndims], data[ndims:]
        keys = zip(*key_data)
        values = zip(*value_data)
        inherited = dict(kdims=kdims,
                         vdims=vdims, label=self.label)
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
        kwargs = dict(kwargs, kdims=key_dims, vdims=val_dims,
                      label=self.label)
        if isinstance(heatmap, HoloMap):
            return heatmap.map(lambda x: Surface(x.data, **kwargs), ['HeatMap'])
        else:
            return Surface(heatmap.data, **kwargs)
