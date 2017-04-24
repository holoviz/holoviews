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
    DFrame is a specialized Dataset type useful as an interface for
    pandas plots.
    """
