import itertools

import param
import numpy as np

from ..core import Dataset, OrderedDict
from ..core.operation import ElementOperation
from ..core.util import (is_nan, sort_topologically, one_to_one,
                         cartesian_product, is_cyclic)

try:
    import pandas as pd
    from ..core.data import PandasInterface
except:
    pd = None

try:
    import dask
except:
    dask = None

try:
    import xarray as xr
except:
    xr = None


def toarray(v, index_value=False):
    """
    Interface helper function to turn dask Arrays into numpy arrays as
    necessary. If index_value is True, a value is returned instead of
    an array holding a single value.
    """
    if dask and isinstance(v, dask.array.Array):
        arr =  v.compute()
        return arr[()] if index_value else arr
    else:
        return v

def compute_edges(edges):
    """
    Computes edges from a number of bin centers,
    throwing an exception if the edges are not
    evenly spaced.
    """
    widths = np.diff(edges)
    if np.allclose(widths, widths[0]):
        width = widths[0]
    else:
        raise ValueError('Centered bins have to be of equal width.')
    edges -= width/2.
    return np.concatenate([edges, [edges[-1]+width]])


def reduce_fn(x):
    """
    Aggregation function to get the first non-zero value.
    """
    values = x.values if pd and isinstance(x, pd.Series) else x
    for v in values:
        if not is_nan(v):
            return v
    return np.NaN


class categorical_aggregate2d(ElementOperation):
    """
    Generates a gridded Dataset of 2D aggregate arrays indexed by the
    first two dimensions of the passed Element, turning all remaining
    dimensions into value dimensions. The key dimensions of the
    gridded array are treated as categorical indices. Useful for data
    indexed by two independent categorical variables such as a table
    of population values indexed by country and year. Data that is
    indexed by continuous dimensions should be binned before
    aggregation. The aggregation will retain the global sorting order
    of both dimensions.

    >> table = Table([('USA', 2000, 282.2), ('UK', 2005, 58.89)],
                     kdims=['Country', 'Year'], vdims=['Population'])
    >> categorical_aggregate2d(table)
    Dataset({'Country': ['USA', 'UK'], 'Year': [2000, 2005],
             'Population': [[ 282.2 , np.NaN], [np.NaN,   58.89]]},
            kdims=['Country', 'Year'], vdims=['Population'])
    """

    datatype = param.List(['xarray', 'grid'] if xr else ['grid'], doc="""
        The grid interface types to use when constructing the gridded Dataset.""")

    def _get_coords(self, obj):
        """
        Get the coordinates of the 2D aggregate, maintaining the correct
        sorting order.
        """
        xdim, ydim = obj.dimensions(label=True)[:2]
        xcoords = obj.dimension_values(xdim, False)
        ycoords = obj.dimension_values(ydim, False)

        # Determine global orderings of y-values using topological sort
        grouped = obj.groupby(xdim, container_type=OrderedDict,
                              group_type=Dataset).values()
        orderings = OrderedDict()
        sort = True
        for group in grouped:
            vals = group.dimension_values(ydim, False)
            if len(vals) == 1:
                orderings[vals[0]] = [vals[0]]
            else:
                for i in range(len(vals)-1):
                    p1, p2 = vals[i:i+2]
                    orderings[p1] = [p2]
            if sort:
                if vals.dtype.kind in ('i', 'f'):
                    sort = (np.diff(vals)>=0).all()
                else:
                    sort = np.array_equal(np.sort(vals), vals)
        if sort or one_to_one(orderings, ycoords):
            ycoords = np.sort(ycoords)
        elif not is_cyclic(orderings):
            ycoords = list(itertools.chain(*sort_topologically(orderings)))
        return xcoords, ycoords


    def _aggregate_dataset(self, obj, xcoords, ycoords):
        """
        Generates a gridded Dataset from a column-based dataset and
        lists of xcoords and ycoords
        """
        dim_labels = obj.dimensions(label=True)
        vdims = obj.dimensions()[2:]
        xdim, ydim = dim_labels[:2]
        shape = (len(ycoords), len(xcoords))
        nsamples = np.product(shape)

        ys, xs = cartesian_product([ycoords, xcoords], copy=True)
        data = {xdim: xs, ydim: ys}
        for vdim in vdims:
            values = np.empty(nsamples)
            values[:] = np.NaN
            data[vdim.name] = values
        dtype = 'dataframe' if pd else 'dictionary'
        dense_data = Dataset(data, kdims=obj.kdims, vdims=obj.vdims, datatype=[dtype])
        concat_data = obj.interface.concatenate([dense_data, obj], datatype=[dtype])
        reindexed = concat_data.reindex([xdim, ydim], vdims)
        if pd:
            df = PandasInterface.as_dframe(reindexed)
            df = df.groupby([xdim, ydim], sort=False).first().reset_index()
            agg = reindexed.clone(df)
        else:
            agg = reindexed.aggregate([xdim, ydim], reduce_fn)

        # Convert data to a gridded dataset
        grid_data = {xdim: xcoords, ydim: ycoords}
        for vdim in vdims:
            grid_data[vdim.name] = agg.dimension_values(vdim).reshape(shape)
        return agg.clone(grid_data, kdims=[xdim, ydim], vdims=vdims,
                         datatype=self.p.datatype)


    def _process(self, obj, key=None):
        """
        Generates a categorical 2D aggregate by inserting NaNs at all
        cross-product locations that do not already have a value assigned.
        Returns a 2D gridded Dataset object.
        """
        if isinstance(obj, Dataset) and obj.interface.gridded:
            return obj
        elif obj.ndims > 2:
            raise ValueError("Cannot aggregate more than two dimensions")
        elif len(obj.dimensions()) < 3:
            raise ValueError("Must have at two dimensions to aggregate over"
                             "and one value dimension to aggregate on.")

        dtype = 'dataframe' if pd else 'dictionary'
        obj = Dataset(obj, datatype=[dtype])
        xcoords, ycoords = self._get_coords(obj)
        return self._aggregate_dataset(obj, xcoords, ycoords)
