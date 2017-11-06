import itertools

import param
import numpy as np

from ..core import Dataset, OrderedDict
from ..core.boundingregion import BoundingBox
from ..core.operation import Operation
from ..core.sheetcoords import Slice
from ..core.util import (is_nan, sort_topologically, one_to_one,
                         cartesian_product, is_cyclic, datetime_types)

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


def compute_edges(edges):
    """
    Computes edges as midpoints of the bin centers.
    The first and last boundaries are equidistant from the first and last
    midpoints respectively.
    """
    edges = np.asarray(edges)
    if edges.dtype.kind == 'i':
        edges = edges.astype('f')
    midpoints = (edges[:-1] + edges[1:])/2.0
    boundaries = (2*edges[0] - midpoints[0], 2*edges[-1] - midpoints[-1])
    return np.concatenate([boundaries[:1], midpoints, boundaries[-1:]])


def split_path(path):
    """
    Split a Path type containing a single NaN separated path into
    multiple subpaths.
    """
    path = path.split()[0]
    values = path.dimension_values(0)
    splits = np.concatenate([[0], np.where(np.isnan(values))[0]+1, [0]])
    subpaths = []
    data = PandasInterface.as_dframe(path) if pd else path.array()
    for i in range(len(splits)-1):
        slc = slice(splits[i], splits[i+1]-1)
        subpath = data.iloc[slc] if pd else data[slc]
        if len(subpath):
            subpaths.append(subpath)
    return subpaths


def compute_slice_bounds(slices, scs, shape):
    """
    Given a 2D selection consisting of slices/coordinates, a
    SheetCoordinateSystem and the shape of the array returns a new
    BoundingBox representing the sliced region.
    """
    xidx, yidx = slices
    ys, xs = shape
    l, b, r, t = scs.bounds.lbrt()
    xdensity, ydensity = scs.xdensity, scs.ydensity
    xunit = (1./xdensity)
    yunit = (1./ydensity)
    if isinstance(l, datetime_types):
        xunit = np.timedelta64(int(round(xunit)), scs._time_unit)
    if isinstance(b, datetime_types):
        yunit = np.timedelta64(int(round(yunit)), scs._time_unit)
    if isinstance(xidx, slice):
        l = l if xidx.start is None else max(l, xidx.start)
        r = r if xidx.stop is None else min(r, xidx.stop)
    if isinstance(yidx, slice):
        b = b if yidx.start is None else max(b, yidx.start)
        t = t if yidx.stop is None else min(t, yidx.stop)
    bounds = BoundingBox(points=((l, b), (r, t)))

    # Apply new bounds
    slc = Slice(bounds, scs)

    # Apply scalar and list indices
    l, b, r, t = slc.compute_bounds(scs).lbrt()
    if not isinstance(xidx, slice):
        if not isinstance(xidx, (list, set)): xidx = [xidx]
        if len(xidx) > 1:
            xdensity = xdensity*(float(len(xidx))/xs)
        ls, rs = [], []
        for idx in xidx:
            xc, _ = scs.closest_cell_center(idx, b)
            ls.append(xc-xunit/2)
            rs.append(xc+xunit/2)
        l, r = np.min(ls), np.max(rs)
    elif not isinstance(yidx, slice):
        if not isinstance(yidx, (set, list)): yidx = [yidx]
        if len(yidx) > 1:
            ydensity = ydensity*(float(len(yidx))/ys)
        bs, ts = [], []
        for idx in yidx:
            _, yc = scs.closest_cell_center(l, idx)
            bs.append(yc-yunit/2)
            ts.append(yc+yunit/2)
        b, t = np.min(bs), np.max(ts)
    return BoundingBox(points=((l, b), (r, t)))


def reduce_fn(x):
    """
    Aggregation function to get the first non-zero value.
    """
    values = x.values if pd and isinstance(x, pd.Series) else x
    for v in values:
        if not is_nan(v):
            return v
    return np.NaN


class categorical_aggregate2d(Operation):
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
            coords = list(itertools.chain(*sort_topologically(orderings)))
            ycoords = coords if len(coords) == len(ycoords) else np.sort(ycoords)
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
