from __future__ import absolute_import

import sys
import warnings

from collections import defaultdict

import numpy as np

from ..dimension import dimension_name
from .interface import DataError, Interface
from .multipath import MultiInterface
from .pandas import PandasInterface


class SpatialPandasInterface(MultiInterface):

    types = ()

    datatype = 'spatialpandas'

    multi = True

    @classmethod
    def loaded(cls):
        return 'spatialpandas' in sys.modules

    @classmethod
    def applies(cls, obj):
        if not cls.loaded():
            return False
        from spatialpandas import GeoDataFrame, GeoSeries
        return isinstance(obj, (GeoDataFrame, GeoSeries))

    @classmethod
    def geo_column(cls, data):
        from spatialpandas import GeoSeries
        col = 'geometry'
        if col in data and isinstance(data[col], GeoSeries):
            return col
        cols = [c for c in data.columns if isinstance(data[c], GeoSeries)]
        if not cols:
            raise ValueError('No geometry column found in spatialpandas.GeoDataFrame, '
                             'use the PandasInterface instead.')
        return cols[0]

    @classmethod
    def init(cls, eltype, data, kdims, vdims):
        import pandas as pd
        from spatialpandas import GeoDataFrame, GeoSeries

        # Handle conversion from geopandas
        if 'geopandas' in sys.modules:
            import geopandas as gpd
            if isinstance(data, gpd.GeoSeries):
                data = data.to_frame()
            if isinstance(data, gpd.GeoDataFrame):
                data = GeoDataFrame(data)

        if isinstance(data, GeoSeries):
            data = data.to_frame()
        if isinstance(data, list):
            if 'shapely' in sys.modules:
                # Handle conversion from shapely geometries
                from shapely.geometry.base import BaseGeometry
                if all(isinstance(d, BaseGeometry) for d in data):
                    data = GeoSeries(data).to_frame()
                elif all(isinstance(d, dict) and 'geometry' in d and isinstance(d['geometry'], BaseGeometry)
                         for d in data):
                    new_data = {col: [] for col in data[0]}
                    for d in data:
                        for col, val in d.items():
                            new_data[col] = val
                    new_data['geometry'] = GeoSeries(new_data['geometry'])
                    data = GeoDataFrame(new_data)
            if isinstance(data, list):
                dims = (kd.name for kd in (kdims or eltype.kdims)[:2])
                data = to_spatialpandas(data, *dims, ring=hasattr(eltype, 'holes'))
        elif not isinstance(data, GeoDataFrame):
            raise ValueError("SpatialPandasInterface only support spatialpandas DataFrames.")
        elif 'geometry' not in data:
            cls.geo_column(data)

        if kdims is None:
            kdims = eltype.kdims

        if vdims is None:
            vdims = eltype.vdims

        index_names = data.index.names if isinstance(data, pd.DataFrame) else [data.index.name]
        if index_names == [None]:
            index_names = ['index']

        for kd in kdims+vdims:
            kd = dimension_name(kd)
            if kd in data.columns:
                continue
            if any(kd == ('index' if name is None else name)
                   for name in index_names):
                data = data.reset_index()
                break

        return data, {'kdims': kdims, 'vdims': vdims}, {}

    @classmethod
    def validate(cls, dataset, vdims=True):
        dim_types = 'key' if vdims else 'all'
        geom_dims = cls.geom_dims(dataset)
        if len(geom_dims) != 2:
            raise DataError('Expected %s instance to declare two key '
                            'dimensions corresponding to the geometry '
                            'coordinates but %d dimensions were found '
                            'which did not refer to any columns.'
                            % (type(dataset).__name__, len(geom_dims)), cls)
        not_found = [d.name for d in dataset.dimensions(dim_types)
                     if d not in geom_dims and d.name not in dataset.data]
        if not_found:
            raise DataError("Supplied data does not contain specified "
                             "dimensions, the following dimensions were "
                             "not found: %s" % repr(not_found), cls)


    @classmethod
    def dtype(cls, dataset, dimension):
        name = dataset.get_dimension(dimension, strict=True).name
        if name not in dataset.data:
            return np.dtype('float') # Geometry dimension
        return dataset.data[name].dtype


    @classmethod
    def has_holes(cls, dataset):
        from spatialpandas.geometry import MultiPolygonDtype, PolygonDtype
        col = cls.geo_column(dataset.data)
        series = dataset.data[col]
        if isinstance(series.dtype, (MultiPolygonDtype, PolygonDtype)):
            return False
        for geom in series:
            if isinstance(geom, Polygon) and geom.interiors:
                return True
            elif isinstance(geom, MultiPolygon):
                for g in geom:
                    if isinstance(g, Polygon) and g.interiors:
                        return True
        return False

    @classmethod
    def holes(cls, dataset):
        from spatialpandas.geometry import MultiPolygonDtype, PolygonDtype
        holes = []
        if not len(dataset.data):
            return holes
        col = cls.geo_column(dataset.data)
        series = dataset.data[col]
        return [geom_to_holes(geom) for geom in series]

    @classmethod
    def select(cls, dataset, selection_mask=None, **selection):
        if cls.geom_dims(dataset):
            df = cls.shape_mask(dataset, selection)
        else:
            df = dataset.data
        if not selection:
            return df
        elif selection_mask is None:
            selection_mask = cls.select_mask(dataset, selection)
        indexed = cls.indexed(dataset, selection)
        df = df.iloc[selection_mask]
        if indexed and len(df) == 1 and len(dataset.vdims) == 1:
            return df[dataset.vdims[0].name].iloc[0]
        return df

    @classmethod
    def shape_mask(cls, dataset, selection):
        xdim, ydim = cls.geom_dims(dataset)
        xsel = selection.pop(xdim.name, None)
        ysel = selection.pop(ydim.name, None)
        if xsel is None and ysel is None:
            return dataset.data

        from shapely.geometry import box

        if xsel is None:
            x0, x1 = cls.range(dataset, xdim)
        elif isinstance(xsel, slice):
            x0, x1 = xsel.start, xsel.stop
        elif isinstance(xsel, tuple):
            x0, x1 = xsel
        else:
            raise ValueError("Only slicing is supported on geometries, %s "
                             "selection is of type %s."
                             % (xdim, type(xsel).__name__))

        if ysel is None:
            y0, y1 = cls.range(dataset, ydim)
        elif isinstance(ysel, slice):
            y0, y1 = ysel.start, ysel.stop
        elif isinstance(ysel, tuple):
            y0, y1 = ysel
        else:
            raise ValueError("Only slicing is supported on geometries, %s "
                             "selection is of type %s."
                             % (ydim, type(ysel).__name__))

        bounds = box(x0, y0, x1, y1)
        col = cls.geo_column(dataset.data)
        df = dataset.data.copy()
        df[col] = df[col].intersection(bounds)
        return df[df[col].area > 0]

    @classmethod
    def select_mask(cls, dataset, selection):
        mask = np.ones(len(dataset.data), dtype=np.bool)
        for dim, k in selection.items():
            if isinstance(k, tuple):
                k = slice(*k)
            arr = dataset.data[dim].values
            if isinstance(k, slice):
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', r'invalid value encountered')
                    if k.start is not None:
                        mask &= k.start <= arr
                    if k.stop is not None:
                        mask &= arr < k.stop
            elif isinstance(k, (set, list)):
                iter_slcs = []
                for ik in k:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', r'invalid value encountered')
                        iter_slcs.append(arr == ik)
                mask &= np.logical_or.reduce(iter_slcs)
            elif callable(k):
                mask &= k(arr)
            else:
                index_mask = arr == k
                if dataset.ndims == 1 and np.sum(index_mask) == 0:
                    data_index = np.argmin(np.abs(arr - k))
                    mask = np.zeros(len(dataset), dtype=np.bool)
                    mask[data_index] = True
                else:
                    mask &= index_mask
        return mask

    @classmethod
    def geom_dims(cls, dataset):
        return [d for d in dataset.kdims + dataset.vdims
                if d.name not in dataset.data]

    @classmethod
    def dimension_type(cls, dataset, dim):
        col = cls.geo_column(dataset.data)
        arr = geom_to_array(dataset.data[col].iloc[0])
        ds = dataset.clone(arr, datatype=cls.subtypes, vdims=[])
        return ds.interface.dimension_type(ds, dim)

    @classmethod
    def isscalar(cls, dataset, dim):
        """
        Tests if dimension is scalar in each subpath.
        """
        idx = dataset.get_dimension_index(dim)
        return idx not in [0, 1]

    @classmethod
    def range(cls, dataset, dim):
        dim = dataset.get_dimension(dim)
        geom_dims = cls.geom_dims(dataset)
        if dim in geom_dims:
            col = cls.geo_column(dataset.data)
            idx = geom_dims.index(dim)
            bounds = dataset.data[col].total_bounds
            if idx == 0:
                return (bounds[0], bounds[2])
            else:
                return (bounds[1], bounds[3])
        else:
            vals = dataset.data[dim.name]
            return vals.min(), vals.max()

    @classmethod
    def aggregate(cls, columns, dimensions, function, **kwargs):
        raise NotImplementedError

    @classmethod
    def groupby(cls, columns, dimensions, container_type, group_type, **kwargs):
        return PandasInterface.groupby(columns, dimensions, container_type, group_type, **kwargs)

    @classmethod
    def reindex(cls, dataset, kdims=None, vdims=None):
        return dataset.data

    @classmethod
    def sample(cls, columns, samples=[]):
        raise NotImplementedError

    @classmethod
    def shape(cls, dataset):
        return PandasInterface.shape(dataset)

    @classmethod
    def length(cls, dataset):
        return PandasInterface.length(dataset)

    @classmethod
    def nonzero(cls, dataset):
        return bool(cls.length(dataset))

    @classmethod
    def redim(cls, dataset, dimensions):
        return PandasInterface.redim(dataset, dimensions)

    @classmethod
    def values(cls, dataset, dimension, expanded=True, flat=True, compute=True, keep_index=False):
        dimension = dataset.get_dimension(dimension)
        geom_dims = dataset.interface.geom_dims(dataset)
        data = dataset.data
        if dimension not in geom_dims and not expanded:
            data = data[dimension.name]
            return data if keep_index else data.values
        elif not len(data):
            return np.array([])

        index = geom_dims.index(dimension)
        col = cls.geo_column(dataset.data)
        return geom_array_to_array(data[col].values, index)

    @classmethod
    def split(cls, dataset, start, end, datatype, **kwargs):
        objs = []
        xdim, ydim = dataset.kdims[:2]
        if not len(dataset.data):
            return []
        row = dataset.data.iloc[0]
        col = cls.geo_column(dataset.data)
        arr = geom_to_array(row[col])
        d = {(xdim.name, ydim.name): arr}
        d.update({vd.name: row[vd.name] for vd in dataset.vdims})
        ds = dataset.clone(d, datatype=['dictionary'])
        for i, row in dataset.data.iterrows():
            if datatype == 'geom':
                objs.append(row[col])
                continue
            geom = row[col]
            arr = geom_to_array(geom)
            d = {xdim.name: arr[:, 0], ydim.name: arr[:, 1]}
            d.update({vd.name: row[vd.name] for vd in dataset.vdims})
            ds.data = d
            if datatype == 'array':
                obj = ds.array(**kwargs)
            elif datatype == 'dataframe':
                obj = ds.dframe(**kwargs)
            elif datatype == 'columns':
                obj = ds.columns(**kwargs)
            elif datatype is None:
                obj = ds.clone()
            else:
                raise ValueError("%s datatype not support" % datatype)
            objs.append(obj)
        return objs


def geom_to_array(geom, index=None, multi=False):
    """Converts spatialpandas geometry to an array.

    Args:
        geom: spatialpandas geometry
        index: The column index to return
        multi: Whether to concatenate multiple arrays or not

    Returns:
        Array or list of arrays.
    """
    from spatialpandas.geometry import Point, Polygon, Line, Ring, MultiPolygon
    if isinstance(geom, Point):
        if index is None:
            return np.array([geom.x, geom.y])
        arrays = [np.array([geom.y if index else geom.x])]
    elif isinstance(geom, (Polygon, Line, Ring)):
        exterior = geom.data[0] if isinstance(geom, Polygon) else geom.data
        arr = np.array(exterior.as_py()).reshape(-1, 2)
        arrays = [arr if index is None else arr[:, index]]
    else:
        arrays = []
        for g in geom.data:
            exterior = g[0] if isinstance(geom, MultiPolygon) else g
            arr = np.array(exterior.as_py()).reshape(-1, 2)
            arrays.append(arr if index is None else arr[:, index])
            arrays.append([[np.nan, np.nan]] if index is None else [np.nan])
        arrays = arrays[:-1]

    if multi:
        return arrays
    elif len(arrays) == 1:
        return arrays[0]
    else:
        return np.concatenate(arrays)


def geom_to_holes(geom):
    """Extracts holes from spatialpandas Polygon geometries.

    Args:
        geom: spatialpandas geometry

    Returns:
        List of arrays representing holes
    """
    from spatialpandas.geometry import Polygon, MultiPolygon
    if isinstance(geom, Polygon):
        holes = []
        for i, hole in enumerate(geom.data):
            if i == 0:
                continue
            holes.append(np.array(hole.as_py()).reshape(-1, 2))
        return [holes]
    elif isinstance(geom, MultiPolygon):
        holes = []
        for poly in geom.data:
            poly_holes = []
            for i, hole in enumerate(poly):
                if i == 0:
                    continue
                arr = np.array(hole.as_py()).reshape(-1, 2)
                poly_holes.append(arr)
            holes.append(poly_holes)
        return holes
    elif 'Multi' in type(geom).__name__:
        return [[]]*len(geom)
    else:
        return [[]]


def geom_array_to_array(geom_array, index):
    """Converts spatialpandas extension arrays to a flattened array.

    Args:
        geom: spatialpandas geometry
        index: The column index to return

    Returns:
        Flattened array
    """
    from spatialpandas.geometry import PointArray
    if isinstance(geom_array, PointArray):
        return geom_array.y if index else geom_array.x
    else:
        arrays = []
        for poly in geom_array:
            arrays.extend(geom_to_array(poly, index, multi=True))
            arrays.append([np.nan])
        return np.concatenate(arrays[:-1]) if arrays else np.array([])


def to_spatialpandas(data, xdim, ydim, ring=False):
    """Converts list of dictionary format geometries to spatialpandas line geometries.

    Args:
        data: List of dictionaries representing individual geometries
        xdim: Name of x-coordinates column
        ydim: Name of y-coordinates column
        ring: Whether the data represents a closed ring

    Returns:
        A spatialpandas.GeoDataFrame version of the data
    """
    from spatialpandas import GeoSeries, GeoDataFrame
    from spatialpandas.geometry import (
        Line, Polygon, Ring, MultiPolygon, MultiLine
    )
    poly = any('holes' in d for d in data)
    if poly:
        single_type, multi_type = Polygon, MultiPolygon
    elif ring:
        single_type, multi_type = Ring, MultiLine
    else:
        single_type, multi_type = Line, MultiLine

    lines = defaultdict(list)
    for path in data:
        path = dict(path)
        geom = np.column_stack([path.pop(xdim), path.pop(ydim)])
        splits = np.where(np.isnan(geom[:, :2].astype('float')).sum(axis=1))[0]
        paths = np.split(geom, splits+1) if len(splits) else [geom]
        parts = []
        for i, p in enumerate(paths):
            if i != (len(paths)-1):
                p = p[:-1]
            if len(p) < (3 if poly else 2):
                continue
            holes = path.pop('holes', None)
            for c, v in path.items():
                lines[c].append(v)
            if poly:
                parts.append([])
                subparts = parts[-1]
            else:
                subparts = parts
            subparts.append(p[:, :2])
            if poly:
                subparts += [np.array(h) for h in holes[i]]

        if len(parts) > 1:
            geom_type = multi_type
            parts = [[p.flatten() for p in sp] if poly else sp.flatten() for sp in parts]
        else:
            geom_type = single_type
            parts = [p.flatten() for p in parts[0]] if poly else parts[0].flatten()

        lines['geometry'].append(geom_type(parts))
    lines['geometry'] = GeoSeries(lines['geometry'])
    return GeoDataFrame(lines)


Interface.register(SpatialPandasInterface)
