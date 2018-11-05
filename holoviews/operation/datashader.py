from __future__ import absolute_import, division

from collections import Callable, Iterable
from distutils.version import LooseVersion
import warnings

import param
import numpy as np
import pandas as pd
import xarray as xr
import datashader as ds
import datashader.reductions as rd
import datashader.transfer_functions as tf
import dask.dataframe as dd
from param.parameterized import bothmethod

ds_version = LooseVersion(ds.__version__)

try:
    from datashader.bundling import (directly_connect_edges as connect_edges,
                                     hammer_bundle)
except:
    hammer_bundle, connect_edges = object, object

from ..core import (Operation, Element, Dimension, NdOverlay,
                    CompositeOverlay, Dataset, Overlay)
from ..core.data import PandasInterface, XArrayInterface
from ..core.sheetcoords import BoundingBox
from ..core.util import get_param_values, basestring, datetime_types, dt_to_int
from ..element import (Image, Path, Curve, RGB, Graph, TriMesh, QuadMesh, Contours)
from ..streams import RangeXY, PlotSize


class LinkableOperation(Operation):
    """
    Abstract baseclass for operations supporting linked inputs.
    """

    link_inputs = param.Boolean(default=True, doc="""
        By default, the link_inputs parameter is set to True so that
        when applying an operation, backends that support linked
        streams update RangeXY streams on the inputs of the operation.
        Disable when you do not want the resulting plot to be
        interactive, e.g. when trying to display an interactive plot a
        second time.""")


class ResamplingOperation(LinkableOperation):
    """
    Abstract baseclass for resampling operations
    """

    dynamic = param.Boolean(default=True, doc="""
       Enables dynamic processing by default.""")

    expand = param.Boolean(default=True, doc="""
       Whether the x_range and y_range should be allowed to expand
       beyond the extent of the data.  Setting this value to True is
       useful for the case where you want to ensure a certain size of
       output grid, e.g. if you are doing masking or other arithmetic
       on the grids.  A value of False ensures that the grid is only
       just as large as it needs to be to contain the data, which will
       be faster and use less memory if the resulting aggregate is
       being overlaid on a much larger background.""")

    height = param.Integer(default=400, doc="""
       The height of the output image in pixels.""")

    width = param.Integer(default=400, doc="""
       The width of the output image in pixels.""")

    x_range  = param.NumericTuple(default=None, length=2, doc="""
       The x_range as a tuple of min and max x-value. Auto-ranges
       if set to None.""")

    y_range  = param.NumericTuple(default=None, length=2, doc="""
       The x_range as a tuple of min and max y-value. Auto-ranges
       if set to None.""")

    x_sampling = param.Number(default=None, doc="""
        Specifies the smallest allowed sampling interval along the y-axis.""")

    y_sampling = param.Number(default=None, doc="""
        Specifies the smallest allowed sampling interval along the y-axis.""")

    target = param.ClassSelector(class_=Image, doc="""
        A target Image which defines the desired x_range, y_range,
        width and height.
    """)

    streams = param.List(default=[PlotSize, RangeXY], doc="""
        List of streams that are applied if dynamic=True, allowing
        for dynamic interaction with the plot.""")

    element_type = param.ClassSelector(class_=(Dataset,), instantiate=False,
                                        is_instance=False, default=Image,
                                        doc="""
        The type of the returned Elements, must be a 2D Dataset type.""")

    precompute = param.Boolean(default=False, doc="""
        Whether to apply precomputing operations. Precomputing can
        speed up resampling operations by avoiding unnecessary
        recomputation if the supplied element does not change between
        calls. The cost of enabling this option is that the memory
        used to represent this internal state is not freed between
        calls.""")

    @bothmethod
    def instance(self_or_cls,**params):
        inst = super(ResamplingOperation, self_or_cls).instance(**params)
        inst._precomputed = {}
        return inst

    def _get_sampling(self, element, x, y):
        target = self.p.target
        if target:
            x_range, y_range = target.range(x), target.range(y)
            height, width = target.dimension_values(2, flat=False).shape
        else:
            if x is None or y is None:
                x_range = self.p.x_range or (-0.5, 0.5)
                y_range = self.p.y_range or (-0.5, 0.5)
            else:
                if self.p.expand or not self.p.x_range:
                    x_range = self.p.x_range or element.range(x)
                else:
                    x0, x1 = self.p.x_range
                    ex0, ex1 = element.range(x)
                    x_range = (np.min([np.max([x0, ex0]), ex1]),
                               np.max([np.min([x1, ex1]), ex0]))

                if self.p.expand or not self.p.y_range:
                    y_range = self.p.y_range or element.range(y)
                else:
                    y0, y1 = self.p.y_range
                    ey0, ey1 = element.range(y)
                    y_range = (np.min([np.max([y0, ey0]), ey1]),
                               np.max([np.min([y1, ey1]), ey0]))
            width, height = self.p.width, self.p.height
        (xstart, xend), (ystart, yend) = x_range, y_range

        xtype = 'numeric'
        if isinstance(xstart, datetime_types) or isinstance(xend, datetime_types):
            xstart, xend = dt_to_int(xstart, 'ns'), dt_to_int(xend, 'ns')
            xtype = 'datetime'
        elif not np.isfinite(xstart) and not np.isfinite(xend):
            xstart, xend = 0, 0
            if element.get_dimension_type(x) in datetime_types:
                xtype = 'datetime'
        x_range = (xstart, xend)

        ytype = 'numeric'
        if isinstance(ystart, datetime_types) or isinstance(yend, datetime_types):
            ystart, yend = dt_to_int(ystart, 'ns'), dt_to_int(yend, 'ns')
            ytype = 'datetime'
        elif not np.isfinite(ystart) and not np.isfinite(yend):
            ystart, yend = 0, 0
            if element.get_dimension_type(y) in datetime_types:
                ytype = 'datetime'
        y_range = (ystart, yend)

        # Compute highest allowed sampling density
        xspan = xend - xstart
        yspan = yend - ystart
        if self.p.x_sampling:
            width = int(min([(xspan/self.p.x_sampling), width]))
        if self.p.y_sampling:
            height = int(min([(yspan/self.p.y_sampling), height]))
        if xstart == xend or width == 0:
            xunit, width = 0, 0
        else:
            xunit = float(xspan)/width
        if ystart == yend or height == 0:
            yunit, height = 0, 0
        else:
            yunit = float(yspan)/height
        xs, ys = (np.linspace(xstart+xunit/2., xend-xunit/2., width),
                  np.linspace(ystart+yunit/2., yend-yunit/2., height))

        return (x_range, y_range), (xs, ys), (width, height), (xtype, ytype)



class AggregationOperation(ResamplingOperation):
    """
    AggregationOperation extends the ResamplingOperation defining an
    aggregator parameter used to define a datashader Reduction.
    """

    aggregator = param.ClassSelector(class_=(ds.reductions.Reduction, basestring),
                                     default=ds.count(), doc="""
        Datashader reduction function used for aggregating the data.
        The aggregator may also define a column to aggregate; if
        no column is defined the first value dimension of the element
        will be used. May also be defined as a string.""")

    _agg_methods = {
        'any':   rd.any,
        'count': rd.count,
        'first': rd.first,
        'last':  rd.last,
        'mode':  rd.mode,
        'mean':  rd.mean,
        'sum':   rd.sum,
        'var':   rd.var,
        'std':   rd.std,
        'min':   rd.min,
        'max':   rd.max
    }

    def _get_aggregator(self, element, add_field=True):
        agg = self.p.aggregator
        if isinstance(agg, basestring):
            if agg not in self._agg_methods:
                agg_methods = sorted(self._agg_methods)
                raise ValueError('Aggregation method %r is not known; '
                                 'aggregator must be one of: %r' %
                                 (agg, agg_methods))
            agg = self._agg_methods[agg]()

        elements = element.traverse(lambda x: x, [Element])
        if add_field and agg.column is None and not isinstance(agg, (rd.count, rd.any)):
            if not elements:
                raise ValueError('Could not find any elements to apply '
                                 '%s operation to.' % type(self).__name__)
            inner_element = elements[0]
            if isinstance(inner_element, TriMesh) and inner_element.nodes.vdims:
                field = inner_element.nodes.vdims[0].name
            elif inner_element.vdims:
                field = inner_element.vdims[0].name
            elif isinstance(element, NdOverlay):
                field = element.kdims[0].name
            else:
                raise ValueError('Could not determine dimension to apply '
                                 '%s operation to. Declare the dimension '
                                 'to aggregate as part of the datashader '
                                 'aggregator.' % type(self).__name__)
            agg = type(agg)(field)
        return agg


class aggregate(AggregationOperation):
    """
    aggregate implements 2D binning for any valid HoloViews Element
    type using datashader. I.e., this operation turns a HoloViews
    Element or overlay of Elements into an Image or an overlay of
    Images by rasterizing it. This allows quickly aggregating large
    datasets computing a fixed-sized representation independent
    of the original dataset size.

    By default it will simply count the number of values in each bin
    but other aggregators can be supplied implementing mean, max, min
    and other reduction operations.

    The bins of the aggregate are defined by the width and height and
    the x_range and y_range. If x_sampling or y_sampling are supplied
    the operation will ensure that a bin is no smaller than the minimum
    sampling distance by reducing the width and height when zoomed in
    beyond the minimum sampling distance.

    By default, the PlotSize stream is applied when this operation
    is used dynamically, which means that the height and width
    will automatically be set to match the inner dimensions of
    the linked plot.
    """


    @classmethod
    def get_agg_data(cls, obj, category=None):
        """
        Reduces any Overlay or NdOverlay of Elements into a single
        xarray Dataset that can be aggregated.
        """
        paths = []
        if isinstance(obj, Graph):
            obj = obj.edgepaths
        kdims = list(obj.kdims)
        vdims = list(obj.vdims)
        dims = obj.dimensions()[:2]
        if isinstance(obj, Path):
            glyph = 'line'
            for p in obj.split(datatype='dataframe'):
                paths.append(p)
        elif isinstance(obj, CompositeOverlay):
            element = None
            for key, el in obj.data.items():
                x, y, element, glyph = cls.get_agg_data(el)
                dims = (x, y)
                df = PandasInterface.as_dframe(element)
                if isinstance(obj, NdOverlay):
                    df = df.assign(**dict(zip(obj.dimensions('key', True), key)))
                paths.append(df)
            if element is None:
                dims = None
            else:
                kdims += element.kdims
                vdims = element.vdims
        elif isinstance(obj, Element):
            glyph = 'line' if isinstance(obj, Curve) else 'points'
            paths.append(PandasInterface.as_dframe(obj))

        if dims is None or len(dims) != 2:
            return None, None, None, None
        else:
            x, y = dims

        if len(paths) > 1:
            if glyph == 'line':
                path = paths[0][:1]
                if isinstance(path, dd.DataFrame):
                    path = path.compute()
                empty = path.copy()
                empty.iloc[0, :] = (np.NaN,) * empty.shape[1]
                paths = [elem for p in paths for elem in (p, empty)][:-1]
            if all(isinstance(path, dd.DataFrame) for path in paths):
                df = dd.concat(paths)
            else:
                paths = [p.compute() if isinstance(p, dd.DataFrame) else p for p in paths]
                df = pd.concat(paths)
        else:
            df = paths[0] if paths else pd.DataFrame([], columns=[x.name, y.name])
        if category and df[category].dtype.name != 'category':
            df[category] = df[category].astype('category')

        if any(df[d.name].dtype.kind == 'M' for d in (x, y)):
            df = df.copy()
        for d in (x, y):
            if df[d.name].dtype.kind == 'M':
                df[d.name] = df[d.name].astype('datetime64[ns]').astype('int64') * 1000.
        return x, y, Dataset(df, kdims=kdims, vdims=vdims), glyph


    def _aggregate_ndoverlay(self, element, agg_fn):
        """
        Optimized aggregation for NdOverlay objects by aggregating each
        Element in an NdOverlay individually avoiding having to concatenate
        items in the NdOverlay. Works by summing sum and count aggregates and
        applying appropriate masking for NaN values. Mean aggregation
        is also supported by dividing sum and count aggregates. count_cat
        aggregates are grouped by the categorical dimension and a separate
        aggregate for each category is generated.
        """
        # Compute overall bounds
        x, y = element.last.dimensions()[0:2]
        info = self._get_sampling(element, x, y)
        (x_range, y_range), (xs, ys), (width, height), (xtype, ytype) = info
        if xtype == 'datetime':
            x_range = tuple((np.array(x_range)/10e5).astype('datetime64[us]'))
        if ytype == 'datetime':
            y_range = tuple((np.array(y_range)/10e5).astype('datetime64[us]'))
        agg_params = dict({k: v for k, v in dict(self.get_param_values(), **self.p).items()
                           if k in aggregate.params()},
                          x_range=x_range, y_range=y_range)
        bbox = BoundingBox(points=[(x_range[0], y_range[0]), (x_range[1], y_range[1])])

        # Optimize categorical counts by aggregating them individually
        if isinstance(agg_fn, ds.count_cat):
            agg_params.update(dict(dynamic=False, aggregator=ds.count()))
            agg_fn1 = aggregate.instance(**agg_params)
            if element.ndims == 1:
                grouped = element
            else:
                grouped = element.groupby([agg_fn.column], container_type=NdOverlay,
                                          group_type=NdOverlay)
            groups = []
            for k, v in grouped.items():
                agg = agg_fn1(v)
                groups.append((k, agg.clone(agg.data, bounds=bbox)))
            return grouped.clone(groups)

        # Create aggregate instance for sum, count operations, breaking mean
        # into two aggregates
        column = agg_fn.column or 'Count'
        if isinstance(agg_fn, ds.mean):
            agg_fn1 = aggregate.instance(**dict(agg_params, aggregator=ds.sum(column)))
            agg_fn2 = aggregate.instance(**dict(agg_params, aggregator=ds.count()))
        else:
            agg_fn1 = aggregate.instance(**agg_params)
            agg_fn2 = None
        is_sum = isinstance(agg_fn1.aggregator, ds.sum)

        # Accumulate into two aggregates and mask
        agg, agg2, mask = None, None, None
        mask = None
        for v in element:
            # Compute aggregates and mask
            new_agg = agg_fn1.process_element(v, None)
            if is_sum:
                new_mask = np.isnan(new_agg.data[column].values)
                new_agg.data = new_agg.data.fillna(0)
            if agg_fn2:
                new_agg2 = agg_fn2.process_element(v, None)

            if agg is None:
                agg = new_agg
                if is_sum: mask = new_mask
                if agg_fn2: agg2 = new_agg2
            else:
                agg.data += new_agg.data
                if is_sum: mask &= new_mask
                if agg_fn2: agg2.data += new_agg2.data

        # Divide sum by count to compute mean
        if agg2 is not None:
            agg2.data.rename({'Count': agg_fn.column}, inplace=True)
            with np.errstate(divide='ignore', invalid='ignore'):
                agg.data /= agg2.data

        # Fill masked with with NaNs
        if is_sum:
            agg.data[column].values[mask] = np.NaN

        return agg.clone(bounds=bbox)


    def _process(self, element, key=None):
        agg_fn = self._get_aggregator(element)
        category = agg_fn.column if isinstance(agg_fn, ds.count_cat) else None

        if (isinstance(element, NdOverlay) and
            ((isinstance(agg_fn, (ds.count, ds.sum, ds.mean)) and
              (agg_fn.column is None or agg_fn.column not in element.kdims)) or
             (isinstance(agg_fn, ds.count_cat) and agg_fn.column in element.kdims))):
            return self._aggregate_ndoverlay(element, agg_fn)

        if element._plot_id in self._precomputed:
            x, y, data, glyph = self._precomputed[element._plot_id]
        else:
            x, y, data, glyph = self.get_agg_data(element, category)

        if self.p.precompute:
            self._precomputed[element._plot_id] = x, y, data, glyph
        (x_range, y_range), (xs, ys), (width, height), (xtype, ytype) = self._get_sampling(element, x, y)

        (x0, x1), (y0, y1) = x_range, y_range
        if xtype == 'datetime':
            x0, x1 = (np.array([x0, x1])/10e5).astype('datetime64[us]')
            xs = (xs/10e5).astype('datetime64[us]')
        if ytype == 'datetime':
            y0, y1 = (np.array([y0, y1])/10e5).astype('datetime64[us]')
            ys = (ys/10e5).astype('datetime64[us]')
        bounds = (x0, y0, x1, y1)
        params = dict(get_param_values(element), kdims=[x, y],
                      datatype=['xarray'], bounds=bounds)

        column = agg_fn.column if agg_fn else None
        if column:
            dims = [d for d in element.dimensions('ranges') if d == column]
            if not dims:
                raise ValueError("Aggregation column %s not found on %s element. "
                                 "Ensure the aggregator references an existing "
                                 "dimension." % (column,element))
            name = '%s Count' % column if isinstance(agg_fn, ds.count_cat) else column
            vdims = [dims[0](name)]
        else:
            vdims = Dimension('Count')
        params['vdims'] = vdims

        if x is None or y is None or width == 0 or height == 0:
            x = x.name if x else 'x'
            y = y.name if x else 'y'
            xarray = xr.DataArray(np.full((height, width), np.NaN),
                                  dims=[y, x], coords={x: xs, y: ys})
            if width == 0:
                params['xdensity'] = 1
            if height == 0:
                params['ydensity'] = 1
            el = self.p.element_type(xarray, **params)
            if isinstance(agg_fn, ds.count_cat):
                vals = element.dimension_values(agg_fn.column, expanded=False)
                dim = element.get_dimension(agg_fn.column)
                return NdOverlay({v: el for v in vals}, dim)
            return el
        elif not len(data):
            xarray = xr.DataArray(np.full((height, width), np.NaN),
                                  dims=[y.name, x.name], coords={x.name: xs, y.name: ys})
            return self.p.element_type(xarray, **params)

        cvs = ds.Canvas(plot_width=width, plot_height=height,
                        x_range=x_range, y_range=y_range)

        dfdata = PandasInterface.as_dframe(data)
        agg = getattr(cvs, glyph)(dfdata, x.name, y.name, agg_fn)
        if 'x_axis' in agg.coords and 'y_axis' in agg.coords:
            agg = agg.rename({'x_axis': x, 'y_axis': y})
        if xtype == 'datetime':
            agg[x.name] = (agg[x.name]/10e5).astype('datetime64[us]')
        if ytype == 'datetime':
            agg[y.name] = (agg[y.name]/10e5).astype('datetime64[us]')

        if agg.ndim == 2:
            # Replacing x and y coordinates to avoid numerical precision issues
            eldata = agg if ds_version > '0.5.0' else (xs, ys, agg.data)
            return self.p.element_type(eldata, **params)
        else:
            layers = {}
            for c in agg.coords[column].data:
                cagg = agg.sel(**{column: c})
                eldata = cagg if ds_version > '0.5.0' else (xs, ys, cagg.data)
                layers[c] = self.p.element_type(eldata, **dict(params, vdims=vdims))
            return NdOverlay(layers, kdims=[data.get_dimension(column)])



class regrid(AggregationOperation):
    """
    regrid allows resampling a HoloViews Image type using specified
    up- and downsampling functions defined using the aggregator and
    interpolation parameters respectively. By default upsampling is
    disabled to avoid unnecessarily upscaling an image that has to be
    sent to the browser. Also disables expanding the image beyond its
    original bounds avoiding unnecessarily padding the output array
    with NaN values.
    """

    aggregator = param.ClassSelector(default=ds.mean(),
                                     class_=(ds.reductions.Reduction, basestring))

    expand = param.Boolean(default=False, doc="""
       Whether the x_range and y_range should be allowed to expand
       beyond the extent of the data.  Setting this value to True is
       useful for the case where you want to ensure a certain size of
       output grid, e.g. if you are doing masking or other arithmetic
       on the grids.  A value of False ensures that the grid is only
       just as large as it needs to be to contain the data, which will
       be faster and use less memory if the resulting aggregate is
       being overlaid on a much larger background.""")

    interpolation = param.ObjectSelector(default='nearest',
        objects=['linear', 'nearest'], doc="""
        Interpolation method""")

    upsample = param.Boolean(default=False, doc="""
        Whether to allow upsampling if the source array is smaller
        than the requested array. Setting this value to True will
        enable upsampling using the interpolation method, when the
        requested width and height are larger than what is available
        on the source grid. If upsampling is disabled (the default)
        the width and height are clipped to what is available on the
        source array.""")

    def _get_xarrays(self, element, coords, xtype, ytype):
        x, y = element.kdims
        dims = [y.name, x.name]
        irregular = any(element.interface.irregular(element, d)
                        for d in dims)
        if irregular:
            coord_dict = {x.name: (('y', 'x'), coords[0]),
                          y.name: (('y', 'x'), coords[1])}
        else:
            coord_dict = {x.name: coords[0], y.name: coords[1]}

        arrays = {}
        for vd in element.vdims:
            if element.interface is XArrayInterface:
                xarr = element.data[vd.name]
                if 'datetime' in (xtype, ytype):
                    xarr = xarr.copy()
                if dims != xarr.dims and not irregular:
                    xarr = xarr.transpose(*dims)
            elif irregular:
                arr = element.dimension_values(vd, flat=False)
                xarr = xr.DataArray(arr, coords=coord_dict, dims=['y', 'x'])
            else:
                arr = element.dimension_values(vd, flat=False)
                xarr = xr.DataArray(arr, coords=coord_dict, dims=dims)
            if xtype == "datetime":
                xarr[x.name] = [dt_to_int(v, 'ns') for v in xarr[x.name].values]
            if ytype == "datetime":
                xarr[y.name] = [dt_to_int(v, 'ns') for v in xarr[y.name].values]
            arrays[vd.name] = xarr
        return arrays


    def _process(self, element, key=None):
        if ds_version <= '0.5.0':
            raise RuntimeError('regrid operation requires datashader>=0.6.0')

        # Compute coords, anges and size
        x, y = element.kdims
        coords = tuple(element.dimension_values(d, expanded=False) for d in [x, y])
        info = self._get_sampling(element, x, y)
        (x_range, y_range), (xs, ys), (width, height), (xtype, ytype) = info

        # Disable upsampling by clipping size and ranges
        (xstart, xend), (ystart, yend) = (x_range, y_range)
        xspan, yspan = (xend-xstart), (yend-ystart)
        if not self.p.upsample and self.p.target is None:
            (x0, x1), (y0, y1) = element.range(0), element.range(1)
            if isinstance(x0, datetime_types):
                x0, x1 = dt_to_int(x0, 'ns'), dt_to_int(x1, 'ns')
            if isinstance(y0, datetime_types):
                y0, y1 = dt_to_int(y0, 'ns'), dt_to_int(y1, 'ns')
            exspan, eyspan = (x1-x0), (y1-y0)
            if np.isfinite(exspan) and exspan > 0:
                width = min([int((xspan/exspan) * len(coords[0])), width])
            else:
                width = 0
            if np.isfinite(eyspan) and eyspan > 0:
                height = min([int((yspan/eyspan) * len(coords[1])), height])
            else:
                height = 0
            xunit = float(xspan)/width if width else 0
            yunit = float(yspan)/height if height else 0
            xs, ys = (np.linspace(xstart+xunit/2., xend-xunit/2., width),
                      np.linspace(ystart+yunit/2., yend-yunit/2., height))

        # Compute bounds (converting datetimes)
        if xtype == 'datetime':
            xstart, xend = (np.array([xstart, xend])/10e5).astype('datetime64[us]')
            xs = (xs/10e5).astype('datetime64[us]')
        if ytype == 'datetime':
            ystart, yend = (np.array([ystart, yend])/10e5).astype('datetime64[us]')
            ys = (ys/10e5).astype('datetime64[us]')
        bbox = BoundingBox(points=[(xstart, ystart), (xend, yend)])

        params = dict(bounds=bbox)
        if width == 0 or height == 0:
            if width == 0: params['xdensity'] = 1
            if height == 0: params['ydensity'] = 1
            return element.clone((xs, ys, np.zeros((height, width))), **params)

        cvs = ds.Canvas(plot_width=width, plot_height=height,
                        x_range=x_range, y_range=y_range)

        # Apply regridding to each value dimension
        regridded = {}
        arrays = self._get_xarrays(element, coords, xtype, ytype)
        agg_fn = self._get_aggregator(element, add_field=False)
        for vd, xarr in arrays.items():
            rarray = cvs.raster(xarr, upsample_method=self.p.interpolation,
                                downsample_method=agg_fn)

            # Convert datetime coordinates
            if xtype == "datetime":
                rarray[x.name] = (rarray[x.name]/10e5).astype('datetime64[us]')
            if ytype == "datetime":
                rarray[y.name] = (rarray[y.name]/10e5).astype('datetime64[us]')
            regridded[vd] = rarray
        regridded = xr.Dataset(regridded)

        return element.clone(regridded, bounds=bbox, datatype=['xarray']+element.datatype)



class contours_rasterize(aggregate):
    """
    Rasterizes the Contours element by weighting the aggregation by
    the iso-contour levels if a value dimension is defined, otherwise
    default to any aggregator.
    """

    aggregator = param.ClassSelector(default=ds.mean(),
                                     class_=(ds.reductions.Reduction, basestring))

    def _get_aggregator(self, element, add_field=True):
        agg = self.p.aggregator
        if not element.vdims and agg.column is None and not isinstance(agg, (rd.count, rd.any)):
            return ds.any()
        return super(contours_rasterize, self)._get_aggregator(element, add_field)



class trimesh_rasterize(aggregate):
    """
    Rasterize the TriMesh element using the supplied aggregator. If
    the TriMesh nodes or edges define a value dimension, will plot
    filled and shaded polygons; otherwise returns a wiremesh of the
    data.
    """

    aggregator = param.ClassSelector(default=ds.mean(),
                                     class_=(ds.reductions.Reduction, basestring))

    interpolation = param.ObjectSelector(default='bilinear',
                                         objects=['bilinear', None], doc="""
        The interpolation method to apply during rasterization.""")

    def _precompute(self, element, agg):
        from datashader.utils import mesh
        if element.vdims and getattr(agg, 'column', None) not in element.nodes.vdims:
            simplices = element.dframe([0, 1, 2, 3])
            verts = element.nodes.dframe([0, 1])
        elif element.nodes.vdims:
            simplices = element.dframe([0, 1, 2])
            verts = element.nodes.dframe([0, 1, 3])
        for c, dtype in zip(simplices.columns[:3], simplices.dtypes):
            if dtype.kind != 'i':
                simplices[c] = simplices[c].astype('int')
        return {'mesh': mesh(verts, simplices), 'simplices': simplices,
                'vertices': verts}


    def _process(self, element, key=None):
        if isinstance(element, TriMesh):
            x, y = element.nodes.kdims[:2]
        else:
            x, y = element.kdims
        info = self._get_sampling(element, x, y)
        (x_range, y_range), (xs, ys), (width, height), (xtype, ytype) = info

        agg = self.p.aggregator
        if getattr(agg, 'column', None):
            if agg.column in element.vdims:
                vdim = element.get_dimension(agg.column)
            elif isinstance(element, TriMesh) and agg.column in element.nodes.vdims:
                vdim = element.nodes.get_dimension(agg.column)
            else:
                raise ValueError("Aggregation column %s not found on TriMesh element."
                                 % agg.column)
        elif not (element.vdims or (isinstance(element, TriMesh) and element.nodes.vdims)):
            self.p.aggregator = ds.count() if not isinstance(agg, ds.any) else agg
            return aggregate._process(self, element, key)
        else:
            if isinstance(element, TriMesh) and element.nodes.vdims:
                vdim = element.nodes.vdims[0]
            else:
                vdim = element.vdims[0]
            agg = self._get_aggregator(element)

        if element._plot_id in self._precomputed:
            precomputed = self._precomputed[element._plot_id]
        else:
            precomputed = self._precompute(element, agg)

        params = dict(get_param_values(element), kdims=[x, y],
                      datatype=['xarray'], vdims=[vdim])

        if width == 0 or height == 0:
            if width == 0: params['xdensity'] = 1
            if height == 0: params['ydensity'] = 1
            bounds = (x_range[0], y_range[0], x_range[1], y_range[1])
            return Image((xs, ys, np.zeros((height, width))), bounds=bounds, **params)

        simplices = precomputed['simplices']
        pts = precomputed['vertices']
        mesh = precomputed['mesh']
        if self.p.precompute:
            self._precomputed = {element._plot_id: precomputed}

        cvs = ds.Canvas(plot_width=width, plot_height=height,
                        x_range=x_range, y_range=y_range)
        interpolate = bool(self.p.interpolation)
        agg = cvs.trimesh(pts, simplices, agg=agg,
                          interp=interpolate, mesh=mesh)
        return Image(agg, **params)



class quadmesh_rasterize(trimesh_rasterize):
    """
    Rasterize the QuadMesh element using the supplied aggregator.
    Simply converts to a TriMesh and lets trimesh_rasterize
    handle the actual rasterization.
    """

    def _precompute(self, element, agg):
        return super(quadmesh_rasterize, self)._precompute(element.trimesh(), agg)



class rasterize(AggregationOperation):
    """
    Rasterize is a high-level operation that will rasterize any
    Element or combination of Elements, aggregating them with the supplied
    aggregator and interpolation method.

    The default aggregation method depends on the type of Element but
    usually defaults to the count of samples in each bin. Other
    aggregators can be supplied implementing mean, max, min and other
    reduction operations.

    The bins of the aggregate are defined by the width and height and
    the x_range and y_range. If x_sampling or y_sampling are supplied
    the operation will ensure that a bin is no smaller than the minimum
    sampling distance by reducing the width and height when zoomed in
    beyond the minimum sampling distance.

    By default, the PlotSize and RangeXY streams are applied when this
    operation is used dynamically, which means that the width, height,
    x_range and y_range will automatically be set to match the inner
    dimensions of the linked plot and the ranges of the axes.
    """

    aggregator = param.ClassSelector(class_=(ds.reductions.Reduction, basestring),
                                     default=None)

    interpolation = param.ObjectSelector(default='bilinear',
                                         objects=['bilinear', None], doc="""
        The interpolation method to apply during rasterization.""")

    _transforms = [(Image, regrid),
                   (TriMesh, trimesh_rasterize),
                   (QuadMesh, quadmesh_rasterize),
                   (lambda x: (isinstance(x, NdOverlay) and
                               issubclass(x.type, Dataset)
                               and not issubclass(x.type, Image)),
                    aggregate),
                   (Contours, contours_rasterize),
                   (lambda x: (isinstance(x, Dataset) and
                               (not isinstance(x, Image))),
                    aggregate)]

    def _process(self, element, key=None):
        for predicate, transform in self._transforms:
            op_params = dict({k: v for k, v in self.p.items()
                              if k in transform.params() and v is not None},
                             dynamic=False)
            op = transform.instance(**op_params)
            op._precomputed = self._precomputed
            element = element.map(op, predicate)
            self._precomputed = op._precomputed
        return element



class shade(LinkableOperation):
    """
    shade applies a normalization function followed by colormapping to
    an Image or NdOverlay of Images, returning an RGB Element.
    The data must be in the form of a 2D or 3D DataArray, but NdOverlays
    of 2D Images will be automatically converted to a 3D array.

    In the 2D case data is normalized and colormapped, while a 3D
    array representing categorical aggregates will be supplied a color
    key for each category. The colormap (cmap) for the 2D case may be
    supplied as an Iterable or a Callable.
    """

    cmap = param.ClassSelector(class_=(Iterable, Callable, dict), doc="""
        Iterable or callable which returns colors as hex colors
        or web color names (as defined by datashader), to be used
        for the colormap of single-layer datashader output.
        Callable type must allow mapping colors between 0 and 1.
        The default value of None reverts to Datashader's default
        colormap.""")

    color_key = param.ClassSelector(class_=(Iterable, Callable, dict), doc="""
        Iterable or callable that returns colors as hex colors, to
        be used for the color key of categorical datashader output.
        Callable type must allow mapping colors for supplied values
        between 0 and 1.""")

    normalization = param.ClassSelector(default='eq_hist',
                                        class_=(basestring, Callable),
                                        doc="""
        The normalization operation applied before colormapping.
        Valid options include 'linear', 'log', 'eq_hist', 'cbrt',
        and any valid transfer function that accepts data, mask, nbins
        arguments.""")

    clims = param.NumericTuple(default=None, length=2, doc="""
        Min and max data values to use for colormap interpolation, when
        wishing to override autoranging.
        """)

    min_alpha = param.Number(default=40, doc="""
        The minimum alpha value to use for non-empty pixels when doing
        colormapping, in [0, 255].  Use a higher value to avoid
        undersaturation, i.e. poorly visible low-value datapoints, at
        the expense of the overall dynamic range..""")

    @classmethod
    def concatenate(cls, overlay):
        """
        Concatenates an NdOverlay of Image types into a single 3D
        xarray Dataset.
        """
        if not isinstance(overlay, NdOverlay):
            raise ValueError('Only NdOverlays can be concatenated')
        xarr = xr.concat([v.data.transpose() for v in overlay.values()],
                         pd.Index(overlay.keys(), name=overlay.kdims[0].name))
        params = dict(get_param_values(overlay.last),
                      vdims=overlay.last.vdims,
                      kdims=overlay.kdims+overlay.last.kdims)
        return Dataset(xarr.transpose(), datatype=['xarray'], **params)


    @classmethod
    def uint32_to_uint8(cls, img):
        """
        Cast uint32 RGB image to 4 uint8 channels.
        """
        return np.flipud(img.view(dtype=np.uint8).reshape(img.shape + (4,)))


    @classmethod
    def rgb2hex(cls, rgb):
        """
        Convert RGB(A) tuple to hex.
        """
        if len(rgb) > 3:
            rgb = rgb[:-1]
        return "#{0:02x}{1:02x}{2:02x}".format(*(int(v*255) for v in rgb))


    @classmethod
    def to_xarray(cls, element):
        if issubclass(element.interface, XArrayInterface):
            return element
        data = tuple(element.dimension_values(kd, expanded=False)
                     for kd in element.kdims)
        data += tuple(element.dimension_values(vd, flat=False)
                      for vd in element.vdims)
        dtypes = [dt for dt in element.datatype if dt != 'xarray']
        return element.clone(data, datatype=['xarray']+dtypes,
                             bounds=element.bounds,
                             xdensity=element.xdensity,
                             ydensity=element.ydensity)


    def _process(self, element, key=None):
        element = element.map(self.to_xarray, Image)
        if isinstance(element, NdOverlay):
            bounds = element.last.bounds
            xdensity = element.last.xdensity
            ydensity = element.last.ydensity
            element = self.concatenate(element)
        elif isinstance(element, Overlay):
            return element.map(self._process, [Element])
        else:
            xdensity = element.xdensity
            ydensity = element.ydensity
            bounds = element.bounds

        vdim = element.vdims[0].name
        array = element.data[vdim]
        kdims = element.kdims

        # Compute shading options depending on whether
        # it is a categorical or regular aggregate
        shade_opts = dict(how=self.p.normalization, min_alpha=self.p.min_alpha)
        if element.ndims > 2:
            kdims = element.kdims[1:]
            categories = array.shape[-1]
            if not self.p.color_key:
                pass
            elif isinstance(self.p.color_key, dict):
                shade_opts['color_key'] = self.p.color_key
            elif isinstance(self.p.color_key, Iterable):
                shade_opts['color_key'] = [c for i, c in
                                           zip(range(categories), self.p.color_key)]
            else:
                colors = [self.p.color_key(s) for s in np.linspace(0, 1, categories)]
                shade_opts['color_key'] = map(self.rgb2hex, colors)
        elif not self.p.cmap:
            pass
        elif isinstance(self.p.cmap, Callable):
            colors = [self.p.cmap(s) for s in np.linspace(0, 1, 256)]
            shade_opts['cmap'] = map(self.rgb2hex, colors)
        else:
            shade_opts['cmap'] = self.p.cmap

        if self.p.clims:
            shade_opts['span'] = self.p.clims
        elif ds_version > '0.5.0' and self.p.normalization != 'eq_hist':
            shade_opts['span'] = element.range(vdim)

        for d in kdims:
            if array[d.name].dtype.kind == 'M':
                array[d.name] = array[d.name].astype('datetime64[ns]').astype('int64') * 10e-4

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'invalid value encountered in true_divide')
            if np.isnan(array.data).all():
                arr = np.zeros(array.data.shape, dtype=np.uint32)
                img = array.copy()
                img.data = arr
            else:
                img = tf.shade(array, **shade_opts)
        params = dict(get_param_values(element), kdims=kdims,
                      bounds=bounds, vdims=RGB.vdims[:],
                      xdensity=xdensity, ydensity=ydensity)
        return RGB(self.uint32_to_uint8(img.data), **params)



class datashade(rasterize, shade):
    """
    Applies the aggregate and shade operations, aggregating all
    elements in the supplied object and then applying normalization
    and colormapping the aggregated data returning RGB elements.

    See aggregate and shade operations for more details.
    """

    def _process(self, element, key=None):
        agg = rasterize._process(self, element, key)
        shaded = shade._process(self, agg, key)
        return shaded



class stack(Operation):
    """
    The stack operation allows compositing multiple RGB Elements using
    the defined compositing operator.
    """

    compositor = param.ObjectSelector(objects=['add', 'over', 'saturate', 'source'],
                                      default='over', doc="""
        Defines how the compositing operation combines the images""")

    def uint8_to_uint32(self, element):
        img = np.dstack([element.dimension_values(d, flat=False)
                         for d in element.vdims])
        if img.shape[2] == 3: # alpha channel not included
            alpha = np.ones(img.shape[:2])
            if img.dtype.name == 'uint8':
                alpha = (alpha*255).astype('uint8')
            img = np.dstack([img, alpha])
        if img.dtype.name != 'uint8':
            img = (img*255).astype(np.uint8)
        N, M, _ = img.shape
        return img.view(dtype=np.uint32).reshape((N, M))

    def _process(self, overlay, key=None):
        if not isinstance(overlay, CompositeOverlay):
            return overlay
        elif len(overlay) == 1:
            return overlay.last if isinstance(overlay, NdOverlay) else overlay.get(0)

        imgs = []
        for rgb in overlay:
            if not isinstance(rgb, RGB):
                raise TypeError('stack operation expect RGB type elements, '
                                'not %s name.' % type(rgb).__name__)
            rgb = rgb.rgb
            dims = [kd.name for kd in rgb.kdims][::-1]
            coords = {kd.name: rgb.dimension_values(kd, False)
                      for kd in rgb.kdims}
            imgs.append(tf.Image(self.uint8_to_uint32(rgb), coords=coords, dims=dims))

        try:
            imgs = xr.align(*imgs, join='exact')
        except ValueError:
            raise ValueError('RGB inputs to stack operation could not be aligned, '
                             'ensure they share the same grid sampling.')

        stacked = tf.stack(*imgs, how=self.p.compositor)
        arr = shade.uint32_to_uint8(stacked.data)[::-1]
        data = (coords[dims[1]], coords[dims[0]], arr[:, :, 0],
                arr[:, :, 1], arr[:, :, 2])
        if arr.shape[-1] == 4:
            data = data + (arr[:, :, 3],)
        return rgb.clone(data, datatype=[rgb.interface.datatype]+rgb.datatype)



class SpreadingOperation(LinkableOperation):
    """
    Spreading expands each pixel in an Image based Element a certain
    number of pixels on all sides according to a given shape, merging
    pixels using a specified compositing operator. This can be useful
    to make sparse plots more visible.
    """

    how = param.ObjectSelector(default='source',
            objects=['source', 'over', 'saturate', 'add'], doc="""
        The name of the compositing operator to use when combining
        pixels.""")

    shape = param.ObjectSelector(default='circle', objects=['circle', 'square'],
                                 doc="""
        The shape to spread by. Options are 'circle' [default] or 'square'.""")

    @classmethod
    def uint8_to_uint32(cls, img):
        shape = img.shape
        flat_shape = np.multiply.reduce(shape[:2])
        rgb = img.reshape((flat_shape, 4)).view('uint32').reshape(shape[:2])
        return rgb

    def _apply_spreading(self, array):
        """Apply the spread function using the indicated parameters."""
        raise NotImplementedError

    def _process(self, element, key=None):
        if not isinstance(element, RGB):
            raise ValueError('spreading can only be applied to RGB Elements.')
        rgb = element.rgb
        new_data = {kd.name: rgb.dimension_values(kd, expanded=False)
                    for kd in rgb.kdims}
        rgbarray = np.dstack([element.dimension_values(vd, flat=False)
                              for vd in element.vdims])
        data = self.uint8_to_uint32(rgbarray)
        array = self._apply_spreading(data)
        img = datashade.uint32_to_uint8(array)
        for i, vd in enumerate(element.vdims):
            if i < img.shape[-1]:
                new_data[vd.name] = np.flipud(img[..., i])
        return element.clone(new_data)



class spread(SpreadingOperation):
    """
    Spreading expands each pixel in an Image based Element a certain
    number of pixels on all sides according to a given shape, merging
    pixels using a specified compositing operator. This can be useful
    to make sparse plots more visible.

    See the datashader documentation for more detail:

    http://datashader.org/api.html#datashader.transfer_functions.spread
    """

    px = param.Integer(default=1, doc="""
        Number of pixels to spread on all sides.""")

    def _apply_spreading(self, array):
        img = tf.Image(array)
        return tf.spread(img, px=self.p.px,
                         how=self.p.how, shape=self.p.shape).data


class dynspread(SpreadingOperation):
    """
    Spreading expands each pixel in an Image based Element a certain
    number of pixels on all sides according to a given shape, merging
    pixels using a specified compositing operator. This can be useful
    to make sparse plots more visible. Dynamic spreading determines
    how many pixels to spread based on a density heuristic.

    See the datashader documentation for more detail:

    http://datashader.org/api.html#datashader.transfer_functions.dynspread
    """

    max_px = param.Integer(default=3, doc="""
        Maximum number of pixels to spread on all sides.""")

    threshold = param.Number(default=0.5, bounds=(0,1), doc="""
        When spreading, determines how far to spread.
        Spreading starts at 1 pixel, and stops when the fraction
        of adjacent non-empty pixels reaches this threshold.
        Higher values give more spreading, up to the max_px
        allowed.""")

    def _apply_spreading(self, array):
        img = tf.Image(array)
        return tf.dynspread(img, max_px=self.p.max_px,
                            threshold=self.p.threshold,
                            how=self.p.how, shape=self.p.shape).data


def split_dataframe(path_df):
    """
    Splits a dataframe of paths separated by NaNs into individual
    dataframes.
    """
    splits = np.where(path_df.iloc[:, 0].isnull())[0]+1
    return [df for df in np.split(path_df, splits) if len(df) > 1]


class _connect_edges(Operation):

    split = param.Boolean(default=False, doc="""
        Determines whether bundled edges will be split into individual edges
        or concatenated with NaN separators.""")

    def _bundle(self, position_df, edges_df):
        raise NotImplementedError('_connect_edges is an abstract baseclass '
                                  'and does not implement any actual bundling.')

    def _process(self, element, key=None):
        index = element.nodes.kdims[2].name
        rename_edges = {d.name: v for d, v in zip(element.kdims[:2], ['source', 'target'])}
        rename_nodes = {d.name: v for d, v in zip(element.nodes.kdims[:2], ['x', 'y'])}
        position_df = element.nodes.redim(**rename_nodes).dframe([0, 1, 2]).set_index(index)
        edges_df = element.redim(**rename_edges).dframe([0, 1])
        paths = self._bundle(position_df, edges_df)
        paths = paths.rename(columns={v: k for k, v in rename_nodes.items()})
        paths = split_dataframe(paths) if self.p.split else [paths]
        return element.clone((element.data, element.nodes, paths))


class bundle_graph(_connect_edges, hammer_bundle):
    """
    Iteratively group edges and return as paths suitable for datashading.

    Breaks each edge into a path with multiple line segments, and
    iteratively curves this path to bundle edges into groups.
    """

    def _bundle(self, position_df, edges_df):
        from datashader.bundling import hammer_bundle
        return hammer_bundle.__call__(self, position_df, edges_df, **self.p)


class directly_connect_edges(_connect_edges, connect_edges):
    """
    Given a Graph object will directly connect all nodes.
    """

    def _bundle(self, position_df, edges_df):
        return connect_edges.__call__(self, position_df, edges_df)
