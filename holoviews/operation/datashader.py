import warnings

from collections.abc import Callable
from functools import partial

import param
import numpy as np
import pandas as pd
import xarray as xr
import datashader as ds
import datashader.reductions as rd
import datashader.transfer_functions as tf
import dask.dataframe as dd

from datashader.colors import color_lookup
from param.parameterized import bothmethod

try:
    from datashader.bundling import (directly_connect_edges as connect_edges,
                                     hammer_bundle)
except:
    hammer_bundle, connect_edges = object, object

from ..core import (Operation, Element, Dimension, NdOverlay,
                    CompositeOverlay, Dataset, Overlay, OrderedDict, Store)
from ..core.data import PandasInterface, XArrayInterface, DaskInterface, cuDFInterface
from ..core.util import (
    Iterable, LooseVersion, cast_array_to_int64, cftime_types, cftime_to_timestamp,
    datetime_types, dt_to_int, isfinite, get_param_values, max_range
)
from ..element import (Image, Path, Curve, RGB, Graph, TriMesh,
                       QuadMesh, Contours, Spikes, Area, Rectangles,
                       Spread, Segments, Scatter, Points, Polygons)
from ..element.util import connect_tri_edges_pd
from ..streams import RangeXY, PlotSize, PointerXY

ds_version = LooseVersion(ds.__version__)


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

    _allow_extra_keywords=True

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

    pixel_ratio = param.Number(default=1, bounds=(1,None), doc="""
       Pixel ratio applied to the height and width. Useful for higher
       resolution screens where the PlotSize stream reports 'nominal'
       dimensions in pixels that do not match the physical pixels. For
       instance, setting pixel_ratio=2 can give better results on Retina
       displays.""")

    x_range  = param.Tuple(default=None, length=2, doc="""
       The x_range as a tuple of min and max x-value. Auto-ranges
       if set to None.""")

    y_range  = param.Tuple(default=None, length=2, doc="""
       The y-axis range as a tuple of min and max y value. Auto-ranges
       if set to None.""")

    x_sampling = param.Number(default=None, doc="""
        Specifies the smallest allowed sampling interval along the x axis.""")

    y_sampling = param.Number(default=None, doc="""
        Specifies the smallest allowed sampling interval along the y axis.""")

    target = param.ClassSelector(class_=Dataset, doc="""
        A target Dataset which defines the desired x_range, y_range,
        width and height.
    """)

    streams = param.ClassSelector(default=[PlotSize, RangeXY], class_=(dict, list), doc="""
       List or dictionary of streams that are applied if dynamic=True,
       allowing for dynamic interaction with the plot.""")

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

    _transfer_options = []

    @bothmethod
    def instance(self_or_cls,**params):
        filtered = {k:v for k,v in params.items() if k in self_or_cls.param}
        inst = super(ResamplingOperation, self_or_cls).instance(**filtered)
        inst._precomputed = {}
        return inst

    def _get_sampling(self, element, x, y, ndim=2, default=None):
        target = self.p.target
        if not isinstance(x, list) and x is not None:
            x = [x]
        if not isinstance(y, list) and y is not None:
            y = [y]

        if target:
            x0, y0, x1, y1 = target.bounds.lbrt()
            x_range, y_range = (x0, x1), (y0, y1)
            height, width = target.dimension_values(2, flat=False).shape
        else:
            if x is None:
                x_range = self.p.x_range or (-0.5, 0.5)
            elif self.p.expand or not self.p.x_range:
                if self.p.x_range and all(isfinite(v) for v in self.p.x_range):
                    x_range = self.p.x_range
                else:
                    x_range = max_range([element.range(xd) for xd in x])
            else:
                x0, x1 = self.p.x_range
                ex0, ex1 = max_range([element.range(xd) for xd in x])
                x_range = (np.nanmin([np.nanmax([x0, ex0]), ex1]),
                           np.nanmax([np.nanmin([x1, ex1]), ex0]))

            if (y is None and ndim == 2):
                y_range = self.p.y_range or default or (-0.5, 0.5)
            elif self.p.expand or not self.p.y_range:
                if self.p.y_range and all(isfinite(v) for v in self.p.y_range):
                    y_range = self.p.y_range
                elif default is None:
                    y_range = max_range([element.range(yd) for yd in y])
                else:
                    y_range = default
            else:
                y0, y1 = self.p.y_range
                if default is None:
                    ey0, ey1 = max_range([element.range(yd) for yd in y])
                else:
                    ey0, ey1 = default
                y_range = (np.nanmin([np.nanmax([y0, ey0]), ey1]),
                           np.nanmax([np.nanmin([y1, ey1]), ey0]))
            width, height = self.p.width, self.p.height
        (xstart, xend), (ystart, yend) = x_range, y_range

        xtype = 'numeric'
        if isinstance(xstart, datetime_types) or isinstance(xend, datetime_types):
            xstart, xend = dt_to_int(xstart, 'ns'), dt_to_int(xend, 'ns')
            xtype = 'datetime'
        elif not np.isfinite(xstart) and not np.isfinite(xend):
            xstart, xend = 0, 0
            if x and element.get_dimension_type(x[0]) in datetime_types:
                xtype = 'datetime'

        ytype = 'numeric'
        if isinstance(ystart, datetime_types) or isinstance(yend, datetime_types):
            ystart, yend = dt_to_int(ystart, 'ns'), dt_to_int(yend, 'ns')
            ytype = 'datetime'
        elif not np.isfinite(ystart) and not np.isfinite(yend):
            ystart, yend = 0, 0
            if y and element.get_dimension_type(y[0]) in datetime_types:
                ytype = 'datetime'

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

        width = int(width * self.p.pixel_ratio)
        height = int(height * self.p.pixel_ratio)
        return ((xstart, xend), (ystart, yend)), (xs, ys), (width, height), (xtype, ytype)


    def _dt_transform(self, x_range, y_range, xs, ys, xtype, ytype):
        (xstart, xend), (ystart, yend) = x_range, y_range
        if xtype == 'datetime':
            xstart, xend = (np.array([xstart, xend])/1e3).astype('datetime64[us]')
            xs = (xs/1e3).astype('datetime64[us]')
        if ytype == 'datetime':
            ystart, yend = (np.array([ystart, yend])/1e3).astype('datetime64[us]')
            ys = (ys/1e3).astype('datetime64[us]')
        return ((xstart, xend), (ystart, yend)), (xs, ys)


class AggregationOperation(ResamplingOperation):
    """
    AggregationOperation extends the ResamplingOperation defining an
    aggregator parameter used to define a datashader Reduction.
    """

    aggregator = param.ClassSelector(class_=(ds.reductions.Reduction, str),
                                     default=ds.count(), doc="""
        Datashader reduction function used for aggregating the data.
        The aggregator may also define a column to aggregate; if
        no column is defined the first value dimension of the element
        will be used. May also be defined as a string.""")

    vdim_prefix = param.String(default='{kdims} ', allow_None=True, doc="""
        Prefix to prepend to value dimension name where {kdims}
        templates in the names of the input element key dimensions.""")

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
        'max':   rd.max,
        'count_cat': rd.count_cat
    }

    @classmethod
    def _get_aggregator(cls, element, agg, add_field=True):
        if isinstance(agg, str):
            if agg not in cls._agg_methods:
                agg_methods = sorted(cls._agg_methods)
                raise ValueError("Aggregation method '%r' is not known; "
                                 "aggregator must be one of: %r" %
                                 (agg, agg_methods))
            if agg == 'count_cat':
                agg = cls._agg_methods[agg]('__temp__')
            else:
                agg = cls._agg_methods[agg]()

        elements = element.traverse(lambda x: x, [Element])
        if (add_field and getattr(agg, 'column', False) in ('__temp__', None) and
            not isinstance(agg, (rd.count, rd.any))):
            if not elements:
                raise ValueError('Could not find any elements to apply '
                                 '%s operation to.' % cls.__name__)
            inner_element = elements[0]
            if isinstance(inner_element, TriMesh) and inner_element.nodes.vdims:
                field = inner_element.nodes.vdims[0].name
            elif inner_element.vdims:
                field = inner_element.vdims[0].name
            elif isinstance(element, NdOverlay):
                field = element.kdims[0].name
            else:
                raise ValueError("Could not determine dimension to apply "
                                 "'%s' operation to. Declare the dimension "
                                 "to aggregate as part of the datashader "
                                 "aggregator." % cls.__name__)
            agg = type(agg)(field)
        return agg

    def _empty_agg(self, element, x, y, width, height, xs, ys, agg_fn, **params):
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

    def _get_agg_params(self, element, x, y, agg_fn, bounds):
        params = dict(get_param_values(element), kdims=[x, y],
                      datatype=['xarray'], bounds=bounds)

        if self.vdim_prefix:
            kdim_list = '_'.join(str(kd) for kd in params['kdims'])
            vdim_prefix = self.vdim_prefix.format(kdims=kdim_list)
        else:
            vdim_prefix = ''

        category = None
        if hasattr(agg_fn, 'reduction'):
            category = agg_fn.cat_column
            agg_fn = agg_fn.reduction
        column = agg_fn.column if agg_fn else None
        if column:
            dims = [d for d in element.dimensions('ranges') if d == column]
            if not dims:
                raise ValueError("Aggregation column '%s' not found on '%s' element. "
                                 "Ensure the aggregator references an existing "
                                 "dimension." % (column,element))
            if isinstance(agg_fn, (ds.count, ds.count_cat)):
                if vdim_prefix:
                    vdim_name = '%s%s Count' % (vdim_prefix, column)
                else:
                    vdim_name = '%s Count' % column
                vdims = dims[0].clone(vdim_name, nodata=0)
            else:
                vdims = dims[0].clone(vdim_prefix + column)
        elif category:
            agg_name = type(agg_fn).__name__.title()
            agg_label = '%s %s' % (category, agg_name)
            vdims = Dimension('%s%s' % (vdim_prefix, agg_label), label=agg_label)
            if agg_name in ('Count', 'Any'):
                vdims.nodata = 0
        else:
            agg_name = type(agg_fn).__name__.title()
            vdims = Dimension('%s%s' % (vdim_prefix, agg_name), label=agg_name, nodata=0)
        params['vdims'] = vdims
        return params



class LineAggregationOperation(AggregationOperation):

    line_width = param.Number(default=None, bounds=(0, None), doc="""
        Width of the line to draw, in pixels. If zero, the default,
        lines are drawn using a simple algorithm with a blocky
        single-pixel width based on whether the line passes through
        each pixel or does not. If greater than one, lines are drawn
        with the specified width using a slower and more complex
        antialiasing algorithm with fractional values along each edge,
        so that lines have a more uniform visual appearance across all
        angles. Line widths between 0 and 1 effectively use a
        line_width of 1 pixel but with a proportionate reduction in
        the strength of each pixel, approximating the visual
        appearance of a subpixel line width.""")



class aggregate(LineAggregationOperation):
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

        is_custom = isinstance(df, dd.DataFrame) or cuDFInterface.applies(df)
        if any((not is_custom and len(df[d.name]) and isinstance(df[d.name].values[0], cftime_types)) or
               df[d.name].dtype.kind in ["M", "u"] for d in (x, y)):
            df = df.copy()

        for d in (x, y):
            vals = df[d.name]
            if not is_custom and len(vals) and isinstance(vals.values[0], cftime_types):
                vals = cftime_to_timestamp(vals, 'ns')
            elif vals.dtype.kind == 'M':
                vals = vals.astype('datetime64[ns]')
            elif vals.dtype == np.uint64:
                raise TypeError(f"Dtype of uint64 for column {d.name} is not supported.")
            elif vals.dtype.kind == 'u':
                pass  # To convert to int64
            else:
                continue
            df[d.name] = cast_array_to_int64(vals)
        return x, y, Dataset(df, kdims=kdims, vdims=vdims), glyph


    def _process(self, element, key=None):
        agg_fn = self._get_aggregator(element, self.p.aggregator)
        if hasattr(agg_fn, 'cat_column'):
            category = agg_fn.cat_column
        else:
            category = agg_fn.column if isinstance(agg_fn, ds.count_cat) else None

        if overlay_aggregate.applies(element, agg_fn, line_width=self.p.line_width):
            params = dict(
                {p: v for p, v in self.param.get_param_values() if p != 'name'},
                dynamic=False, **{p: v for p, v in self.p.items()
                                  if p not in ('name', 'dynamic')})
            return overlay_aggregate(element, **params)

        if element._plot_id in self._precomputed:
            x, y, data, glyph = self._precomputed[element._plot_id]
        else:
            x, y, data, glyph = self.get_agg_data(element, category)

        if self.p.precompute:
            self._precomputed[element._plot_id] = x, y, data, glyph
        (x_range, y_range), (xs, ys), (width, height), (xtype, ytype) = self._get_sampling(element, x, y)
        ((x0, x1), (y0, y1)), (xs, ys) = self._dt_transform(x_range, y_range, xs, ys, xtype, ytype)

        params = self._get_agg_params(element, x, y, agg_fn, (x0, y0, x1, y1))

        if x is None or y is None or width == 0 or height == 0:
            return self._empty_agg(element, x, y, width, height, xs, ys, agg_fn, **params)
        elif not getattr(data, 'interface', None) is DaskInterface and not len(data):
            empty_val = 0 if isinstance(agg_fn, ds.count) else np.NaN
            xarray = xr.DataArray(np.full((height, width), empty_val),
                                  dims=[y.name, x.name], coords={x.name: xs, y.name: ys})
            return self.p.element_type(xarray, **params)

        cvs = ds.Canvas(plot_width=width, plot_height=height,
                        x_range=x_range, y_range=y_range)

        agg_kwargs = {}
        if self.p.line_width and glyph == 'line' and ds_version >= LooseVersion('0.14.0'):
            agg_kwargs['line_width'] = self.p.line_width

        dfdata = PandasInterface.as_dframe(data)
        # Suppress numpy warning emitted by dask:
        # https://github.com/dask/dask/issues/8439
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action='ignore', message='casting datetime64',
                category=FutureWarning
            )
            agg = getattr(cvs, glyph)(dfdata, x.name, y.name, agg_fn, **agg_kwargs)
        if 'x_axis' in agg.coords and 'y_axis' in agg.coords:
            agg = agg.rename({'x_axis': x, 'y_axis': y})
        if xtype == 'datetime':
            agg[x.name] = (agg[x.name]/1e3).astype('datetime64[us]')
        if ytype == 'datetime':
            agg[y.name] = (agg[y.name]/1e3).astype('datetime64[us]')

        if agg.ndim == 2:
            # Replacing x and y coordinates to avoid numerical precision issues
            eldata = agg if ds_version > LooseVersion('0.5.0') else (xs, ys, agg.data)
            return self.p.element_type(eldata, **params)
        else:
            layers = {}
            for c in agg.coords[agg_fn.column].data:
                cagg = agg.sel(**{agg_fn.column: c})
                eldata = cagg if ds_version > LooseVersion('0.5.0') else (xs, ys, cagg.data)
                layers[c] = self.p.element_type(eldata, **params)
            return NdOverlay(layers, kdims=[data.get_dimension(agg_fn.column)])



class overlay_aggregate(aggregate):
    """
    Optimized aggregation for NdOverlay objects by aggregating each
    Element in an NdOverlay individually avoiding having to concatenate
    items in the NdOverlay. Works by summing sum and count aggregates and
    applying appropriate masking for NaN values. Mean aggregation
    is also supported by dividing sum and count aggregates. count_cat
    aggregates are grouped by the categorical dimension and a separate
    aggregate for each category is generated.
    """

    @classmethod
    def applies(cls, element, agg_fn, line_width=None):
        return (isinstance(element, NdOverlay) and
                (element.type is not Curve or line_width is None) and
                ((isinstance(agg_fn, (ds.count, ds.sum, ds.mean, ds.any)) and
                  (agg_fn.column is None or agg_fn.column not in element.kdims)) or
                 (isinstance(agg_fn, ds.count_cat) and agg_fn.column in element.kdims)))

    def _process(self, element, key=None):
        agg_fn = self._get_aggregator(element, self.p.aggregator)

        if not self.applies(element, agg_fn, line_width=self.p.line_width):
            raise ValueError(
                'overlay_aggregate only handles aggregation of NdOverlay types '
                'with count, sum or mean reduction.'
            )

        # Compute overall bounds
        dims = element.last.dimensions()[0:2]
        ndims = len(dims)
        if ndims == 1:
            x, y = dims[0], None
        else:
            x, y = dims

        info = self._get_sampling(element, x, y, ndims)
        (x_range, y_range), (xs, ys), (width, height), (xtype, ytype) = info
        ((x0, x1), (y0, y1)), _ = self._dt_transform(x_range, y_range, xs, ys, xtype, ytype)
        agg_params = dict({k: v for k, v in dict(self.param.get_param_values(),
                                                 **self.p).items()
                           if k in aggregate.param},
                          x_range=(x0, x1), y_range=(y0, y1))
        bbox = (x0, y0, x1, y1)

        # Optimize categorical counts by aggregating them individually
        if isinstance(agg_fn, ds.count_cat):
            agg_params.update(dict(dynamic=False, aggregator=ds.count()))
            agg_fn1 = aggregate.instance(**agg_params)
            if element.ndims == 1:
                grouped = element
            else:
                grouped = element.groupby(
                    [agg_fn.column], container_type=NdOverlay,
                    group_type=NdOverlay
                )
            groups = []
            for k, el in grouped.items():
                if width == 0 or height == 0:
                    agg = self._empty_agg(el, x, y, width, height, xs, ys, ds.count())
                    groups.append((k, agg))
                else:
                    agg = agg_fn1(el)
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
        is_sum = isinstance(agg_fn, ds.sum)
        is_any = isinstance(agg_fn, ds.any)

        # Accumulate into two aggregates and mask
        agg, agg2, mask = None, None, None
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
                if is_any:
                    agg.data |= new_agg.data
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



class area_aggregate(AggregationOperation):
    """
    Aggregates Area elements by filling the area between zero and
    the y-values if only one value dimension is defined and the area
    between the curves if two are provided.
    """

    def _process(self, element, key=None):
        x, y = element.dimensions()[:2]
        agg_fn = self._get_aggregator(element, self.p.aggregator)

        default = None
        if not self.p.y_range:
            y0, y1 = element.range(1)
            if len(element.vdims) > 1:
                y0, _ = element.range(2)
            elif y0 >= 0:
                y0 = 0
            elif y1 <= 0:
                y1 = 0
            default = (y0, y1)

        ystack = element.vdims[1].name if len(element.vdims) > 1 else None
        info = self._get_sampling(element, x, y, ndim=2, default=default)
        (x_range, y_range), (xs, ys), (width, height), (xtype, ytype) = info
        ((x0, x1), (y0, y1)), (xs, ys) = self._dt_transform(x_range, y_range, xs, ys, xtype, ytype)

        df = PandasInterface.as_dframe(element)

        cvs = ds.Canvas(plot_width=width, plot_height=height,
                        x_range=x_range, y_range=y_range)

        params = self._get_agg_params(element, x, y, agg_fn, (x0, y0, x1, y1))

        if width == 0 or height == 0:
            return self._empty_agg(element, x, y, width, height, xs, ys, agg_fn, **params)

        agg = cvs.area(df, x.name, y.name, agg_fn, axis=0, y_stack=ystack)
        if xtype == "datetime":
            agg[x.name] = (agg[x.name]/1e3).astype('datetime64[us]')

        return self.p.element_type(agg, **params)



class spread_aggregate(area_aggregate):
    """
    Aggregates Spread elements by filling the area between the lower
    and upper error band.
    """

    def _process(self, element, key=None):
        x, y = element.dimensions()[:2]
        df = PandasInterface.as_dframe(element)
        if df is element.data:
            df = df.copy()

        pos, neg = element.vdims[1:3] if len(element.vdims) > 2 else element.vdims[1:2]*2
        yvals = df[y.name]
        df[y.name] = yvals+df[pos.name]
        df['_lower'] = yvals-df[neg.name]
        area = element.clone(df, vdims=[y, '_lower']+element.vdims[3:], new_type=Area)
        return super()._process(area, key=None)



class spikes_aggregate(LineAggregationOperation):
    """
    Aggregates Spikes elements by drawing individual line segments
    over the entire y_range if no value dimension is defined and
    between zero and the y-value if one is defined.
    """

    spike_length = param.Number(default=None, allow_None=True, doc="""
      If numeric, specifies the length of each spike, overriding the
      vdims values (if present).""")

    offset = param.Number(default=0., doc="""
      The offset of the lower end of each spike.""")

    def _process(self, element, key=None):
        agg_fn = self._get_aggregator(element, self.p.aggregator)
        x, y = element.kdims[0], None

        spike_length = 0.5 if self.p.spike_length is None else self.p.spike_length
        if element.vdims and self.p.spike_length is None:
            x, y = element.dimensions()[:2]
            rename_dict = {'x': x.name, 'y':y.name}
            if not self.p.y_range:
                y0, y1 = element.range(1)
                if y0 >= 0:
                    default = (0, y1)
                elif y1 <= 0:
                    default = (y0, 0)
                else:
                    default = (y0, y1)
            else:
                default = None
        else:
             x, y = element.kdims[0], None
             default = (float(self.p.offset),
                        float(self.p.offset + spike_length))
             rename_dict = {'x': x.name}
        info = self._get_sampling(element, x, y, ndim=1, default=default)
        (x_range, y_range), (xs, ys), (width, height), (xtype, ytype) = info
        ((x0, x1), (y0, y1)), (xs, ys) = self._dt_transform(x_range, y_range, xs, ys, xtype, ytype)

        value_cols = [] if agg_fn.column is None else [agg_fn.column]
        if y is None:
            df = element.dframe([x]+value_cols).copy()
            y = Dimension('y')
            df['y0']  = float(self.p.offset)
            df['y1']  = float(self.p.offset + spike_length)
            yagg = ['y0', 'y1']
            if not self.p.expand: height = 1
        else:
            df = element.dframe([x, y]+value_cols).copy()
            df['y0'] = np.array(0, df.dtypes[y.name])
            yagg = ['y0', y.name]
        if xtype == 'datetime':
            df[x.name] = cast_array_to_int64(df[x.name].astype('datetime64[us]'))

        params = self._get_agg_params(element, x, y, agg_fn, (x0, y0, x1, y1))

        if width == 0 or height == 0:
            return self._empty_agg(element, x, y, width, height, xs, ys, agg_fn, **params)

        cvs = ds.Canvas(plot_width=width, plot_height=height,
                        x_range=x_range, y_range=y_range)

        agg_kwargs = {}
        if ds_version >= LooseVersion('0.14.0'):
            agg_kwargs['line_width'] = self.p.line_width

        agg = cvs.line(df, x.name, yagg, agg_fn, axis=1, **agg_kwargs).rename(rename_dict)
        if xtype == "datetime":
            agg[x.name] = (agg[x.name]/1e3).astype('datetime64[us]')

        return self.p.element_type(agg, **params)



class geom_aggregate(AggregationOperation):
    """
    Baseclass for aggregation of Geom elements.
    """

    __abstract = True

    def _aggregate(self, cvs, df, x0, y0, x1, y1, agg):
        raise NotImplementedError

    def _process(self, element, key=None):
        agg_fn = self._get_aggregator(element, self.p.aggregator)
        x0d, y0d, x1d, y1d = element.kdims
        info = self._get_sampling(element, [x0d, x1d], [y0d, y1d], ndim=1)
        (x_range, y_range), (xs, ys), (width, height), (xtype, ytype) = info
        ((x0, x1), (y0, y1)), (xs, ys) = self._dt_transform(x_range, y_range, xs, ys, xtype, ytype)

        df = element.interface.as_dframe(element)
        if xtype == 'datetime':
            df[x0d.name] = cast_array_to_int64(df[x0d.name].astype('datetime64[us]'))
            df[x1d.name] = cast_array_to_int64(df[x1d.name].astype('datetime64[us]'))
        if ytype == 'datetime':
            df[y0d.name] = cast_array_to_int64(df[y0d.name].astype('datetime64[us]'))
            df[y1d.name] = cast_array_to_int64(df[y1d.name].astype('datetime64[us]'))

        if isinstance(agg_fn, ds.count_cat) and df[agg_fn.column].dtype.name != 'category':
            df[agg_fn.column] = df[agg_fn.column].astype('category')

        params = self._get_agg_params(element, x0d, y0d, agg_fn, (x0, y0, x1, y1))

        if width == 0 or height == 0:
            return self._empty_agg(element, x0d, y0d, width, height, xs, ys, agg_fn, **params)

        cvs = ds.Canvas(plot_width=width, plot_height=height,
                        x_range=x_range, y_range=y_range)

        agg = self._aggregate(cvs, df, x0d.name, y0d.name, x1d.name, y1d.name, agg_fn)

        xdim, ydim = list(agg.dims)[:2][::-1]
        if xtype == "datetime":
            agg[xdim] = (agg[xdim]/1e3).astype('datetime64[us]')
        if ytype == "datetime":
            agg[ydim] = (agg[ydim]/1e3).astype('datetime64[us]')

        params['kdims'] = [xdim, ydim]

        if agg.ndim == 2:
            # Replacing x and y coordinates to avoid numerical precision issues
            eldata = agg if ds_version > LooseVersion('0.5.0') else (xs, ys, agg.data)
            return self.p.element_type(eldata, **params)
        else:
            layers = {}
            for c in agg.coords[agg_fn.column].data:
                cagg = agg.sel(**{agg_fn.column: c})
                eldata = cagg if ds_version > LooseVersion('0.5.0') else (xs, ys, cagg.data)
                layers[c] = self.p.element_type(eldata, **params)
            return NdOverlay(layers, kdims=[element.get_dimension(agg_fn.column)])


class segments_aggregate(geom_aggregate, LineAggregationOperation):
    """
    Aggregates Segments elements.
    """

    def _aggregate(self, cvs, df, x0, y0, x1, y1, agg_fn):
        agg_kwargs = {}
        if ds_version >= LooseVersion('0.14.0'):
            agg_kwargs['line_width'] = self.p.line_width

        return cvs.line(df, [x0, x1], [y0, y1], agg_fn, axis=1, **agg_kwargs)


class rectangle_aggregate(geom_aggregate):
    """
    Aggregates Rectangle elements.
    """

    def _aggregate(self, cvs, df, x0, y0, x1, y1, agg_fn):
        return cvs.area(df, x=[x0, x1], y=y0, y_stack=y1, agg=agg_fn, axis=1)



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
                                     class_=(ds.reductions.Reduction, str))

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
        objects=['linear', 'nearest', 'bilinear', None, False], doc="""
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
        for i, vd in enumerate(element.vdims):
            if element.interface is XArrayInterface:
                if element.interface.packed(element):
                    xarr = element.data[..., i]
                else:
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
        if ds_version <= LooseVersion('0.5.0'):
            raise RuntimeError('regrid operation requires datashader>=0.6.0')

        # Compute coords, anges and size
        x, y = element.kdims
        coords = tuple(element.dimension_values(d, expanded=False) for d in [x, y])
        info = self._get_sampling(element, x, y)
        (x_range, y_range), (xs, ys), (width, height), (xtype, ytype) = info

        # Disable upsampling by clipping size and ranges
        (xstart, xend), (ystart, yend) = (x_range, y_range)
        xspan, yspan = (xend-xstart), (yend-ystart)
        interp = self.p.interpolation or None
        if interp == 'bilinear': interp = 'linear'
        if not (self.p.upsample or interp is None) and self.p.target is None:
            (x0, x1), (y0, y1) = element.range(0), element.range(1)
            if isinstance(x0, datetime_types):
                x0, x1 = dt_to_int(x0, 'ns'), dt_to_int(x1, 'ns')
            if isinstance(y0, datetime_types):
                y0, y1 = dt_to_int(y0, 'ns'), dt_to_int(y1, 'ns')
            exspan, eyspan = (x1-x0), (y1-y0)
            if np.isfinite(exspan) and exspan > 0 and xspan > 0:
                width = max([min([int((xspan/exspan) * len(coords[0])), width]), 1])
            else:
                width = 0
            if np.isfinite(eyspan) and eyspan > 0 and yspan > 0:
                height = max([min([int((yspan/eyspan) * len(coords[1])), height]), 1])
            else:
                height = 0
            xunit = float(xspan)/width if width else 0
            yunit = float(yspan)/height if height else 0
            xs, ys = (np.linspace(xstart+xunit/2., xend-xunit/2., width),
                      np.linspace(ystart+yunit/2., yend-yunit/2., height))

        # Compute bounds (converting datetimes)
        ((x0, x1), (y0, y1)), (xs, ys) = self._dt_transform(x_range, y_range, xs, ys, xtype, ytype)

        params = dict(bounds=(x0, y0, x1, y1))
        if width == 0 or height == 0:
            if width == 0:
                params['xdensity'] = 1
            if height == 0:
                params['ydensity'] = 1
            return element.clone((xs, ys, np.zeros((height, width))), **params)

        cvs = ds.Canvas(plot_width=width, plot_height=height,
                        x_range=x_range, y_range=y_range)

        # Apply regridding to each value dimension
        regridded = {}
        arrays = self._get_xarrays(element, coords, xtype, ytype)
        agg_fn = self._get_aggregator(element, self.p.aggregator, add_field=False)
        for vd, xarr in arrays.items():
            rarray = cvs.raster(xarr, upsample_method=interp,
                                downsample_method=agg_fn)

            # Convert datetime coordinates
            if xtype == "datetime":
                rarray[x.name] = (rarray[x.name]/1e3).astype('datetime64[us]')
            if ytype == "datetime":
                rarray[y.name] = (rarray[y.name]/1e3).astype('datetime64[us]')
            regridded[vd] = rarray
        regridded = xr.Dataset(regridded)

        return element.clone(regridded, datatype=['xarray']+element.datatype, **params)



class contours_rasterize(aggregate):
    """
    Rasterizes the Contours element by weighting the aggregation by
    the iso-contour levels if a value dimension is defined, otherwise
    default to any aggregator.
    """

    aggregator = param.ClassSelector(default=ds.mean(),
                                     class_=(ds.reductions.Reduction, str))

    @classmethod
    def _get_aggregator(cls, element, agg, add_field=True):
        if not element.vdims and agg.column is None and not isinstance(agg, (rd.count, rd.any)):
            return ds.any()
        return super()._get_aggregator(element, agg, add_field)



class trimesh_rasterize(aggregate):
    """
    Rasterize the TriMesh element using the supplied aggregator. If
    the TriMesh nodes or edges define a value dimension, will plot
    filled and shaded polygons; otherwise returns a wiremesh of the
    data.
    """

    aggregator = param.ClassSelector(default=ds.mean(),
                                     class_=(ds.reductions.Reduction, str))

    interpolation = param.ObjectSelector(default='bilinear',
                                         objects=['bilinear', 'linear', None, False], doc="""
        The interpolation method to apply during rasterization.""")

    def _precompute(self, element, agg):
        from datashader.utils import mesh
        if element.vdims and getattr(agg, 'column', None) not in element.nodes.vdims:
            simplex_dims = [0, 1, 2, 3]
            vert_dims = [0, 1]
        elif element.nodes.vdims:
            simplex_dims = [0, 1, 2]
            vert_dims = [0, 1, 3]
        else:
            raise ValueError("Cannot shade TriMesh without value dimension.")
        datatypes = [element.interface.datatype, element.nodes.interface.datatype]
        if set(datatypes) == {'dask'}:
            dims, node_dims = element.dimensions(), element.nodes.dimensions()
            simplices = element.data[[dims[sd].name for sd in simplex_dims]]
            verts = element.nodes.data[[node_dims[vd].name for vd in vert_dims]]
        else:
            if 'dask' in datatypes:
                if datatypes[0] == 'dask':
                    p, n = 'simplexes', 'vertices'
                else:
                    p, n = 'vertices', 'simplexes'
                self.param.warning(
                    "TriMesh %s were provided as dask DataFrame but %s "
                    "were not. Datashader will not use dask to parallelize "
                    "rasterization unless both are provided as dask "
                    "DataFrames." % (p, n))
            simplices = element.dframe(simplex_dims)
            verts = element.nodes.dframe(vert_dims)
        for c, dtype in zip(simplices.columns[:3], simplices.dtypes):
            if dtype.kind != 'i':
                simplices[c] = simplices[c].astype('int')
        mesh = mesh(verts, simplices)
        if hasattr(mesh, 'persist'):
            mesh = mesh.persist()
        return {
            'mesh': mesh,
            'simplices': simplices,
            'vertices': verts
        }

    def _precompute_wireframe(self, element, agg):
        if hasattr(element, '_wireframe'):
            segments = element._wireframe.data
        else:
            segments = connect_tri_edges_pd(element)
            element._wireframe = Dataset(segments, datatype=['dataframe', 'dask'])
        return {'segments': segments}

    def _process(self, element, key=None):
        if isinstance(element, TriMesh):
            x, y = element.nodes.kdims[:2]
        else:
            x, y = element.kdims
        info = self._get_sampling(element, x, y)
        (x_range, y_range), (xs, ys), (width, height), (xtype, ytype) = info

        agg = self.p.aggregator
        interp = self.p.interpolation or None
        precompute = self.p.precompute
        if interp == 'linear': interp = 'bilinear'
        wireframe = False
        if (not (element.vdims or (isinstance(element, TriMesh) and element.nodes.vdims))) and ds_version <= LooseVersion('0.6.9'):
            self.p.aggregator = ds.any() if isinstance(agg, ds.any) or agg == 'any' else ds.count()
            return aggregate._process(self, element, key)
        elif ((not interp and (isinstance(agg, (ds.any, ds.count)) or
                               agg in ['any', 'count']))
               or not (element.vdims or element.nodes.vdims)):
            wireframe = True
            precompute = False # TriMesh itself caches wireframe
            if isinstance(agg, (ds.any, ds.count)):
                agg = self._get_aggregator(element, self.p.aggregator)
            else:
                agg = ds.any()
        elif getattr(agg, 'column', None) is None:
            agg = self._get_aggregator(element, self.p.aggregator)

        if element._plot_id in self._precomputed:
            precomputed = self._precomputed[element._plot_id]
        elif wireframe:
            precomputed = self._precompute_wireframe(element, agg)
        else:
            precomputed = self._precompute(element, agg)
        bounds = (x_range[0], y_range[0], x_range[1], y_range[1])
        params = self._get_agg_params(element, x, y, agg, bounds)

        if width == 0 or height == 0:
            if width == 0: params['xdensity'] = 1
            if height == 0: params['ydensity'] = 1
            return Image((xs, ys, np.zeros((height, width))), **params)

        if wireframe:
            segments = precomputed['segments']
        else:
            simplices = precomputed['simplices']
            pts = precomputed['vertices']
            mesh = precomputed['mesh']
        if precompute:
            self._precomputed = {element._plot_id: precomputed}

        cvs = ds.Canvas(plot_width=width, plot_height=height,
                        x_range=x_range, y_range=y_range)
        if wireframe:
            agg = cvs.line(segments, x=['x0', 'x1', 'x2', 'x0'],
                           y=['y0', 'y1', 'y2', 'y0'], axis=1,
                           agg=agg).rename({'x': x.name, 'y': y.name})
        else:
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
        if ds_version <= LooseVersion('0.7.0'):
            return super()._precompute(element.trimesh(), agg)

    def _process(self, element, key=None):
        if ds_version <= LooseVersion('0.7.0'):
            return super()._process(element, key)

        if element.interface.datatype != 'xarray':
            element = element.clone(datatype=['xarray'])
        data = element.data

        x, y = element.kdims
        agg_fn = self._get_aggregator(element, self.p.aggregator)
        info = self._get_sampling(element, x, y)
        (x_range, y_range), (xs, ys), (width, height), (xtype, ytype) = info
        if xtype == 'datetime':
            data[x.name] = data[x.name].astype('datetime64[us]').astype('int64')
        if ytype == 'datetime':
            data[y.name] = data[y.name].astype('datetime64[us]').astype('int64')

        # Compute bounds (converting datetimes)
        ((x0, x1), (y0, y1)), (xs, ys) = self._dt_transform(
            x_range, y_range, xs, ys, xtype, ytype
        )
        params = dict(get_param_values(element), datatype=['xarray'],
                      bounds=(x0, y0, x1, y1))

        if width == 0 or height == 0:
            return self._empty_agg(element, x, y, width, height, xs, ys, agg_fn, **params)

        cvs = ds.Canvas(plot_width=width, plot_height=height,
                        x_range=x_range, y_range=y_range)

        vdim = getattr(agg_fn, 'column', element.vdims[0].name)
        agg = cvs.quadmesh(data[vdim], x.name, y.name, agg_fn)
        xdim, ydim = list(agg.dims)[:2][::-1]
        if xtype == "datetime":
            agg[xdim] = (agg[xdim]/1e3).astype('datetime64[us]')
        if ytype == "datetime":
            agg[ydim] = (agg[ydim]/1e3).astype('datetime64[us]')

        return Image(agg, **params)



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

    alpha = param.Integer(default=255, bounds=(0, 255), doc="""
        Value between 0 - 255 representing the alpha value to use for
        colormapped pixels that contain data (i.e. non-NaN values).
        Regardless of this value, ``NaN`` values are set to be fully
        transparent when doing colormapping.""")

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

    cnorm = param.ClassSelector(default='eq_hist',
                                class_=(str, Callable),
                                doc="""
        The normalization operation applied before colormapping.
        Valid options include 'linear', 'log', 'eq_hist', 'cbrt',
        and any valid transfer function that accepts data, mask, nbins
        arguments.""")

    clims = param.NumericTuple(default=None, length=2, doc="""
        Min and max data values to use for colormap interpolation, when
        wishing to override autoranging.
        """)

    min_alpha = param.Number(default=40, bounds=(0, 255), doc="""
        The minimum alpha value to use for non-empty pixels when doing
        colormapping, in [0, 255].  Use a higher value to avoid
        undersaturation, i.e. poorly visible low-value datapoints, at
        the expense of the overall dynamic range..""")

    rescale_discrete_levels = param.Boolean(default=True, doc="""
        If ``cnorm='eq_hist`` and there are only a few discrete values,
        then ``rescale_discrete_levels=True`` (the default) decreases
        the lower limit of the autoranged span so that the values are
        rendering towards the (more visible) top of the ``cmap`` range, 
        thus avoiding washout of the lower values.  Has no effect if
        ``cnorm!=`eq_hist``. Set this value to False if you need to
        match historical unscaled behavior, prior to HoloViews 1.14.4.""")

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
    def uint32_to_uint8_xr(cls, img):
        """
        Cast uint32 xarray DataArray to 4 uint8 channels.
        """
        new_array = img.values.view(dtype=np.uint8).reshape(img.shape + (4,))
        coords = OrderedDict(list(img.coords.items())+[('band', [0, 1, 2, 3])])
        return xr.DataArray(new_array, coords=coords, dims=img.dims+('band',))


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
        vdims = list(element.vdims)
        # Override nodata temporarily
        element.vdims[:] = [vd.clone(nodata=None) for vd in element.vdims]
        try:
            data += tuple(element.dimension_values(vd, flat=False)
                          for vd in element.vdims)
        finally:
            element.vdims[:] = vdims
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
            return element.map(partial(shade._process, self), [Element])
        else:
            xdensity = element.xdensity
            ydensity = element.ydensity
            bounds = element.bounds

        vdim = element.vdims[0].name
        array = element.data[vdim]
        kdims = element.kdims

        shade_opts = dict(
            how=self.p.cnorm, min_alpha=self.p.min_alpha, alpha=self.p.alpha
        )
        if ds_version >= LooseVersion('0.14.0'):
            shade_opts['rescale_discrete_levels'] = self.p.rescale_discrete_levels

        # Compute shading options depending on whether
        # it is a categorical or regular aggregate
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
        elif isinstance(self.p.cmap, str):
            if self.p.cmap.startswith('#') or self.p.cmap in color_lookup:
                shade_opts['cmap'] = self.p.cmap
            else:
                from ..plotting.util import process_cmap
                shade_opts['cmap'] = process_cmap(self.p.cmap)
        else:
            shade_opts['cmap'] = self.p.cmap

        if self.p.clims:
            shade_opts['span'] = self.p.clims
        elif ds_version > LooseVersion('0.5.0') and self.p.cnorm != 'eq_hist':
            shade_opts['span'] = element.range(vdim)

        params = dict(get_param_values(element), kdims=kdims,
                      bounds=bounds, vdims=RGB.vdims[:],
                      xdensity=xdensity, ydensity=ydensity)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'invalid value encountered in true_divide')
            if np.isnan(array.data).all():
                xd, yd = kdims[:2]
                arr = np.zeros(array.data.shape[:2]+(4,), dtype=np.uint8)
                coords = {xd.name: element.data.coords[xd.name],
                          yd.name: element.data.coords[yd.name],
                          'band': [0, 1, 2, 3]}
                img = xr.DataArray(arr, coords=coords, dims=(yd.name, xd.name, 'band'))
                return RGB(img, **params)
            else:
                img = tf.shade(array, **shade_opts)
        return RGB(self.uint32_to_uint8_xr(img), **params)



class geometry_rasterize(LineAggregationOperation):
    """
    Rasterizes geometries by converting them to spatialpandas.
    """

    aggregator = param.ClassSelector(default=ds.mean(),
                                     class_=(ds.reductions.Reduction, str))

    @classmethod
    def _get_aggregator(cls, element, agg, add_field=True):
        if (not (element.vdims or isinstance(agg, str)) and
            agg.column is None and not isinstance(agg, (rd.count, rd.any))):
            return ds.count()
        return super()._get_aggregator(element, agg, add_field)

    def _process(self, element, key=None):
        agg_fn = self._get_aggregator(element, self.p.aggregator)
        xdim, ydim = element.kdims
        info = self._get_sampling(element, xdim, ydim)
        (x_range, y_range), (xs, ys), (width, height), (xtype, ytype) = info
        x0, x1 = x_range
        y0, y1 = y_range

        params = self._get_agg_params(element, xdim, ydim, agg_fn, (x0, y0, x1, y1))

        if width == 0 or height == 0:
            return self._empty_agg(element, xdim, ydim, width, height, xs, ys, agg_fn, **params)

        cvs = ds.Canvas(plot_width=width, plot_height=height,
                        x_range=x_range, y_range=y_range)

        if element._plot_id in self._precomputed:
            data, col = self._precomputed[element._plot_id]
        else:
            if 'spatialpandas' not in element.interface.datatype:
                element = element.clone(datatype=['spatialpandas'])
            data = element.data
            col = element.interface.geo_column(data)

        if self.p.precompute:
            self._precomputed[element._plot_id] = (data, col)

        if isinstance(agg_fn, ds.count_cat) and data[agg_fn.column].dtype.name != 'category':
            data[agg_fn.column] = data[agg_fn.column].astype('category')

        agg_kwargs = dict(geometry=col, agg=agg_fn)
        if isinstance(element, Polygons):
            agg = cvs.polygons(data, **agg_kwargs)
        elif isinstance(element, Path):
            if self.p.line_width and ds_version >= LooseVersion('0.14.0'):
                agg_kwargs['line_width'] = self.p.line_width
            agg = cvs.line(data, **agg_kwargs)
        elif isinstance(element, Points):
            agg = cvs.points(data, **agg_kwargs)
        agg = agg.rename({'x': xdim.name, 'y': ydim.name})

        if agg.ndim == 2:
            return self.p.element_type(agg, **params)
        else:
            layers = {}
            for c in agg.coords[agg_fn.column].data:
                cagg = agg.sel(**{agg_fn.column: c})
                layers[c] = self.p.element_type(cagg, **params)
            return NdOverlay(layers, kdims=[element.get_dimension(agg_fn.column)])



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

    aggregator = param.ClassSelector(class_=(ds.reductions.Reduction, str),
                                     default='default')

    interpolation = param.ObjectSelector(
        default='default', objects=['default', 'linear', 'nearest', 'bilinear', None, False], doc="""
        The interpolation method to apply during rasterization.
        Default depends on element type""")

    _transforms = [(Image, regrid),
                   (Polygons, geometry_rasterize),
                   (lambda x: (isinstance(x, (Path, Points)) and
                               'spatialpandas' in x.interface.datatype),
                    geometry_rasterize),
                   (TriMesh, trimesh_rasterize),
                   (QuadMesh, quadmesh_rasterize),
                   (lambda x: (isinstance(x, NdOverlay) and
                               issubclass(x.type, (Scatter, Points, Curve, Path))),
                    aggregate),
                   (Spikes, spikes_aggregate),
                   (Area, area_aggregate),
                   (Spread, spread_aggregate),
                   (Segments, segments_aggregate),
                   (Rectangles, rectangle_aggregate),
                   (Contours, contours_rasterize),
                   (Graph, aggregate),
                   (Scatter, aggregate),
                   (Points, aggregate),
                   (Curve, aggregate),
                   (Path, aggregate),
                   (type(None), shade) # To handle parameters of datashade
    ]

    def _process(self, element, key=None):
        # Potentially needs traverse to find element types first?
        all_allowed_kws = set()
        all_supplied_kws = set()
        for predicate, transform in self._transforms:
            merged_param_values = dict(self.param.get_param_values(), **self.p)

            # If aggregator or interpolation are 'default', pop parameter so
            # datashader can choose the default aggregator itself
            for k in ['aggregator', 'interpolation']:
                if merged_param_values.get(k, None) == 'default':
                    merged_param_values.pop(k)

            op_params = dict({k: v for k, v in merged_param_values.items()
                              if not (v is None and k == 'aggregator')},
                             dynamic=False)
            extended_kws = dict(op_params, **self.p.extra_keywords())
            all_supplied_kws |= set(extended_kws)
            all_allowed_kws |= set(transform.param)
            # Collect union set of consumed. Versus union of available.
            op = transform.instance(**{k:v for k,v in extended_kws.items()
                                       if k in transform.param})
            op._precomputed = self._precomputed
            element = element.map(op, predicate)
            self._precomputed = op._precomputed

        unused_params = list(all_supplied_kws - all_allowed_kws)
        if unused_params:
            self.param.warning('Parameter(s) [%s] not consumed by any element rasterizer.'
                         % ', '.join(unused_params))
        return element



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
                raise TypeError("The stack operation expects elements of type RGB, "
                                "not '%s'." % type(rgb).__name__)
            rgb = rgb.rgb
            dims = [kd.name for kd in rgb.kdims][::-1]
            coords = {kd.name: rgb.dimension_values(kd, False)
                      for kd in rgb.kdims}
            imgs.append(tf.Image(self.uint8_to_uint32(rgb), coords=coords, dims=dims))

        try:
            imgs = xr.align(*imgs, join='exact')
        except ValueError:
            raise ValueError('RGB inputs to the stack operation could not be aligned; '
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

    how = param.ObjectSelector(default='source' if ds_version <= LooseVersion('0.11.1') else None,
            objects=[None, 'source', 'over', 'saturate', 'add', 'max', 'min'], doc="""
        The name of the compositing operator to use when combining
        pixels. Default of None uses 'over' operator for RGB elements
        and 'add' operator for aggregate arrays.""")

    shape = param.ObjectSelector(default='circle', objects=['circle', 'square'],
                                 doc="""
        The shape to spread by. Options are 'circle' [default] or 'square'.""")

    _per_element = True

    @classmethod
    def uint8_to_uint32(cls, img):
        shape = img.shape
        flat_shape = np.multiply.reduce(shape[:2])
        if shape[-1] == 3:
            img = np.dstack([img, np.ones(shape[:2], dtype='uint8')*255])
        rgb = img.reshape((flat_shape, 4)).view('uint32').reshape(shape[:2])
        return rgb

    def _apply_spreading(self, array):
        """Apply the spread function using the indicated parameters."""
        raise NotImplementedError

    def _preprocess_rgb(self, element):
        rgbarray = np.dstack([element.dimension_values(vd, flat=False)
                              for vd in element.vdims])
        if rgbarray.dtype.kind == 'f':
            rgbarray = rgbarray * 255
        return tf.Image(self.uint8_to_uint32(rgbarray.astype('uint8')))

    def _process(self, element, key=None):
        if isinstance(element, RGB):
            rgb = element.rgb
            data = self._preprocess_rgb(rgb)
        elif isinstance(element, Image):
            data = element.clone(datatype=['xarray']).data[element.vdims[0].name]
        else:
            raise ValueError('spreading can only be applied to Image or RGB Elements. '
                             'Received object of type %s' % str(type(element)))

        kwargs = {}
        array = self._apply_spreading(data)
        if isinstance(element, RGB):
            img = datashade.uint32_to_uint8(array.data)[::-1]
            new_data = {
                kd.name: rgb.dimension_values(kd, expanded=False)
                for kd in rgb.kdims
            }
            vdims = rgb.vdims+[rgb.alpha_dimension] if len(rgb.vdims) == 3 else rgb.vdims
            kwargs['vdims'] = vdims
            new_data[tuple(vd.name for vd in vdims)] = img
        else:
            new_data = array
        return element.clone(new_data, xdensity=element.xdensity,
                             ydensity=element.ydensity, **kwargs)



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
        return tf.spread(array, px=self.p.px, how=self.p.how, shape=self.p.shape)


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
        return tf.dynspread(
            array, max_px=self.p.max_px, threshold=self.p.threshold,
            how=self.p.how, shape=self.p.shape
        )


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


def identity(x): return x


class inspect_mask(Operation):
    """
    Operation used to display the inspection mask, for use with other
    inspection operations. Can be used directly but is more commonly
    constructed using the mask property of the corresponding inspector
    operation.
    """

    pixels = param.Integer(default=3, doc="""
       Size of the mask that should match the pixels parameter used in
       the associated inspection operation.""")

    streams = param.ClassSelector(default=[PointerXY], class_=(dict, list))
    x = param.Number(default=0)
    y = param.Number(default=0)

    @classmethod
    def _distance_args(cls, element, x_range, y_range,  pixels):
        ycount, xcount =  element.interface.shape(element, gridded=True)
        x_delta = abs(x_range[1] - x_range[0]) / xcount
        y_delta = abs(y_range[1] - y_range[0]) / ycount
        return (x_delta*pixels, y_delta*pixels)

    def _process(self, raster, key=None):
        if isinstance(raster, RGB):
            raster = raster[..., raster.vdims[-1]]
        x_range, y_range = raster.range(0), raster.range(1)
        xdelta, ydelta = self._distance_args(raster, x_range, y_range, self.p.pixels)
        x, y = self.p.x, self.p.y
        return self._indicator(raster.kdims, x, y, xdelta, ydelta)

    def _indicator(self, kdims, x, y, xdelta, ydelta):
        rect = np.array([(x-xdelta/2,y-ydelta/2), (x+xdelta/2, y-ydelta/2),
                         (x+xdelta/2, y+ydelta/2), (x-xdelta/2, y+ydelta/2)])
        data = {(str(kdims[0]),str(kdims[1])):rect}
        return Polygons(data, kdims=kdims)


class inspect(Operation):
    """
    Generalized inspect operation that detects the appropriate indicator
    type.
    """

    pixels = param.Integer(default=3, doc="""
       Number of pixels in data space around the cursor point to search
       for hits in. The hit within this box mask that is closest to the
       cursor's position is displayed.""")

    null_value = param.Number(default=0, doc="""
       Value of raster which indicates no hits. For instance zero for
       count aggregator (default) and commonly NaN for other (float)
       aggregators. For RGBA images, the alpha channel is used which means
       zero alpha acts as the null value.""")

    value_bounds = param.NumericTuple(default=None, length=2, allow_None=True, doc="""
       If not None, a numeric bounds for the pixel under the cursor in
       order for hits to be computed. Useful for count aggregators where
       a value of (1,1000) would make sure no more than a thousand
       samples will be searched.""")

    hits = param.DataFrame(default=pd.DataFrame(), allow_None=True)

    max_indicators = param.Integer(default=1, doc="""
       Maximum number of indicator elements to display within the mask
       of size pixels. Points are prioritized by distance from the
       cursor point. This means that the default value of one shows the
       single closest sample to the cursor. Note that this limit is not
       applies to the hits parameter.""")

    transform = param.Callable(default=identity, doc="""
      Function that transforms the hits dataframe before it is passed to
      the Points element. Can be used to customize the value dimensions
      e.g. to implement custom hover behavior.""")

    # Stream values and overrides
    streams = param.ClassSelector(default=dict(x=PointerXY.param.x,
                                               y=PointerXY.param.y),
                                  class_=(dict, list))

    x = param.Number(default=0, doc="x-position to inspect.")

    y = param.Number(default=0, doc="y-position to inspect.")

    _dispatch = {}

    @property
    def mask(self):
        return inspect_mask.instance(pixels=self.p.pixels)

    def _update_hits(self, event):
        self.hits = event.obj.hits

    @bothmethod
    def instance(self_or_cls, **params):
        inst = super(inspect, self_or_cls).instance(**params)
        inst._op = None
        return inst

    def _process(self, raster, key=None):
        input_type = self._get_input_type(raster.pipeline.operations)
        inspect_operation = self._dispatch[input_type]
        if self._op is None:
            self._op = inspect_operation.instance()
            self._op.param.watch(self._update_hits, 'hits')
        elif not isinstance(self._op, inspect_operation):
            raise ValueError("Cannot reuse inspect instance on different "
                             "datashader input type.")
        self._op.p = self.p
        return self._op._process(raster)

    def _get_input_type(self, operations):
        for op in operations:
            output_type = getattr(op, 'output_type', None)
            if output_type is not None:
                if output_type in [el[0] for el in rasterize._transforms]:
                    # Datashader output types that are also input types e.g for regrid
                    if issubclass(output_type, (Image, RGB)):
                        continue
                    return output_type
        raise RuntimeError('Could not establish input element type '
                           'for datashader pipeline in the inspect operation.')



class inspect_base(inspect):
    """
    Given datashaded aggregate (Image) output, return a set of
    (hoverable) points sampled from those near the cursor.
    """

    def _process(self, raster, key=None):
        self._validate(raster)
        if isinstance(raster, RGB):
            raster = raster[..., raster.vdims[-1]]
        x_range, y_range = raster.range(0), raster.range(1)
        xdelta, ydelta = self._distance_args(raster, x_range, y_range, self.p.pixels)
        x, y = self.p.x, self.p.y
        val = raster[x-xdelta:x+xdelta, y-ydelta:y+ydelta].reduce(function=np.nansum)
        if np.isnan(val):
            val = self.p.null_value

        if ((self.p.value_bounds and
             not (self.p.value_bounds[0] < val < self.p.value_bounds[1]))
             or val == self.p.null_value):
            result = self._empty_df(raster.dataset)
        else:
            masked = self._mask_dataframe(raster, x, y, xdelta, ydelta)
            result = self._sort_by_distance(raster, masked, x, y)

        self.hits = result
        df = self.p.transform(result)
        return self._element(raster, df.iloc[:self.p.max_indicators])

    @classmethod
    def _distance_args(cls, element, x_range, y_range,  pixels):
        ycount, xcount =  element.interface.shape(element, gridded=True)
        x_delta = abs(x_range[1] - x_range[0]) / xcount
        y_delta = abs(y_range[1] - y_range[0]) / ycount
        return (x_delta*pixels, y_delta*pixels)

    @classmethod
    def _empty_df(cls, dataset):
        if 'dask' in dataset.interface.datatype:
            return dataset.data._meta.iloc[:0]
        elif dataset.interface.datatype in ['pandas', 'geopandas', 'spatialpandas']:
            return dataset.data.head(0)
        return dataset.iloc[:0].dframe()

    @classmethod
    def _mask_dataframe(cls, raster, x, y, xdelta, ydelta):
        """
        Mask the dataframe around the specified x and y position with
        the given x and y deltas
        """
        ds = raster.dataset
        x0, x1, y0, y1 = x-xdelta, x+xdelta, y-ydelta, y+ydelta
        if 'spatialpandas' in ds.interface.datatype:
            df = ds.data.cx[x0:x1, y0:y1]
            return df.compute() if hasattr(df, 'compute') else df
        xdim, ydim = raster.kdims
        query = {xdim.name: (x0, x1), ydim.name: (y0, y1)}
        return ds.select(**query).dframe()

    @classmethod
    def _validate(cls, raster):
        pass

    @classmethod
    def _vdims(cls, raster, df):
        ds = raster.dataset
        if 'spatialpandas' in ds.interface.datatype:
            coords = [ds.interface.geo_column(ds.data)]
        else:
            coords = [kd.name for kd in raster.kdims]
        return [col for col in df.columns if col not in coords]



class inspect_points(inspect_base):

    @classmethod
    def _element(cls, raster, df):
        return Points(df, kdims=raster.kdims, vdims=cls._vdims(raster, df))

    @classmethod
    def _sort_by_distance(cls, raster, df, x, y):
        """
        Returns a dataframe of hits within a given mask around a given
        spatial location, sorted by distance from that location.
        """
        ds = raster.dataset.clone(df)
        xs, ys = (ds.dimension_values(kd) for kd in raster.kdims)
        dx, dy = xs - x, ys - y
        distances = pd.Series(dx*dx + dy*dy)
        return df.iloc[distances.argsort().values]



class inspect_polygons(inspect_base):

    @classmethod
    def _validate(cls, raster):
        if 'spatialpandas' not in raster.dataset.interface.datatype:
            raise ValueError("inspect_polygons only supports spatialpandas datatypes.")

    @classmethod
    def _element(cls, raster, df):
        polygons = Polygons(df, kdims=raster.kdims, vdims=cls._vdims(raster, df))
        if Store.loaded_backends() != []:
            return polygons.opts(color_index=None)
        else:
            return polygons

    @classmethod
    def _sort_by_distance(cls, raster, df, x, y):
        """
        Returns a dataframe of hits within a given mask around a given
        spatial location, sorted by distance from that location.
        """
        xs, ys = [], []
        for geom in df.geometry.array:
            gxs, gys = geom.flat_values[::2], geom.flat_values[1::2]
            if not len(gxs):
                xs.append(np.nan)
                ys.append(np.nan)
            else:
                xs.append((np.min(gxs)+np.max(gxs))/2)
                ys.append((np.min(gys)+np.max(gys))/2)
        dx, dy = np.array(xs) - x, np.array(ys) - y
        distances = pd.Series(dx*dx + dy*dy)
        return df.iloc[distances.argsort().values]


inspect._dispatch = {
    Points: inspect_points,
    Polygons: inspect_polygons
}
