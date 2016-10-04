from __future__ import absolute_import

from collections import Callable, Iterable

import param
import numpy as np
import pandas as pd
import xarray as xr
import datashader as ds
import datashader.transfer_functions as tf

from datashader.core import bypixel
from datashader.pandas import pandas_pipeline
from datashape.dispatch import dispatch
from datashape import discover as dsdiscover

from ..core import (ElementOperation, Element, Dimension, NdOverlay,
                    Overlay, CompositeOverlay, Dataset)
from ..core.data import ArrayInterface, PandasInterface
from ..core.util import get_param_values
from ..element import GridImage, Path, Curve, Contours, RGB
from ..streams import RangeXY


@dispatch(Element)
def discover(dataset):
    """
    Allows datashader to correctly discover the dtypes of the data
    in a holoviews Element.
    """
    if isinstance(dataset.interface, (PandasInterface, ArrayInterface)):
        return dsdiscover(dataset.data)
    else:
        return dsdiscover(dataset.dframe())


@bypixel.pipeline.register(Element)
def dataset_pipeline(dataset, schema, canvas, glyph, summary):
    """
    Defines how to apply a datashader pipeline to a holoviews Element,
    using multidispatch. Returns an Image type with the appropriate
    bounds and dimensions. Passing the returned Image to datashader
    transfer functions is not yet supported.
    """
    x0, x1 = canvas.x_range
    y0, y1 = canvas.y_range
    kdims = [dataset.get_dimension(d) for d in (glyph.x, glyph.y)]

    column = summary.column
    if column and isinstance(summary, ds.count_cat):
        name = '%s Count' % summary.column
    else:
        name = column
    vdims = [dataset.get_dimension(column)(name) if column
             else Dimension('Count')]

    aggregate = pandas_pipeline(dataset.dframe(), schema, canvas,
                                glyph, summary)
    aggregate = aggregate.rename({'x_axis': kdims[0].name,
                                  'y_axis': kdims[1].name})

    params = dict(get_param_values(dataset), kdims=kdims,
                  datatype=['xarray'], vdims=vdims)

    if aggregate.ndim == 2:
        return GridImage(aggregate, **params)
    else:
        return NdOverlay({c: GridImage(aggregate.sel(**{column: c}),
                                       **params)
                          for c in aggregate.coords[column].data},
                         kdims=[dataset.get_dimension(column)])


class Aggregate(ElementOperation):
    """
    Aggregate implements 2D binning for any valid HoloViews Element
    type using datashader. By default it will simply count the number
    of values in each bin but other aggregators can be supplied
    implementing mean, max, min and other reduction operations.

    The bins of the aggregate are defined by the width and height and
    the x_range and y_range. If x_sampling or y_sampling are supplied
    the operation will ensure that a bin is no smaller than the
    minimum sampling distance by reducing the width and height when
    the zoomed in beyond the minimum sampling distance.
    """

    aggregator = param.ClassSelector(class_=ds.reductions.Reduction,
                                     default=ds.count())

    height = param.Integer(default=800, doc="""
       The height of the aggregated image in pixels.""")

    width = param.Integer(default=600, doc="""
       The width of the aggregated image in pixels.""")

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

    streams = param.List(default=[RangeXY], doc="""
        List of streams that are applied if dynamic=True, allowing
        for dynamic interaction with the plot.""")

    @classmethod
    def get_agg_data(cls, obj, category=None):
        """
        Reduces any Overlay or NdOverlay of Elements into a single
        xarray Dataset that can be aggregated.
        """
        paths = []
        kdims = obj.kdims
        vdims = obj.vdims
        x, y = obj.dimensions(label=True)[:2]
        if isinstance(obj, Path):
            glyph = 'line'
            for p in obj.data:
                df = pd.DataFrame(p, columns=obj.dimensions('key', True))
                if isinstance(obj, Contours) and obj.vdims and obj.level:
                    df[obj.vdims[0].name] = p.level
                paths.append(df)
        elif isinstance(obj, CompositeOverlay):
            for key, el in obj.data.items():
                x, y, element, glyph = cls.get_agg_data(el)
                df = element.dframe()
                if isinstance(obj, NdOverlay):
                    df = df.assign(**dict(zip(obj.dimensions('key', True), key)))
                paths.append(df)
            kdims += element.kdims
            vdims = element.vdims
        elif isinstance(obj, Element):
            glyph = 'line' if isinstance(obj, Curve) else 'points'
            paths.append(obj.dframe())
        if glyph == 'line':
            empty = paths[0][:1].copy()
            empty.loc[0, :] = (np.NaN,) * empty.shape[1]
            paths = [elem for path in paths for elem in (path, empty)][:-1]
        df = pd.concat(paths).reset_index(drop=True)
        if category and df[category].dtype.name != 'category':
            df[category] = df[category].astype('category')
        return x, y, Dataset(df, kdims=kdims, vdims=vdims), glyph


    def _process(self, element, key=None):
        agg_fn = self.p.aggregator
        category = agg_fn.column if isinstance(agg_fn, ds.count_cat) else None
        x, y, data, glyph = self.get_agg_data(element, category)

        xstart, xend = self.p.x_range if self.p.x_range else data.range(x)
        ystart, yend = self.p.y_range if self.p.y_range else data.range(y)

        # Compute highest allowed sampling density
        width, height = self.p.width, self.p.height
        if self.x_sampling:
            x_range = xend - xstart
            width = int(min([(x_range/self.p.x_sampling), width]))
        if self.y_sampling:
            y_range = yend - ystart
            height = int(min([(y_range/self.p.y_sampling), height]))

        cvs = ds.Canvas(plot_width=width, plot_height=height,
                        x_range=(xstart, xend), y_range=(ystart, yend))
        return getattr(cvs, glyph)(data, x, y, self.p.aggregator)



class Shade(ElementOperation):
    """
    Shade applies a normalization function to the data and then
    applies colormapping to an Image or NdOverlay of Images, returning
    an RGB Element.
    """

    cmap = param.ClassSelector(class_=(Iterable, Callable), doc="""
        Iterable or callable which returns colors as hex colors.
        Callable type must allow mapping colors between 0 and 1.""")

    normalization = param.ObjectSelector(default='eq_hist',
                                         objects=['linear', 'log',
                                                  'eq_hist', 'cbrt'],
                                         doc="""
        The normalization operation applied before colormapping.""")

    @classmethod
    def concatenate(cls, overlay):
        """
        Concatenates an NdOverlay of GridImage types into a single 3D
        xarray Dataset.
        """
        if not isinstance(overlay, NdOverlay):
            raise ValueError('Only NdOverlays can be concatenated')
        xarr = xr.concat([v.data.T for v in overlay.values()],
                         dim=overlay.kdims[0].name)
        params = dict(get_param_values(overlay.last),
                      vdims=overlay.last.vdims,
                      kdims=overlay.kdims+overlay.last.kdims)
        return Dataset(xarr.T, **params)


    @classmethod
    def uint32_to_uint8(cls, img):
        """
        Cast uint32 RGB image to 4 uint8 channels.
        """
        return np.flipud(img.view(dtype=np.uint8).reshape(img.shape + (4,)))


    def _process(self, element, key=None):
        if isinstance(element, NdOverlay):
            bounds = element.last.bounds
            element = self.concatenate(element)
        else:
            bounds = element.bounds

        array = element.data[element.vdims[0].name]
        kdims = element.kdims

        # Compute shading options depending on whether
        # it is a categorical or regular aggregate
        shade_opts = dict(how=self.p.normalization)
        if element.ndims > 2:
            kdims = element.kdims[1:]
            categories = array.shape[-1]
            if not self.p.cmap:
                pass
            elif isinstance(self.p.cmap, Iterator):
                shade_opts['color_key'] = [c for i, c in
                                           zip(range(categories), self.p.cmap)]
            else:
                shade_opts['color_key'] = [self.p.cmap(s) for s in
                                           np.linspace(0, 1, categories)]
        elif not self.p.cmap:
            pass
        elif isinstance(self.p.cmap, Callable):
            shade_opts['cmap'] = [self.p.cmap(s) for s in np.linspace(0, 1, 256)]
        else:
            shade_opts['cmap'] = self.p.cmap


        img = tf.shade(array, **shade_opts)
        params = dict(get_param_values(element), kdims=kdims,
                      bounds=bounds, vdims=RGB.vdims[:])
        return RGB(self.uint32_to_uint8(img.data), **params)



class Datashade(Aggregate, Shade):
    """
    Applies the Aggregate and Shade operations, aggregating all
    elements in the supplied object and then applying normalization
    and colormapping the aggregated data returning RGB elements.

    See Aggregate and Shade operations for more details.
    """

    def _process(self, element, key=None):
        aggregate = Aggregate._process(self, element, key)
        shaded = Shade._process(self, aggregate, key)
        return shaded
