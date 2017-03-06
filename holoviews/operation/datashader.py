from __future__ import absolute_import

from collections import Callable, Iterable
import warnings

import param
import numpy as np
import pandas as pd
import xarray as xr
import datashader as ds
import datashader.transfer_functions as tf
import dask.dataframe as dd

from datashader.core import bypixel
from datashader.pandas import pandas_pipeline
from datashader.dask import dask_pipeline
from datashape.dispatch import dispatch
from datashape import discover as dsdiscover
import datashader.transfer_functions as tf

from ..core import (ElementOperation, Element, Dimension, NdOverlay,
                    Overlay, CompositeOverlay, Dataset)
from ..core.data import PandasInterface, DaskInterface
from ..core.util import get_param_values, basestring
from ..element import GridImage, Image, Path, Curve, Contours, RGB
from ..streams import RangeXY


@dispatch(Element)
def discover(dataset):
    """
    Allows datashader to correctly discover the dtypes of the data
    in a holoviews Element.
    """
    return dsdiscover(PandasInterface.as_dframe(dataset))


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

    if dataset.interface is PandasInterface:
        agg = pandas_pipeline(dataset.data, schema, canvas,
                              glyph, summary)
    elif dataset.interface is DaskInterface:
        agg = dask_pipeline(dataset.data, schema, canvas,
                            glyph, summary)

    agg = agg.rename({'x_axis': kdims[0].name,
                      'y_axis': kdims[1].name})
    return agg


class aggregate(ElementOperation):
    """
    aggregate implements 2D binning for any valid HoloViews Element
    type using datashader. I.e., this operation turns a HoloViews
    Element or overlay of Elements into an hv.Image or an overlay of
    hv.Images by rasterizing it, which provides a fixed-sized
    representation independent of the original dataset size.

    By default it will simply count the number of values in each bin
    but other aggregators can be supplied implementing mean, max, min
    and other reduction operations.

    The bins of the aggregate are defined by the width and height and
    the x_range and y_range. If x_sampling or y_sampling are supplied
    the operation will ensure that a bin is no smaller than theminimum
    sampling distance by reducing the width and height when the zoomed
    in beyond the minimum sampling distance.
    """

    aggregator = param.ClassSelector(class_=ds.reductions.Reduction,
                                     default=ds.count())

    dynamic = param.Boolean(default=True, doc="""
       Enables dynamic processing by default.""")

    height = param.Integer(default=400, doc="""
       The height of the aggregated image in pixels.""")

    width = param.Integer(default=400, doc="""
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

    element_type = param.ClassSelector(class_=(Dataset,), instantiate=False,
                                        is_instance=False, default=GridImage,
                                        doc="""
        The type of the returned Elements, must be a 2D Dataset type.""")

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
                df = PandasInterface.as_dframe(element)
                if isinstance(obj, NdOverlay):
                    df = df.assign(**dict(zip(obj.dimensions('key', True), key)))
                paths.append(df)
            kdims += element.kdims
            vdims = element.vdims
        elif isinstance(obj, Element):
            glyph = 'line' if isinstance(obj, Curve) else 'points'
            paths.append(PandasInterface.as_dframe(obj))
        if len(paths) > 1:
            if glyph == 'line':
                path = paths[0][:1]
                if isinstance(path, dd.DataFrame):
                    path = path.compute()
                empty = path.copy()
                empty.iloc[0, :] = (np.NaN,) * empty.shape[1]
                paths = [elem for path in paths for elem in (path, empty)][:-1]
            if all(isinstance(path, dd.DataFrame) for path in paths):
                df = dd.concat(paths)
            else:
                paths = [path.compute() if isinstance(path, dd.DataFrame) else path
                         for path in paths]
                df = pd.concat(paths)
        else:
            df = paths[0]
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
        if self.p.x_sampling:
            x_range = xend - xstart
            width = int(min([(x_range/self.p.x_sampling), width]))
        if self.p.y_sampling:
            y_range = yend - ystart
            height = int(min([(y_range/self.p.y_sampling), height]))

        cvs = ds.Canvas(plot_width=width, plot_height=height,
                        x_range=(xstart, xend), y_range=(ystart, yend))

        column = agg_fn.column
        if column and isinstance(agg_fn, ds.count_cat):
            name = '%s Count' % agg_fn.column
        else:
            name = column
        vdims = [element.get_dimension(column)(name) if column
                 else Dimension('Count')]
        params = dict(get_param_values(element), kdims=element.dimensions()[:2],
                      datatype=['xarray'], vdims=vdims)

        agg = getattr(cvs, glyph)(data, x, y, self.p.aggregator)
        if agg.ndim == 2:
            return self.p.element_type(agg, **params)
        else:
            return NdOverlay({c: self.p.element_type(agg.sel(**{column: c}),
                                                     **params)
                              for c in agg.coords[column].data},
                             kdims=[data.get_dimension(column)])



class shade(ElementOperation):
    """
    shade applies a normalization function followed by colormapping to
    an Image or NdOverlay of Images, returning an RGB Element.
    The data must be in the form of a 2D or 3D DataArray, but NdOverlays
    of 2D Images will be automatically converted to a 3D array.

    In the 2D case data is normalized and colormapped, while a 3D
    array representing categorical aggregates will be supplied a color
    key for each category. The colormap (cmap) may be supplied as an
    Iterable or a Callable.
    """

    cmap = param.ClassSelector(class_=(Iterable, Callable, dict), doc="""
        Iterable or callable which returns colors as hex colors.
        Callable type must allow mapping colors between 0 and 1.""")

    normalization = param.ClassSelector(default='eq_hist',
                                        class_=(basestring, Callable),
                                        doc="""
        The normalization operation applied before colormapping.
        Valid options include 'linear', 'log', 'eq_hist', 'cbrt',
        and any valid transfer function that accepts data, mask, nbins
        arguments.""")

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
        return Dataset(xarr.T, datatype=['xarray'], **params)


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
            elif isinstance(self.p.cmap, dict):
                shade_opts['color_key'] = self.p.cmap
            elif isinstance(self.p.cmap, Iterable):
                shade_opts['color_key'] = [c for i, c in
                                           zip(range(categories), self.p.cmap)]
            else:
                colors = [self.p.cmap(s) for s in np.linspace(0, 1, categories)]
                shade_opts['color_key'] = map(self.rgb2hex, colors)
        elif not self.p.cmap:
            pass
        elif isinstance(self.p.cmap, Callable):
            colors = [self.p.cmap(s) for s in np.linspace(0, 1, 256)]
            shade_opts['cmap'] = map(self.rgb2hex, colors)
        else:
            shade_opts['cmap'] = self.p.cmap

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'invalid value encountered in true_divide')
            img = tf.shade(array, **shade_opts)
        params = dict(get_param_values(element), kdims=kdims,
                      bounds=bounds, vdims=RGB.vdims[:])
        return RGB(self.uint32_to_uint8(img.data), **params)



class datashade(aggregate, shade):
    """
    Applies the aggregate and shade operations, aggregating all
    elements in the supplied object and then applying normalization
    and colormapping the aggregated data returning RGB elements.

    See aggregate and shade operations for more details.
    """

    def _process(self, element, key=None):
        agg = aggregate._process(self, element, key)
        shaded = shade._process(self, agg, key)
        return shaded



class dynspread(ElementOperation):
    """
    Spreading expands each pixel in an Image based Element a certain
    number of pixels on all sides according to a given shape, merging
    pixels using a specified compositing operator. This can be useful
    to make sparse plots more visible. Dynamic spreading determines
    how many pixels to spread based on a density heuristic.

    See the datashader documentation for more detail:

    http://datashader.readthedocs.io/en/latest/api.html#datashader.transfer_functions.dynspread
    """

    how = param.ObjectSelector(default='source',
                               objects=['source', 'over',
                                        'saturate', 'add'], doc="""
        The name of the compositing operator to use when combining
        pixels.""")

    max_px = param.Integer(default=3, doc="""
        Maximum number of pixels to spread on all sides.""")
    
    shape = param.ObjectSelector(default='circle', objects=['circle', 'square'],
                                 doc="""
        The shape to spread by. Options are 'circle' [default] or 'square'.""")

    threshold = param.Number(default=0.5, bounds=(0,1), doc="""
        When spreading, determines how far to spread.
        Spreading starts at 1 pixel, and stops when the fraction
        of adjacent non-empty pixels reaches this threshold.
        Higher values give more spreading, up to the max_px
        allowed.""")

    @classmethod
    def uint8_to_uint32(cls, img):
        shape = img.shape
        flat_shape = np.multiply.reduce(shape[:2])
        rgb = img.reshape((flat_shape, 4)).view('uint32').reshape(shape[:2])
        return rgb

    def _apply_dynspread(self, array):
        img = tf.Image(array)
        return tf.dynspread(img, max_px=self.p.max_px,
                            threshold=self.p.threshold,
                            how=self.p.how, shape=self.p.shape).data

    def _process(self, element, key=None):
        if not isinstance(element, (Image, GridImage)):
            raise ValueError('dynspread can only be applied to Image Elements.')

        if isinstance(element, GridImage):
            new_data = {kd.name: element.dimension_values(kd, expanded=False)
                        for kd in element.kdims}
            for vd in element.vdims:
                array = element.dimension_values(vd, flat=False)
                new_data[vd.name] = self._apply_dynspread(array)
            return element.clone(element.data)
        else:
            img = np.flipud(element.data)
            isrgb = isinstance(element, RGB)
            data = self.uint8_to_uint32(img) if isrgb else img
            array = self._apply_dynspread(data)
            img = datashade.uint32_to_uint8(array) if isrgb else np.flipud(array)
            return element.clone(img)
