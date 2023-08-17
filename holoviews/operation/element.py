"""
Collection of either extremely generic or simple Operation
examples.
"""
import warnings

import numpy as np
import param

from packaging.version import Version
from param import _is_number

from ..core import (Operation, NdOverlay, Overlay, GridMatrix,
                    HoloMap, Dataset, Element, Collator, Dimension)
from ..core.data import ArrayInterface, DictInterface, PandasInterface, default_datatype
from ..core.data.util import dask_array_module
from ..core.util import (
    group_sanitizer, label_sanitizer, datetime_types, isfinite,
    dt_to_int, isdatetime, is_dask_array, is_cupy_array, is_ibis_expr
)
from ..element.chart import Histogram, Scatter
from ..element.raster import Image, RGB
from ..element.path import Contours, Polygons
from ..element.util import categorical_aggregate2d # noqa (API import)
from ..streams import RangeXY

column_interfaces = [ArrayInterface, DictInterface, PandasInterface]


def identity(x,k): return x

class operation(Operation):
    """
    The most generic operation that wraps any callable into an
    Operation. The callable needs to accept an HoloViews
    component and a key (that may be ignored) and must return a new
    HoloViews component.

    This class may be useful for turning a HoloViews method into an
    operation to define as compositor operation. For instance, the
    following definition:

    operation.instance(op=lambda x, k: x.collapse(np.subtract))

    Could be used to implement a collapse operation to subtracts the
    data between Rasters in an Overlay.
    """

    output_type = param.Parameter(default=None, doc="""
       The output element type which may be None to disable type
       checking.

       May be used to declare useful information to other code in
       HoloViews, e.g. required for tab-completion support of operations
       registered with compositors.""")

    group = param.String(default='Operation', doc="""
        The group assigned to the result after having applied the
        operator.""")

    op = param.Callable(default=identity, doc="""
        The operation used to generate a new HoloViews object returned
        by the operation. By default, the identity operation is
        applied.""")

    def _process(self, view, key=None):
        retval = self.p.op(view, key)
        if (self.p.output_type is not None):
            assert isinstance(retval, self.p.output_type), \
                              "Return value does not match the declared output type."
        return retval.relabel(group=self.p.group)


class factory(Operation):
    """
    Simple operation that constructs any element that accepts some
    other element as input. For instance, RGB and HSV elements can be
    created from overlays of Image elements.
    """

    output_type = param.Parameter(default=RGB, doc="""
        The output type of the factor operation.

        By default, if three overlaid Images elements are supplied,
        the corresponding RGB element will be returned. """)

    args = param.List(default=[], doc="""
        The list of positional argument to pass to the factory""")

    kwargs = param.Dict(default={}, doc="""
        The dict of keyword arguments to pass to the factory""")

    def _process(self, view, key=None):
        return self.p.output_type(view, *self.p.args, **self.p.kwargs)


class function(Operation):

    output_type = param.ClassSelector(class_=type, doc="""
        The output type of the method operation""")

    input_type = param.ClassSelector(class_=type, doc="""
        The object type the method is defined on""")

    fn = param.Callable(default=lambda el, *args, **kwargs: el, doc="""
        The function to apply.""")

    args = param.List(default=[], doc="""
        The list of positional argument to pass to the method""")

    kwargs = param.Dict(default={}, doc="""
        The dict of keyword arguments to pass to the method""")

    def _process(self, element, key=None):
        return self.p.fn(element, *self.p.args, **self.p.kwargs)


class method(Operation):
    """
    Operation that wraps a method call
    """

    output_type = param.ClassSelector(class_=type, doc="""
        The output type of the method operation""")

    input_type = param.ClassSelector(class_=type, doc="""
        The object type the method is defined on""")

    method_name = param.String(default='__call__', doc="""
        The method name""")

    args = param.List(default=[], doc="""
        The list of positional argument to pass to the method""")

    kwargs = param.Dict(default={}, doc="""
        The dict of keyword arguments to pass to the method""")

    def _process(self, element, key=None):
        fn = getattr(self.p.input_type, self.p.method_name)
        return fn(element, *self.p.args, **self.p.kwargs)


class apply_when(param.ParameterizedFunction):
    """
    Applies a selection depending on the current zoom range. If the
    supplied predicate function returns a True it will apply the
    operation otherwise it will return the raw element after the
    selection. For example the following will apply datashading if
    the number of points in the current viewport exceed 1000 otherwise
    just returning the selected points element:

       apply_when(points, operation=datashade, predicate=lambda x: x > 1000)
    """

    operation = param.Callable(default=lambda x: x)

    predicate = param.Callable(default=None)

    def _apply(self, element, x_range, y_range, invert=False):
        selected = element
        if x_range is not None and y_range is not None:
            selected = element[x_range, y_range]
        condition = self.predicate(selected)
        if (not invert and condition) or (invert and not condition):
            return selected
        elif selected.interface.gridded:
            return selected.clone([])
        else:
            return selected.iloc[:0]

    def __call__(self, obj, **params):
        if 'streams' in params:
            streams = params.pop('streams')
        else:
            streams = [RangeXY()]
        self.param.update(**params)
        if not self.predicate:
            raise ValueError(
                'Must provide a predicate function to determine when '
                'to apply the operation and when to return the selected '
                'data.'
            )
        applied = self.operation(obj.apply(self._apply, streams=streams))
        raw = obj.apply(self._apply, streams=streams, invert=True)
        return applied * raw


class chain(Operation):
    """
    Defining an Operation chain is an easy way to define a new
    Operation from a series of existing ones. The argument is a
    list of Operation (or Operation instances) that are
    called in sequence to generate the returned element.

    chain(operations=[gradient, threshold.instance(level=2)])

    This operation can accept an Image instance and would first
    compute the gradient before thresholding the result at a level of
    2.0.

    Instances are only required when arguments need to be passed to
    individual operations so the resulting object is a function over a
    single argument.
    """

    output_type = param.Parameter(default=Image, doc="""
        The output type of the chain operation. Must be supplied if
        the chain is to be used as a channel operation.""")

    group = param.String(default='', doc="""
        The group assigned to the result after having applied the chain.
        Defaults to the group produced by the last operation in the chain""")

    operations = param.List(default=[], item_type=Operation, doc="""
       A list of Operations (or Operation instances)
       that are applied on the input from left to right.""")

    def _process(self, view, key=None):
        processed = view
        for operation in self.p.operations:
            processed = operation.process_element(
                processed, key, input_ranges=self.p.input_ranges
            )

        if not self.p.group:
            return processed
        else:
            return processed.clone(group=self.p.group)

    def find(self, operation, skip_nonlinked=True):
        """
        Returns the first found occurrence of an operation while
        performing a backward traversal of the chain pipeline.
        """
        found = None
        for op in self.operations[::-1]:
            if isinstance(op, operation):
                found = op
                break
            if not op.link_inputs and skip_nonlinked:
                break
        return found


class transform(Operation):
    """
    Generic Operation to transform an input Image or RGBA
    element into an output Image. The transformation is defined by
    the supplied callable that accepts the data of the input Image
    (typically a numpy array) and returns the transformed data of the
    output Image.

    This operator is extremely versatile; for instance, you could
    implement an alternative to the explicit threshold operator with:

    operator=lambda x: np.clip(x, 0, 0.5)

    Alternatively, you can implement a transform computing the 2D
    autocorrelation using the scipy library with:

    operator=lambda x: scipy.signal.correlate2d(x, x)
    """

    output_type = Image

    group = param.String(default='Transform', doc="""
        The group assigned to the result after applying the
        transform.""")

    operator = param.Callable(doc="""
       Function of one argument that transforms the data in the input
       Image to the data in the output Image. By default, acts as
       the identity function such that the output matches the input.""")

    def _process(self, img, key=None):
        processed = (img.data if not self.p.operator
                     else self.p.operator(img.data))
        return img.clone(processed, group=self.p.group)


class image_overlay(Operation):
    """
    Operation to build a overlay of images to a specification from a
    subset of the required elements.

    This is useful for reordering the elements of an overlay,
    duplicating layers of an overlay or creating blank image elements
    in the appropriate positions.

    For instance, image_overlay may build a three layered input
    suitable for the RGB factory operation even if supplied with one
    or two of the required channels (creating blank channels for the
    missing elements).

    Note that if there is any ambiguity regarding the match, the
    strongest match will be used. In the case of a tie in match
    strength, the first layer in the input is used. One successful
    match is always required.
    """

    output_type = Overlay

    spec = param.String(doc="""
       Specification of the output Overlay structure. For instance:

       Image.R * Image.G * Image.B

       Will ensure an overlay of this structure is created even if
       (for instance) only (Image.R * Image.B) is supplied.

       Elements in the input overlay that match are placed in the
       appropriate positions and unavailable specification elements
       are created with the specified fill group.""")

    fill = param.Number(default=0)

    default_range = param.Tuple(default=(0,1), doc="""
        The default range that will be set on the value_dimension of
        any automatically created blank image elements.""")

    group = param.String(default='Transform', doc="""
        The group assigned to the resulting overlay.""")


    @classmethod
    def _match(cls, el, spec):
        "Return the strength of the match (None if no match)"
        spec_dict = dict(zip(['type', 'group', 'label'], spec.split('.')))
        if not isinstance(el, Image) or spec_dict['type'] != 'Image':
            raise NotImplementedError("Only Image currently supported")

        sanitizers = {'group':group_sanitizer, 'label':label_sanitizer}
        strength = 1
        for key in ['group', 'label']:
            attr_value = sanitizers[key](getattr(el, key))
            if key in spec_dict:
                if spec_dict[key] != attr_value: return None
                strength += 1
        return strength


    def _match_overlay(self, raster, overlay_spec):
        """
        Given a raster or input overlay, generate a list of matched
        elements (None if no match) and corresponding tuple of match
        strength values.
        """
        ordering = [None]*len(overlay_spec) # Elements to overlay
        strengths = [0]*len(overlay_spec)   # Match strengths

        elements = raster.values() if isinstance(raster, Overlay) else [raster]

        for el in elements:
            for pos in range(len(overlay_spec)):
                strength = self._match(el, overlay_spec[pos])
                if strength is None:               continue  # No match
                elif (strength <= strengths[pos]): continue  # Weaker match
                else:                                        # Stronger match
                    ordering[pos] = el
                    strengths[pos] = strength
        return ordering, strengths


    def _process(self, raster, key=None):
        specs = tuple(el.strip() for el in self.p.spec.split('*'))
        ordering, strengths = self._match_overlay(raster, specs)
        if all(el is None for el in ordering):
            raise Exception("The image_overlay operation requires at least one match")

        completed = []
        strongest = ordering[np.argmax(strengths)]
        for el, spec in zip(ordering, specs):
            if el is None:
                spec_dict = dict(zip(['type', 'group', 'label'], spec.split('.')))
                el = Image(np.ones(strongest.data.shape) * self.p.fill,
                            group=spec_dict.get('group','Image'),
                            label=spec_dict.get('label',''))
                el.vdims[0].range = self.p.default_range
            completed.append(el)
        return np.prod(completed)



class threshold(Operation):
    """
    Threshold a given Image whereby all values higher than a given
    level map to the specified high value and all values lower than
    that level map to the specified low value.
    """
    output_type = Image

    level = param.Number(default=0.5, doc="""
       The value at which the threshold is applied. Values lower than
       the threshold map to the 'low' value and values above map to
       the 'high' value.""")

    high = param.Number(default=1.0, doc="""
      The value given to elements greater than (or equal to) the
      threshold.""")

    low = param.Number(default=0.0, doc="""
      The value given to elements below the threshold.""")

    group = param.String(default='Threshold', doc="""
       The group assigned to the thresholded output.""")

    _per_element = True

    def _process(self, matrix, key=None):

        if not isinstance(matrix, Image):
            raise TypeError("The threshold operation requires a Image as input.")

        arr = matrix.data
        high = np.ones(arr.shape) * self.p.high
        low = np.ones(arr.shape) * self.p.low
        thresholded = np.where(arr > self.p.level, high, low)

        return matrix.clone(thresholded, group=self.p.group)



class gradient(Operation):
    """
    Compute the gradient plot of the supplied Image.

    If the Image value dimension is cyclic, the smallest step is taken
    considered the cyclic range
    """

    output_type = Image

    group = param.String(default='Gradient', doc="""
    The group assigned to the output gradient matrix.""")

    _per_element = True

    def _process(self, matrix, key=None):

        if len(matrix.vdims) != 1:
            raise ValueError("Input matrix to gradient operation must "
                             "have single value dimension.")

        matrix_dim = matrix.vdims[0]

        data = np.flipud(matrix.dimension_values(matrix_dim, flat=False))
        r, c = data.shape

        if  matrix_dim.cyclic and (None in matrix_dim.range):
            raise Exception("Cyclic range must be specified to compute "
                            "the gradient of cyclic quantities")
        cyclic_range = None if not matrix_dim.cyclic else np.diff(matrix_dim.range)
        if cyclic_range is not None:
            # shift values such that wrapping works ok
            data = data - matrix_dim.range[0]

        dx = np.diff(data, 1, axis=1)[0:r-1, 0:c-1]
        dy = np.diff(data, 1, axis=0)[0:r-1, 0:c-1]

        if cyclic_range is not None: # Wrap into the specified range
            # Convert negative differences to an equivalent positive value
            dx = dx % cyclic_range
            dy = dy % cyclic_range
            #
            # Prefer small jumps
            dx_negatives = dx - cyclic_range
            dy_negatives = dy - cyclic_range
            dx = np.where(np.abs(dx_negatives)<dx, dx_negatives, dx)
            dy = np.where(np.abs(dy_negatives)<dy, dy_negatives, dy)

        return Image(np.sqrt(dx * dx + dy * dy), bounds=matrix.bounds, group=self.p.group)



class convolve(Operation):
    """
    Apply a convolution to an overlay using the top layer as the
    kernel for convolving the bottom layer. Both Image elements in
    the input overlay should have a single value dimension.
    """

    output_type = Image

    group = param.String(default='Convolution', doc="""
        The group assigned to the convolved output.""")

    kernel_roi = param.NumericTuple(default=(0,0,0,0), length=4, doc="""
        A 2-dimensional slice of the kernel layer to use in the
        convolution in lbrt (left, bottom, right, top) format. By
        default, no slicing is applied.""")

    _per_element = True

    def _process(self, overlay, key=None):
        if len(overlay) != 2:
            raise Exception("Overlay must contain at least to items.")

        [target, kernel] = overlay.get(0), overlay.get(1)

        if len(target.vdims) != 1:
            raise Exception("Convolution requires inputs with single value dimensions.")

        xslice = slice(self.p.kernel_roi[0], self.p.kernel_roi[2])
        yslice = slice(self.p.kernel_roi[1], self.p.kernel_roi[3])

        k = kernel.data if self.p.kernel_roi == (0,0,0,0) else kernel[xslice, yslice].data

        data = np.flipud(target.dimension_values(2, flat=False))
        fft1 = np.fft.fft2(data)
        fft2 = np.fft.fft2(k, s=data.shape)
        convolved_raw = np.fft.ifft2(fft1 * fft2).real

        k_rows, k_cols = k.shape
        rolled = np.roll(np.roll(convolved_raw, -(k_cols//2), axis=-1), -(k_rows//2), axis=-2)
        convolved = rolled / float(k.sum())

        return Image(convolved, bounds=target.bounds, group=self.p.group)



class contours(Operation):
    """
    Given a Image with a single channel, annotate it with contour
    lines for a given set of contour levels.

    The return is an NdOverlay with a Contours layer for each given
    level, overlaid on top of the input Image.
    """

    output_type = Overlay

    levels = param.ClassSelector(default=10, class_=(list, int), doc="""
        A list of scalar values used to specify the contour levels.""")

    group = param.String(default='Level', doc="""
        The group assigned to the output contours.""")

    filled = param.Boolean(default=False, doc="""
        Whether to generate filled contours""")

    overlaid = param.Boolean(default=False, doc="""
        Whether to overlay the contour on the supplied Element.""")

    _per_element = True

    def _process(self, element, key=None):
        try:
            from matplotlib.contour import QuadContourSet
            from matplotlib.axes import Axes
            from matplotlib.figure import Figure
            from matplotlib.dates import num2date, date2num
        except ImportError:
            raise ImportError("contours operation requires matplotlib.")
        extent = element.range(0) + element.range(1)[::-1]

        xs = element.dimension_values(0, True, flat=False)
        ys = element.dimension_values(1, True, flat=False)
        zs = element.dimension_values(2, flat=False)

        # Ensure that coordinate arrays specify bin centers
        if xs.shape[0] != zs.shape[0]:
            xs = xs[:-1] + np.diff(xs, axis=0)/2.
        if xs.shape[1] != zs.shape[1]:
            xs = xs[:, :-1] + (np.diff(xs, axis=1)/2.)
        if ys.shape[0] != zs.shape[0]:
            ys = ys[:-1] + np.diff(ys, axis=0)/2.
        if ys.shape[1] != zs.shape[1]:
            ys = ys[:, :-1] + (np.diff(ys, axis=1)/2.)
        data = (xs, ys, zs)

        # if any data is a datetime, transform to matplotlib's numerical format
        data_is_datetime = tuple(isdatetime(arr) for k, arr in enumerate(data))
        if any(data_is_datetime):
            data = tuple(
                date2num(d) if is_datetime else d
                for d, is_datetime in zip(data, data_is_datetime)
            )

        xdim, ydim = element.dimensions('key', label=True)
        if self.p.filled:
            contour_type = Polygons
        else:
            contour_type = Contours
        vdims = element.vdims[:1]

        kwargs = {}
        levels = self.p.levels
        zmin, zmax = element.range(2)
        if isinstance(self.p.levels, int):
            if zmin == zmax:
                contours = contour_type([], [xdim, ydim], vdims)
                return (element * contours) if self.p.overlaid else contours
            data += (levels,)
        else:
            kwargs = {'levels': levels}

        fig = Figure()
        ax = Axes(fig, [0, 0, 1, 1])
        contour_set = QuadContourSet(ax, *data, filled=self.p.filled,
                                     extent=extent, **kwargs)
        levels = np.array(contour_set.get_array())
        crange = levels.min(), levels.max()
        if self.p.filled:
            levels = levels[:-1] + np.diff(levels)/2.
            vdims = [vdims[0].clone(range=crange)]

        paths = []
        empty = np.array([[np.nan, np.nan]])
        for level, cset in zip(levels, contour_set.collections):
            exteriors = []
            interiors = []
            for geom in cset.get_paths():
                interior = []
                polys = geom.to_polygons(closed_only=False)
                for ncp, cp in enumerate(polys):
                    if any(data_is_datetime[0:2]):
                        # transform x/y coordinates back to datetimes
                        xs, ys = np.split(cp, 2, axis=1)
                        if data_is_datetime[0]:
                            xs = np.array(num2date(xs))
                        if data_is_datetime[1]:
                            ys = np.array(num2date(ys))
                        cp = np.concatenate((xs, ys), axis=1)
                    if ncp == 0:
                        exteriors.append(cp)
                        exteriors.append(empty)
                    else:
                        interior.append(cp)
                if len(polys):
                    interiors.append(interior)
            if not exteriors:
                continue
            geom = {
                element.vdims[0].name:
                num2date(level) if data_is_datetime[2] else level,
                (xdim, ydim): np.concatenate(exteriors[:-1])
            }
            if self.p.filled and interiors:
                geom['holes'] = interiors
            paths.append(geom)
        contours = contour_type(paths, label=element.label, kdims=element.kdims, vdims=vdims)
        if self.p.overlaid:
            contours = element * contours
        return contours


class histogram(Operation):
    """
    Returns a Histogram of the input element data, binned into
    num_bins over the bin_range (if specified) along the specified
    dimension.
    """

    bin_range = param.NumericTuple(default=None, length=2,  doc="""
      Specifies the range within which to compute the bins.""")

    bins = param.ClassSelector(default=None, class_=(np.ndarray, list, tuple, str), doc="""
      An explicit set of bin edges or a method to find the optimal
      set of bin edges, e.g. 'auto', 'fd', 'scott' etc. For more
      documentation on these approaches see the np.histogram_bin_edges
      documentation.""")

    cumulative = param.Boolean(default=False, doc="""
      Whether to compute the cumulative histogram""")

    dimension = param.String(default=None, doc="""
      Along which dimension of the Element to compute the histogram.""")

    frequency_label = param.String(default=None, doc="""
      Format string defining the label of the frequency dimension of the Histogram.""")

    groupby = param.ClassSelector(default=None, class_=(str, Dimension), doc="""
      Defines a dimension to group the Histogram returning an NdOverlay of Histograms.""")

    log = param.Boolean(default=False, doc="""
      Whether to use base 10 logarithmic samples for the bin edges.""")

    mean_weighted = param.Boolean(default=False, doc="""
      Whether the weighted frequencies are averaged.""")

    normed = param.ObjectSelector(default=False,
                                  objects=[True, False, 'integral', 'height'],
                                  doc="""
      Controls normalization behavior.  If `True` or `'integral'`, then
      `density=True` is passed to np.histogram, and the distribution
      is normalized such that the integral is unity.  If `False`,
      then the frequencies will be raw counts. If `'height'`, then the
      frequencies are normalized such that the max bin height is unity.""")

    nonzero = param.Boolean(default=False, doc="""
      Whether to use only nonzero values when computing the histogram""")

    num_bins = param.Integer(default=20, doc="""
      Number of bins in the histogram .""")

    weight_dimension = param.String(default=None, doc="""
       Name of the dimension the weighting should be drawn from""")

    style_prefix = param.String(default=None, allow_None=None, doc="""
      Used for setting a common style for histograms in a HoloMap or AdjointLayout.""")

    def _process(self, element, key=None):
        if self.p.groupby:
            if not isinstance(element, Dataset):
                raise ValueError('Cannot use histogram groupby on non-Dataset Element')
            grouped = element.groupby(self.p.groupby, group_type=Dataset, container_type=NdOverlay)
            self.p.groupby = None
            return grouped.map(self._process, Dataset)

        normed = False if self.p.mean_weighted and self.p.weight_dimension else self.p.normed
        if self.p.dimension:
            selected_dim = self.p.dimension
        else:
            selected_dim = next(d.name for d in element.vdims + element.kdims)
        dim = element.get_dimension(selected_dim)

        if hasattr(element, 'interface'):
            data = element.interface.values(element, selected_dim, compute=False)
        else:
            data = element.dimension_values(selected_dim)

        is_datetime = isdatetime(data)
        if is_datetime:
            data = data.astype('datetime64[ns]').astype('int64')

        # Handle different datatypes
        is_finite = isfinite
        is_cupy = is_cupy_array(data)
        if is_cupy:
            import cupy
            full_cupy_support = Version(cupy.__version__) > Version('8.0')
            if not full_cupy_support and (normed or self.p.weight_dimension):
                data = cupy.asnumpy(data)
                is_cupy = False
            else:
                is_finite = cupy.isfinite

        # Mask data
        if is_ibis_expr(data):
            mask = data.notnull()
            if self.p.nonzero:
                mask = mask & (data != 0)
            data = data.to_projection()
            data = data[mask]
            no_data = not len(data.head(1).execute())
            data = data[dim.name]
        else:
            mask = is_finite(data)
            if self.p.nonzero:
                mask = mask & (data != 0)
            data = data[mask]
            da = dask_array_module()
            no_data = False if da and isinstance(data, da.Array) else not len(data)

        # Compute weights
        if self.p.weight_dimension:
            if hasattr(element, 'interface'):
                weights = element.interface.values(element, self.p.weight_dimension, compute=False)
            else:
                weights = element.dimension_values(self.p.weight_dimension)
            weights = weights[mask]
        else:
            weights = None

        # Compute bins
        if isinstance(self.p.bins, str):
            bin_data = cupy.asnumpy(data) if is_cupy else data
            edges = np.histogram_bin_edges(bin_data, bins=self.p.bins)
        elif isinstance(self.p.bins, (list, np.ndarray)):
            edges = self.p.bins
            if isdatetime(edges):
                edges = edges.astype('datetime64[ns]').astype('int64')
        else:
            hist_range = self.p.bin_range or element.range(selected_dim)
            # Suppress a warning emitted by Numpy when datetime or timedelta scalars
            # are compared. See https://github.com/numpy/numpy/issues/10095 and
            # https://github.com/numpy/numpy/issues/9210.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    action='ignore', message='elementwise comparison failed',
                    category=DeprecationWarning
                )
                null_hist_range = hist_range == (0, 0)
            # Avoids range issues including zero bin range and empty bins
            if null_hist_range or any(not isfinite(r) for r in hist_range):
                hist_range = (0, 1)
            steps = self.p.num_bins + 1
            start, end = hist_range
            if isinstance(start, str) or isinstance(end, str) or isinstance(steps, str):
                raise ValueError("Categorical data found. Cannot create histogram from categorical data.")
            if is_datetime:
                start, end = dt_to_int(start, 'ns'), dt_to_int(end, 'ns')
            if self.p.log:
                bin_min = max([abs(start), data[data>0].min()])
                edges = np.logspace(np.log10(bin_min), np.log10(end), steps)
            else:
                edges = np.linspace(start, end, steps)
        if is_cupy:
            edges = cupy.asarray(edges)

        if not is_dask_array(data) and no_data:
            nbins = self.p.num_bins if self.p.bins is None else len(self.p.bins)-1
            hist = np.zeros(nbins)
        elif hasattr(element, 'interface'):
            density = True if normed else False
            hist, edges = element.interface.histogram(
                data, edges, density=density, weights=weights
            )
            if normed == 'height':
                hist /= hist.max()
            if self.p.weight_dimension and self.p.mean_weighted:
                hist_mean, _ = element.interface.histogram(
                    data, density=False, bins=edges
                )
                hist /= hist_mean
        elif normed:
            # This covers True, 'height', 'integral'
            hist, edges = np.histogram(data, density=True,
                                       weights=weights, bins=edges)
            if normed == 'height':
                hist /= hist.max()
        else:
            hist, edges = np.histogram(data, normed=normed, weights=weights, bins=edges)
            if self.p.weight_dimension and self.p.mean_weighted:
                hist_mean, _ = np.histogram(data, density=False, bins=self.p.num_bins)
                hist /= hist_mean

        hist[np.isnan(hist)] = 0
        if is_datetime:
            edges = (edges/1e3).astype('datetime64[us]')

        params = {}
        if self.p.weight_dimension:
            params['vdims'] = [element.get_dimension(self.p.weight_dimension)]
        elif self.p.frequency_label:
            label = self.p.frequency_label.format(dim=dim.pprint_label)
            params['vdims'] = [Dimension('Frequency', label=label)]
        else:
            label = 'Frequency' if normed else 'Count'
            params['vdims'] = [Dimension(f'{dim.name}_{label.lower()}',
                                         label=label)]

        if element.group != element.__class__.__name__:
            params['group'] = element.group

        if self.p.cumulative:
            hist = np.cumsum(hist)
            if self.p.normed in (True, 'integral'):
                hist *= edges[1]-edges[0]

        # Save off the computed bin edges so that if this operation instance
        # is used to compute another histogram, it will default to the same
        # bin edges.
        self.bins = list(edges)
        return Histogram((edges, hist), kdims=[element.get_dimension(selected_dim)],
                         label=element.label, **params)


class decimate(Operation):
    """
    Decimates any column based Element to a specified number of random
    rows if the current element defined by the x_range and y_range
    contains more than max_samples. By default the operation returns a
    DynamicMap with a RangeXY stream allowing dynamic downsampling.
    """

    dynamic = param.Boolean(default=True, doc="""
       Enables dynamic processing by default.""")

    link_inputs = param.Boolean(default=True, doc="""
         By default, the link_inputs parameter is set to True so that
         when applying shade, backends that support linked streams
         update RangeXY streams on the inputs of the shade operation.""")

    max_samples = param.Integer(default=5000, doc="""
        Maximum number of samples to display at the same time.""")

    random_seed = param.Integer(default=42, doc="""
        Seed used to initialize randomization.""")

    streams = param.ClassSelector(default=[RangeXY], class_=(dict, list),
                                   doc="""
        List of streams that are applied if dynamic=True, allowing
        for dynamic interaction with the plot.""")

    x_range  = param.NumericTuple(default=None, length=2, doc="""
       The x_range as a tuple of min and max x-value. Auto-ranges
       if set to None.""")

    y_range  = param.NumericTuple(default=None, length=2, doc="""
       The x_range as a tuple of min and max y-value. Auto-ranges
       if set to None.""")

    _per_element = True

    def _process_layer(self, element, key=None):
        if not isinstance(element, Dataset):
            raise ValueError("Cannot downsample non-Dataset types.")
        if element.interface not in column_interfaces:
            element = element.clone(tuple(element.columns().values()))

        xstart, xend = self.p.x_range if self.p.x_range else element.range(0)
        ystart, yend = self.p.y_range if self.p.y_range else element.range(1)

        # Slice element to current ranges
        xdim, ydim = element.dimensions(label=True)[0:2]
        sliced = element.select(**{xdim: (xstart, xend),
                                   ydim: (ystart, yend)})

        if len(sliced) > self.p.max_samples:
            prng = np.random.RandomState(self.p.random_seed)
            choice = prng.choice(len(sliced), self.p.max_samples, False)
            return sliced.iloc[np.sort(choice)]
        return sliced

    def _process(self, element, key=None):
        return element.map(self._process_layer, Element)


class interpolate_curve(Operation):
    """
    Resamples a Curve using the defined interpolation method, e.g.
    to represent changes in y-values as steps.
    """

    interpolation = param.ObjectSelector(objects=['steps-pre', 'steps-mid',
                                                  'steps-post', 'linear'],
                                         default='steps-mid', doc="""
       Controls the transition point of the step along the x-axis.""")

    _per_element = True

    @classmethod
    def pts_to_prestep(cls, x, values):
        steps = np.zeros(2 * len(x) - 1)
        value_steps = tuple(np.empty(2 * len(x) - 1, dtype=v.dtype) for v in values)

        steps[0::2] = x
        steps[1::2] = steps[0:-2:2]

        val_arrays = []
        for v, s in zip(values, value_steps):
            s[0::2] = v
            s[1::2] = s[2::2]
            val_arrays.append(s)

        return steps, tuple(val_arrays)

    @classmethod
    def pts_to_midstep(cls, x, values):
        steps = np.zeros(2 * len(x))
        value_steps = tuple(np.empty(2 * len(x), dtype=v.dtype) for v in values)

        steps[1:-1:2] = steps[2::2] = x[:-1] + (x[1:] - x[:-1])/2
        steps[0], steps[-1] = x[0], x[-1]

        val_arrays = []
        for v, s in zip(values, value_steps):
            s[0::2] = v
            s[1::2] = s[0::2]
            val_arrays.append(s)

        return steps, tuple(val_arrays)

    @classmethod
    def pts_to_poststep(cls, x, values):
        steps = np.zeros(2 * len(x) - 1)
        value_steps = tuple(np.empty(2 * len(x) - 1, dtype=v.dtype) for v in values)

        steps[0::2] = x
        steps[1::2] = steps[2::2]

        val_arrays = []
        for v, s in zip(values, value_steps):
            s[0::2] = v
            s[1::2] = s[0:-2:2]
            val_arrays.append(s)

        return steps, tuple(val_arrays)

    def _process_layer(self, element, key=None):
        INTERPOLATE_FUNCS = {'steps-pre': self.pts_to_prestep,
                             'steps-mid': self.pts_to_midstep,
                             'steps-post': self.pts_to_poststep}
        if self.p.interpolation not in INTERPOLATE_FUNCS:
            return element
        x = element.dimension_values(0)
        is_datetime = isdatetime(x)
        if is_datetime:
            dt_type = 'datetime64[ns]'
            x = x.astype(dt_type)
        dvals = tuple(element.dimension_values(d) for d in element.dimensions()[1:])
        xs, dvals = INTERPOLATE_FUNCS[self.p.interpolation](x, dvals)
        if is_datetime:
            xs = xs.astype(dt_type)
        return element.clone((xs,)+dvals)

    def _process(self, element, key=None):
        return element.map(self._process_layer, Element)


#==================#
# Other operations #
#==================#


class collapse(Operation):
    """
    Given an overlay of Element types, collapse into single Element
    object using supplied function. Collapsing aggregates over the
    key dimensions of each object applying the supplied fn to each group.

    This is an example of an Operation that does not involve
    any Raster types.
    """

    fn = param.Callable(default=np.mean, doc="""
        The function that is used to collapse the curve y-values for
        each x-value.""")

    def _process(self, overlay, key=None):
        if isinstance(overlay, NdOverlay):
            collapse_map = HoloMap(overlay)
        else:
            collapse_map = HoloMap({i: el for i, el in enumerate(overlay)})
        return collapse_map.collapse(function=self.p.fn)


class gridmatrix(param.ParameterizedFunction):
    """
    The gridmatrix operation takes an Element or HoloMap
    of Elements as input and creates a GridMatrix object,
    which plots each dimension in the Element against
    each other dimension. This provides a very useful
    overview of high-dimensional data and is inspired
    by pandas and seaborn scatter_matrix implementations.
    """

    chart_type = param.Parameter(default=Scatter, doc="""
        The Element type used to display bivariate distributions
        of the data.""")

    diagonal_type = param.Parameter(default=None, doc="""
       The Element type along the diagonal, may be a Histogram or any
       other plot type which can visualize a univariate distribution.
       This parameter overrides diagonal_operation.""")

    diagonal_operation = param.Parameter(default=histogram, doc="""
       The operation applied along the diagonal, may be a histogram-operation
       or any other function which returns a viewable element.""")

    overlay_dims = param.List(default=[], doc="""
       If a HoloMap is supplied, this will allow overlaying one or
       more of its key dimensions.""")

    def __call__(self, data, **params):
        p = param.ParamOverrides(self, params)

        if isinstance(data, (HoloMap, NdOverlay)):
            ranges = {d.name: data.range(d) for d in data.dimensions()}
            data = data.clone({k: GridMatrix(self._process(p, v, ranges))
                               for k, v in data.items()})
            data = Collator(data, merge_type=type(data))()
            if p.overlay_dims:
                data = data.map(lambda x: x.overlay(p.overlay_dims), (HoloMap,))
            return data
        elif isinstance(data, Element):
            data = self._process(p, data)
            return GridMatrix(data)


    def _process(self, p, element, ranges={}):
        # Creates a unified Dataset.data attribute
        # to draw the data from
        if isinstance(element.data, np.ndarray):
            el_data = element.table(default_datatype)
        else:
            el_data = element.data

        # Get dimensions to plot against each other
        types = (str, np.str_, np.object_)+datetime_types
        dims = [d for d in element.dimensions()
                if _is_number(element.range(d)[0]) and
                not issubclass(element.get_dimension_type(d), types)]
        permuted_dims = [(d1, d2) for d1 in dims
                         for d2 in dims[::-1]]

        # Convert Histogram type to operation to avoid one case in the if below.
        if p.diagonal_type is Histogram:
            p.diagonal_type = None
            p.diagonal_operation = histogram

        data = {}
        for d1, d2 in permuted_dims:
            if d1 == d2:
                if p.diagonal_type is not None:
                    if p.diagonal_type._auto_indexable_1d:
                        el = p.diagonal_type(el_data, kdims=[d1], vdims=[d2],
                                             datatype=[default_datatype])
                    else:
                        values = element.dimension_values(d1)
                        el = p.diagonal_type(values, kdims=[d1])
                elif p.diagonal_operation is None:
                    continue
                elif p.diagonal_operation is histogram or isinstance(p.diagonal_operation, histogram):
                    bin_range = ranges.get(d1.name, element.range(d1))
                    el = p.diagonal_operation(element, dimension=d1.name, bin_range=bin_range)
                else:
                    el = p.diagonal_operation(element, dimension=d1.name)
            else:
                kdims, vdims = ([d1, d2], []) if len(p.chart_type.kdims) == 2 else (d1, d2)
                el = p.chart_type(el_data, kdims=kdims, vdims=vdims,
                                  datatype=[default_datatype])
            data[(d1.name, d2.name)] = el
        return data
