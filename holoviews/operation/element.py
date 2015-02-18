import numpy as np

import param

from ..core import Dimension, ElementOperation, CompositeOverlay, \
                   NdOverlay, Overlay, BoundingBox
from ..core.util import find_minmax
from ..element.chart import Histogram, VectorField, Curve
from ..element.raster import Matrix
from ..element.tabular import ItemTable
from ..element.path import Contours
from .normalization import raster_normalization


class chain(ElementOperation):
    """
    Defining an ElementOperation chain is an easy way to define a new
    ElementOperation from a series of existing ones. The argument is a
    list of ElementOperation (or ElementOperation instances) that are
    called in turn until the final, transformed element is returned.

    chain(operations=[collapse.instance(operator=np.add), colormap])

    This first sums the data in the input with collapse (using np.add)
    and then returns an RGB Matrix applying the default colormap of
    the colormap operator.

    Instances are only required when arguments need to be passed to
    individual operators so that the result is a function over one
    argument.
    """

    output_type = param.Parameter(Matrix, doc="""
        The output type of the chain operation. Must be supplied if
        the chain is to be used as a channel operation.""")

    value = param.String(default='Chain', doc="""
        The value assigned to the result after having applied the chain.""")


    operations = param.List(default=[], class_=ElementOperation, doc="""
       A list of ElementOperations (or ElementOperation instances)
       that are applied on the input from left to right..""")

    def _process(self, view, key=None):
        processed = view
        for operation in self.p.operations:
            processed = operation.process_element(processed, key, input_ranges=self.p.input_ranges)

        return processed.clone(value=self.p.value)


class transform(ElementOperation):
    """
    Generic ElementOperation to transform an input Matrix or RGBA
    element into an output Matrix. The transformation is defined by
    the supplied callable that accepts the data of the input Matrix
    (typically a numpy array) and returns the transformed data of the
    output Matrix.

    This operator is extremely versatile; for instance, you could
    implement an alternative to the explict threshold operator with:

    operator=lambda x: np.clip(x, 0, 0.5)

    Alternatively, you can implement a transform computing the 2D
    autocorrelation using the scipy library with:

    operator=lambda x: scipy.signal.correlate2d(x, x)
    """

    output_type = Matrix

    value = param.String(default='Transform', doc="""
        The value assigned to the result after applying the
        transform.""")

    operator = param.Callable(doc="""
       Function of one argument that transforms the data in the input
       Matrix to the data in the output Matrix. By default, acts as
       the identity function such that the output matches the input.""")

    def _process(self, matrix, key=None):
        processed = (matrix.data if not self.p.operator
                     else self.p.operator(matrix.data))
        return Matrix(processed, matrix.bounds, value=self.p.value)


#==============================#
# Raster processing operations #
#==============================#


class matrix_overlay(ElementOperation):
    """
    Operation to build a matrix overlay to a specification from a
    subset of the required elements.

    This is useful for reordering the elements of an overlay,
    duplicating layers of an overlay or creating blank matrix elements
    in the appropriate positions.

    For instance, matrix_overlay may build a three layered input
    suitable for the toRGB operation even if supplied with one or two
    of the required channels (creating blank channels for the missing
    elements).

    Note that if there is any ambiguity regarding the match, the
    strongest match will be used. In the case of a tie in match
    strength, the first layer in the input is used. One successful
    match is always required.
    """

    output_type = Overlay

    spec = param.String(doc="""
       Specification of the output Overlay structure. For instance:

       Matrix.R * Matrix.G * Matrix.B

       Will ensure an overlay of this structure is created even if
       (for instance) only (Matrix.R * Matrix.B) is supplied.

       Elements in the input overlay that match are placed in the
       appropriate positions and unavailable specification elements
       are created with the specified fill value.""")

    fill = param.Number(default=0)

    value = param.String(default='Transform', doc="""
        The value assigned to the resulting overlay.""")


    @classmethod
    def _match(cls, el, spec):
        "Return the strength of the match (None if no match)"
        spec_dict = dict(zip(['type', 'value', 'label'], spec.split('.')))
        if not isinstance(el, Matrix) or spec_dict['type'] != 'Matrix':
            raise NotImplementedError("Only Matrix currently supported")

        strength = 1
        for key in ['value', 'label']:
            attr_value = getattr(el, key)
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
            raise Exception("The matrix_overlay operation requires at least one match")

        completed = []
        strongest = ordering[np.argmax(strengths)]
        for el, spec in zip(ordering, specs):
            if el is None:
                spec_dict = dict(zip(['type', 'value', 'label'], spec.split('.')))
                el = Matrix(np.ones(strongest.data.shape) * self.p.fill,
                            value=spec_dict.get('value','Matrix'),
                            label=spec_dict.get('label',''))
            completed.append(el)
        return np.prod(completed)


class collapse(ElementOperation):
    """
    Applies any arbitrary collapsing operator across the data elements
    of the input overlay and returns the result.

    As applying collapse operations on arbitrary data works very
    naturally using arrays, the result is a Matrix containing the
    computed result data.
    """

    output_type = Matrix

    operator = param.Callable(np.add, doc="""
        The collapsing operator to apply between the data attributes
        of the supplied Views used to collapse the data.

        By default applies elementwise addition across the input
        data. In unpack is set to True, needs to be used with
        operators than can take an arbitrary number of inputs.

       Simple example operators include:
          np.add, np.subtract, np.multiply np.divide

       For more complex example see the documentation for unpack.""")

    unpack = param.Boolean(default=True, doc="""
       Whether the operator is supplied the .data attributes as an
       unpack collection of arguments or as a list.

       Using unpack=False, more complex operators may be used such as:

        lambda x: np.mean(x, axis=0)
        lambda x: np.std(x, axis=0)
        lambda x: np.var(x, axis=0)
        """)

    value = param.String(default='Operation', doc="""
        The value assigned to the result after having applied the operator.""")

    def _process(self, overlay, key=None):
        if not isinstance(overlay, CompositeOverlay):
            raise Exception("Operation requires an Overlay type as input")

        if self.p.unpack:
            new_data = self.p.operator(*[el.data for el in overlay])
        else:
            new_data = self.p.operator([el.data for el in overlay])

        return Matrix(new_data, bounds=overlay[0].bounds,
                      label=self.get_overlay_label(overlay))



class threshold(ElementOperation):
    """
    Threshold a given Matrix whereby all values higher than a given
    level map to the specified high value and all values lower than
    that level map to the specified low value.
    """
    output_type = Matrix

    level = param.Number(default=0.5, doc="""
       The value at which the threshold is applied. Values lower than
       the threshold map to the 'low' value and values above map to
       the 'high' value.""")

    high = param.Number(default=1.0, doc="""
      The value given to elements greater than (or equal to) the
      threshold.""")

    low = param.Number(default=0.0, doc="""
      The value given to elements below the threshold.""")

    value = param.String(default='Threshold', doc="""
       The value assigned to the thresholded output.""")

    def _process(self, matrix, key=None):

        if not isinstance(matrix, Matrix):
            raise TypeError("The threshold operation requires a Matrix as input.")

        arr = matrix.data
        high = np.ones(arr.shape) * self.p.high
        low = np.ones(arr.shape) * self.p.low
        thresholded = np.where(arr > self.p.level, high, low)

        return matrix.clone(thresholded, value=self.p.value)



class gradient(ElementOperation):
    """
    Compute the gradient plot of the supplied Matrix.

    If the Matrix value dimension is cyclic, negative differences will
    be wrapped into the cyclic range.
    """

    output_type = Matrix

    value = param.String(default='Gradient', doc="""
    The value assigned to the output gradient matrix.""")


    @classmethod
    def wrap(cls, lower, upper, x):
        """
        Circularly alias the numeric value x into the range
        [lower,upper). Corresponds to a function of the same name in
        the ImaGen project.

        Valid for cyclic quantities.
        """
        range_=upper-lower
        return lower + np.fmod(x-lower + 2*range_*(1-np.floor(x/(2*range_))), range_)


    def _process(self, matrix, key=None):

        if len(matrix.value_dimensions) != 1:
            raise ValueError("Input matrix to gradient operation must "
                             "have single value dimension.")

        matrix_dim = matrix.value_dimensions[0]

        data = matrix.data
        r, c = data.shape
        dx = np.diff(data, 1, axis=1)[0:r-1, 0:c-1]
        dy = np.diff(data, 1, axis=0)[0:r-1, 0:c-1]

        cyclic_range = 1.0 if not matrix_dim.cyclic else matrix_dim.range
        if cyclic_range is not None: # Wrap into the specified range
            # Convert negative differences to an equivalent positive value
            dx = self.wrap(0, cyclic_range, dx)
            dy = self.wrap(0, cyclic_range, dy)
            #
            # Make it increase as gradient reaches the halfway point,
            # and decrease from there
            dx = 0.5 * cyclic_range - np.abs(dx - 0.5 * cyclic_range)
            dy = 0.5 * cyclic_range - np.abs(dy - 0.5 * cyclic_range)

        return Matrix(np.sqrt(dx * dx + dy * dy), matrix.bounds, value=self.p.value)



class fft_power(ElementOperation):
    """
    Given a Matrix element, compute the power of the 2D Fast Fourier
    Transform (FFT).
    """

    output_type = Curve

    max_power = param.Number(default=1.0, doc="""
    The maximum power value of the output power spectrum.""")

    value = param.String(default='FFT Power', doc="""
    The value assigned to the output power spectrum.""")


    def _process(self, matrix, key=None):

        if self.p.input_ranges:
            normfn = raster_normalization.instance()
            matrix = normfn.process_element(matrix, key, *self.p.input_ranges)

        fft_spectrum = abs(np.fft.fftshift(np.fft.fft2(matrix.data - 0.5, s=None, axes=(-2, -1))))
        fft_spectrum = 1 - fft_spectrum # Inverted spectrum by convention
        zero_min_spectrum = fft_spectrum - fft_spectrum.min()
        spectrum_range = fft_spectrum.max() - fft_spectrum.min()
        spectrum = (self.p.max_power * zero_min_spectrum) / spectrum_range

        l, b, r, t = matrix.bounds.lbrt()
        density = matrix.xdensity
        bounds = BoundingBox(radius=(density/2)/(r-l))

        return Matrix(spectrum, bounds, label=matrix.label, value=self.p.value)



class convolve(ElementOperation):
    """
    Apply a convolution to an overlay using the top layer as the
    kernel for convolving the bottom layer. Both Matrix elements in
    the input overlay should have a single value dimension.
    """

    output_type = Matrix

    value = param.String(default='Convolution', doc="""
        The value assigned to the convolved output.""")

    kernel_roi = param.NumericTuple(default=(0,0,0,0), length=4, doc="""
        A 2-dimensional slice of the kernel layer to use in the
        convolution in lbrt (left, bottom, right, top) format. By
        default, no slicing is applied.""")

    def _process(self, overlay, key=None):
        if len(overlay) != 2:
            raise Exception("Overlay must contain at least to items.")

        [target, kernel] = overlay[0], overlay[1]

        if len(target.value_dimensions) != 1:
            raise Exception("Convolution requires inputs with single value dimensions.")

        xslice = slice(self.p.kernel_roi[0], self.p.kernel_roi[2])
        yslice = slice(self.p.kernel_roi[1], self.p.kernel_roi[3])

        k = kernel.data if self.p.kernel_roi == (0,0,0,0) else kernel[xslice, yslice].data

        fft1 = np.fft.fft2(target.data)
        fft2 = np.fft.fft2(k, s= target.data.shape)
        convolved_raw = np.fft.ifft2(fft1 * fft2).real

        k_rows, k_cols = k.shape
        rolled = np.roll(np.roll(convolved_raw, -(k_cols//2), axis=-1), -(k_rows//2), axis=-2)
        convolved = rolled / float(k.sum())

        return Matrix(convolved, bounds=target.bounds, value=self.p.value)



class split_raster(ElementOperation):
    """
    Given a Raster element, return the individual value dimensions as
    an overlay of Matrix elements.
    """

    value = param.String(default='', doc="""
       Optional suffix appended to the value dimensions in the
       components of the output overlay. Default keeps the value
       strings identical to those in the input raster.""")

    def _process(self, raster, key=None):
        matrices = []
        for i, dim in enumerate(raster.value_dimensions):
            matrix = Matrix(raster.data[:, :, i],
                            value_dimensions = [dim(name=dim.name+self.p.value)])
            matrices.append(matrix)
        return np.product(matrices)



class contours(ElementOperation):
    """
    Given a Matrix with a single channel, annotate it with contour
    lines for a given set of contour levels.

    The return is an NdOverlay with a Contours layer for each given
    level, overlaid on top of the input Matrix.
    """

    output_type = Overlay

    levels = param.NumericTuple(default=(0.5,), doc="""
        A list of scalar values used to specify the contour levels.""")

    value = param.String(default='Level', doc="""
        The value assigned to the output contours.""")


    def _process(self, matrix, key=None):
        from matplotlib import pyplot as plt

        figure_handle = plt.figure()
        (l, b, r, t) = matrix.extents
        contour_set = plt.contour(matrix.data, extent=(l, r, t, b),
                                  levels=self.p.levels)

        contours = NdOverlay(None, key_dimensions=['Levels'])
        for level, cset in zip(self.p.levels, contour_set.collections):
            paths = cset.get_paths()
            lines = [path.vertices for path in paths]
            contours[level] = Contours(lines, value=self.p.value)

        plt.close(figure_handle)
        return Overlay.from_values(matrix * contours)



class histogram(ElementOperation):
    """
    Returns a Histogram of the Raster data, binned into
    num_bins over the bin_range (if specified).

    If adjoin is True, the histogram will be returned adjoined to
    the Raster as a side-plot.
    """

    adjoin = param.Boolean(default=True, doc="""
      Whether to adjoin the histogram to the ViewableElement.""")

    bin_range = param.NumericTuple(default=(0, 0), doc="""
      Specifies the range within which to compute the bins.""")

    dimension = param.String(default=None, doc="""
      Along which dimension of the ViewableElement to compute the histogram.""")

    normed = param.Boolean(default=True, doc="""
      Whether the histogram frequencies are normalized.""")

    individually = param.Boolean(default=True, doc="""
      Specifies whether the histogram will be rescaled for each Raster in a UniformNdMapping.""")

    num_bins = param.Integer(default=20, doc="""
      Number of bins in the histogram .""")

    style_prefix = param.String(default=None, allow_None=None, doc="""
      Used for setting a common style for histograms in a HoloMap or AdjointLayout.""")

    def _process(self, view, key=None):
        if self.p.dimension:
            selected_dim = self.p.dimension
        else:
            selected_dim = [d.name for d in view.value_dimensions][0]
        data = np.array(view.dimension_values(selected_dim))
        range = find_minmax((np.min(data), np.max(data)), (0, -float('inf')))\
            if self.p.bin_range is None else self.p.bin_range

        # Avoids range issues including zero bin range and empty bins
        if range == (0, 0):
            range = (0.0, 0.1)
        try:
            data = data[np.invert(np.isnan(data))]
            hist, edges = np.histogram(data, normed=self.p.normed,
                                       range=range, bins=self.p.num_bins)
        except:
            edges = np.linspace(range[0], range[1], self.p.num_bins + 1)
            hist = np.zeros(self.p.num_bins)
        hist[np.isnan(hist)] = 0

        hist_view = Histogram(hist, edges, key_dimensions=[view.get_dimension(selected_dim)],
                              label=view.label)

        return (view << hist_view) if self.p.adjoin else hist_view



class vectorfield(ElementOperation):
    """
    Given a Matrix with a single channel, convert it to a VectorField
    object at a given spatial sampling interval. The values in the
    Matrix are assumed to correspond to the vector angle in radians
    and the value is assumed to be cyclic.

    If supplied with an Overlay, the second sheetview in the overlay
    will be interpreted as the third vector dimension.
    """

    output_type = VectorField

    rows = param.Integer(default=10, doc="""
       The number of rows in the vector field.""")

    cols = param.Integer(default=10, doc="""
       The number of columns in the vector field.""")

    value = param.String(default='Vectors', doc="""
       The value assigned to the output vector field.""")


    def _process(self, view, key=None):

        if isinstance(view, CompositeOverlay) and len(view) >= 2:
            radians, lengths = view[0], view[1]
        else:
            radians, lengths = view, None

        cyclic_dim = radians.value_dimensions[0]
        if not cyclic_dim.cyclic:
            raise Exception("First input Matrix must be declared cyclic")

        l, b, r, t = radians.bounds.lbrt()
        X, Y = np.meshgrid(np.linspace(l, r, self.p.cols+2)[1:-1],
                           np.linspace(b, t, self.p.rows+2)[1:-1])

        vector_data = []
        for x, y in zip(X.flat, Y.flat):
            components = (x,y, radians[x,y])
            if lengths is not None:
                components += (lengths[x,y],)

            vector_data.append(components)

        value_dimensions = [Dimension('Magnitude'),
                            Dimension('Angle', cyclic=True, range=cyclic_dim.range)]
        return VectorField(vector_data, label=radians.label, value=self.p.value,
                           value_dimensions=value_dimensions)



class analyze_roi(ElementOperation):
    """
    Compute a table of information from a Matrix within the indicated
    region-of-interest (ROI). The function applied must accept a numpy
    array and return either a single value or a dictionary of values
    which is returned as a ItemTable or HoloMap.

    The roi is specified using a boolean Matrix overlay (e.g as
    generated by threshold) where all zero values are excluded from
    the analysis.
    """

    output_type = ItemTable

    fn = param.Callable(default=np.mean, doc="""
        The function that is applied within the mask area to supply
        the relevant information.

        The function may return a single value (e.g np.sum, np.median,
        np.std) or a dictionary of values.""")

    heading = param.String(default='result', doc="""
       If the output of the function is not a dictionary, this is the
       row label given to the resulting value.""")

    value = param.String(default='ROI', doc="""
       The value assigned to the output analysis table.""")

    def _process(self, overlay, key=None):

        if not isinstance(overlay, CompositeOverlay) or len(overlay) != 2:
            raise Exception("A CompositeOverlay object of two Matrix elements is required.")

        matrix, mask = overlay[0], overlay[1]

        if matrix.data.shape != mask.data.shape:
            raise Exception("The data array shape of the mask layer"
                            " must match that of the data layer.")

        bit_mask = mask.data.astype(np.bool)
        roi = matrix.data[bit_mask]

        if len(roi) == 0:
            raise Exception("No region of interest defined in ROI mask" )

        results = self.p.fn(roi)
        if not isinstance(results, dict):
            results = {self.p.heading:results}

        return ItemTable(results,
                         label=self.get_overlay_label(overlay),
                         value=self.p.value)


#==================#
# Other operations #
#==================#


class collapse_curve(ElementOperation):
    """
    Given an overlay of Curves, compute a new curve which is collapsed
    for each x-value given a specified function.

    This is an example of an ElementOperation that does not involve
    any Raster types.
    """

    output_type = Curve

    fn = param.Callable(default=np.mean, doc="""
        The function that is used to collapse the curve y-values for
        each x-value.""")

    value = param.String(default='Collapses', doc="""
       The value assigned to the collapsed curve output.""")

    def _process(self, overlay, key=None):

        for curve in overlay:
            if not isinstance(curve, Curve):
                raise ValueError("The collapse_curve operation requires Curves as input.")
            if not all(curve.data[:,0] == overlay[0].data[:,0]):
                raise ValueError("All input curves must have same x-axis values.")

        data = []
        for i, xval in enumerate(overlay[0].data[:,0]):
            yval = self.p.fn([c.data[i,1]  for c in overlay])
            data.append((xval, yval))

        return Curve(np.array(data), value=self.p.value,
                     label=self.get_overlay_label(overlay))




