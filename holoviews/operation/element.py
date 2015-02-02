import numpy as np

import param

from ..core import Dimension, ElementOperation, CompositeOverlay, NdOverlay, Overlay
from ..core.util import find_minmax
from ..element.chart import Histogram, VectorField
from ..element.annotation import Contours
from ..element.raster import Matrix
from ..element.tabular import ItemTable



class chain(ElementOperation):
    """
    Defining an ElementOperation chain is an easy way to define a new
    ElementOperation from a series of existing ones. The single
    argument is a callable that accepts an input element and returns
    the final, transformed element. To create the custom
    ElementOperation, you will need to supply this argument to a new
    instance of chain. For example:

    chain.instance(
       chain=lambda x: colormap(operator(x, operator=np.add), cmap='jet'))

    This defines an ElementOperation that sums the data in the input
    and turns it into an RGB Matrix using the 'jet' colormap.
    """

    value = param.String(default='Chain', doc="""
        The value assigned to the result after having applied the chain.""")


    chain = param.Callable(doc="""A chain of existing ViewOperations.""")

    def _process(self, view, key=None):
        return self.p.chain(view).clone(shared_data=True, value=self.p.value)



class operator(ElementOperation):
    """
    Applies any arbitrary collapsing operator across the data elements
    of the input overlay and returns the result.

    As applying collapse operations on arbitrary data works very
    naturally using arrays, the result is a Matrix containing the
    computed result data.
    """

    output_type = Matrix

    operator = param.Callable(np.add, doc="""
        The commutative operator to apply between the data attributes
        of the supplied Views used to collapse the data.

        By default applies elementwise addition across the input data.""")

    unpack = param.Boolean(default=True, doc="""
       Whether the operator is supplied the .data attributes as an
       unpack collection of arguments or as a list.""")

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
        return matrix * contours



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

    fn = param.Callable(default=np.mean, doc="""
        The function that is applied within the mask area to supply
        the relevant information.

        The function may return a single value (e.g np.sum, np.median,
        np.std) or a dictionary of values..""")

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
