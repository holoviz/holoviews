from matplotlib import pyplot as plt
import numpy as np

import param

from ..core import Dimension, ViewOperation, Overlay
from ..views import ItemTable, SheetMatrix, VectorField, Contours


class chain(ViewOperation):
    """
    Definining a viewoperation chain is an easy way to define a new
    ViewOperation from a series of existing ones. The single argument
    is a callable that accepts an input view and returns a list of
    output views. To create the custom ViewOperation, you will need to
    supply this argument to a new instance of chain. For example:

    chain.instance(chain=lambda x: [cmap2rgb(operator(x).N, cmap='jet')])

    This is now a ViewOperation that sums the data in the input
    overlay and turns it into an RGB SheetMatrix with the 'jet'
    colormap.
    """

    chain = param.Callable(doc="""A chain of existing ViewOperations.""")

    def _process(self, view, key=None):
        return self.p.chain(view)


class operator(ViewOperation):
    """
    Applies any arbitrary operator on the data (currently only
    supports SheetViews) and returns the result.
    """

    operator = param.Callable(np.add, doc="""
        The binary operator to apply between the data attributes of
        the supplied Views. By default, performs elementwise addition
        across the SheetMatrix arrays.""")

    label = param.String(default='Operation', doc="""
        The label for the result after having applied the operator.""")

    def _process(self, overlay, key=None):

        if not isinstance(overlay, Overlay):
            raise Exception("Operation requires an Overlay as input")

        new_data = self.p.operator(*[el.data for el in overlay.data])
        return [SheetMatrix(new_data, bounds=overlay[0].bounds, label=self.p.label,
                            roi_bounds=overlay[0].roi_bounds)]


class contours(ViewOperation):
    """
    Given a SheetMatrix with a single channel, annotate it with contour
    lines for a given set of contour levels.

    The return is a overlay with a Contours layer for each given
    level, overlaid on top of the input SheetMatrix.
    """

    levels = param.NumericTuple(default=(0.5,), doc="""
         A list of scalar values used to specify the contour levels.""")

    label = param.String(default='Level', doc="""
      The label suffix used to label the resulting contour curves
      where the suffix is added to the label of the  input SheetMatrix""")

    def _process(self, sheetview, key=None):

        figure_handle = plt.figure()
        (l, b, r, t) = sheetview.bounds.lbrt()
        contour_set = plt.contour(sheetview.data, extent=(l, r, t, b),
                                  levels=self.p.levels)

        contours = []
        for level, cset in zip(self.p.levels, contour_set.collections):
            paths = cset.get_paths()
            lines = [path.vertices for path in paths]
            contours.append(Contours(lines, label=sheetview.label + ' ' + self.p.label))

        plt.close(figure_handle)

        if len(contours) == 1:
            return [(sheetview * contours[0])]
        else:
            return [sheetview * Overlay(contours, sheetview.bounds)]


class vectorfield(ViewOperation):
    """
    Given a SheetMatrix with a single channel, convert it to a
    VectorField object at a given spatial sampling interval. The
    values in the SheetMatrix are assumed to correspond to the vector
    angle in radians and the value is assumed to be cyclic.

    If supplied with an overlay, the second sheetview in the overlay
    will be interpreted as the third vector dimension.
    """

    rows = param.Integer(default=10, doc="""
         Number of rows in the vector field.""")

    cols = param.Integer(default=10, doc="""
         Number of columns in the vector field.""")

    label = param.String(default='Vectors', doc="""
      The label suffix used to label the resulting vector field
      where the suffix is added to the label of the  input SheetMatrix""")


    def _process(self, view, key=None):

        if isinstance(view, Overlay) and len(view) >= 2:
            radians, lengths = view[0], view[1]
        else:
            radians, lengths = view, None

        if not radians.value.cyclic:
            raise Exception("First input SheetMatrix must be declared cyclic")

        l, b, r, t = radians.bounds.lbrt()
        X, Y = np.meshgrid(np.linspace(l, r, self.p.cols+2)[1:-1],
                           np.linspace(b, t, self.p.rows+2)[1:-1])

        vector_data = []
        for x, y in zip(X.flat, Y.flat):

            components = (x,y, radians[x,y])
            if lengths is not None:
                components += (lengths[x,y],)

            vector_data.append(components)

        value_dimension = Dimension('VectorField', cyclic=True, range=radians.range)
        return [VectorField(vector_data,
                            label=radians.label + ' ' + self.p.label,
                            value=value_dimension)]


class threshold(ViewOperation):
    """
    Threshold a given SheetMatrix at a given level into the specified
    low and high values.  """

    level = param.Number(default=0.5, doc="""
       The value at which the threshold is applied. Values lower than
       the threshold map to the 'low' value and values above map to
       the 'high' value.""")

    high = param.Number(default=1.0, doc="""
      The value given to elements greater than (or equal to) the
      threshold.""")

    low = param.Number(default=0.0, doc="""
      The value given to elements below the threshold.""")

    label = param.String(default='Thresholded', doc="""
       The label suffix used to label the resulting sheetview where
       the suffix is added to the label of the input SheetMatrix""")

    def _process(self, view, key=None):
        arr = view.data
        high = np.ones(arr.shape) * self.p.high
        low = np.ones(arr.shape) * self.p.low
        thresholded = np.where(arr > self.p.level, high, low)

        return [SheetMatrix(thresholded,
                          label=view.label + ' ' + self.p.label)]


class roi_table(ViewOperation):
    """
    Compute a table of information from a SheetMatrix within the
    indicated region-of-interest (ROI). The function applied must
    accept a numpy array and return either a single value or a
    dictionary of values which is returned as a ItemTable or ViewMap.

    The roi is specified using a boolean SheetMatrix overlay (e.g as
    generated by threshold) where zero values are excluded from the
    ROI.
    """

    fn = param.Callable(default=np.mean, doc="""
        The function that is applied within the mask area to supply
        the relevant information. Other valid examples include np.sum,
        np.median, np.std etc.""")

    heading = param.String(default='result', doc="""
       If the output of the function is not a dictionary, this is the
       label given to the resulting value.""")

    label = param.String(default='ROI', doc="""
       The label suffix that labels the resulting table where this
       suffix is added to the label of the input data SheetMatrix""")


    def _process(self, view, key=None):

        if not isinstance(view, Overlay) or len(view) != 2:
            raise Exception("A Overlay of two SheetViews is required.")

        sview, mask = view[0], view[1]

        if sview.data.shape != mask.data.shape:
            raise Exception("The data array shape of the mask layer"
                            " must match that of the data layer.")

        bit_mask = mask.data.astype(np.bool)
        roi = sview.data[bit_mask]

        results = self.p.fn(roi)
        if not isinstance(results, dict):
            results = {self.p.heading:results}

        return [ItemTable(results, label=sview.label + ' ' + self.p.label)]