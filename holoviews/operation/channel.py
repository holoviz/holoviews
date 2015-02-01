import colorsys

import numpy as np

import param

from ..core.operation import ElementOperation
from ..core.options import Options, Cycle
from ..element import Matrix, RGB
from .normalization import raster_normalization
from .element import split_raster

rgb_to_hsv = np.vectorize(colorsys.rgb_to_hsv)
hsv_to_rgb = np.vectorize(colorsys.hsv_to_rgb)


class toRGB(ElementOperation):
    """
    Accepts an overlay containing either 3 or 4 layers. The first
    three layers are the R,G, B channels and the last layer (if
    supplied) is the (optional) alpha channel.

    Note that the values in the input Matrix elements are expected to
    be bounded between 0.0-1.0. If this isn't the case, the values of
    these channels will be clipped into this range.
    """

    output_type = RGB

    value = param.String(default='RGB', doc="""
        The value string for the output RGB element.""")


    def _process(self, overlay, key=None):
        if len(overlay) not in [3, 4]:
            raise Exception("Requires 3 or 4 layers to convert to RGB(A)")
        if not all(isinstance(el, Matrix) for el in overlay):
            raise Exception("All layers must be of type Matrix to convert to RGB")
        if not all(el.depth == 1 for el in overlay):
            raise Exception("All Matrix elements must have single value"
                            " dimension for conversion to RGB")
        if not all(el.bounds == overlay[0].bounds for el in overlay):
            raise Exception("All input Matrix elements must have the same bounds.")
        if not all(el.data.shape == overlay[0].data.shape for el in overlay):
            raise Exception("All input Matrix must have data with matching shape.")

        if self.p.input_ranges:
            normfn = raster_normalization.instance()
            overlay = normfn.process_element(overlay, key, *self.p.input_ranges)

        arrays = []
        for el in overlay:
            if el.data.max() > 1.0 or el.data.min() < 0:
                self.warning("Clipping data into the interval [0, 1]")
                data = el.data.clip(0,1.0)
                arrays.append(data)
            else:
                arrays.append(el.data.copy())

        return RGB(np.dstack(arrays),
                   bounds = self.get_overlay_extents(overlay),
                   label  = self.get_overlay_label(overlay),
                   value  = self.p.value)



class toHCS(ElementOperation):
    """
    Hue-Confidence-Strength plot.

    Accepts an overlay containing either 2 or 3 layers. The first two
    layers are hue and confidence and the third layer (if available)
    is the strength channel.
    """

    output_type = RGB

    S_multiplier = param.Number(default=1.0, bounds=(0.0,None), doc="""
        Post-normalization multiplier for the strength value.

        Note that if the result is outside the bounds 0.0-1.0, it will
        be clipped. """)

    C_multiplier = param.Number(default=1.0, bounds=(0.0,None), doc="""
        Post-normalization multiplier for the confidence value.

        Note that if the result is outside the bounds 0.0-1.0, it will
        be clipped.""")

    flipSC = param.Boolean(default=False, doc="""
        Whether to flip the strength and confidence channels""")

    value = param.String(default='HCS', doc="""
        The value string for the output (an RGB element).""")

    def _process(self, overlay, key=None):

        if self.p.input_ranges:
            normfn = raster_normalization.instance()
            overlay = normfn.process_element(overlay, key, *self.p.input_ranges)

        hue, confidence = overlay[0], overlay[1]
        strength_data = overlay[2].data if (len(overlay) == 3) else np.ones(hue.shape)

        hue_range = hue.value_dimensions[0].range
        if (not hue.value_dimensions[0].cyclic) or (None in hue_range):
            raise Exception("The input hue channel must be declared cyclic with a defined range.")
        else:
            hue_data = hue.data - hue_range[0]
            hue_data /= (hue_range[1] - hue_range[0])

        if hue.shape != confidence.shape:
            raise Exception("Cannot combine input Matrices with different shapes.")

        (h,s,v)= (hue_data,
                  (confidence.data * self.p.C_multiplier).clip(0.0, 1.0),
                  (strength_data * self.p.S_multiplier).clip(0.0, 1.0))

        if self.p.flipSC:
            (h,s,v) = (h,v,s.clip(0,1.0))

        return RGB(np.dstack(hsv_to_rgb(h,s,v)),
                   bounds = self.get_overlay_extents(overlay),
                   label =  self.get_overlay_label(overlay),
                   value =  self.p.value)



class colormap(ElementOperation):
    """
    Applies a colormap on a Matrix input, returning the result as an
    RGB element.
    """

    output_type = RGB

    cmap = param.String(default='jet', doc="""
        The name of matplotlib color map to apply.""")

    value = param.String(default='Colormap', doc="""
        The value string for the output (an RGB element).""")

    def _process(self, matrix, key=None):
        import matplotlib

        if len(matrix.value_dimensions) != 1:
            raise Exception("Can only apply colour maps to Matrix"
                            " with single value dimension.")

        return RGB(matplotlib.cm.get_cmap(self.p.cmap)(matrix.data),
                   bounds = matrix.bounds,
                   label = matrix.label,
                   value=self.p.value)



class alpha_overlay(ElementOperation):
    """
    Accepts an overlay of a Matrix defined with a cmap and converts it
    to an RGB element whereby the alpha channel of the result is
    obtained from the second layer of the overlay.
    """

    value = param.String(default='AlphaOverlay', doc="""
        The value string for the output (an RGB element).""")

    cmap = param.String(default='jet', doc="""
        The name of matplotlib color map to apply.""")

    def _process(self, overlay, key=None):
        R,G,B,_ = split_raster(colormap(overlay[0], cmap=self.p.cmap))
        return RGB(toRGB(R*G*B*overlay[1]).data,
                   bounds=self.get_overlay_extents(overlay),
                   label=self.get_overlay_label(overlay),
                   value=self.p.value)




class colorize(ElementOperation):
    """
    Given a CompositeOverlay object consisting of a grayscale colormap and a
    second Sheetview with some specified colour map, use the second
    layer to colorize the data of the first layer.

    Currently, colorize only support the 'hsv' color map and is just a
    shortcut to the HCS operation using a constant confidence
    value. Arbitrary colorization will be supported in future.
    """

    label = param.String(default='Colorized', doc="""
        The label suffix to use for the resulting colorized plot where
        the suffix is added to the label of the first layer.""")

    def _process(self, overlay, key=None):

         if len(overlay) != 2 and overlay[0].mode != 'cmap':
             raise Exception("Can only colorize grayscale overlayed with colour map.")
         if [overlay[0].depth, overlay[1].depth ] != [1,1]:
             raise Exception("Depth one layers required.")
         if overlay[0].shape != overlay[1].shape:
             raise Exception("Shapes don't match.")

         # Needs a general approach which works with any color map
         C = Matrix(np.ones(overlay[0].data.shape),
                       bounds=overlay[0].bounds)
         hcs = toHCS(overlay[1] * C * overlay[0].N)

         return [Matrix(hcs.data, hcs.bounds, roi_bounds=hcs.roi_bounds,
                        label=hcs.label, value=self.p.label)]








Plot.options.RGBA.Red_Channel = GrayNearest
Plot.options.RGBA.Green_Channel = GrayNearest
Plot.options.RGBA.Blue_Channel = GrayNearest
Plot.options.Contours.Level = Options('style', color=Cycle(['b', 'g', 'r']))
