"""
ElementOperations that output RGB elements are commonly used for
displaying colored images.

The element operations can often be used with option.Compositor to
define operations that should automatically process data or generate
visualization upon display.
"""

import colorsys

import numpy as np

import param

from ..core.operation import ElementOperation
from ..element import Image, RGB
from .normalization import raster_normalization
from .element import split_raster

rgb_to_hsv = np.vectorize(colorsys.rgb_to_hsv)
hsv_to_rgb = np.vectorize(colorsys.hsv_to_rgb)




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

    group = param.String(default='HCS', doc="""
        The group string for the output (an RGB element).""")

    def _process(self, overlay, key=None):

        normfn = raster_normalization.instance()
        if self.p.input_ranges:
            overlay = normfn.process_element(overlay, key, *self.p.input_ranges)
        else:
            overlay = normfn.process_element(overlay, key)

        hue, confidence = overlay[0], overlay[1]
        strength_data = overlay[2].data if (len(overlay) == 3) else np.ones(hue.shape)

        hue_data = hue.data
        hue_range = hue.value_dimensions[0].range
        if (not hue.value_dimensions[0].cyclic) or (None in hue_range):
             raise Exception("The input hue channel must be declared cyclic with a defined range.")
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
                   group =  self.p.group)



class colormap(ElementOperation):
    """
    Applies a colormap on a Image input, returning the result as an
    RGB element.
    """

    output_type = RGB

    cmap = param.String(default='jet', doc="""
        The name of matplotlib color map to apply.""")

    group = param.String(default='Colormap', doc="""
        The group string for the output (an RGB element).""")

    def _process(self, matrix, key=None):
        import matplotlib

        normfn = raster_normalization.instance()
        if self.p.input_ranges:
            matrix = normfn.process_element(matrix, key, *self.p.input_ranges)
        else:
            matrix = normfn.process_element(matrix, key)

        if len(matrix.value_dimensions) != 1:
            raise Exception("Can only apply colour maps to Image"
                            " with single value dimension.")

        return RGB(matplotlib.cm.get_cmap(self.p.cmap)(matrix.data),
                   bounds = matrix.bounds,
                   label = matrix.label,
                   group=self.p.group)



class alpha_overlay(ElementOperation):
    """
    Accepts an overlay of a Image defined with a cmap and converts it
    to an RGB element whereby the alpha channel of the result is
    obtained from the second layer of the overlay.
    """

    group = param.String(default='AlphaOverlay', doc="""
        The group string for the output (an RGB element).""")

    cmap = param.String(default='jet', doc="""
        The name of matplotlib color map to apply.""")

    def _process(self, overlay, key=None):
        R,G,B,_ = split_raster(colormap(overlay[0], cmap=self.p.cmap))
        return RGB(toRGB(R*G*B*overlay[1]).data,
                   bounds=self.get_overlay_extents(overlay),
                   label=self.get_overlay_label(overlay),
                   group=self.p.group)


class colorizeHSV(ElementOperation):
    """
    Given an Overlay consisting of two Image elements, colorize the
    data in the bottom Image with the data in the top Image using
    the HSV color space.
    """

    group = param.String(default='ColorizedHSV', doc="""
        The group string for the colorized output (an RGB element)""")

    output_type = RGB

    def _process(self, overlay, key=None):
        if len(overlay) != 2:
            raise Exception("colorizeHSV required an overlay of two Image elements as input.")
        if (len(overlay[0].value_dimensions), len(overlay[1].value_dimensions)) != (1,1):
            raise Exception("Each Image element must have single value dimension.")
        if overlay[0].shape != overlay[1].shape:
            raise Exception("Mismatch in the shapes of the data in the Image elements.")


        hue = overlay[1]
        Hdim = hue.value_dimensions[0]
        H = hue.clone(hue.data.copy(),
                      value_dimensions=[Hdim(cyclic=True, range=hue.range(Hdim.name))])

        normfn = raster_normalization.instance()
        if self.p.input_ranges:
            S = normfn.process_element(overlay[0], key, *self.p.input_ranges)
        else:
            S = normfn.process_element(overlay[0], key)

        C = Image(np.ones(hue.data.shape),
                   bounds=self.get_overlay_extents(overlay), group='F', label='G')
        return toHCS(H * C * S).clone(group=self.p.group)
