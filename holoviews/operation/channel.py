import colorsys

import numpy as np

import param

from ..core.operation import ElementOperation
from ..core.options import Options, Cycle
from ..element import Matrix, RGBA
from .normalization import raster_normalization

rgb_to_hsv = np.vectorize(colorsys.rgb_to_hsv)
hsv_to_rgb = np.vectorize(colorsys.hsv_to_rgb)


class toRGBA(ElementOperation):
    """
    Accepts an overlay containing either 3 or 4 layers. The first
    three layers are the R,G, B channels and the last layer (if
    supplied) is the alpha channel.

    Note that the values in the input Matrix elements are expected to
    be bounded between 0.0-1.0. If this isn't the case, the values of
    these channels will be clipped into this range.
    """

    output_type = RGBA

    value = param.String(default='RGBA', doc="""
        The value string for the output RGBA element.""")


    def _process(self, overlay, key=None):
        if len(overlay) not in [3, 4]:
            raise Exception("Requires 3 or 4 layers to convert to RGB(A)")
        if not all(isinstance(el, Matrix) for el in overlay):
            raise Exception("All layers must be of type Matrix to convert to RGBA")
        if not all(el.depth == 1 for el in overlay):
            raise Exception("All Matrix elements must have single value"
                            " dimension for conversion to RGBA")
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

        return RGBA(np.dstack(arrays),
                    bounds = self.get_overlay_extents(overlay),
                    label  = self.get_overlay_label(overlay),
                    value  = self.p.value)



class alpha_overlay(ElementOperation):
    """
    Accepts an overlay of a Matrix defined with a cmap and converts
    it to an RGBA Matrix. The alpha channel of the result is
    defined by the second layer of the overlay.
    """

    label = param.String(default='AlphaOverlay', doc="""
        The label suffix to use for the alpha overlay result where the
        suffix is added to the label of the first layer.""")

    def _process(self, overlay, key=None):
        R,G,B,_ = split(cmap2rgb(overlay[0]))
        return [Matrix(toRGBA(R*G*B*overlay[1]).data, overlay[0].bounds,
                       label=overlay[0].label, value=self.p.label,
                       value_dimensions=overlay[0].value_dimensions)]


class toHCS(ElementOperation):
    """
    Hue-Confidence-Strength plot.

    Accepts an overlay containing either 2 or 3 layers. The first two
    layers are hue and confidence and the third layer (if available)
    is the strength channel.
    """

    S_multiplier = param.Number(default=1.0, bounds=(0.0,None), doc="""
        Multiplier for the strength value.""")

    C_multiplier = param.Number(default=1.0, bounds=(0.0,None), doc="""
        Multiplier for the confidence value.""")

    flipSC = param.Boolean(default=False, doc="""
        Whether to flip the strength and confidence channels""")

    label = param.String(default='HCS', doc="""
        The label suffix to use for the resulting HCS plot where the
        suffix is added to the label of the Hue channel.""")

    def _process(self, overlay, key=None):
        hue = overlay[0]
        confidence = overlay[1]

        strength_data = overlay[2].data if (len(overlay) == 3) else np.ones(hue.shape)

        if hue.shape != confidence.shape:
            raise Exception("Cannot combine plots with different shapes")

        (h,s,v)= (hue.N.data.clip(0.0, 1.0),
                  (confidence.data * self.p.C_multiplier).clip(0.0, 1.0),
                  (strength_data * self.p.S_multiplier).clip(0.0, 1.0))

        if self.p.flipSC:
            (h,s,v) = (h,v,s.clip(0,1.0))

        r, g, b = hsv_to_rgb(h, s, v)
        rgb = np.dstack([r,g,b])
        return [Matrix(rgb, hue.bounds, roi_bounds=hue.roi_bounds,
                       label=hue.label, value=self.p.label)]


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


class cmap2rgb(ElementOperation):
    """
    Convert Matrix Views using colormaps to RGBA mode. The colormap of
    the style is used, if available. Otherwise, the colormap may be
    forced as a parameter.
    """

    cmap = param.String(default=None, allow_None=True, doc="""
          Force the use of a specific color map. Otherwise, the cmap
          property of the applicable style is used.""")

    label = param.String(default='RGB', doc="""
        The label suffix to use for the resulting RGB Matrix where
        the suffix is added to the label of the Matrix to be
        colored.""")

    def _process(self, sheetview, key=None):
        import matplotlib

        if sheetview.depth != 1:
            raise Exception("Can only apply colour maps to Matrix with depth of 1.")

        style_cmap = options.style(sheetview)[0].get('cmap', None)
        if not any([self.p.cmap, style_cmap]):
            raise Exception("No color map supplied and no cmap in the active style.")

        cmap = matplotlib.cm.get_cmap(style_cmap if self.p.cmap is None else self.p.cmap)
        return [sheetview.clone(cmap(sheetview.data), value=self.p.label)]


class split(ElementOperation):
    """
    Given Matrix in RGBA mode, return the R,G,B and A channels as
    a NdLayout.
    """

    label = param.String(default='Channel', doc="""
      The label suffix used to label the components of the split
      following the character selected from output_names.""")

    def _process(self, sheetview, key=None):
        if sheetview.mode not in ['rgb', 'rgba']:
            raise Exception("Can only split Matrix with a depth of 3 or 4")
        return [sheetview.clone(sheetview.data[:, :, i],
                                value='RGBA'[i] + ' ' + self.p.label)
                for i in range(sheetview.depth)]


Plot.options.RGBA.Red_Channel = GrayNearest
Plot.options.RGBA.Green_Channel = GrayNearest
Plot.options.RGBA.Blue_Channel = GrayNearest
Plot.options.Contours.Level = Options('style', color=Cycle(['b', 'g', 'r']))
