"""
Transforms are SheetOperations that manipulate dataviews, typically
for the purposes of visualization. Such transformations often apply to
SheetViews or SheetStacks and compose the data together in ways that
can be viewed conveniently, often by creating or manipulating color
channels.
"""

import colorsys
import param

import numpy as np
import matplotlib
from imagen.analysis import ViewOperation
from sheetviews import SheetView

from styles import Styles, Style

rgb_to_hsv = np.vectorize(colorsys.rgb_to_hsv)
hsv_to_rgb = np.vectorize(colorsys.hsv_to_rgb)



class HCS(ViewOperation):
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

    def _process(self, overlay):
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
        return [SheetView(rgb, hue.bounds, roi_bounds=overlay.roi_bounds,
                          mode='rgb', label=hue.label+' HCS Plot')]



class colorize(ViewOperation):
    """
    Given a SheetOverlay consisting of a grayscale colormap and a
    second Sheetview with some specified colour map, use the second
    layer to colorize the data of the first layer.

    Currently, colorize only support the 'hsv' color map and is just a
    shortcut to the HCS transform using a constant confidence
    value. Arbitrary colorization will be supported in future.
    """

    def _process(self, overlay):

         if len(overlay) != 2 and overlay[0].mode != 'gray':
             raise Exception("Can only colorize grayscale overlayed with colour map.")
         if [overlay[0].depth, overlay[1].depth ] != [1,1]:
             raise Exception("Depth one layers required.")
         if overlay[0].shape != overlay[1].shape:
             raise Exception("Shapes don't match.")

         if overlay[1].mode != 'hsv':
             raise NotImplementedError

         C = SheetView(np.ones(overlay[1].data.shape),
                       bounds=overlay.bounds)
         hcs = HCS(overlay[1] * C * overlay[0].N)
         return [SheetView(hsc.data, hsc.bounds,
                           mode='rgb',
                           roi_bounds=hcs.roi_bounds,
                           label=sheetview.label+' Colorized')]



class cmap2rgb(ViewOperation):
    """
    Convert SheetViews using colormaps to RGBA mode.  The colormap of
    the style is used, if available. Otherwise, the colormap may be
    forced.
    """

    cmap = param.String(default=None, allow_None=True, doc="""
          Force the use of a specific color map. Otherwise, the cmap
          property of the applicable style is used.""")

    def _process(self, sheetview):
        if sheetview.depth != 1:
            raise Exception("Can only apply colour maps to SheetViews with depth of 1.")

        style_cmap = Styles[sheetview][0].get('cmap', None)
        if not any([self.p.cmap, style_cmap]):
            raise Exception("No color map supplied and no cmap in the active style.")

        cmap = matplotlib.cm.get_cmap(style_cmap if self.p.cmap is None else self.p.cmap)
        return [SheetView(cmap(sheetview.data),
                         bounds=sheetview.bounds,
                         cyclic_range=sheetview.cyclic_range,
                         style=sheetview.style,
                         metadata=sheetview.metadata,
                         label = sheetview.label+' RGB',
                         mode='rgba')]



class split(ViewOperation):
    """
    Given SheetViews in RGBA mode, return the R,G,B and A channels as
    a GridLayout.
    """
    def _process(self, sheetview):
        if sheetview.depth not in [3,4]:
            raise Exception("Can only split SheetViews with a depth of 3 or 4")
        return [SheetView(sheetview.data[:,:,i],
                          bounds=sheetview.bounds,
                          label='RGBA'[i] + ' Channel')
                for i in range(sheetview.depth)]



Styles.R_Channel = Style(cmap='gray')
Styles.G_Channel = Style(cmap='gray')
Styles.B_Channel = Style(cmap='gray')
Styles.A_Channel = Style(cmap='gray')
