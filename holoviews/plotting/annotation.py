from matplotlib import patches as patches
from matplotlib.collections import LineCollection
from matplotlib.path import Path
import numpy as np

import param

from ..core import Element
from ..element import Annotation, Contours, VLine, HLine, Arrow, Spline, Text
from .plot import Plot


class AnnotationPlot(Plot):
    """
    AnnotationPlot handles the display of all annotation elements.
    """

    def __init__(self, annotation, **params):
        self._annotation = annotation
        super(AnnotationPlot, self).__init__(annotation, **params)
        self.handles['annotations'] = []

    def __call__(self, lbrt=None):
        annotation = self._map.last
        opts = self.settings.closest(annotation, 'style')[self.cyclic_index]
        handle = self.draw_annotation(annotation, annotation.data, opts)
        self.handles['annotations'].append(handle)
        return self._finalize_axis(self._keys[-1])


    def update_handles(self, annotation, key, lbrt=None):
        # Clear all existing annotations
        for element in self.handles['annotations']:
            element.remove()

        opts = self.settings.closest(annotation, 'style')[self.cyclic_index]
        self.draw_annotation(annotation, annotation.data, opts)



class VLinePlot(AnnotationPlot):
    "Draw a vertical line on the axis"

    def __init__(self, annotation, **params):
        super(VLinePlot, self).__init__(annotation, **params)

    def draw_annotation(self, annotation, position, opts):
        return self.ax.axvline(position, **opts)



class HLinePlot(AnnotationPlot):
    "Draw a horizontal line on the axis"

    def __init__(self, annotation, **params):
        super(HLinePlot, self).__init__(annotation, **params)

    def draw_annotation(self, annotation, position, opts):
        "Draw a horizontal line on the axis"
        return self.ax.axhline(position, **opts)



class ArrowPlot(AnnotationPlot):
    "Draw an arrow using the information supplied to the Arrow annotation"

    def __init__(self, annotation, **params):
        super(ArrowPlot, self).__init__(annotation, **params)

    def draw_annotation(self, annotation, data, opts):
        direction, text, xy, points, arrowstyle = data
        arrowprops = {'arrowstyle':arrowstyle}
        if 'color' in opts:
            arrowprops['color'] = opts['color']
        if direction in ['v', '^']:
            xytext = (0, points if direction=='v' else -points)
        elif direction in ['>', '<']:
            xytext = (points if direction=='<' else -points, 0)
        return self.ax.annotate(text, xy=xy, textcoords='offset points',
                                xytext=xytext, ha="center", va="center",
                                arrowprops=arrowprops, **opts)



class SplinePlot(AnnotationPlot):
    "Draw the supplied Spline annotation (see Spline docstring)"

    def __init__(self, annotation, **params):
        super(SplinePlot, self).__init__(annotation, **params)

    def draw_annotation(self, annotation, data, opts):
        verts, codes = data
        patch = patches.PathPatch(Path(verts, codes),
                                  facecolor='none', edgecolor='b', **opts)
        self.ax.add_patch(patch)
        return patch



class TextPlot(AnnotationPlot):
    "Draw the Text annotation object"

    def __init__(self, annotation, **params):
        super(TextPlot, self).__init__(annotation, **params)

    def draw_annotation(self, annotation, data, opts):
        (x,y, text, fontsize,
         horizontalalignment, verticalalignment, rotation) = data
        return self.ax.text(x,y, text,
                            horizontalalignment = horizontalalignment,
                            verticalalignment = verticalalignment,
                            rotation=rotation,
                            fontsize=fontsize, **opts)



class ContourPlot(Plot):

    style_opts = ['alpha', 'color', 'linestyle', 'linewidth', 'visible']

    def __init__(self, *args, **params):
        self.aspect = 'equal'
        super(ContourPlot, self).__init__(*args, **params)


    def __call__(self, lbrt=None):
        lines = self._map.last

        style = self.settings.closest(lines, 'style')[self.cyclic_index]
        line_segments = LineCollection(lines.data, zorder=self.zorder, **style)
        self.handles['line_segments'] = line_segments
        self.ax.add_collection(line_segments)

        return self._finalize_axis(self._keys[-1])


    def update_handles(self, view, key, lbrt=None):
        self.handles['line_segments'].set_paths(view.data)



Plot.defaults.update({
    Contours: ContourPlot,

    VLine:VLinePlot,
    HLine:HLinePlot,
    Arrow:ArrowPlot,
    Spline:SplinePlot,

    Text:TextPlot
})
