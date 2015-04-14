import matplotlib
from matplotlib import patches as patches

from ..core.options import Store
from ..core.util import match_spec
from ..element import VLine, HLine, Arrow, Spline, Text
from .element import ElementPlot


class AnnotationPlot(ElementPlot):
    """
    AnnotationPlot handles the display of all annotation elements.
    """

    def __init__(self, annotation, **params):
        self._annotation = annotation
        super(AnnotationPlot, self).__init__(annotation, **params)
        self.handles['annotations'] = []

    def __call__(self, ranges=None):
        annotation = self.map.last
        key = self.keys[-1]
        ranges = self.compute_ranges(self.map, key, ranges)
        ranges = match_spec(annotation, ranges)
        axis = self.handles['axis']
        opts = self.style[self.cyclic_index]
        handles = self.draw_annotation(axis, annotation, annotation.data, opts)
        self.handles['annotations'] = handles
        return self._finalize_axis(key, ranges=ranges)


    def update_handles(self, axis, annotation, key, ranges=None):
        # Clear all existing annotations
        for element in self.handles['annotations']:
            element.remove()

        self.handles['annotations']=[]
        opts = self.style[self.cyclic_index]
        self.handles['annotations'] = self.draw_annotation(axis, annotation,
                                                           annotation.data, opts)



class VLinePlot(AnnotationPlot):
    "Draw a vertical line on the axis"

    def __init__(self, annotation, **params):
        super(VLinePlot, self).__init__(annotation, **params)

    def draw_annotation(self, axis, annotation, position, opts):
        return [axis.axvline(position, **opts)]



class HLinePlot(AnnotationPlot):
    "Draw a horizontal line on the axis"

    def __init__(self, annotation, **params):
        super(HLinePlot, self).__init__(annotation, **params)

    def draw_annotation(self, axis, annotation, position, opts):
        "Draw a horizontal line on the axis"
        return [axis.axhline(position, **opts)]



class ArrowPlot(AnnotationPlot):
    "Draw an arrow using the information supplied to the Arrow annotation"

    def __init__(self, annotation, **params):
        super(ArrowPlot, self).__init__(annotation, **params)

    def draw_annotation(self, axis, annotation, data, opts):
        direction, text, xy, points, arrowstyle = data
        arrowprops = {'arrowstyle':arrowstyle, 'lw':opts.pop('lw',2)}
        if 'color' in opts:
            arrowprops['color'] = opts['color']
        if direction in ['v', '^']:
            xytext = (0, points if direction=='v' else -points)
        elif direction in ['>', '<']:
            xytext = (points if direction=='<' else -points, 0)
        return [axis.annotate(text, xy=xy, textcoords='offset points',
                             xytext=xytext, ha="center", va="center",
                             arrowprops=arrowprops, **opts)]



class SplinePlot(AnnotationPlot):
    "Draw the supplied Spline annotation (see Spline docstring)"

    def __init__(self, annotation, **params):
        super(SplinePlot, self).__init__(annotation, **params)

    def draw_annotation(self, axis, annotation, data, opts):
        verts, codes = data
        patch = patches.PathPatch(matplotlib.path.Path(verts, codes),
                                  facecolor='none', edgecolor='b', **opts)
        axis.add_patch(patch)
        return [patch]



class TextPlot(AnnotationPlot):
    "Draw the Text annotation object"

    def __init__(self, annotation, **params):
        super(TextPlot, self).__init__(annotation, **params)

    def draw_annotation(self, axis, annotation, data, opts):
        (x,y, text, fontsize,
         horizontalalignment, verticalalignment, rotation) = data
        return [axis.text(x,y, text,
                          horizontalalignment = horizontalalignment,
                          verticalalignment = verticalalignment,
                          rotation=rotation,
                          fontsize=opts.pop('fontsize', fontsize), **opts)]



Store.registry.update({
    VLine: VLinePlot,
    HLine: HLinePlot,
    Arrow: ArrowPlot,
    Spline: SplinePlot,
    Text: TextPlot})
