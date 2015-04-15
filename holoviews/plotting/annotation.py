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
        handles = self.draw_annotation(axis, annotation.data, opts)
        self.handles['annotations'] = handles
        return self._finalize_axis(key, ranges=ranges)


    def update_handles(self, axis, annotation, key, ranges=None):
        # Clear all existing annotations
        for element in self.handles['annotations']:
            element.remove()

        self.handles['annotations']=[]
        opts = self.style[self.cyclic_index]
        self.handles['annotations'] = self.draw_annotation(axis, annotation.data, opts)


class VLinePlot(AnnotationPlot):
    "Draw a vertical line on the axis"

    style_opts = ['alpha', 'color', 'linewidth', 'linestyle', 'visible']

    def __init__(self, annotation, **params):
        super(VLinePlot, self).__init__(annotation, **params)

    def draw_annotation(self, axis, position, opts):
        return [axis.axvline(position, **opts)]



class HLinePlot(AnnotationPlot):
    "Draw a horizontal line on the axis"

    style_opts = ['alpha', 'color', 'linewidth', 'linestyle', 'visible']

    def __init__(self, annotation, **params):
        super(HLinePlot, self).__init__(annotation, **params)

    def draw_annotation(self, axis, position, opts):
        "Draw a horizontal line on the axis"
        return [axis.axhline(position, **opts)]


class TextPlot(AnnotationPlot):
    "Draw the Text annotation object"

    style_opts = ['alpha', 'color', 'family', 'weight', 'rotation', 'fontsize', 'visible']

    def __init__(self, annotation, **params):
        super(TextPlot, self).__init__(annotation, **params)

    def draw_annotation(self, axis, data, opts):
        (x,y, text, fontsize,
         horizontalalignment, verticalalignment, rotation) = data
        return [axis.text(x,y, text,
                          horizontalalignment = horizontalalignment,
                          verticalalignment = verticalalignment,
                          rotation=rotation,
                          fontsize=opts.pop('fontsize', fontsize), **opts)]



class ArrowPlot(AnnotationPlot):
    "Draw an arrow using the information supplied to the Arrow annotation"

    _arrow_style_opts = ['alpha', 'color', 'lw', 'linewidth', 'visible']
    _text_style_opts = TextPlot.style_opts

    style_opts = sorted(set(_arrow_style_opts + _text_style_opts))

    def __init__(self, annotation, **params):
        super(ArrowPlot, self).__init__(annotation, **params)

    def draw_annotation(self, axis, data, opts):
        direction, text, xy, points, arrowstyle = data
        arrowprops = dict({'arrowstyle':arrowstyle},
                          **{k: opts[k] for k in self._arrow_style_opts if k in opts})
        textopts = {k: opts[k] for k in self._text_style_opts if k in opts}
        if direction in ['v', '^']:
            xytext = (0, points if direction=='v' else -points)
        elif direction in ['>', '<']:
            xytext = (points if direction=='<' else -points, 0)
        return [axis.annotate(text, xy=xy, textcoords='offset points',
                              xytext=xytext, ha="center", va="center",
                              arrowprops=arrowprops, **textopts)]



class SplinePlot(AnnotationPlot):
    "Draw the supplied Spline annotation (see Spline docstring)"

    style_opts = ['alpha', 'edgecolor', 'linewidth', 'linestyle', 'visible']

    def __init__(self, annotation, **params):
        super(SplinePlot, self).__init__(annotation, **params)

    def draw_annotation(self, axis, data, opts):
        verts, codes = data
        patch = patches.PathPatch(matplotlib.path.Path(verts, codes),
                                  facecolor='none', **opts)
        axis.add_patch(patch)
        return [patch]



Store.registry.update({
    VLine: VLinePlot,
    HLine: HLinePlot,
    Arrow: ArrowPlot,
    Spline: SplinePlot,
    Text: TextPlot})
