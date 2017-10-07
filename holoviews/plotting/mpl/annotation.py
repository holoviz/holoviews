import matplotlib
from matplotlib import patches as patches

from ...core.util import match_spec
from ...core.options import abbreviated_exception
from .element import ElementPlot
from .plot import mpl_rc_context


class AnnotationPlot(ElementPlot):
    """
    AnnotationPlot handles the display of all annotation elements.
    """

    def __init__(self, annotation, **params):
        self._annotation = annotation
        super(AnnotationPlot, self).__init__(annotation, **params)
        self.handles['annotations'] = []

    @mpl_rc_context
    def initialize_plot(self, ranges=None):
        annotation = self.hmap.last
        key = self.keys[-1]
        ranges = self.compute_ranges(self.hmap, key, ranges)
        ranges = match_spec(annotation, ranges)
        axis = self.handles['axis']
        opts = self.style[self.cyclic_index]
        with abbreviated_exception():
            handles = self.draw_annotation(axis, annotation.data, opts)
        self.handles['annotations'] = handles
        return self._finalize_axis(key, element=annotation, ranges=ranges)

    def update_handles(self, key, axis, annotation, ranges, style):
        # Clear all existing annotations
        for element in self.handles['annotations']:
            element.remove()

        with abbreviated_exception():
            self.handles['annotations'] = self.draw_annotation(axis, annotation.data, style)


class VLinePlot(AnnotationPlot):
    "Draw a vertical line on the axis"

    style_opts = ['alpha', 'color', 'linewidth', 'linestyle', 'visible']

    def draw_annotation(self, axis, position, opts):
        if self.invert_axes:
            return [axis.axhline(position, **opts)]
        else:
            return [axis.axvline(position, **opts)]



class HLinePlot(AnnotationPlot):
    "Draw a horizontal line on the axis"

    style_opts = ['alpha', 'color', 'linewidth', 'linestyle', 'visible']

    def draw_annotation(self, axis, position, opts):
        "Draw a horizontal line on the axis"
        if self.invert_axes:
            return [axis.axvline(position, **opts)]
        else:
            return [axis.axhline(position, **opts)]


class TextPlot(AnnotationPlot):
    "Draw the Text annotation object"

    style_opts = ['alpha', 'color', 'family', 'weight', 'rotation', 'fontsize', 'visible']

    def draw_annotation(self, axis, data, opts):
        (x,y, text, fontsize,
         horizontalalignment, verticalalignment, rotation) = data
        if self.invert_axes: x, y = y, x
        opts['fontsize'] = fontsize
        return [axis.text(x,y, text,
                          horizontalalignment = horizontalalignment,
                          verticalalignment = verticalalignment,
                          rotation=rotation, **opts)]



class ArrowPlot(AnnotationPlot):
    "Draw an arrow using the information supplied to the Arrow annotation"

    _arrow_style_opts = ['alpha', 'color', 'lw', 'linewidth', 'visible']
    _text_style_opts = TextPlot.style_opts

    style_opts = sorted(set(_arrow_style_opts + _text_style_opts))

    def draw_annotation(self, axis, data, opts):
        x, y, text, direction, points, arrowstyle = data
        if self.invert_axes: x, y = y, x
        direction = direction.lower()
        arrowprops = dict({'arrowstyle':arrowstyle},
                          **{k: opts[k] for k in self._arrow_style_opts if k in opts})
        textopts = {k: opts[k] for k in self._text_style_opts if k in opts}
        if direction in ['v', '^']:
            xytext = (0, points if direction=='v' else -points)
        elif direction in ['>', '<']:
            xytext = (points if direction=='<' else -points, 0)
        return [axis.annotate(text, xy=(x, y), textcoords='offset points',
                              xytext=xytext, ha="center", va="center",
                              arrowprops=arrowprops, **textopts)]



class SplinePlot(AnnotationPlot):
    "Draw the supplied Spline annotation (see Spline docstring)"

    style_opts = ['alpha', 'edgecolor', 'linewidth', 'linestyle', 'visible']

    def draw_annotation(self, axis, data, opts):
        verts, codes = data
        if not len(verts):
            return []
        patch = patches.PathPatch(matplotlib.path.Path(verts, codes),
                                  facecolor='none', **opts)
        axis.add_patch(patch)
        return [patch]
