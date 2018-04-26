import param
import numpy as np
import matplotlib
from matplotlib import patches as patches

from ...core.util import match_spec, basestring
from ...core.options import abbreviated_exception
from .element import ElementPlot, ColorbarPlot
from .plot import mpl_rc_context


class AnnotationPlot(ElementPlot):
    """
    AnnotationPlot handles the display of all annotation elements.
    """

    show_legend = param.Boolean(default=False, doc="""
        Whether to show legend for the plot.""")

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

    style_opts = ['alpha', 'color', 'family', 'weight', 'visible']

    def draw_annotation(self, axis, data, opts):
        (x,y, text, fontsize,
         horizontalalignment, verticalalignment, rotation) = data
        if self.invert_axes: x, y = y, x
        opts['fontsize'] = fontsize
        return [axis.text(x,y, text,
                          horizontalalignment = horizontalalignment,
                          verticalalignment = verticalalignment,
                          rotation=rotation, **opts)]


class LabelsPlot(ColorbarPlot):

    color_index = param.ClassSelector(default=None, class_=(basestring, int),
                                      allow_None=True, doc="""
      Index of the dimension from which the color will the drawn""")

    xoffset = param.Number(default=None, doc="""
      Amount of offset to apply to labels along x-axis.""")

    yoffset = param.Number(default=None, doc="""
      Amount of offset to apply to labels along x-axis.""")

    style_opts = ['alpha', 'color', 'family', 'weight', 'size', 'visible',
                  'horizontalalignment', 'verticalalignment', 'cmap']

    _plot_methods = dict(single='annotate')

    def get_data(self, element, ranges, style):
        xs, ys = (element.dimension_values(i) for i in range(2))
        tdim = element.get_dimension(2)
        text = [tdim.pprint_value(v) for v in element.dimension_values(tdim)]
        positions = (ys, xs) if self.invert_axes else (xs, ys)
        if self.xoffset is not None:
            xs += self.xoffset
        if self.yoffset is not None:
            ys += self.yoffset

        cdim = element.get_dimension(self.color_index)
        if cdim:
            self._norm_kwargs(element, ranges, style, cdim)
        cs = element.dimension_values(cdim) if cdim else None
        if 'size' in style: style['fontsize'] = style.pop('size')
        if 'horizontalalignment' not in style: style['horizontalalignment'] = 'center'
        if 'verticalalignment' not in style: style['verticalalignment'] = 'center'
        return positions + (text, cs), style, {}

    def init_artists(self, ax, plot_args, plot_kwargs):
        if plot_args[-1] is not None:
            cmap = plot_kwargs.pop('cmap', None)
            colors = list(np.unique(plot_args[-1]))
            vmin, vmax = plot_kwargs.pop('vmin'), plot_kwargs.pop('vmax')
        else:
            cmap = None
            plot_args = plot_args[:-1]

        texts = []
        for item in zip(*plot_args):
            x, y, text = item[:3]
            if len(item) == 4 and cmap is not None:
                color = item[3]
                if plot_args[-1].dtype.kind in 'if':
                    color = (color - vmin) / (vmax-vmin)
                    plot_kwargs['color'] = cmap(color)
                else:
                    color = colors.index(color) if color in colors else np.NaN
                    plot_kwargs['color'] = cmap(color)
            texts.append(ax.text(x, y, text, **plot_kwargs))
        return {'artist': texts}

    def teardown_handles(self):
        if 'artist' in self.handles:
            for artist in self.handles['artist']:
                artist.remove()


class ArrowPlot(AnnotationPlot):
    "Draw an arrow using the information supplied to the Arrow annotation"

    _arrow_style_opts = ['alpha', 'color', 'lw', 'linewidth', 'visible']
    _text_style_opts = TextPlot.style_opts + ['textsize', 'fontsize']

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
        if 'textsize' in textopts:
            textopts['fontsize'] = textopts.pop('textsize')
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
