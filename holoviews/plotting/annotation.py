from matplotlib import patches as patches
from matplotlib.collections import LineCollection
from matplotlib.path import Path
import numpy as np

import param

from ..core import Element
from ..element import Annotation, Contours
from .plot import Plot


class AnnotationPlot(Plot):
    """
    Draw the Annotation element on the supplied axis. Supports axis
    vlines, hlines, arrows (with or without labels), boxes and
    arbitrary polygonal lines. Note, unlike other Plot types,
    AnnotationPlot must always operate on a supplied axis as
    Annotations may only be used as part of Overlays.
    """

    style_opts = param.List(default=['alpha', 'color', 'edgecolors',
                                     'facecolors', 'linewidth',
                                     'linestyle', 'rotation', 'family',
                                     'weight', 'fontsize', 'visible',
                                     'edgecolor'],
                            constant=True, doc="""
     Box annotations, hlines and vlines and lines all accept
     matplotlib line style options. Arrow annotations also accept
     additional text options.""")

    def __init__(self, annotation, **params):
        self._annotation = annotation
        super(AnnotationPlot, self).__init__(annotation, **params)
        self._warn_invalid_intervals(self._map)
        self.handles['annotations'] = []

        line_only = ['linewidth', 'linestyle']
        arrow_opts = [opt for opt in self.style_opts if opt not in line_only]
        line_opts = line_only + ['color']
        self.opt_filter = {'hline': line_opts, 'vline': line_opts,
                           'line': line_opts,
                           '<': arrow_opts, '^': arrow_opts,
                           '>': arrow_opts, 'v': arrow_opts,
                           'spline': line_opts + ['edgecolor']}


    def _warn_invalid_intervals(self, vmap):
        "Check if the annotated intervals have appropriate keys"
        dim_labels = [d.name for d in self._map.key_dimensions]

        mismatch_set = set()
        for annotation in vmap.values():
            for spec in annotation.data:
                interval = spec[-1]
                if interval is None or dim_labels == ['Default']:
                    continue
                mismatches = set(interval.keys()) - set(dim_labels)
                mismatch_set = mismatch_set | mismatches

        if mismatch_set:
            mismatch_list= ', '.join('%r' % el for el in mismatch_set)
            self.warning("Invalid annotation interval key(s) ignored: %r" % mismatch_list)


    def _active_interval(self, key, interval):
        """
        Given an interval specification, determine whether the
        annotation should be shown or not.
        """
        dim_labels = [d.name for d in self._map.key_dimensions]
        if (interval is None) or dim_labels == ['Default']:
            return True

        key = key if isinstance(key, tuple) else (key,)
        key_dict = dict(zip(dim_labels, key))
        for key, (start, end) in interval.items():
            if (start is not None) and key_dict.get(key, -float('inf')) <= start:
                return False
            if (end is not None) and key_dict.get(key, float('inf')) > end:
                return False

        return True


    def _draw_annotations(self, annotation, key):
        """
        Draw the elements specified by the Annotation ViewableElement on the
        axis, return a list of handles.
        """
        handles = []
        opts = self.settings.closest(annotation, 'style').settings
        color = opts.get('color', 'k')

        for spec in annotation.data:
            mode, info, interval = spec[0], spec[1:-1], spec[-1]
            opts = dict(el for el in opts.items()
                        if el[0] in self.opt_filter[mode])

            if not self._active_interval(key, interval):
                continue
            if mode == 'vline':
                handles.append(self.ax.axvline(spec[1], **opts))
                continue
            elif mode == 'hline':
                handles.append(self.ax.axhline(spec[1], **opts))
                continue
            elif mode == 'line':
                line = LineCollection([np.array(info[0])], **opts)
                self.ax.add_collection(line)
                handles.append(line)
                continue
            elif mode == 'spline':
                verts, codes = info
                patch = patches.PathPatch(Path(verts, codes),
                                          facecolor='none', **opts)
                self.ax.add_patch(patch)
                continue


            text, xy, points, arrowstyle = info
            arrowprops = dict(arrowstyle=arrowstyle, color=color)
            if mode in ['v', '^']:
                xytext = (0, points if mode=='v' else -points)
            elif mode in ['>', '<']:
                xytext = (points if mode=='<' else -points, 0)
            arrow = self.ax.annotate(text, xy=xy, textcoords='offset points',
                                     xytext=xytext, ha="center", va="center",
                                     arrowprops=arrowprops, **opts)
            handles.append(arrow)
        return handles


    def __call__(self, axis=None, lbrt=None):
        self.ax = self._init_axis(axis)
        handles = self._draw_annotations(self._map.last, list(self._map.keys())[-1])
        self.handles['annotations'] = handles
        return self._finalize_axis(self._keys[-1])


    def update_handles(self, annotation, key, lbrt=None):
        # Clear all existing annotations
        for element in self.handles['annotations']:
            element.remove()

        self.handles['annotations'] = self._draw_annotations(annotation, key)


class ContourPlot(Plot):

    style_opts = param.List(default=['alpha', 'color', 'linestyle',
                                     'linewidth', 'visible'],
                            constant=True, doc="""
        The style options for ContourPlot match those of matplotlib's
        LineCollection class.""")

    def __init__(self, *args, **params):
        self.aspect = 'equal'
        super(ContourPlot, self).__init__(*args, **params)


    def __call__(self, axis=None, cyclic_index=0, lbrt=None):
        lines = self._map.last
        self.ax = self._init_axis(axis)

        style = self.settings.closest(lines, 'style')[self.cyclic_index]
        line_segments = LineCollection(lines.data, zorder=self.zorder, **style)
        self.handles['line_segments'] = line_segments
        self.ax.add_collection(line_segments)

        return self._finalize_axis(self._keys[-1], lbrt=lbrt)


    def update_handles(self, view, key, lbrt=None):
        self.handles['line_segments'].set_paths(view.data)



Plot.defaults.update({Annotation: AnnotationPlot,
                      Contours: ContourPlot})