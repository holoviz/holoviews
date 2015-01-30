from collections import OrderedDict
from itertools import groupby
from matplotlib.font_manager import FontProperties
import numpy as np

import param

from ..core import NdOverlay, Overlay, HoloMap, CompositeOverlay, Element, Element3D
from ..core.util import valid_identifier
from .plot import Plot


class ElementPlot(Plot):

    apply_databounds = param.Boolean(default=True, doc="""
        Whether to compute the plot bounds from the data itself.""")

    orientation = param.ObjectSelector(default='horizontal',
                                       objects=['horizontal', 'vertical'], doc="""
        The orientation of the plot. Note that this parameter may not
        always be respected by all plots but should be respected by
        adjoined plots when appropriate.""")

    rescale_individually = param.Boolean(default=False, doc="""
        Whether to use redraw the axes per map or per element.""")

    show_grid = param.Boolean(default=False, doc="""
        Whether to show a Cartesian grid on the plot.""")

    show_xaxis = param.ObjectSelector(default='bottom',
                                      objects=['top', 'bottom', None], doc="""
        Whether and where to display the xaxis.""")

    show_yaxis = param.ObjectSelector(default='left',
                                      objects=['left', 'right', None], doc="""
        Whether and where to display the yaxis.""")

    # Element Plots should declare the valid style options for matplotlib call
    style_opts = []

    def __init__(self, element, keys=None, cyclic_index=0, zorder=0, **params):
        self.map = self._check_map(element)
        self.cyclic_index = cyclic_index
        self.zorder = zorder
        keys = keys if keys else self.map.keys()
        super(ElementPlot, self).__init__(keys=keys, **params)


    def _check_map(self, view, element_type=Element):
        """
        Helper method that ensures a given element is always returned as
        an HoloMap object.
        """
        if not isinstance(view, HoloMap):
            vmap = HoloMap(initial_items=(0, view), id=view.id)
        else:
            vmap = view

        check = vmap.last
        if issubclass(vmap.type, CompositeOverlay):
            check = vmap.last.values()[0]
        if isinstance(check, Element3D):
            self.projection = '3d'

        return vmap


    def _get_frame(self, key, obj):
        return self.map[key]


    def get_extents(self, view, ranges):
        """
        Gets the extents for the axes from the current View. The globally
        computed ranges can optionally override the extents.
        """
        return view.extents if self.rescale_individually else self.map.extents


    def _finalize_axis(self, key, title=None, ranges=None, xticks=None, yticks=None,
                       xlabel=None, ylabel=None):
        """
        Applies all the axis settings before the axis or figure is returned.
        Only plots with zorder 0 get to apply their settings.

        When the number of the frame is supplied as n, this method looks
        up and computes the appropriate title, axis labels and axis bounds.
        """

        axis = self.handles['axis']

        view = self.map.get(key, None)
        if self.zorder == 0 and key is not None:
            if view is not None:
                title = None if self.zorder > 0 else self._format_title(key)
                if hasattr(view, 'xlabel') and xlabel is None:
                    xlabel = view.xlabel
                if hasattr(view, 'ylabel') and ylabel is None:
                    ylabel = view.ylabel
                if self.apply_databounds:
                    extents = self.get_extents(view, ranges)
                    l, b, r, t = [coord if np.isreal(coord) else np.NaN for coord in extents]
                    if not np.NaN in (l, r): axis.set_xlim((l, r))
                    if b == t: t += 1. # Arbitrary y-extent if zero range
                    if not np.NaN in (b, t): axis.set_ylim((b, t))

            if self.show_grid:
                axis.get_xaxis().grid(True)
                axis.get_yaxis().grid(True)

            if xlabel: axis.set_xlabel(xlabel)
            if ylabel: axis.set_ylabel(ylabel)

            disabled_spines = []
            if self.show_xaxis is not None:
                if self.show_xaxis == 'top':
                    axis.xaxis.set_ticks_position("top")
                    axis.xaxis.set_label_position("top")
                elif self.show_xaxis == 'bottom':
                    axis.xaxis.set_ticks_position("bottom")
            else:
                axis.xaxis.set_visible(False)
                disabled_spines.extend(['top', 'bottom'])

            if self.show_yaxis is not None:
                if self.show_yaxis == 'left':
                    axis.yaxis.set_ticks_position("left")
                elif self.show_yaxis == 'right':
                    axis.yaxis.set_ticks_position("right")
                    axis.yaxis.set_label_position("right")
            else:
                axis.yaxis.set_visible(False)
                disabled_spines.extend(['left', 'right'])

            for pos in disabled_spines:
                axis.spines[pos].set_visible(False)

            if not self.show_frame:
                axis.spines['right' if self.show_yaxis == 'left' else 'left'].set_visible(False)
                axis.spines['bottom' if self.show_xaxis == 'top' else 'top'].set_visible(False)

            if self.aspect == 'square':
                axis.set_aspect((1./axis.get_data_ratio()))
            elif self.aspect not in [None, 'square']:
                axis.set_aspect(self.aspect)

            if xticks:
                axis.set_xticks(xticks[0])
                axis.set_xticklabels(xticks[1])

            if yticks:
                axis.set_yticks(yticks[0])
                axis.set_yticklabels(yticks[1])

            if self.show_title and title is not None:
                self.handles['title'] = axis.set_title(title)

        for hook in self.finalize_hooks:
            hook(self, view)

        return super(ElementPlot, self)._finalize_axis(key)


    def match_range(self, element, ranges):
        match_tuple = ()
        match = ranges.get((), {})
        for spec in [type(element).__name__,
                     valid_identifier(element.value),
                     valid_identifier(element.label)]:
            match_tuple += (spec,)
            if match_tuple in ranges:
                match = ranges[match_tuple]
        return match


    def update_frame(self, n, ranges=None):
        """
        Set the plot(s) to the given frame number.  Operates by
        manipulating the matplotlib objects held in the self._handles
        dictionary.

        If n is greater than the number of available frames, update
        using the last available frame.
        """
        n = n if n < len(self) else len(self) - 1
        key = self._keys[n]
        view = self.map.get(key, None)
        axis = self.handles['axis']
        axis.set_visible(view is not None)
        if self.normalize:
            ranges = self.compute_ranges(self.map, key, ranges)
            ranges = self.match_range(view, ranges)
        axis_kwargs = self.update_handles(axis, view, key if view is not None else {}, ranges)
        self._finalize_axis(key, ranges=ranges, **(axis_kwargs if axis_kwargs else {}))


    def update_handles(self, axis, view, key, ranges=None):
        """
        Update the elements of the plot.
        :param axis:
        """
        raise NotImplementedError


class OverlayPlot(ElementPlot):
    """
    OverlayPlot supports processing of channel operations on Overlays
    across maps.
    """

    show_legend = param.Boolean(default=True, doc="""
        Whether to show legend for the plot.""")

    def __init__(self, overlay, **params):
        super(OverlayPlot, self).__init__(overlay, **params)
        self.subplots = self._create_subplots()


    def _create_subplots(self):
        subplots = OrderedDict()

        keys, vmaps = self.map.split_overlays()
        style_groups = dict((k, enumerate(list(v))) for k,v
                            in groupby(vmaps, lambda s: (s.last.value)))
        for zorder, (key, vmap) in enumerate(zip(keys, vmaps)):
            cyclic_index, _ = next(style_groups[(vmap.last.value)])
            plotopts = self.lookup_options(vmap.last, 'plot').options
            if issubclass(vmap.type, NdOverlay):
                plotopts['dimensions'] = zip(vmap.last.key_dimensions, key)
            plotopts = dict(keys=self._keys, axis=self.handles['axis'],
                            cyclic_index=cyclic_index, figure=self.handles['fig'],
                            zorder=zorder, **plotopts)
            plotype = Plot.defaults[type(vmap.last)]
            subplots[key] = plotype(vmap, **plotopts)

        return subplots


    def _adjust_legend(self, axis):
        # If legend enabled update handles and labels
        if not axis or not axis.get_legend(): return
        handles, _ = axis.get_legend_handles_labels()
        labels = self.map.last.legend
        if len(handles) and self.show_legend:
            fontP = FontProperties()
            fontP.set_size('medium')
            leg = axis.legend(handles[::-1], labels[::-1], prop=fontP)
            leg.get_frame().set_alpha(1.0)
        frame = axis.get_legend().get_frame()
        frame.set_facecolor('1.0')
        frame.set_edgecolor('0.0')
        frame.set_linewidth('1.5')

    def _format_title(self, key):
        view = self.map.get(key, None)
        if view is None: return None
        title_format = self.map.get_title(key if isinstance(key, tuple) else (key,), view)
        if title_format is None: return None

        values = [v.value for v in view]
        value = values[0] if len(set(values)) == 1 else ""
        return title_format.format(label=view.label, value=value,
                                   type=view.__class__.__name__)


    def __call__(self, ranges=None):
        axis = self.handles['axis']

        for plot in self.subplots.values():
            plot(ranges=ranges)
        self._adjust_legend(axis)

        key = self._keys[-1]
        return self._finalize_axis(key, ranges=ranges, title=self._format_title(key))


    def update_frame(self, n, ranges=None):
        n = n if n < len(self) else len(self) - 1
        key = self._keys[n]
        if self.projection == '3d':
            self.handles['axis'].clear()

        for plot in self.subplots.values():
            plot.update_frame(n, ranges)
        self._finalize_axis(key, ranges)


Plot.defaults.update({NdOverlay: OverlayPlot,
                      Overlay: OverlayPlot})
