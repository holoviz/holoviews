from itertools import groupby
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

import numpy as np

import param

from ..core.options import Store
from ..core import OrderedDict, Element, NdOverlay, Overlay, HoloMap, CompositeOverlay, Element3D
from ..core.util import match_spec, basestring, safe_unicode, max_extents
from ..element import Annotation, Table, ItemTable
from ..operation import Compositor
from .plot import Plot


class ElementPlot(Plot):

    apply_ranges = param.Boolean(default=True, doc="""
        Whether to compute the plot bounds from the data itself.""")

    apply_extents = param.Boolean(default=True, doc="""
        Whether to apply extent overrides on the Elements""")

    apply_ticks = param.Boolean(default=True, doc="""
        Whether to apply custom ticks.""")

    aspect = param.Parameter(default='square', doc="""
        The aspect ratio mode of the plot. By default, a plot may
        select its own appropriate aspect ratio but sometimes it may
        be necessary to force a square aspect ratio (e.g. to display
        the plot as an element of a grid). The modes 'auto' and
        'equal' correspond to the axis modes of the same name in
        matplotlib, a numeric value may also be passed.""")

    invert_xaxis = param.Boolean(default=False, doc="""
        Whether to invert the plot x-axis.""")

    invert_yaxis = param.Boolean(default=False, doc="""
        Whether to invert the plot y-axis.""")

    logx = param.Boolean(default=False, doc="""
         Whether to apply log scaling to the x-axis of the Chart.""")

    logy  = param.Boolean(default=False, doc="""
         Whether to apply log scaling to the y-axis of the Chart.""")

    logz  = param.Boolean(default=False, doc="""
         Whether to apply log scaling to the y-axis of the Chart.""")

    orientation = param.ObjectSelector(default='horizontal',
                                       objects=['horizontal', 'vertical'], doc="""
        The orientation of the plot. Note that this parameter may not
        always be respected by all plots but should be respected by
        adjoined plots when appropriate.""")

    show_legend = param.Boolean(default=False, doc="""
        Whether to show legend for the plot.""")

    show_grid = param.Boolean(default=False, doc="""
        Whether to show a Cartesian grid on the plot.""")

    xaxis = param.ObjectSelector(default='bottom',
                                      objects=['top', 'bottom', None], doc="""
        Whether and where to display the xaxis.""")

    yaxis = param.ObjectSelector(default='left',
                                      objects=['left', 'right', None], doc="""
        Whether and where to display the yaxis.""")

    xticks = param.Integer(default=5, doc="""
        Number of ticks along the x-axis.""")

    xticker = param.ClassSelector(default=None, class_=ticker.Locator, doc="""
        Allows supplying a matplotlib x-tick locator.""")

    xrotation = param.Integer(default=0, bounds=(0, 360), doc="""
        Rotation angle of the xticks.""")

    yticks = param.Integer(default=5, doc="""
        Number of ticks along the y-axis.""")

    yticker = param.ClassSelector(default=None, class_=ticker.Locator, doc="""
        Allows supplying a matplotlib y-tick locator.""")

    yrotation = param.Integer(default=0, bounds=(0, 360), doc="""
        Rotation angle of the xticks.""")

    zrotation = param.Integer(default=0, bounds=(0, 360), doc="""
        Rotation angle of the xticks.""")

    zticks = param.Integer(default=5, doc="""
        Number of ticks along the z-axis.""")

    zticker = param.ClassSelector(default=None, class_=ticker.Locator, doc="""
        Allows supplying a matplotlib z-tick locator.""")

    # Element Plots should declare the valid style options for matplotlib call
    style_opts = []

    _suppressed = [Table, ItemTable]

    def __init__(self, element, keys=None, ranges=None, dimensions=None, overlaid=0,
                 cyclic_index=0, style=None, zorder=0, adjoined=None, uniform=True, **params):
        self.dimensions = dimensions
        self.keys = keys
        if not isinstance(element, HoloMap):
            self.map = HoloMap(initial_items=(0, element),
                               key_dimensions=['Frame'], id=element.id)
        else:
            self.map = element
        self.uniform = uniform
        self.adjoined = adjoined
        self.map = self._check_map(ranges, keys)
        self.overlaid = overlaid
        self.cyclic_index = cyclic_index
        self.style = Store.lookup_options(self.map.last, 'style') if style is None else style
        self.zorder = zorder
        dimensions = self.map.key_dimensions if dimensions is None else dimensions
        keys = keys if keys else list(self.map.data.keys())
        plot_opts = Store.lookup_options(self.map.last, 'plot').options
        super(ElementPlot, self).__init__(keys=keys, dimensions=dimensions, adjoined=adjoined,
                                          uniform=uniform, **dict(params, **plot_opts))


    def _get_frame(self, key):
        if self.uniform:
            if not isinstance(key, tuple): key = (key,)
            key_dimensions = [d.name for d in self.map.key_dimensions]
            if self.dimensions is None:
                dimensions = key_dimensions
            else:
                dimensions = [d.name for d in self.dimensions]
            if key_dimensions == ['Frame'] and key_dimensions != dimensions:
                select = dict(Frame=0)
            else:
                select = {d: key[dimensions.index(d)]
                          for d in key_dimensions}
        elif isinstance(key, int):
            return self.map.values()[min([key, len(self.map)-1])]
        else:
            select = dict(zip(self.map.dimensions('key', label=True), key))
        try:
            selection = self.map.select(ignore_invalid=True, **select)
        except KeyError:
            selection = None
        return selection.last if isinstance(selection, HoloMap) else selection


    def _check_map(self, ranges=None, keys=None):
        """
        Helper method that ensures a given element is always returned as
        an HoloMap object.
        """
        # Apply data collapse
        if issubclass(self.map.type, CompositeOverlay):
            holomap = Compositor.collapse(self.map, None, mode='data')
        else:
            holomap = self.map

        # Compute framewise normalization
        mapwise_ranges = self.compute_ranges(holomap, None, None)
        if keys and isinstance(holomap, HoloMap) and ranges:
            frame_ranges = OrderedDict([(tuple(key),
                                         self.compute_ranges(holomap, key, ranges[key]))
                                        for key in keys])
            ranges = frame_ranges.values()
        elif isinstance(holomap, HoloMap):
            frame_ranges = OrderedDict([(key, self.compute_ranges(holomap, key, mapwise_ranges))
                                        for key in (keys if keys else holomap.keys())])
            ranges = frame_ranges.values()
            keys = holomap.data.keys()

        check = holomap.last
        if issubclass(holomap.type, CompositeOverlay):
            check = holomap.last.values()[0]
            holomap = Compositor.collapse(holomap, (ranges, keys if keys else None),
                                          mode='display')
        if isinstance(check, Element3D):
            self.projection = '3d'

        return holomap


    def get_extents(self, view, ranges):
        """
        Gets the extents for the axes from the current View. The globally
        computed ranges can optionally override the extents.
        """
        num = 6 if self.projection == '3d' else 4
        if self.apply_ranges and ranges:
            dims = view.dimensions()
            x0, x1 = ranges[dims[0].name]
            y0, y1 = ranges[dims[1].name]
            if self.projection == '3d':
                z0, z1 = ranges[dims[2].name]
                range_extents = (x0, y0, z0, x1, y1, z1)
            else:
                range_extents = (x0, y0, x1, y1)
        else:
            range_extents = (np.NaN,) * num

        if self.apply_extents:
            norm_opts = Store.lookup_options(view, 'norm').options
            if norm_opts.get('framewise', False):
                extents = view.extents
            else:
                extent_list = self.map.traverse(lambda x: x.extents, [Element])
                extents = max_extents(extent_list, self.projection == '3d')
        else:
            extents = (np.NaN,) * num
        return tuple(l2 if l2 is None or np.isfinite(l2) else l1 for l1, l2 in zip(range_extents, extents))


    def _format_title(self, key):
        frame = self._get_frame(key)
        if frame is None: return None
        type_name = type(frame).__name__
        group = frame.group if frame.group != type_name else ''
        label = frame.label
        if self.layout_dimensions:
            title = ''
        else:
            title_format = safe_unicode(self.title_format)
            title = title_format.format(label=safe_unicode(label),
                                        group=safe_unicode(group),
                                        type=type_name)
        dim_title = self._frame_title(key, 2)
        if not title or title.isspace():
            return dim_title
        elif not dim_title or dim_title.isspace():
            return title
        else:
            return '\n'.join([title, dim_title])


    def _draw_colorbar(self, artist):
        if 'cax' not in self.handles:
            axis = self.handles['axis']
            divider = make_axes_locatable(axis)
            self.handles['cax'] = divider.append_axes('right', size="5%", pad=0.05)
        plt.colorbar(artist, cax=self.handles['cax'])


    def _finalize_axis(self, key, title=None, ranges=None, xticks=None, yticks=None,
                       zticks=None, xlabel=None, ylabel=None, zlabel=None):
        """
        Applies all the axis settings before the axis or figure is returned.
        Only plots with zorder 0 get to apply their settings.

        When the number of the frame is supplied as n, this method looks
        up and computes the appropriate title, axis labels and axis bounds.
        """

        axis = self.handles['axis']

        view = self._get_frame(key)
        subplots = list(self.subplots.values()) if self.subplots else []
        if self.zorder == 0 and key is not None:
            title = None if self.zorder > 0 else self._format_title(key)
            suppress = any(sp.map.type in self._suppressed for sp in [self] + subplots
                           if isinstance(sp.map, HoloMap))
            if view is not None and not suppress:
                xlabel, ylabel, zlabel = self._axis_labels(view, subplots, xlabel, ylabel, zlabel)
                self._finalize_limits(axis, view, subplots, ranges)

                # Tick formatting
                xdim, ydim = view.get_dimension(0), view.get_dimension(1)
                xformat, yformat = None, None
                if xdim.formatter:
                    xformat = xdim.formatter
                elif xdim.type_formatters.get(xdim.type):
                    xformat = xdim.type_formatters[xdim.type]
                if xformat:
                    axis.xaxis.set_major_formatter(xformat)

                if ydim.formatter:
                    yformat = ydim.formatter
                elif ydim.type_formatters.get(ydim.type):
                    yformat = ydim.type_formatters[ydim.type]
                if yformat:
                    axis.yaxis.set_major_formatter(yformat)

            if not self.show_legend:
                legend = axis.get_legend()
                if legend: legend.set_visible(False)

            if self.show_grid:
                axis.get_xaxis().grid(True)
                axis.get_yaxis().grid(True)

            if xlabel: axis.set_xlabel(xlabel)
            if ylabel: axis.set_ylabel(ylabel)
            if zlabel: axis.set_zlabel(zlabel)

            self._apply_aspect(axis)
            self._subplot_label(axis)
            self._finalize_axes(axis)
            if self.apply_ticks:
                self._finalize_ticks(axis, xticks, yticks, zticks)

            if self.show_title and title is not None:
                self.handles['title'] = axis.set_title(title)

        for hook in self.finalize_hooks:
            try:
                hook(self, view)
            except Exception as e:
                self.warning("Plotting hook %r could not be applied:\n\n %s" % (hook, e))

        return super(ElementPlot, self)._finalize_axis(key)


    def _axis_labels(self, view, subplots, xlabel, ylabel, zlabel):
        # Axis labels
        dims = view.dimensions()
        if isinstance(view, CompositeOverlay):
            dims = dims[view.ndims:]
        if dims and xlabel is None:
            xlabel = str(dims[0])
        if len(dims) >= 2 and ylabel is None:
            ylabel = str(dims[1])
        if self.projection == '3d' and len(dims) >= 3 and zlabel is None:
            zlabel = str(dims[2])
        return xlabel, ylabel, zlabel


    def _apply_aspect(self, axis):
        if self.logx or self.logy:
            pass
        elif self.aspect == 'square':
            axis.set_aspect((1./axis.get_data_ratio()))
        elif self.aspect not in [None, 'square']:
            if isinstance(self.aspect, basestring):
                axis.set_aspect(self.aspect)
            else:
                axis.set_aspect(((1./axis.get_data_ratio()))/self.aspect)


    def _finalize_limits(self, axis, view, subplots, ranges):
        # Extents
        extents = self.get_extents(view, ranges)
        if extents and not self.overlaid:
            coords = [coord if np.isreal(coord) else np.NaN for coord in extents]
            if isinstance(view, Element3D):
                l, b, zmin, r, t, zmax = coords
                if not np.NaN in (zmin, zmax) and not zmin==zmax: axis.set_zlim((zmin, zmax))
            else:
                l, b, r, t = [coord if np.isreal(coord) else np.NaN for coord in extents]
            if not np.NaN in (l, r) and not l==r: axis.set_xlim((l, r))
            if not np.NaN in (b, t) and not b==t: axis.set_ylim((b, t))


    def _finalize_axes(self, axis):
        if self.logx:
            axis.set_xscale('log')
        elif self.logy:
            axis.set_yscale('log')

        if self.invert_xaxis:
            axis.invert_xaxis()
        if self.invert_yaxis:
            axis.invert_yaxis()


    def _finalize_ticks(self, axis, xticks, yticks, zticks):
        if not self.projection == '3d':
            disabled_spines = []
            if self.xaxis is not None:
                if self.xaxis == 'top':
                    axis.xaxis.set_ticks_position("top")
                    axis.xaxis.set_label_position("top")
                elif self.xaxis == 'bottom':
                    axis.xaxis.set_ticks_position("bottom")
            else:
                axis.xaxis.set_visible(False)
                disabled_spines.extend(['top', 'bottom'])

            if self.yaxis is not None:
                if self.yaxis == 'left':
                    axis.yaxis.set_ticks_position("left")
                elif self.yaxis == 'right':
                    axis.yaxis.set_ticks_position("right")
                    axis.yaxis.set_label_position("right")
            else:
                axis.yaxis.set_visible(False)
                disabled_spines.extend(['left', 'right'])

            for pos in disabled_spines:
                axis.spines[pos].set_visible(False)

        if not self.show_frame and self.projection != 'polar':
            axis.spines['right' if self.yaxis == 'left' else 'left'].set_visible(False)
            axis.spines['bottom' if self.xaxis == 'top' else 'top'].set_visible(False)

        if self.xticker:
            axis.xaxis.set_major_locator(self.xticker)
        elif xticks:
            axis.set_xticks(xticks[0])
            axis.set_xticklabels(xticks[1])
        elif self.logx:
            log_locator = ticker.LogLocator(numticks=self.xticks,
                                            subs=range(1,10))
            axis.xaxis.set_major_locator(log_locator)
        elif self.xticks:
            axis.xaxis.set_major_locator(ticker.MaxNLocator(self.xticks))

        for tick in axis.get_xticklabels():
            tick.set_rotation(self.xrotation)

        if self.yticker:
            axis.yaxis.set_major_locator(self.yticker)
        elif yticks:
            axis.set_yticks(yticks[0])
            axis.set_yticklabels(yticks[1])
        elif self.logy:
            log_locator = ticker.LogLocator(numticks=self.yticks,
                                            subs=range(1,10))
            axis.yaxis.set_major_locator(log_locator)
        elif self.yticks:
            axis.yaxis.set_major_locator(ticker.MaxNLocator(self.yticks))

        if not self.projection == '3d':
            pass
        elif self.zticker:
            axis.zaxis.set_major_locator(self.zticker)
        elif zticks:
            axis.set_zticks(zticks[0])
            axis.set_zticklabels(zticks[1])
        elif self.logz:
            log_locator = ticker.LogLocator(numticks=self.zticks,
                                            subs=range(1,10))
            axis.zaxis.set_major_locator(log_locator)
        else:
            axis.zaxis.set_major_locator(ticker.MaxNLocator(self.zticks))

        if self.projection == '3d':
            for tick in axis.get_zticklabels():
                tick.set_rotation(self.zrotation)


    def update_frame(self, key, ranges=None):
        """
        Set the plot(s) to the given frame number.  Operates by
        manipulating the matplotlib objects held in the self._handles
        dictionary.

        If n is greater than the number of available frames, update
        using the last available frame.
        """
        view = self._get_frame(key)
        axis = self.handles['axis']

        axes_visible = view is not None or self.overlaid
        axis.xaxis.set_visible(axes_visible and self.xaxis)
        axis.yaxis.set_visible(axes_visible and self.yaxis)
        axis.patch.set_alpha(np.min([int(axes_visible), 1]))

        for hname, handle in self.handles.items():
            hideable = hasattr(handle, 'set_visible')
            if hname not in ['axis', 'fig'] and hideable:
                handle.set_visible(view is not None)
        if view is None:
            return
        if self.normalize:
            ranges = self.compute_ranges(self.map, key, ranges)
            ranges = match_spec(view, ranges)
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
    OverlayPlot supports compositors processing of Overlays across maps.
    """

    style_group = param.List(['group'], bounds=(1, 2), doc="""Which Element parts of the
        Element specification the Elements will be grouped by for styling.
        Accepts any combination of label and group.""")

    show_legend = param.Boolean(default=True, doc="""
        Whether to show legend for the plot.""")

    legend_position = param.ObjectSelector(objects=['inner', 'right',
                                                    'bottom', 'top',
                                                    'left'],
                                           default='inner', doc="""
        Allows selecting between a number of predefined legend position
        options. The predefined options may be customized in the
        legend_specs class attribute.""")

    legend_specs = {'inner': {},
                    'left':   dict(bbox_to_anchor=(-.15, 1)),
                    'right':  dict(bbox_to_anchor=(1.25, 1)),
                    'top':    dict(bbox_to_anchor=(0., 1.02, 1., .102),
                                   ncol=3, mode="expand", borderaxespad=0.),
                    'bottom': dict(ncol=3, mode="expand",
                                   bbox_to_anchor=(0., -0.25, 1., .102),
                                   borderaxespad=0.1)}

    def __init__(self, overlay, ranges=None, **params):
        super(OverlayPlot, self).__init__(overlay, ranges=ranges, **params)
        self.subplots = self._create_subplots(ranges)


    def _create_subplots(self, ranges):
        subplots = OrderedDict()

        keys, vmaps = self.map.split_overlays()
        group_fn = lambda s: tuple(getattr(s.last, sg) for sg in self.style_group)
        style_groups = {k: list(v) for k,v in groupby(vmaps, group_fn)}
        style_lengths = {k: len(v) for k, v, in style_groups.items()}
        style_iter = {k: enumerate(v) for k, v, in style_groups.items()}
        overlay_type = 1 if self.map.type == Overlay else 2
        for zorder, (key, vmap) in enumerate(zip(keys, vmaps)):
            style_key = group_fn(vmap)
            cyclic_index,  _ = next(style_iter[style_key])
            length = style_lengths[style_key]
            style = Store.lookup_options(vmap.last, 'style').max_cycles(length)
            plotopts = dict(keys=self.keys, axis=self.handles['axis'], style=style,
                            cyclic_index=cyclic_index, figure=self.handles['fig'],
                            zorder=self.zorder+zorder, ranges=ranges, overlaid=overlay_type,
                            layout_dimensions=self.layout_dimensions,
                            show_title=self.show_title, dimensions=self.dimensions,
                            uniform=self.uniform, show_legend=False)
            plotype = Store.registry[type(vmap.last)]
            if not isinstance(key, tuple): key = (key,)
            subplots[key] = plotype(vmap, **plotopts)

        return subplots


    def _adjust_legend(self, axis):
        """
        Accumulate the legend handles and labels for all subplots
        and set up the legend
        """

        title = ''
        legend_data = []
        if issubclass(self.map.type, NdOverlay):
            dimensions = self.map.last.key_dimensions
            for key in self.map.last.data.keys():
                subplot = self.subplots[key]
                key = (dim.pprint_value(k) for k, dim in zip(key, dimensions))
                label = ','.join([str(k) + dim.unit if dim.unit else str(k) for dim, k in
                                  zip(dimensions, key)])
                handle = subplot.handles.get('legend_handle', False)
                if handle:
                    legend_data.append((handle, label))
            title = ', '.join([d.name for d in dimensions])
        else:
            for key, subplot in self.subplots.items():
                if isinstance(subplot, OverlayPlot):
                    legend_data += subplot.handles.get('legend_data', {}).items()
                else:
                    layer = self.map.last.data.get(key, False)
                    handle = subplot.handles.get('legend_handle', False)
                    if layer and layer.label and handle:
                        legend_data.append((handle, layer.label))
        autohandles, autolabels = axis.get_legend_handles_labels()
        legends = list(zip(*legend_data)) if legend_data else ([], [])
        all_handles = list(legends[0]) + list(autohandles)
        all_labels = list(legends[1]) + list(autolabels)
        data = OrderedDict()
        for handle, label in zip(all_handles, all_labels):
            if handle and (handle not in data) and label:
                data[handle] = label
        if not len(data) > 1 or not self.show_legend:
            legend = axis.get_legend()
            if legend:
                legend.set_visible(False)
        else:
            leg_spec = self.legend_specs[self.legend_position]
            leg = axis.legend(data.keys(), data.values(),
                              title=title, scatterpoints=1,
                              **leg_spec)
            frame = leg.get_frame()
            frame.set_facecolor('1.0')
            frame.set_edgecolor('0.0')
            frame.set_linewidth('1.0')
            self.handles['legend'] = leg
        self.handles['legend_data'] = data


    def __call__(self, ranges=None):
        axis = self.handles['axis']
        key = self.keys[-1]

        ranges = self.compute_ranges(self.map, key, ranges)
        for plot in self.subplots.values():
            plot(ranges=ranges)
        self._adjust_legend(axis)

        return self._finalize_axis(key, ranges=ranges, title=self._format_title(key))


    def _axis_labels(self, view, subplots, xlabel, ylabel, zlabel):
        return xlabel, ylabel, zlabel


    def get_extents(self, overlay, ranges):
        extents = []
        for key, subplot in self.subplots.items():
            layer = overlay.data.get(key, False)
            if layer and subplot.apply_ranges and not isinstance(layer, Annotation):
                if isinstance(layer, CompositeOverlay):
                    sp_ranges = ranges
                else:
                    sp_ranges = match_spec(layer, ranges) if ranges else {}
                extents.append(subplot.get_extents(layer, sp_ranges))
        return max_extents(extents, self.projection == '3d')


    def _format_title(self, key):
        frame = self._get_frame(key)
        if frame is None: return None

        type_name = type(frame).__name__
        group = frame.group if frame.group != type_name else ''
        label = frame.label
        if self.layout_dimensions:
            title = ''
        else:
            title_format = safe_unicode(self.title_format)
            title = title_format.format(label=safe_unicode(label),
                                        group=safe_unicode(group),
                                        type=type_name)
        dim_title = self._frame_title(key, 2)
        if not title or title.isspace():
            return dim_title
        elif not dim_title or dim_title.isspace():
            return title
        else:
            return '\n'.join([title, dim_title])


    def update_frame(self, key, ranges=None):
        if self.projection == '3d':
            self.handles['axis'].clear()

        ranges = self.compute_ranges(self.map, key, ranges)
        for plot in self.subplots.values():
            plot.update_frame(key, ranges)

        self._finalize_axis(key, ranges=ranges)


Store.registry.update({NdOverlay: OverlayPlot,
                       Overlay: OverlayPlot})
