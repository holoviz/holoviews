import math

from matplotlib import ticker
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import param

from ...core import util
from ...core import (OrderedDict, Collator, NdOverlay, HoloMap,
                     CompositeOverlay, Element3D, Columns, NdElement)
from ...element import Table, ItemTable, Raster
from ..plot import GenericElementPlot, GenericOverlayPlot
from .plot import MPLPlot
from .util import wrap_formatter


class ElementPlot(GenericElementPlot, MPLPlot):

    apply_ticks = param.Boolean(default=True, doc="""
        Whether to apply custom ticks.""")

    aspect = param.Parameter(default='square', doc="""
        The aspect ratio mode of the plot. By default, a plot may
        select its own appropriate aspect ratio but sometimes it may
        be necessary to force a square aspect ratio (e.g. to display
        the plot as an element of a grid). The modes 'auto' and
        'equal' correspond to the axis modes of the same name in
        matplotlib, a numeric value may also be passed.""")

    bgcolor = param.ClassSelector(class_=(str, tuple), default=None, doc="""
        If set bgcolor overrides the background color of the axis.""")

    invert_axes = param.ObjectSelector(default=False, doc="""
        Inverts the axes of the plot. Note that this parameter may not
        always be respected by all plots but should be respected by
        adjoined plots when appropriate.""")

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

    show_legend = param.Boolean(default=False, doc="""
        Whether to show legend for the plot.""")

    show_grid = param.Boolean(default=False, doc="""
        Whether to show a Cartesian grid on the plot.""")

    xaxis = param.ObjectSelector(default='bottom',
                                 objects=['top', 'bottom', 'bare', 'top-bare',
                                          'bottom-bare', None], doc="""
        Whether and where to display the xaxis, bare options allow suppressing
        all axis labels including ticks and xlabel. Valid options are 'top',
        'bottom', 'bare', 'top-bare' and 'bottom-bare'.""")

    yaxis = param.ObjectSelector(default='left',
                                      objects=['left', 'right', 'bare', 'left-bare',
                                               'right-bare', None], doc="""
        Whether and where to display the yaxis, bare options allow suppressing
        all axis labels including ticks and ylabel. Valid options are 'left',
        'right', 'bare' 'left-bare' and 'right-bare'.""")

    zaxis = param.Boolean(default=True, doc="""
        Whether to display the z-axis.""")

    xticks = param.Parameter(default=None, doc="""
        Ticks along x-axis specified as an integer, explicit list of
        tick locations, list of tuples containing the locations and
        labels or a matplotlib tick locator object. If set to None
        default matplotlib ticking behavior is applied.""")

    xrotation = param.Integer(default=0, bounds=(0, 360), doc="""
        Rotation angle of the xticks.""")

    yticks = param.Parameter(default=None, doc="""
        Ticks along y-axis specified as an integer, explicit list of
        tick locations, list of tuples containing the locations and
        labels or a matplotlib tick locator object. If set to None
        default matplotlib ticking behavior is applied.""")

    yrotation = param.Integer(default=0, bounds=(0, 360), doc="""
        Rotation angle of the yticks.""")

    zrotation = param.Integer(default=0, bounds=(0, 360), doc="""
        Rotation angle of the zticks.""")

    zticks = param.Parameter(default=None, doc="""
        Ticks along z-axis specified as an integer, explicit list of
        tick locations, list of tuples containing the locations and
        labels or a matplotlib tick locator object. If set to None
        default matplotlib ticking behavior is applied.""")

    # Element Plots should declare the valid style options for matplotlib call
    style_opts = []

    _suppressed = [Table, NdElement, Collator, Columns, ItemTable]

    def __init__(self, element, **params):
        super(ElementPlot, self).__init__(element, **params)
        check = self.hmap.last
        if isinstance(check, CompositeOverlay):
            check = check.values()[0] # Should check if any are 3D plots
        if isinstance(check, Element3D):
            self.projection = '3d'


    def _finalize_axis(self, key, title=None, ranges=None, xticks=None, yticks=None,
                       zticks=None, xlabel=None, ylabel=None, zlabel=None):
        """
        Applies all the axis settings before the axis or figure is returned.
        Only plots with zorder 0 get to apply their settings.

        When the number of the frame is supplied as n, this method looks
        up and computes the appropriate title, axis labels and axis bounds.
        """

        axis = self.handles['axis']
        if self.bgcolor:
            axis.set_axis_bgcolor(self.bgcolor)

        element = self._get_frame(key)
        subplots = list(self.subplots.values()) if self.subplots else []
        if self.zorder == 0 and key is not None:
            title = None if self.zorder > 0 else self._format_title(key)
            suppress = any(sp.hmap.type in self._suppressed for sp in [self] + subplots
                           if isinstance(sp.hmap, HoloMap))
            if element is not None and not suppress:
                xlabel, ylabel, zlabel = self._axis_labels(element, subplots, xlabel, ylabel, zlabel)
                if self.invert_axes:
                    xlabel, ylabel = ylabel, xlabel

                self._finalize_limits(axis, element, subplots, ranges)

                # Tick formatting
                xdim, ydim = element.get_dimension(0), element.get_dimension(1)
                xformat, yformat = None, None
                if xdim is None:
                    pass
                elif xdim.value_format:
                    xformat = xdim.value_format
                elif xdim.type in xdim.type_formatters:
                    xformat = xdim.type_formatters[xdim.type]
                if xformat:
                    axis.xaxis.set_major_formatter(wrap_formatter(xformat))

                if ydim is None:
                    pass
                elif ydim.value_format:
                    yformat = ydim.value_format
                elif ydim.type in ydim.type_formatters:
                    yformat = ydim.type_formatters[ydim.type]
                if yformat:
                    axis.yaxis.set_major_formatter(wrap_formatter(yformat))

            if self.zorder == 0 and not subplots:
                legend = axis.get_legend()
                if legend: legend.set_visible(self.show_legend)

                axis.get_xaxis().grid(self.show_grid)
                axis.get_yaxis().grid(self.show_grid)

            if xlabel and self.xaxis: axis.set_xlabel(xlabel, **self._fontsize('xlabel'))
            if ylabel and self.yaxis: axis.set_ylabel(ylabel, **self._fontsize('ylabel'))
            if zlabel and self.zaxis: axis.set_zlabel(zlabel, **self._fontsize('ylabel'))

            self._apply_aspect(axis)
            self._subplot_label(axis)
            if self.apply_ticks:
                self._finalize_ticks(axis, element, xticks, yticks, zticks)

            if self.show_title and title is not None:
                self.handles['title'] = axis.set_title(title,
                                                **self._fontsize('title'))
        # Always called to ensure log and inverted axes are applied
        self._finalize_axes(axis)
        if not self.overlaid and not self.drawn:
            self._finalize_artist(key)

        for hook in self.finalize_hooks:
            try:
                hook(self, element)
            except Exception as e:
                self.warning("Plotting hook %r could not be applied:\n\n %s" % (hook, e))

        return super(ElementPlot, self)._finalize_axis(key)


    def _finalize_artist(self, element):
        """
        Allows extending the _finalize_axis method with Element
        specific options.
        """
        pass


    def _apply_aspect(self, axis):
        if self.logx or self.logy:
            pass
        elif self.aspect == 'square':
            axis.set_aspect((1./axis.get_data_ratio()))
        elif self.aspect not in [None, 'square']:
            if isinstance(self.aspect, util.basestring):
                axis.set_aspect(self.aspect)
            else:
                axis.set_aspect(((1./axis.get_data_ratio()))/self.aspect)


    def _finalize_limits(self, axis, view, subplots, ranges):
        # Extents
        extents = self.get_extents(view, ranges)
        if extents and not self.overlaid:
            coords = [coord if np.isreal(coord) else np.NaN for coord in extents]
            if isinstance(view, Element3D) or self.projection == '3d':
                l, b, zmin, r, t, zmax = coords
                zmin, zmax = (c if np.isfinite(c) else None for c in (zmin, zmax))
                if not zmin == zmax:
                    axis.set_zlim((zmin, zmax))
            else:
                l, b, r, t = [coord if np.isreal(coord) else np.NaN for coord in extents]
            if self.invert_axes:
                l, b, r, t = b, l, t, r
            l, r = (c if np.isfinite(c) else None for c in (l, r))
            if self.invert_xaxis or any(p.invert_xaxis for p in subplots):
                r, l = l, r
            if not l == r:
                axis.set_xlim((l, r))
            b, t = (c if np.isfinite(c) else None for c in (b, t))
            if self.invert_yaxis or any(p.invert_yaxis for p in subplots):
                t, b = b, t
            if not b == t:
                axis.set_ylim((b, t))


    def _finalize_axes(self, axis):
        if self.logx:
            axis.set_xscale('log')
        elif self.logy:
            axis.set_yscale('log')


    def _finalize_ticks(self, axis, view, xticks, yticks, zticks):
        if not self.projection == '3d':
            disabled_spines = []
            if self.xaxis is not None:
                if 'bare' in self.xaxis:
                    axis.set_xticklabels([])
                    axis.xaxis.set_ticks_position('none')
                    axis.set_xlabel('')
                if 'top' in self.xaxis:
                    axis.xaxis.set_ticks_position("top")
                    axis.xaxis.set_label_position("top")
                elif 'bottom' in self.xaxis:
                    axis.xaxis.set_ticks_position("bottom")
            else:
                axis.xaxis.set_visible(False)
                disabled_spines.extend(['top', 'bottom'])

            if self.yaxis is not None:
                if 'bare' in self.yaxis:
                    axis.set_yticklabels([])
                    axis.yaxis.set_ticks_position('none')
                    axis.set_ylabel('')
                if 'left' in self.yaxis:
                    axis.yaxis.set_ticks_position("left")
                elif 'right' in self.yaxis:
                    axis.yaxis.set_ticks_position("right")
                    axis.yaxis.set_label_position("right")
            else:
                axis.yaxis.set_visible(False)
                disabled_spines.extend(['left', 'right'])

            for pos in disabled_spines:
                axis.spines[pos].set_visible(False)

        if not self.overlaid and not self.show_frame and self.projection != 'polar':
            xaxis = self.xaxis if self.xaxis else ''
            yaxis = self.yaxis if self.yaxis else ''
            axis.spines['top' if self.xaxis == 'bare' or 'bottom' in xaxis else 'bottom'].set_visible(False)
            axis.spines['right' if self.yaxis == 'bare' or 'left' in yaxis else 'left'].set_visible(False)

        if xticks:
            axis.set_xticks(xticks[0])
            axis.set_xticklabels(xticks[1])
        elif self.xticks is not None:
            if isinstance(self.xticks, ticker.Locator):
                axis.xaxis.set_major_locator(self.xticks)
            elif self.xticks == 0:
                axis.set_xticks([])
            elif isinstance(self.xticks, int):
                if self.logx:
                    locator = ticker.LogLocator(numticks=self.xticks,
                                                subs=range(1,10))
                else:
                    locator = ticker.MaxNLocator(self.xticks)
                axis.xaxis.set_major_locator(locator)
            elif isinstance(self.xticks, (list, tuple)):
                if all(isinstance(t, tuple) for t in self.xticks):
                    xticks, xlabels = zip(*self.xticks)
                else:
                    xdim = view.get_dimension(0)
                    xticks, xlabels = zip(*[(t, xdim.pprint_value(t))
                                            for t in self.xticks])
                axis.set_xticks(xticks)
                axis.set_xticklabels(xlabels)

        if self.xticks != 0 or xticks:
            for tick in axis.get_xticklabels():
                tick.set_rotation(self.xrotation)

        if yticks:
            axis.set_yticks(yticks[0])
            axis.set_yticklabels(yticks[1])
        elif self.yticks is not None:
            if isinstance(self.yticks, ticker.Locator):
                axis.yaxis.set_major_locator(self.yticks)
            elif self.yticks == 0:
                axis.set_yticks([])
            elif isinstance(self.yticks, int):
                if self.logy:
                    locator = ticker.LogLocator(numticks=self.yticks,
                                                subs=range(1,10))
                else:
                    locator = ticker.MaxNLocator(self.yticks)
                axis.yaxis.set_major_locator(locator)
            elif isinstance(self.yticks, (list, tuple)):
                if all(isinstance(t, tuple) for t in self.yticks):
                    yticks, ylabels = zip(*self.yticks)
                else:
                    ydim = view.get_dimension(1)
                    yticks, ylabels = zip(*[(t, ydim.pprint_value(t))
                                            for t in self.yticks])
                axis.set_yticks(yticks)
                axis.set_yticklabels(ylabels)

        if self.yticks != 0 or yticks:
            for tick in axis.get_yticklabels():
                tick.set_rotation(self.yrotation)

        if not self.projection == '3d':
            pass
        elif zticks:
            axis.set_zticks(zticks[0])
            axis.set_zticklabels(zticks[1])
        elif self.zticks is not None:
            if isinstance(self.zticks, ticker.Locator):
                axis.zaxis.set_major_locator(self.zticks)
            elif self.zticks == 0:
                axis.set_zticks([])
            elif isinstance(self.zticks, int):
                if self.logz:
                    locator = ticker.LogLocator(numticks=self.zticks,
                                                subs=range(1,10))
                else:
                    locator = ticker.MaxNLocator(self.zticks)
                axis.zaxis.set_major_locator(locator)
            elif isinstance(self.zticks, (list, tuple)):
                if all(isinstance(t, tuple) for t in self.zticks):
                    zticks, zlabels = zip(*self.zticks)
                else:
                    zdim = view.get_dimension(2)
                    zticks, zlabels = zip(*[(t, zdim.pprint_value(t))
                                            for t in self.zticks])
                axis.set_zticks(zticks)
                axis.set_zticklabels(zlabels)

        if self.projection == '3d' and self.zticks != 0:
            for tick in axis.get_zticklabels():
                tick.set_rotation(self.zrotation)

        tick_fontsize = self._fontsize('ticks','labelsize',common=False)
        if tick_fontsize:  axis.tick_params(**tick_fontsize)


    def update_frame(self, key, ranges=None, element=None):
        """
        Set the plot(s) to the given frame number.  Operates by
        manipulating the matplotlib objects held in the self._handles
        dictionary.

        If n is greater than the number of available frames, update
        using the last available frame.
        """
        if not element:
            if self.dynamic and self.overlaid:
                self.current_key = key
                element = self.current_frame
            else:
                element = self._get_frame(key)
        else:
            self.current_key = key
            self.current_frame = element

        if element is not None:
            self.set_param(**self.lookup_options(element, 'plot').options)
        axis = self.handles['axis']

        axes_visible = element is not None or self.overlaid
        axis.xaxis.set_visible(axes_visible and self.xaxis)
        axis.yaxis.set_visible(axes_visible and self.yaxis)
        axis.patch.set_alpha(np.min([int(axes_visible), 1]))

        for hname, handle in self.handles.items():
            hideable = hasattr(handle, 'set_visible')
            if hname not in ['axis', 'fig'] and hideable:
                handle.set_visible(element is not None)
        if element is None:
            return
        ranges = self.compute_ranges(self.hmap, key, ranges)
        if not self.adjoined:
            ranges = util.match_spec(element, ranges)
        axis_kwargs = self.update_handles(axis, element, key if element is not None else {}, ranges)
        self._finalize_axis(key, ranges=ranges, **(axis_kwargs if axis_kwargs else {}))


    def update_handles(self, axis, view, key, ranges=None):
        """
        Update the elements of the plot.
        :param axis:
        """
        raise NotImplementedError



class ColorbarPlot(ElementPlot):

    colorbar = param.Boolean(default=False, doc="""
        Whether to draw a colorbar.""")

    cbar_width = param.Number(default=0.05, doc="""
        Width of the colorbar as a fraction of the main plot""")

    cbar_padding = param.Number(default=0.01, doc="""
        Padding between colorbar and other plots.""")

    cbar_ticks = param.Parameter(default=None, doc="""
        Ticks along colorbar-axis specified as an integer, explicit
        list of tick locations, list of tuples containing the
        locations and labels or a matplotlib tick locator object. If
        set to None default matplotlib ticking behavior is
        applied.""")

    _colorbars = {}

    def _adjust_cbar(self, cbar, label, dim):
        if math.floor(self.style[self.cyclic_index].get('alpha', 1)) == 1:
            cbar.solids.set_edgecolor("face")
        cbar.set_label(label)
        if isinstance(self.cbar_ticks, ticker.Locator):
            cbar.set_major_locator(self.cbar_ticks)
        elif self.cbar_ticks == 0:
            cbar.set_ticks([])
        elif isinstance(self.cbar_ticks, int):
            locator = ticker.MaxNLocator(self.cbar_ticks)
            cbar.set_major_locator(locator)
        elif isinstance(self.cbar_ticks, list):
            if all(isinstance(t, tuple) for t in self.cbar_ticks):
                ticks, labels = zip(*self.cbar_ticks)
            else:
                ticks, labels = zip(*[(t, dim.pprint_value(t))
                                        for t in self.cbar_ticks])
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(labels)


    def _finalize_artist(self, key):
        element = self.hmap.last
        artist = self.handles.get('artist', None)
        if artist and self.colorbar:
            self._draw_colorbar(artist, element)


    def _draw_colorbar(self, artist, element, dim=None):
        fig = self.handles['fig']
        axis = self.handles['axis']
        ax_colorbars, position = ColorbarPlot._colorbars.get(id(axis), ([], None))
        specs = [spec[:2] for _, _, spec, _ in ax_colorbars]
        spec = util.get_spec(element)

        if position is None:
            fig.canvas.draw()
            bbox = axis.get_position()
            l, b, w, h = bbox.x0, bbox.y0, bbox.width, bbox.height
        else:
            l, b, w, h = position

        # Get colorbar label
        dim = element.get_dimension(dim)
        if dim is None:
            dim = element.vdims[0]
        label = str(dim)

        padding = self.cbar_padding
        width = self.cbar_width
        if spec[:2] not in specs:
            offset = len(ax_colorbars)
            scaled_w = w*width
            cax = fig.add_axes([l+w+padding+(scaled_w+padding+w*0.15)*offset,
                                b, scaled_w, h])
            cbar = plt.colorbar(artist, cax=cax)
            self._adjust_cbar(cbar, label, dim)
            self.handles['cax'] = cax
            self.handles['cbar'] = cbar
            ax_colorbars.append((artist, cax, spec, label))

        for i, (artist, cax, spec, label) in enumerate(ax_colorbars[:-1]):
            scaled_w = w*width
            cax.set_position([l+w+padding+(scaled_w+padding+w*0.15)*i,
                              b, scaled_w, h])

        ColorbarPlot._colorbars[id(axis)] = (ax_colorbars, (l, b, w, h))



    def _norm_kwargs(self, element, ranges, opts):
        """
        Returns valid color normalization kwargs
        to be passed to matplotlib plot function.
        """
        norm = None
        clim = opts.pop('clims', None)
        if clim is None:
            val_dim = [d.name for d in element.vdims][0]
            clim = ranges.get(val_dim)
            if self.symmetric:
                clim = -np.abs(clim).max(), np.abs(clim).max()
        if self.logz:
            if self.symmetric:
                norm = colors.SymLogNorm(vmin=clim[0], vmax=clim[1],
                                         linthresh=clim[1]/np.e)
            else:
                norm = colors.LogNorm(vmin=clim[0], vmax=clim[1])
        return clim, norm, opts



class LegendPlot(ElementPlot):

    show_legend = param.Boolean(default=True, doc="""
        Whether to show legend for the plot.""")

    legend_cols = param.Integer(default=None, doc="""
       Number of legend columns in the legend.""")

    legend_position = param.ObjectSelector(objects=['inner', 'right',
                                                    'bottom', 'top',
                                                    'left', 'best',
                                                    'top_right',
                                                    'top_left',
                                                    'bottom_left',
                                                    'bottom_right'],
                                           default='inner', doc="""
        Allows selecting between a number of predefined legend position
        options. The predefined options may be customized in the
        legend_specs class attribute. By default, 'inner', 'right',
        'bottom', 'top', 'left', 'best', 'top_right', 'top_left',
        'bottom_right' and 'bottom_left' are supported.""")

    legend_specs = {'inner': {},
                    'best': {},
                    'left':   dict(bbox_to_anchor=(-.15, 1), loc=1),
                    'right':  dict(bbox_to_anchor=(1.05, 1), loc=2),
                    'top':    dict(bbox_to_anchor=(0., 1.02, 1., .102),
                                   ncol=3, loc=3, mode="expand", borderaxespad=0.),
                    'bottom': dict(ncol=3, mode="expand", loc=2,
                                   bbox_to_anchor=(0., -0.25, 1., .102),
                                   borderaxespad=0.1),
                    'top_right': dict(loc=1),
                    'top_left': dict(loc=2),
                    'bottom_left': dict(loc=3),
                    'bottom_right': dict(loc=4)}



class OverlayPlot(LegendPlot, GenericOverlayPlot):
    """
    OverlayPlot supports compositors processing of Overlays across maps.
    """

    _passed_handles = ['fig', 'axis']

    def __init__(self, overlay, ranges=None, **params):
        if overlay.traverse(lambda x: x, (Element3D,)):
            params['projection'] = '3d'
        super(OverlayPlot, self).__init__(overlay, ranges=ranges, **params)

    def _finalize_artist(self, key):
        for subplot in self.subplots.values():
            subplot._finalize_artist(key)

    def _adjust_legend(self, axis):
        """
        Accumulate the legend handles and labels for all subplots
        and set up the legend
        """

        title = ''
        legend_data = []
        if issubclass(self.hmap.type, NdOverlay):
            dimensions = self.hmap.last.kdims
            for key in self.hmap.last.data.keys():
                subplot = self.subplots[key]
                key = (dim.pprint_value(k) for k, dim in zip(key, dimensions))
                label = ','.join([str(k) + dim.unit if dim.unit else str(k) for dim, k in
                                  zip(dimensions, key)])
                handle = subplot.handles.get('artist', False)
                if handle:
                    legend_data.append((handle, label))
            title = ', '.join([d.name for d in dimensions])
        else:
            for key, subplot in self.subplots.items():
                if isinstance(subplot, OverlayPlot):
                    legend_data += subplot.handles.get('legend_data', {}).items()
                else:
                    layer = self.hmap.last.data.get(key, False)
                    handle = subplot.handles.get('artist', False)
                    if layer and not isinstance(layer, Raster) and layer.label and handle:
                        legend_data.append((handle, layer.label))
        autohandles, autolabels = axis.get_legend_handles_labels()
        legends = list(zip(*legend_data)) if legend_data else ([], [])
        all_handles = list(legends[0]) + list(autohandles)
        all_labels = list(legends[1]) + list(autolabels)
        data = OrderedDict()
        show_legend = self.lookup_options(self.hmap.last, 'plot').options.get('show_legend', None)
        used_labels = []
        for handle, label in zip(all_handles, all_labels):
            if handle and (handle not in data) and label and label not in used_labels:
                data[handle] = label
                used_labels.append(label)
        if (not len(set(data.values())) > 1 and not show_legend) or not self.show_legend:
            legend = axis.get_legend()
            if legend:
                legend.set_visible(False)
        else:
            leg_spec = self.legend_specs[self.legend_position]
            if self.legend_cols: leg_spec['ncol'] = self.legend_cols
            leg = axis.legend(data.keys(), data.values(),
                              title=title, scatterpoints=1,
                              **leg_spec)
            frame = leg.get_frame()
            frame.set_facecolor('1.0')
            frame.set_edgecolor('0.0')
            frame.set_linewidth('1.0')
            self.handles['legend'] = leg
        self.handles['legend_data'] = data


    def initialize_plot(self, ranges=None):
        axis = self.handles['axis']
        key = self.keys[-1]

        ranges = self.compute_ranges(self.hmap, key, ranges)
        for plot in self.subplots.values():
            plot.initialize_plot(ranges=ranges)
        self._adjust_legend(axis)

        return self._finalize_axis(key, ranges=ranges, title=self._format_title(key))


    def update_frame(self, key, ranges=None, element=None):
        if self.projection == '3d':
            self.handles['axis'].clear()

        if element is None:
            element = self._get_frame(key)
        else:
            self.current_frame = element
            self.current_key = key
        ranges = self.compute_ranges(self.hmap, key, ranges)
        for k, plot in self.subplots.items():
            plot.update_frame(key, ranges, element.get(k, None))

        self._finalize_axis(key, ranges=ranges)



class DrawPlot(ElementPlot):
    """
    A DrawPlot is an ElementPlot that uses a draw method for
    rendering. The draw method is also called per update such that a
    full redraw is triggered per frame.

    Although not optimized for HoloMaps (due to the full redraw),
    DrawPlot is very easy to subclass to interface HoloViews with any
    third-party libraries offering matplotlib plotting functionality.
    """

    _abstract = True

    def draw(self, axis, element, ranges=None):
        """
        The only method that needs to be overridden in subclasses.

        The current axis and element are supplied as arguments. The
        job of this function is to apply the appropriate matplotlib
        commands to render the element to the supplied axis.
        """
        raise NotImplementedError

    def initialize_plot(self, ranges=None):
        element = self.hmap.last
        key = self.keys[-1]
        ranges = self.compute_ranges(self.hmap, key, ranges)
        ranges = util.match_spec(element, ranges)
        self.draw(self.handles['axis'], self.hmap.last, ranges)
        return self._finalize_axis(self.keys[-1], ranges=ranges)

    def update_handles(self, axis, element, key, ranges=None):
        if self.zorder == 0 and axis: axis.cla()
        self.draw(axis, element, ranges)
