import math

from matplotlib import ticker
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import param

from ...core import util
from ...core import (OrderedDict, NdOverlay, DynamicMap,
                     CompositeOverlay, Element3D, Element)
from ...core.options import abbreviated_exception
from ..plot import GenericElementPlot, GenericOverlayPlot
from ..util import dynamic_update
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

    invert_zaxis = param.Boolean(default=False, doc="""
        Whether to invert the plot z-axis.""")

    labelled = param.List(default=['x', 'y'], doc="""
        Whether to plot the 'x' and 'y' labels.""")

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

    # Whether plot has axes, disables setting axis limits, labels and ticks
    _has_axes = True

    def __init__(self, element, **params):
        super(ElementPlot, self).__init__(element, **params)
        check = self.hmap.last
        if isinstance(check, CompositeOverlay):
            check = check.values()[0] # Should check if any are 3D plots
        if isinstance(check, Element3D):
            self.projection = '3d'

        for hook in self.initial_hooks:
            try:
                hook(self, element)
            except Exception as e:
                self.warning("Plotting hook %r could not be applied:\n\n %s" % (hook, e))


    def _finalize_axis(self, key, title=None, dimensions=None, ranges=None, xticks=None,
                       yticks=None, zticks=None, xlabel=None, ylabel=None, zlabel=None):
        """
        Applies all the axis settings before the axis or figure is returned.
        Only plots with zorder 0 get to apply their settings.

        When the number of the frame is supplied as n, this method looks
        up and computes the appropriate title, axis labels and axis bounds.
        """
        element = self._get_frame(key)
        self.current_frame = element
        if not dimensions and element and not self.subplots:
            el = element.traverse(lambda x: x, [Element])
            if el: dimensions = el[0].dimensions()
        axis = self.handles['axis']

        subplots = list(self.subplots.values()) if self.subplots else []
        if self.zorder == 0 and key is not None:
            if self.bgcolor:
                axis.set_axis_bgcolor(self.bgcolor)

            # Apply title
            title = None if self.zorder > 0 else self._format_title(key)
            if self.show_title and title is not None:
                fontsize = self._fontsize('title')
                self.handles['title'] = axis.set_title(title, **fontsize)

            # Apply subplot label
            self._subplot_label(axis)

            # Apply axis options if axes are enabled
            if element and not any(not sp._has_axes for sp in [self] + subplots):
                # Set axis labels
                if dimensions:
                    self._set_labels(axis, dimensions, xlabel, ylabel, zlabel)

                # Set axes limits
                self._set_axis_limits(axis, element, subplots, ranges)

                if not subplots:
                    legend = axis.get_legend()
                    if legend: legend.set_visible(self.show_legend)
                    axis.xaxis.grid(self.show_grid)
                    axis.yaxis.grid(self.show_grid)

                # Apply log axes
                if self.logx:
                    axis.set_xscale('log')
                if self.logy:
                    axis.set_yscale('log')

                if not self.projection == '3d':
                    self._set_axis_position(axis, 'x', self.xaxis)
                    self._set_axis_position(axis, 'y', self.yaxis)

                # Apply ticks
                if self.apply_ticks:
                    self._finalize_ticks(axis, dimensions, xticks, yticks, zticks)

            # Apply aspects
            if not (self.logx or self.logy):
                self._set_aspect(axis, self.aspect)

        if not subplots and not self.drawn:
            self._finalize_artist(key)

        for hook in self.finalize_hooks:
            try:
                hook(self, element)
            except Exception as e:
                self.warning("Plotting hook %r could not be applied:\n\n %s" % (hook, e))

        return super(ElementPlot, self)._finalize_axis(key)


    def _finalize_ticks(self, axis, dimensions, xticks, yticks, zticks):
        """
        Finalizes the ticks on the axes based on the supplied ticks
        and Elements. Sets the axes position as well as tick positions,
        labels and fontsize.
        """
        ndims = len(dimensions) if dimensions else 0
        xdim = dimensions[0] if ndims else None
        ydim = dimensions[1] if ndims > 1 else None

        # Tick formatting
        if xdim:
            self._set_axis_formatter(axis.xaxis, xdim)
        if ydim:
            self._set_axis_formatter(axis.yaxis, ydim)
        if self.projection == '3d':
            zdim = dimensions[2] if ndims > 2 else None
            if zdim:
                self._set_axis_formatter(axis.zaxis, zdim)

        xticks = xticks if xticks else self.xticks
        self._set_axis_ticks(axis.xaxis, xticks, log=self.logx,
                             rotation=self.xrotation)

        yticks = yticks if yticks else self.yticks
        self._set_axis_ticks(axis.yaxis, yticks, log=self.logy,
                             rotation=self.yrotation)

        if self.projection == '3d':
            zticks = zticks if zticks else self.zticks
            self._set_axis_ticks(axis.zaxis, zticks, log=self.logz,
                                 rotation=self.zrotation)

        tick_fontsize = self._fontsize('ticks','labelsize',common=False)
        if tick_fontsize: axis.tick_params(**tick_fontsize)


    def _finalize_artist(self, element):
        """
        Allows extending the _finalize_axis method with Element
        specific options.
        """
        pass


    def _set_labels(self, axes, dimensions, xlabel=None, ylabel=None, zlabel=None):
        """
        Sets the labels of the axes using the supplied list of dimensions.
        Optionally explicit labels may be supplied to override the dimension
        label.
        """
        xlabel, ylabel, zlabel = self._get_axis_labels(dimensions, xlabel, ylabel, zlabel)
        if self.invert_axes:
            xlabel, ylabel = ylabel, xlabel
        if xlabel and self.xaxis and 'x' in self.labelled:
            axes.set_xlabel(xlabel, **self._fontsize('xlabel'))
        if ylabel and self.yaxis and 'y' in self.labelled:
            axes.set_ylabel(ylabel, **self._fontsize('ylabel'))
        if zlabel and self.zaxis and 'z' in self.labelled:
            axes.set_zlabel(zlabel, **self._fontsize('zlabel'))


    def _set_axis_formatter(self, axis, dim):
        """
        Set axis formatter based on dimension formatter.
        """
        if isinstance(dim, list): dim = dim[0]
        formatter = None
        if dim.value_format:
            formatter = dim.value_format
        elif dim.type in dim.type_formatters:
            formatter = dim.type_formatters[dim.type]
        if formatter:
            axis.set_major_formatter(wrap_formatter(formatter))


    def _set_aspect(self, axes, aspect):
        """
        Set the aspect on the axes based on the aspect setting.
        """
        if aspect and aspect == 'square':
            axes.set_aspect((1./axes.get_data_ratio()))
        elif aspect not in [None, 'square']:
            if isinstance(aspect, util.basestring):
                axes.set_aspect(aspect)
            else:
                axes.set_aspect(((1./axes.get_data_ratio()))/aspect)


    def _set_axis_limits(self, axis, view, subplots, ranges):
        """
        Compute extents for current view and apply as axis limits
        """
        # Extents
        scalex, scaley = True, True
        extents = self.get_extents(view, ranges)
        if extents and not self.overlaid:
            coords = [coord if np.isreal(coord) else np.NaN for coord in extents]
            valid_lim = lambda c: util.isnumeric(c) and not np.isnan(c)
            if self.projection == '3d' or len(extents) == 6:
                l, b, zmin, r, t, zmax = coords
                if self.invert_zaxis or any(p.invert_zaxis for p in subplots):
                    zmin, zmax = zmax, zmin
                if zmin != zmax:
                    if valid_lim(zmin):
                        axis.set_zlim(bottom=zmin)
                    if valid_lim(zmax):
                        axis.set_zlim(top=zmax)
            else:
                l, b, r, t = coords

            if self.invert_axes:
                l, b, r, t = b, l, t, r

            if self.invert_xaxis or any(p.invert_xaxis for p in subplots):
                r, l = l, r
            if l != r:
                if valid_lim(l):
                    axis.set_xlim(left=l)
                    scalex = False
                if valid_lim(r):
                    axis.set_xlim(right=r)
                    scalex = False

            if self.invert_yaxis or any(p.invert_yaxis for p in subplots):
                t, b = b, t
            if b != t:
                if valid_lim(b):
                    axis.set_ylim(bottom=b)
                    scaley = False
                if valid_lim(t):
                    axis.set_ylim(top=t)
                    scaley = False
        axis.autoscale_view(scalex=scalex, scaley=scaley)


    def _set_axis_position(self, axes, axis, option):
        """
        Set the position and visibility of the xaxis or yaxis by
        supplying the axes object, the axis to set, i.e. 'x' or 'y'
        and an option to specify the position and visibility of the axis.
        The option may be None, 'bare' or positional, i.e. 'left' and
        'right' for the yaxis and 'top' and 'bottom' for the xaxis.
        May also combine positional and 'bare' into for example 'left-bare'.
        """
        positions = {'x': ['bottom', 'top'], 'y': ['left', 'right']}[axis]
        axis = axes.xaxis if axis == 'x' else axes.yaxis
        if option is None:
            axis.set_visible(False)
            for pos in positions:
                axes.spines[pos].set_visible(False)
        else:
            if 'bare' in option:
                axis.set_ticklabels([])
                axis.set_label_text('')
            if option != 'bare':
                option = option.split('-')[0]
                axis.set_ticks_position(option)
                axis.set_label_position(option)
        if not self.overlaid and not self.show_frame and self.projection != 'polar':
            pos = (positions[1] if (option and (option == 'bare' or positions[0] in option))
                   else positions[0])
            axes.spines[pos].set_visible(False)


    def _set_axis_ticks(self, axis, ticks, log=False, rotation=0):
        """
        Allows setting the ticks for a particular axis either with
        a tuple of ticks, a tick locator object, an integer number
        of ticks, a list of tuples containing positions and labels
        or a list of positions. Also supports enabling log ticking
        if an integer number of ticks is supplied and setting a
        rotation for the ticks.
        """
        if isinstance(ticks, (list, tuple)) and all(isinstance(l, list) for l in ticks):
            axis.set_ticks(ticks[0])
            axis.set_ticklabels(ticks[1])
        elif isinstance(ticks, ticker.Locator):
            axis.set_major_locator(ticks)
        elif not ticks and ticks is not None:
            axis.set_ticks([])
        elif isinstance(ticks, int):
            if log:
                locator = ticker.LogLocator(numticks=ticks,
                                            subs=range(1,10))
            else:
                locator = ticker.MaxNLocator(ticks)
            axis.set_major_locator(locator)
        elif isinstance(ticks, (list, tuple)):
            labels = None
            if all(isinstance(t, tuple) for t in ticks):
                ticks, labels = zip(*ticks)
            axis.set_ticks(ticks)
            if labels:
                axis.set_ticklabels(labels)
        for tick in axis.get_ticklabels():
            tick.set_rotation(rotation)


    def update_frame(self, key, ranges=None, element=None):
        """
        Set the plot(s) to the given frame number.  Operates by
        manipulating the matplotlib objects held in the self._handles
        dictionary.

        If n is greater than the number of available frames, update
        using the last available frame.
        """
        reused = isinstance(self.hmap, DynamicMap) and self.overlaid
        if not reused and element is None:
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

        label = element.label if self.show_legend else ''
        style = dict(label=label, zorder=self.zorder, **self.style[self.cyclic_index])
        axis_kwargs = self.update_handles(key, axis, element, ranges, style)
        self._finalize_axis(key, ranges=ranges, **(axis_kwargs if axis_kwargs else {}))


    def initialize_plot(self, ranges=None):
        element = self.hmap.last
        ax = self.handles['axis']
        key = list(self.hmap.data.keys())[-1]
        dim_map = dict(zip((d.name for d in self.hmap.kdims), key))
        key = tuple(dim_map.get(d.name, None) for d in self.dimensions)

        ranges = self.compute_ranges(self.hmap, key, ranges)
        ranges = util.match_spec(element, ranges)

        style = dict(zorder=self.zorder, **self.style[self.cyclic_index])
        if self.show_legend:
            style['label'] = element.label

        plot_data, plot_kwargs, axis_kwargs = self.get_data(element, ranges, style)

        with abbreviated_exception():
            handles = self.init_artists(ax, plot_data, plot_kwargs)
        self.handles.update(handles)

        return self._finalize_axis(self.keys[-1], ranges=ranges, **axis_kwargs)


    def update_handles(self, key, axis, element, ranges, style):
        """
        Update the elements of the plot.
        """
        self.teardown_handles()
        plot_data, plot_kwargs, axis_kwargs = self.get_data(element, ranges, style)

        with abbreviated_exception():
            handles = self.init_artists(axis, plot_data, plot_kwargs)
        self.handles.update(handles)
        return axis_kwargs

    def teardown_handles(self):
        """
        If no custom update_handles method is supplied this method
        is called to tear down any previous handles before replacing
        them.
        """
        if 'artist' in self.handles:
            self.handles['artist'].remove()




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

    symmetric = param.Boolean(default=False, doc="""
        Whether to make the colormap symmetric around zero.""")

    _colorbars = {}

    def _adjust_cbar(self, cbar, label, dim):
        noalpha = math.floor(self.style[self.cyclic_index].get('alpha', 1)) == 1
        if (cbar.solids and noalpha):
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
            ylabel = cax.yaxis.get_label()
            self.handles['bbox_extra_artists'] += [cax, ylabel]
            ax_colorbars.append((artist, cax, spec, label))

        for i, (artist, cax, spec, label) in enumerate(ax_colorbars[:-1]):
            scaled_w = w*width
            cax.set_position([l+w+padding+(scaled_w+padding+w*0.15)*i,
                              b, scaled_w, h])

        ColorbarPlot._colorbars[id(axis)] = (ax_colorbars, (l, b, w, h))



    def _norm_kwargs(self, element, ranges, opts, vdim):
        """
        Returns valid color normalization kwargs
        to be passed to matplotlib plot function.
        """
        clim = opts.pop('clims', None)
        if clim is None:
            clim = ranges[vdim.name] if vdim.name in ranges else element.range(vdim)
            if self.symmetric:
                clim = -np.abs(clim).max(), np.abs(clim).max()
        if self.logz:
            if self.symmetric:
                norm = colors.SymLogNorm(vmin=clim[0], vmax=clim[1],
                                         linthresh=clim[1]/np.e)
            else:
                norm = colors.LogNorm(vmin=clim[0], vmax=clim[1])
            opts['norm'] = norm
        opts['vmin'] = clim[0]
        opts['vmax'] = clim[1]


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
        if 'projection' not in params:
            params['projection'] = self._get_projection(overlay)
        super(OverlayPlot, self).__init__(overlay, ranges=ranges, **params)


    def _finalize_artist(self, key):
        for subplot in self.subplots.values():
            subplot._finalize_artist(key)

    def _adjust_legend(self, overlay, axis):
        """
        Accumulate the legend handles and labels for all subplots
        and set up the legend
        """
        legend_data = []
        dimensions = overlay.kdims
        title = ', '.join([d.name for d in dimensions])
        for key, subplot in self.subplots.items():
            element = overlay.data.get(key, False)
            if not subplot.show_legend or not element: continue
            title = ', '.join([d.name for d in dimensions])
            handle = subplot.handles.get('artist', False)
            if isinstance(overlay, NdOverlay):
                key = (dim.pprint_value(k) for k, dim in zip(key, dimensions))
                label = ','.join([str(k) + dim.unit if dim.unit else str(k) for dim, k in
                                  zip(dimensions, key)])
                if handle:
                    legend_data.append((handle, label))
            else:
                if isinstance(subplot, OverlayPlot):
                    legend_data += subplot.handles.get('legend_data', {}).items()
                if element.label and handle:
                    legend_data.append((handle, element.label))
        all_handles, all_labels = list(zip(*legend_data)) if legend_data else ([], [])
        data = OrderedDict()
        used_labels = []
        for handle, label in zip(all_handles, all_labels):
            if handle and (handle not in data) and label and label not in used_labels:
                data[handle] = label
                used_labels.append(label)
        if (not len(set(data.values())) > 0) or not self.show_legend:
            legend = axis.get_legend()
            if legend:
                legend.set_visible(False)
        else:
            leg_spec = self.legend_specs[self.legend_position]
            if self.legend_cols: leg_spec['ncol'] = self.legend_cols
            leg = axis.legend(data.keys(), data.values(),
                              title=title, scatterpoints=1,
                              **dict(leg_spec, **self._fontsize('legend')))
            title_fontsize = self._fontsize('legend_title')
            if title_fontsize:
                leg.get_title().set_fontsize(title_fontsize['fontsize'])
            frame = leg.get_frame()
            frame.set_facecolor('1.0')
            frame.set_edgecolor('0.0')
            frame.set_linewidth('1.0')
            leg.set_zorder(10e6)
            self.handles['legend'] = leg
            self.handles['bbox_extra_artists'].append(leg)
        self.handles['legend_data'] = data


    def initialize_plot(self, ranges=None):
        axis = self.handles['axis']
        key = self.keys[-1]
        element = self._get_frame(key)

        ranges = self.compute_ranges(self.hmap, key, ranges)
        for k, subplot in self.subplots.items():
            subplot.initialize_plot(ranges=ranges)
            if isinstance(element, CompositeOverlay):
                frame = element.get(k, None)
                subplot.current_frame = frame

        if self.show_legend:
            self._adjust_legend(element, axis)

        return self._finalize_axis(key, ranges=ranges, title=self._format_title(key))


    def update_frame(self, key, ranges=None, element=None):
        axis = self.handles['axis']
        if element is None:
            element = self._get_frame(key)
        else:
            self.current_frame = element
            self.current_key = key

        if isinstance(self.hmap, DynamicMap):
            range_obj = element
            items = element.items()
        else:
            range_obj = self.hmap
            items = element.items()
        ranges = self.compute_ranges(range_obj, key, ranges)

        for k, subplot in self.subplots.items():
            el = element.get(k, None)
            if isinstance(self.hmap, DynamicMap):
                idx = dynamic_update(self, subplot, k, element, items)
                if idx is not None:
                    _, el = items.pop(idx)
            subplot.update_frame(key, ranges, el)

        if isinstance(self.hmap, DynamicMap) and items:
            raise Exception("Some Elements returned by the dynamic callback "
                            "were not initialized correctly and could not be "
                            "rendered.")

        if self.show_legend:
            self._adjust_legend(element, axis)

        self._finalize_axis(key, ranges=ranges)
