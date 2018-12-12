from __future__ import absolute_import, division, unicode_literals

import math
import warnings
from types import FunctionType

import param
import numpy as np
import matplotlib.colors as mpl_colors

from matplotlib import ticker
from matplotlib.dates import date2num

from ...core import util
from ...core import (OrderedDict, NdOverlay, DynamicMap, Dataset,
                     CompositeOverlay, Element3D, Element)
from ...core.options import abbreviated_exception
from ...element import Graph, Path
from ...util.transform import dim
from ..plot import GenericElementPlot, GenericOverlayPlot
from ..util import dynamic_update, process_cmap, color_intervals, dim_range_key
from .plot import MPLPlot, mpl_rc_context
from .util import mpl_version, validate, wrap_formatter


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

    invert_zaxis = param.Boolean(default=False, doc="""
        Whether to invert the plot z-axis.""")

    labelled = param.List(default=['x', 'y'], doc="""
        Whether to plot the 'x' and 'y' labels.""")

    logz  = param.Boolean(default=False, doc="""
         Whether to apply log scaling to the y-axis of the Chart.""")

    xformatter = param.ClassSelector(
        default=None, class_=(util.basestring, ticker.Formatter, FunctionType), doc="""
        Formatter for ticks along the x-axis.""")

    yformatter = param.ClassSelector(
        default=None, class_=(util.basestring, ticker.Formatter, FunctionType), doc="""
        Formatter for ticks along the y-axis.""")

    zformatter = param.ClassSelector(
        default=None, class_=(util.basestring, ticker.Formatter, FunctionType), doc="""
        Formatter for ticks along the z-axis.""")

    zaxis = param.Boolean(default=True, doc="""
        Whether to display the z-axis.""")

    zlabel = param.String(default=None, doc="""
        An explicit override of the z-axis label, if set takes precedence
        over the dimension label.""")

    zrotation = param.Integer(default=0, bounds=(0, 360), doc="""
        Rotation angle of the zticks.""")

    zticks = param.Parameter(default=None, doc="""
        Ticks along z-axis specified as an integer, explicit list of
        tick locations, list of tuples containing the locations and
        labels or a matplotlib tick locator object. If set to None
        default matplotlib ticking behavior is applied.""")

    # Element Plots should declare the valid style options for matplotlib call
    style_opts = []

    # Declare which styles cannot be mapped to a non-scalar dimension
    _nonvectorized_styles = ['marker', 'alpha', 'cmap', 'angle', 'visible']

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


    def _finalize_axis(self, key, element=None, title=None, dimensions=None, ranges=None, xticks=None,
                       yticks=None, zticks=None, xlabel=None, ylabel=None, zlabel=None):
        """
        Applies all the axis settings before the axis or figure is returned.
        Only plots with zorder 0 get to apply their settings.

        When the number of the frame is supplied as n, this method looks
        up and computes the appropriate title, axis labels and axis bounds.
        """
        if element is None:
            element = self._get_frame(key)
        self.current_frame = element
        if not dimensions and element and not self.subplots:
            el = element.traverse(lambda x: x, [Element])
            if el:
                el = el[0]
                dimensions = el.nodes.dimensions() if isinstance(el, Graph) else el.dimensions()
        axis = self.handles['axis']

        subplots = list(self.subplots.values()) if self.subplots else []
        if self.zorder == 0 and key is not None:
            if self.bgcolor:
                if mpl_version <= '1.5.9':
                    axis.set_axis_bgcolor(self.bgcolor)
                else:
                    axis.set_facecolor(self.bgcolor)

            # Apply title
            title = self._format_title(key)
            if self.show_title and title is not None:
                fontsize = self._fontsize('title')
                if 'title' in self.handles:
                    self.handles['title'].set_text(title)
                else:
                    self.handles['title'] = axis.set_title(title, **fontsize)

            # Apply subplot label
            self._subplot_label(axis)

            # Apply axis options if axes are enabled
            if element is not None and not any(not sp._has_axes for sp in [self] + subplots):
                # Set axis labels
                if dimensions:
                    self._set_labels(axis, dimensions, xlabel, ylabel, zlabel)
                else:
                    if self.xlabel is not None:
                        axis.set_xlabel(self.xlabel)
                    if self.ylabel is not None:
                        axis.set_ylabel(self.ylabel)
                    if self.zlabel is not None and hasattr(axis, 'set_zlabel'):
                        axis.set_zlabel(self.zlabel)

                if not subplots:
                    legend = axis.get_legend()
                    if legend:
                        legend.set_visible(self.show_legend)
                        self.handles["bbox_extra_artists"] += [legend]
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

                # Set axes limits
                self._set_axis_limits(axis, element, subplots, ranges)

            # Apply aspects
            if self.aspect is not None and self.projection != 'polar' and not self.adjoined:
                self._set_aspect(axis, self.aspect)

        if not subplots and not self.drawn:
            self._finalize_artist(element)

        self._execute_hooks(element)
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
            self._set_axis_formatter(axis.xaxis, xdim, self.xformatter)
        if ydim:
            self._set_axis_formatter(axis.yaxis, ydim, self.yformatter)
        if self.projection == '3d':
            zdim = dimensions[2] if ndims > 2 else None
            if zdim or self.zformatter is not None:
                self._set_axis_formatter(axis.zaxis, zdim, self.zformatter)

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

        axes_str = 'xy'
        axes_list = [axis.xaxis, axis.yaxis]

        if hasattr(axis, 'zaxis'):
            axes_str += 'z'
            axes_list.append(axis.zaxis)

        for ax, ax_obj in zip(axes_str, axes_list):
            tick_fontsize = self._fontsize('%sticks' % ax,'labelsize',common=False)
            if tick_fontsize: ax_obj.set_tick_params(**tick_fontsize)


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


    def _set_axis_formatter(self, axis, dim, formatter):
        """
        Set axis formatter based on dimension formatter.
        """
        if isinstance(dim, list): dim = dim[0]
        if formatter is not None:
            pass
        elif dim.value_format:
            formatter = dim.value_format
        elif dim.type in dim.type_formatters:
            formatter = dim.type_formatters[dim.type]
        if formatter:
            axis.set_major_formatter(wrap_formatter(formatter))


    def get_aspect(self, xspan, yspan):
        """
        Computes the aspect ratio of the plot
        """
        if isinstance(self.aspect, (int, float)):
            return self.aspect
        elif self.aspect == 'square':
            return 1
        elif self.aspect == 'equal':
            return xspan/yspan
        return 1


    def _set_aspect(self, axes, aspect):
        """
        Set the aspect on the axes based on the aspect setting.
        """
        if isinstance(aspect, util.basestring) and aspect != 'square':
            axes.set_aspect(aspect)
            return

        (x0, x1), (y0, y1) = axes.get_xlim(), axes.get_ylim()
        xsize = np.log(x1) - np.log(x0) if self.logx else x1-x0
        ysize = np.log(y1) - np.log(y0) if self.logy else y1-y0
        xsize = max(abs(xsize), 1e-30)
        ysize = max(abs(ysize), 1e-30)
        data_ratio = 1./(ysize/xsize)
        if aspect != 'square':
            data_ratio = data_ratio/aspect
        axes.set_aspect(float(data_ratio))


    def _set_axis_limits(self, axis, view, subplots, ranges):
        """
        Compute extents for current view and apply as axis limits
        """
        # Extents
        scalex, scaley = True, True
        extents = self.get_extents(view, ranges)
        if extents and not self.overlaid:
            coords = [coord if np.isreal(coord) or isinstance(coord, np.datetime64) else np.NaN for coord in extents]
            coords = [date2num(util.dt64_to_dt(c)) if isinstance(c, np.datetime64) else c
                      for c in coords]
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

            if isinstance(l, util.cftime_types) or l != r:
                lims = {}
                if valid_lim(l):
                    lims['left'] = l
                    scalex = False
                if valid_lim(r):
                    lims['right'] = r
                    scalex = False
                if lims:
                    axis.set_xlim(**lims)
            if self.invert_yaxis or any(p.invert_yaxis for p in subplots):
                t, b = b, t
            if isinstance(b, util.cftime_types) or b != t:
                lims = {}
                if valid_lim(b):
                    lims['bottom'] = b
                    scaley = False
                if valid_lim(t):
                    lims['top'] = t
                    scaley = False
                if lims:
                    axis.set_ylim(**lims)
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
        if option in [None, False]:
            axis.set_visible(False)
            for pos in positions:
                axes.spines[pos].set_visible(False)
        else:
            if option is True:
                option = positions[0]
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


    @mpl_rc_context
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
        elif element is not None:
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
        ranges = util.match_spec(element, ranges)

        label = element.label if self.show_legend else ''
        style = dict(label=label, zorder=self.zorder, **self.style[self.cyclic_index])
        axis_kwargs = self.update_handles(key, axis, element, ranges, style)
        self._finalize_axis(key, element=element, ranges=ranges,
                            **(axis_kwargs if axis_kwargs else {}))


    @mpl_rc_context
    def initialize_plot(self, ranges=None):
        element = self.hmap.last
        ax = self.handles['axis']
        key = list(self.hmap.data.keys())[-1]
        dim_map = dict(zip((d.name for d in self.hmap.kdims), key))
        key = tuple(dim_map.get(d.name, None) for d in self.dimensions)
        ranges = self.compute_ranges(self.hmap, key, ranges)
        self.current_ranges = ranges
        self.current_frame = element
        self.current_key = key

        ranges = util.match_spec(element, ranges)

        style = dict(zorder=self.zorder, **self.style[self.cyclic_index])
        if self.show_legend:
            style['label'] = element.label

        plot_data, plot_kwargs, axis_kwargs = self.get_data(element, ranges, style)

        with abbreviated_exception():
            handles = self.init_artists(ax, plot_data, plot_kwargs)
        self.handles.update(handles)

        return self._finalize_axis(self.keys[-1], element=element, ranges=ranges,
                                   **axis_kwargs)


    def init_artists(self, ax, plot_args, plot_kwargs):
        """
        Initializes the artist based on the plot method declared on
        the plot.
        """
        plot_method = self._plot_methods.get('batched' if self.batched else 'single')
        plot_fn = getattr(ax, plot_method)
        artist = plot_fn(*plot_args, **plot_kwargs)
        return {'artist': artist[0] if isinstance(artist, list) and
                len(artist) == 1 else artist}


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


    def _apply_transforms(self, element, ranges, style):
        new_style = dict(style)
        for k, v in style.items():
            if isinstance(v, util.basestring):
                if validate(k, v) == True:
                    continue
                elif v in element or (isinstance(element, Graph) and v in element.nodes):
                    v = dim(v)
                elif any(d==v for d in self.overlay_dims):
                    v = dim([d for d in self.overlay_dims if d==v][0])

            if not isinstance(v, dim):
                continue
            elif (not v.applies(element) and v.dimension not in self.overlay_dims):
                new_style.pop(k)
                self.warning('Specified %s dim transform %r could not be applied, as not all '
                             'dimensions could be resolved.' % (k, v))
                continue

            if len(v.ops) == 0 and v.dimension in self.overlay_dims:
                val = self.overlay_dims[v.dimension]
            elif type(element) is Path:
                val = np.concatenate([v.apply(el, ranges=ranges, flat=True)[:-1]
                                      for el in element.split()])
            else:
                val = v.apply(element, ranges)

            if (not np.isscalar(val) and len(util.unique_array(val)) == 1 and
                (not 'color' in k or validate('color', val))):
                val = val[0]

            if not np.isscalar(val) and k in self._nonvectorized_styles:
                element = type(element).__name__
                raise ValueError('Mapping a dimension to the "{style}" '
                                 'style option is not supported by the '
                                 '{element} element using the {backend} '
                                 'backend. To map the "{dim}" dimension '
                                 'to the {style} use a groupby operation '
                                 'to overlay your data along the dimension.'.format(
                                     style=k, dim=v.dimension, element=element,
                                     backend=self.renderer.backend))

            style_groups = getattr(self, '_style_groups', [])
            groups = [sg for sg in style_groups if k.startswith(sg)]
            group = groups[0] if groups else None
            prefix = '' if group is None else group+'_'
            if (k in (prefix+'c', prefix+'color') and isinstance(val, np.ndarray)
                and not validate('color', val)):
                new_style.pop(k)
                self._norm_kwargs(element, ranges, new_style, v, val, prefix)
                if val.dtype.kind in 'OSUM':
                    range_key = dim_range_key(v)
                    if range_key in ranges and 'factors' in ranges[range_key]:
                        factors = ranges[range_key]['factors']
                    else:
                        factors = util.unique_array(val)
                    val = util.search_indices(val, factors)
                k = prefix+'c'

            new_style[k] = val

        for k, val in list(new_style.items()):
            # If mapped to color/alpha override static fill/line style
            if k == 'c':
                new_style.pop('color', None)

            style_groups = getattr(self, '_style_groups', [])
            groups = [sg for sg in style_groups if k.startswith(sg)]
            group = groups[0] if groups else None
            prefix = '' if group is None else group+'_'
            if k in (prefix+'c', prefix+'color') and isinstance(val, np.ndarray):
                fill_style = new_style.get(prefix+'facecolor')
                if fill_style and validate('color', fill_style):
                    new_style.pop('facecolor')
                line_style = new_style.get(prefix+'edgecolor')
                if line_style and validate('color', line_style):
                    new_style.pop('edgecolor')
            elif k == 'facecolors' and not isinstance(new_style.get('color', new_style.get('c')), np.ndarray):
                # Color overrides facecolors if defined
                new_style.pop('color', None)
                new_style.pop('c', None)

        return new_style


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

    color_levels = param.ClassSelector(default=None, class_=(int, list), doc="""
        Number of discrete colors to use when colormapping or a set of color
        intervals defining the range of values to map each color to.""")

    clipping_colors = param.Dict(default={}, doc="""
        Dictionary to specify colors for clipped values, allows
        setting color for NaN values and for values above and below
        the min and max value. The min, max or NaN color may specify
        an RGB(A) color as a color hex string of the form #FFFFFF or
        #FFFFFFFF or a length 3 or length 4 tuple specifying values in
        the range 0-1 or a named HTML color.""")

    cbar_padding = param.Number(default=0.01, doc="""
        Padding between colorbar and other plots.""")

    cbar_ticks = param.Parameter(default=None, doc="""
        Ticks along colorbar-axis specified as an integer, explicit
        list of tick locations, list of tuples containing the
        locations and labels or a matplotlib tick locator object. If
        set to None default matplotlib ticking behavior is
        applied.""")

    cbar_width = param.Number(default=0.05, doc="""
        Width of the colorbar as a fraction of the main plot""")

    symmetric = param.Boolean(default=False, doc="""
        Whether to make the colormap symmetric around zero.""")

    _colorbars = {}

    _default_nan = '#8b8b8b'

    def __init__(self, *args, **kwargs):
        super(ColorbarPlot, self).__init__(*args, **kwargs)
        self._cbar_extend = 'neither'

    def _adjust_cbar(self, cbar, label, dim):
        noalpha = math.floor(self.style[self.cyclic_index].get('alpha', 1)) == 1
        if (cbar.solids and noalpha):
            cbar.solids.set_edgecolor("face")
        cbar.set_label(label)
        if isinstance(self.cbar_ticks, ticker.Locator):
            cbar.ax.yaxis.set_major_locator(self.cbar_ticks)
        elif self.cbar_ticks == 0:
            cbar.set_ticks([])
        elif isinstance(self.cbar_ticks, int):
            locator = ticker.MaxNLocator(self.cbar_ticks)
            cbar.ax.yaxis.set_major_locator(locator)
        elif isinstance(self.cbar_ticks, list):
            if all(isinstance(t, tuple) for t in self.cbar_ticks):
                ticks, labels = zip(*self.cbar_ticks)
            else:
                ticks, labels = zip(*[(t, dim.pprint_value(t))
                                        for t in self.cbar_ticks])
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(labels)


    def _finalize_artist(self, element):
        artist = self.handles.get('artist', None)
        if artist and self.colorbar:
            color_dim = element.get_dimension(getattr(self, 'color_index', None))
            self._draw_colorbar(color_dim)


    def _draw_colorbar(self, dim=None, redraw=True):
        element = self.hmap.last
        artist = self.handles.get('artist', None)
        fig = self.handles['fig']
        axis = self.handles['axis']
        ax_colorbars, position = ColorbarPlot._colorbars.get(id(axis), ([], None))
        specs = [spec[:2] for _, _, spec, _ in ax_colorbars]
        spec = util.get_spec(element)

        if position is None or not redraw:
            if redraw:
                fig.canvas.draw()
            bbox = axis.get_position()
            l, b, w, h = bbox.x0, bbox.y0, bbox.width, bbox.height
        else:
            l, b, w, h = position

        # Get colorbar label
        dim = element.get_dimension(dim)
        if dim:
            label = dim.pprint_label
        elif element.vdims:
            label = element.vdims[0].pprint_label
        elif dim is None:
            label = ''

        padding = self.cbar_padding
        width = self.cbar_width
        if spec[:2] not in specs:
            offset = len(ax_colorbars)
            scaled_w = w*width
            cax = fig.add_axes([l+w+padding+(scaled_w+padding+w*0.15)*offset,
                                b, scaled_w, h])
            cbar = fig.colorbar(artist, cax=cax, ax=axis, extend=self._cbar_extend)
            self._adjust_cbar(cbar, label, dim)
            self.handles['cax'] = cax
            self.handles['cbar'] = cbar
            ylabel = cax.yaxis.get_label()
            self.handles['bbox_extra_artists'] += [cax, ylabel]
            ax_colorbars.append((artist, cax, spec, label))

        for i, (artist, cax, spec, label) in enumerate(ax_colorbars):
            scaled_w = w*width
            cax.set_position([l+w+padding+(scaled_w+padding+w*0.15)*i,
                              b, scaled_w, h])

        ColorbarPlot._colorbars[id(axis)] = (ax_colorbars, (l, b, w, h))


    def _norm_kwargs(self, element, ranges, opts, vdim, values=None, prefix=''):
        """
        Returns valid color normalization kwargs
        to be passed to matplotlib plot function.
        """
        dim_name = dim_range_key(vdim)
        if values is None:
            if isinstance(vdim, dim):
                values = vdim.apply(element, flat=True)
            else:
                expanded = not (
                    isinstance(element, Dataset) and
                    element.interface.multi and
                    (getattr(element, 'level', None) is not None or
                     element.interface.isscalar(element, vdim.name))
                )
                values = np.asarray(element.dimension_values(vdim, expanded=expanded))

        clim = opts.pop(prefix+'clims', None)
        if clim is None:
            if not len(values):
                clim = (0, 0)
                categorical = False
            elif values.dtype.kind in 'uif':
                if dim_name in ranges:
                    clim = ranges[dim_name]['combined']
                elif isinstance(vdim, dim):
                    if values.dtype.kind == 'M':
                        clim = values.min(), values.max()
                    elif len(values) == 0:
                        clim = np.NaN, np.NaN
                    else:
                        try:
                            with warnings.catch_warnings():
                                warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
                                clim = (np.nanmin(values), np.nanmax(values))
                        except:
                            clim = np.NaN, np.NaN
                else:
                    clim = element.range(vdim)
                if self.logz:
                    # Lower clim must be >0 when logz=True
                    # Choose the maximum between the lowest non-zero value
                    # and the overall range
                    if clim[0] == 0:
                        clim = (values[values!=0].min(), clim[1])
                if self.symmetric:
                    clim = -np.abs(clim).max(), np.abs(clim).max()
                categorical = False
            else:
                range_key = dim_range_key(vdim)
                if range_key in ranges and 'factors' in ranges[range_key]:
                    factors = ranges[range_key]['factors']
                else:
                    factors = util.unique_array(values)
                clim = (0, len(factors)-1)
                categorical = True
        else:
            categorical = values.dtype.kind not in 'uif'

        if self.logz:
            if self.symmetric:
                norm = mpl_colors.SymLogNorm(vmin=clim[0], vmax=clim[1],
                                             linthresh=clim[1]/np.e)
            else:
                norm = mpl_colors.LogNorm(vmin=clim[0], vmax=clim[1])
            opts[prefix+'norm'] = norm
        opts[prefix+'vmin'] = clim[0]
        opts[prefix+'vmax'] = clim[1]

        cmap = opts.get(prefix+'cmap', opts.get('cmap', 'viridis'))
        if values.dtype.kind not in 'OSUM':
            ncolors = None
            if isinstance(self.color_levels, int):
                ncolors = self.color_levels
            elif isinstance(self.color_levels, list):
                ncolors = len(self.color_levels) - 1
                if isinstance(cmap, list) and len(cmap) != ncolors:
                    raise ValueError('The number of colors in the colormap '
                                     'must match the intervals defined in the '
                                     'color_levels, expected %d colors found %d.'
                                     % (ncolors, len(cmap)))
            try:
                el_min, el_max = np.nanmin(values), np.nanmax(values)
            except ValueError:
                el_min, el_max = -np.inf, np.inf
        else:
            ncolors = clim[-1]+1
            el_min, el_max = -np.inf, np.inf
        vmin = -np.inf if opts[prefix+'vmin'] is None else opts[prefix+'vmin']
        vmax = np.inf if opts[prefix+'vmax'] is None else opts[prefix+'vmax']
        if el_min < vmin and el_max > vmax:
            self._cbar_extend = 'both'
        elif el_min < vmin:
            self._cbar_extend = 'min'
        elif el_max > vmax:
            self._cbar_extend = 'max'

        # Define special out-of-range colors on colormap
        colors = {}
        for k, val in self.clipping_colors.items():
            if val == 'transparent':
                colors[k] = {'color': 'w', 'alpha': 0}
            elif isinstance(val, tuple):
                colors[k] = {'color': val[:3],
                             'alpha': val[3] if len(val) > 3 else 1}
            elif isinstance(val, util.basestring):
                color = val
                alpha = 1
                if color.startswith('#') and len(color) == 9:
                    alpha = int(color[-2:], 16)/255.
                    color = color[:-2]
                colors[k] = {'color': color, 'alpha': alpha}

        if not isinstance(cmap, mpl_colors.Colormap):
            if isinstance(cmap, dict):
                factors = util.unique_array(values)
                palette = [cmap.get(f, colors.get('NaN', {'color': self._default_nan})['color'])
                           for f in factors]
            else:
                palette = process_cmap(cmap, ncolors, categorical=categorical)
                if isinstance(self.color_levels, list):
                    palette, (vmin, vmax) = color_intervals(palette, self.color_levels, clip=(vmin, vmax))
            cmap = mpl_colors.ListedColormap(palette)
        if 'max' in colors: cmap.set_over(**colors['max'])
        if 'min' in colors: cmap.set_under(**colors['min'])
        if 'NaN' in colors: cmap.set_bad(**colors['NaN'])
        opts[prefix+'cmap'] = cmap



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

    _propagate_options = ['aspect', 'fig_size', 'xaxis', 'yaxis', 'zaxis',
                          'labelled', 'bgcolor', 'fontsize', 'invert_axes',
                          'show_frame', 'show_grid', 'logx', 'logy', 'logz',
                          'xticks', 'yticks', 'zticks', 'xrotation', 'yrotation',
                          'zrotation', 'invert_xaxis', 'invert_yaxis',
                          'invert_zaxis', 'title', 'title_format', 'padding',
                          'xlabel', 'ylabel', 'zlabel', 'xlim', 'ylim', 'zlim',
                          'xformatter', 'yformatter']

    def __init__(self, overlay, ranges=None, **params):
        if 'projection' not in params:
            params['projection'] = self._get_projection(overlay)
        super(OverlayPlot, self).__init__(overlay, ranges=ranges, **params)


    def _finalize_artist(self, element):
        for subplot in self.subplots.values():
            subplot._finalize_artist(element)

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
            handle = subplot.traverse(lambda p: p.handles['artist'],
                                      [lambda p: 'artist' in p.handles])
            if isinstance(overlay, NdOverlay):
                key = (dim.pprint_value(k) for k, dim in zip(key, dimensions))
                label = ','.join([str(k) + dim.unit if dim.unit else str(k) for dim, k in
                                  zip(dimensions, key)])
                if handle:
                    legend_data.append((handle, label))
            else:
                if isinstance(subplot, OverlayPlot):
                    legend_data += subplot.handles.get('legend_data', {}).items()
                elif element.label and handle:
                    legend_data.append((handle, element.label))
        all_handles, all_labels = list(zip(*legend_data)) if legend_data else ([], [])
        data = OrderedDict()
        used_labels = []
        for handle, label in zip(all_handles, all_labels):
            # Ensure that artists with multiple handles are supported
            if isinstance(handle, list): handle = tuple(handle)
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
            leg = axis.legend(list(data.keys()), list(data.values()),
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


    @mpl_rc_context
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

        if self.show_legend and element is not None:
            self._adjust_legend(element, axis)

        return self._finalize_axis(key, element=element, ranges=ranges,
                                   title=self._format_title(key))


    @mpl_rc_context
    def update_frame(self, key, ranges=None, element=None):
        axis = self.handles['axis']
        reused = isinstance(self.hmap, DynamicMap) and self.overlaid
        if element is None and not reused:
            element = self._get_frame(key)
        elif element is not None:
            self.current_frame = element
            self.current_key = key
        empty = element is None

        if isinstance(self.hmap, DynamicMap):
            range_obj = element
        else:
            range_obj = self.hmap
        items = [] if element is None else list(element.data.items())

        if not empty:
            ranges = self.compute_ranges(range_obj, key, ranges)
        for k, subplot in self.subplots.items():
            el = None if empty else element.get(k, None)
            if isinstance(self.hmap, DynamicMap) and not empty:
                idx, spec, exact = dynamic_update(self, subplot, k, element, items)
                if idx is not None:
                    _, el = items.pop(idx)
                    if not exact:
                        self._update_subplot(subplot, spec)
            subplot.update_frame(key, ranges, el)

        if isinstance(self.hmap, DynamicMap) and items:
            self._create_dynamic_subplots(key, items, ranges)

        # Update plot options
        plot_opts = self.lookup_options(element, 'plot').options
        inherited = self._traverse_options(element, 'plot',
                                           self._propagate_options,
                                           defaults=False)
        plot_opts.update(**{k: v[0] for k, v in inherited.items()
                            if k not in plot_opts})
        self.set_param(**plot_opts)

        if self.show_legend and not empty:
            self._adjust_legend(element, axis)

        self._finalize_axis(key, element=element, ranges=ranges)
