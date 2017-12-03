import numpy as np
import param

from bokeh.models import HoverTool, Range1d
from bokeh.models.glyphs import AnnularWedge
from ...core import Element
from ...core.util import cartesian_product, is_nan, dimension_sanitizer
from ...element import Raster
from .element import (ElementPlot, ColorbarPlot, line_properties,
                      fill_properties, text_properties)
from .util import mpl_to_bokeh, colormesh


class RasterPlot(ColorbarPlot):

    show_legend = param.Boolean(default=False, doc="""
        Whether to show legend for the plot.""")

    style_opts = ['cmap']
    _plot_methods = dict(single='image')

    def __init__(self, *args, **kwargs):
        super(RasterPlot, self).__init__(*args, **kwargs)
        if self.hmap.type == Raster:
            self.invert_yaxis = not self.invert_yaxis

    def get_data(self, element, ranges, style):
        mapping = dict(image='image', x='x', y='y', dw='dw', dh='dh')
        val_dim = [d for d in element.vdims][0]
        style['color_mapper'] = self._get_colormapper(val_dim, element, ranges, style)

        if self.static_source:
            return {}, mapping, style

        img = element.dimension_values(2, flat=False)
        if img.dtype.kind == 'b':
            img = img.astype(np.int8)

        if type(element) is Raster:
            l, b, r, t = element.extents
            if self.invert_axes:
                l, b, r, t = b, l, t, r
            else:
                img = img.T
        else:
            l, b, r, t = element.bounds.lbrt()
            if self.invert_axes:
                img = img.T
                l, b, r, t = b, l, t, r

        if self.invert_xaxis:
            l, r = r, l
            img = img[:, ::-1]
        if self.invert_yaxis:
            img = img[::-1]
            b, t = t, b
        dh, dw = t-b, r-l

        data = dict(image=[img], x=[l], y=[b], dw=[dw], dh=[dh])
        return (data, mapping, style)



class RGBPlot(RasterPlot):

    style_opts = []
    _plot_methods = dict(single='image_rgba')

    def get_data(self, element, ranges, style):
        mapping = dict(image='image', x='x', y='y', dw='dw', dh='dh')
        if self.static_source:
            return {}, mapping, style

        img = np.dstack([element.dimension_values(d, flat=False)
                         for d in element.vdims])
        if img.ndim == 3:
            if img.shape[2] == 3: # alpha channel not included
                alpha = np.ones(img.shape[:2])
                if img.dtype.name == 'uint8':
                    alpha = (alpha*255).astype('uint8')
                img = np.dstack([img, alpha])
            if img.dtype.name != 'uint8':
                img = (img*255).astype(np.uint8)
            N, M, _ = img.shape
            #convert image NxM dtype=uint32
            img = img.view(dtype=np.uint32).reshape((N, M))

        # Ensure axis inversions are handled correctly
        l, b, r, t = element.bounds.lbrt()
        if self.invert_axes:
            img = img.T
            l, b, r, t = b, l, t, r
        if self.invert_xaxis:
            l, r = r, l
            img = img[:, ::-1]
        if self.invert_yaxis:
            img = img[::-1]
            b, t = t, b
        dh, dw = t-b, r-l

        data = dict(image=[img], x=[l], y=[b], dw=[dw], dh=[dh])
        return (data, mapping, style)

    def _glyph_properties(self, plot, element, source, ranges, style):
        return ElementPlot._glyph_properties(self, plot, element,
                                             source, ranges, style)

class HSVPlot(RGBPlot):

    def get_data(self, element, ranges, style):
        return super(HSVPlot, self).get_data(element.rgb, ranges, style)


class HeatMapPlot(ColorbarPlot):

    clipping_colors = param.Dict(default={'NaN': 'white'}, doc="""
        Dictionary to specify colors for clipped values. 
        Allows setting color for NaN values and for values above and below
        the min and max value. The min, max, or NaN color may specify
        an RGB(A) color as a either (1) a color hex string of the form 
        #FFFFFF or #FFFFFFFF, (2) a length-3 or length-4 tuple specifying
        values in the range 0-1, or (3) a named HTML color.""")

    show_legend = param.Boolean(default=False, doc="""
        Whether to show legend for the plot.""")

    _plot_methods = dict(single='rect')
    style_opts = ['cmap', 'color'] + line_properties + fill_properties

    _categorical = True

    def _get_factors(self, element):
        return super(HeatMapPlot, self)._get_factors(element.gridded)

    def get_data(self, element, ranges, style):
        x, y, z = [dimension_sanitizer(d) for d in element.dimensions(label=True)[:3]]
        if self.invert_axes: x, y = y, x
        cmapper = self._get_colormapper(element.vdims[0], element, ranges, style)
        if self.static_source:
            return {}, {'x': x, 'y': y, 'fill_color': {'field': 'zvalues', 'transform': cmapper}}, style

        aggregate = element.gridded
        xdim, ydim = aggregate.dimensions()[:2]
        xvals, yvals = (aggregate.dimension_values(x),
                        aggregate.dimension_values(y))
        zvals = aggregate.dimension_values(2, flat=False)
        if self.invert_axes:
            xdim, ydim = ydim, xdim
            zvals = zvals.T.flatten()
        else:
            zvals = zvals.T.flatten()
        if xvals.dtype.kind not in 'SU':
            xvals = [xdim.pprint_value(xv) for xv in xvals]
        if yvals.dtype.kind not in 'SU':
            yvals = [ydim.pprint_value(yv) for yv in yvals]
        data = {x: xvals, y: yvals, 'zvalues': zvals}

        if any(isinstance(t, HoverTool) for t in self.state.tools) and not self.static_source:
            for vdim in element.vdims:
                sanitized = dimension_sanitizer(vdim.name)
                data[sanitized] = ['-' if is_nan(v) else vdim.pprint_value(v)
                                   for v in aggregate.dimension_values(vdim)]
        return (data, {'x': x, 'y': y, 'fill_color': {'field': 'zvalues', 'transform': cmapper},
                       'height': 1, 'width': 1}, style)


class RadialHeatMapPlot(CompositeElementPlot, ColorbarPlot):


    start_angle = param.Number(default=np.pi/2, doc="""
        Define starting angle of the first annulus segment. By default, begins at 
        12 o'clock.""")

    padding_inner = param.Number(default=0.1, bounds=(0, 0.5), doc="""
        Define the radius fraction of inner, empty space.""")

    padding_outer = param.Number(default=0.05, bounds=(0, 1), doc="""
        Define the radius fraction of outer space including the labels.""")

    separate_nth_segment = param.Number(default=0, doc="""
        Add separation lines between segments for better readability. By
        default, does not show any separation lines.""")

    max_radius = param.Number(default=0.5, doc="""
        Define the maximum radius which is used for the x and y range extents.
        """)

    xticks = param.Parameter(default=4, doc="""
        Ticks along x-axis specified as an integer, explicit list of
        ticks or function. If `None`, no ticks are shown.""")

    yticks = param.Parameter(default=4, doc="""
        Ticks along y-axis specified as an integer, explicit list of
        ticks or function. If `None`, no ticks are shown.""")

    # Force x and y ranges to be numerical
    _y_range_type = Range1d
    _x_range_type = Range1d

    # Map each glyph to a style group
    _style_groups = {'annular_wedge': 'annular',
                     'text': 'ticks',
                     'multi_line': 'separator'}

    _draw_order = ["annular_wedge", "multi_line", "text"]

    style_opts = (['separator_' + p for p in line_properties] + \
                  ['annular_' + p for p in fill_properties + line_properties] + \
                  ['ticks_' + p for p in text_properties] + ['width', 'cmap'])


    def _get_bins(self, kind, order, reverse=False):
        """
        Map elements from given `order` array to bins of start and end values
        for radius or angle dimension.
        """

        if kind == "radius":
            start = self.max_radius * self.padding_inner
            end = self.max_radius

        elif kind == "angle":
            start = self.start_angle
            end = self.start_angle + 2 * np.pi

        bounds = np.linspace(start, end, len(order) + 1)
        bins = np.vstack([bounds[:-1], bounds[1:]]).T

        if reverse:
            bins = bins[::-1]

        return dict(zip(order, bins))


    @staticmethod
    def _get_bounds(mapper, values):
        """
        Extract first and second value from tuples of mapped bins.
        """

        array = np.array([mapper.get(x) for x in values])

        return array[:, 0], array[:, 1]


    @staticmethod
    def _extract_labels(mapper):
        """
        Extracts text label and radiant for segment texts.
        """

        values = [(text, ((end - start) / 2) + start)
                  for text, (start, end) in mapper.items()]

        text, radiants = zip(*values)

        return text, np.array(radiants)

    def _postprocess_hover(self, renderer, source):
        """
        Limit hover tool to annular wedges only.
        """

        if isinstance(renderer.glyph, AnnularWedge):
            super(RadialHeatMapPlot, self)._postprocess_hover(renderer, source)


    def get_extents(self, view, ranges):
        """Supply custom, static extents because radial heatmaps always have
        the same boundaries.

        """

        lower = -self.padding_outer
        upper = 1 + self.padding_outer
        return (lower, lower, upper, upper)


    def _axis_properties(self, *args, **kwargs):
        """Overwrite default axis properties handling due to clashing
        categorical input and numerical output axes.

        Axis properties are handled separately for radial heatmaps because of
        missing radial axes in bokeh.

        """

        return {}


    def get_default_mapping(self, z, cmapper):
        """Create dictionary containing default ColumnDataSource glyph to data
        mappings.

        """

        map_annular = dict(x=0.5, y=0.5,
                           inner_radius="inner_radius",
                           outer_radius="outer_radius",
                           start_angle="start_angle",
                           end_angle="end_angle",
                           fill_color={'field': z, 'transform': cmapper})

        map_seg_label = dict(x="x", y="y", text="text",
                             angle="angle", text_align="center")

        map_ann_label = dict(x="x", y="y", text="text",
                             angle="angle", text_align="center",
                             text_baseline="bottom")

        map_multi_line = dict(xs="xs", ys="ys")

        return {'annular_wedge_1': map_annular,
                'text_1': map_seg_label,
                'text_2': map_ann_label,
                'multi_line_1': map_multi_line}


    def _pprint(self, element, dim_label, vals):
        """Helper function to convert values to corresponding dimension type.

        """

        if vals.dtype.kind not in 'SU':
            dim = element.gridded.get_dimension(dim_label)
            return [dim.pprint_value(v) for v in vals]

        return vals


    def _compute_tick_mapping(self, kind, order_default, bin_default):
        """Helper function to compute tick mappings based on `ticks` and
        default orders and bins.

        """

        if kind == "angle":
            ticks = self.xticks
            reverse = True
        elif kind == "radius":
            ticks = self.yticks
            reverse = False

        nth_label = 1
        order = order_default
        bins = bin_default

        if isinstance(ticks, (tuple, list)):
            order = ticks
            bins = self._get_bins(kind, ticks, reverse)
        elif self.xticks:
            nth_label = np.ceil(len(order_default) / ticks).astype(int)

        text_nth = order[::nth_label]
        return {x: bins[x] for x in text_nth}


    def _get_seg_labels_data(self, order_seg, bins_seg):
        """Generate ColumnDataSource dictionary for segment labels.

        """

        if self.xticks is None:
            return dict(x=[], y=[], text=[], angle=[])

        mapping = self._compute_tick_mapping("angle", order_seg, bins_seg)

        values = [(text, ((end - start) / 2) + start)
                  for text, (start, end) in mapping.items()]

        labels, radiant = zip(*values)
        radiant = np.array(radiant)

        y_coord = np.sin(radiant) * self.max_radius * 1.01 + self.max_radius
        x_coord = np.cos(radiant) * self.max_radius * 1.01 + self.max_radius

        return dict(x=x_coord,
                    y=y_coord,
                    text=labels,
                    angle=1.5 * np.pi + radiant)

    def _get_ann_labels_data(self, order_ann, bins_ann):
        """Generate ColumnDataSource dictionary for annular labels.

        """

        if self.yticks is None:
            return dict(x=[], y=[], text=[], angle=[])

        mapping = self._compute_tick_mapping("radius", order_ann, bins_ann)
        values = [(label, radius[0]) for label, radius in mapping.items()]

        labels, radius = zip(*values)
        radius = np.array(radius)

        y_coord = np.sin(np.pi/2) * radius + 0.5
        x_coord = np.cos(np.pi/2) * radius + 0.5

        return dict(x=x_coord,
                    y=y_coord,
                    text=labels,
                    angle=[0]*len(labels))

    def _get_multiline_data(self, order_seg, bins_seg):
        """Generate ColumnDataSource dictionary for segment separation lines.

        """

        if self.separate_nth_segment > 1:
            separate_nth = order_seg[::self.separate_nth_segment]
            angles = np.array([bins_seg[x][1] for x in separate_nth])

            inner = self.max_radius * self.padding_inner
            outer = self.max_radius

            y_start = np.sin(angles) * inner + self.max_radius
            y_end = np.sin(angles) * outer + self.max_radius

            x_start = np.cos(angles) * inner + self.max_radius
            x_end = np.cos(angles) * outer + self.max_radius

            xs = zip(x_start, x_end)
            ys = zip(y_start, y_end)

        else:
            xs, ys = [], []

        return dict(xs=list(xs), ys=list(ys))


    def get_data(self, element, ranges, style):

        # dimension labels
        dim_labels = element.dimensions(label=True)[:3]
        x, y, z = [dimension_sanitizer(d) for d in dim_labels]
        if self.invert_axes: x, y = y, x

        # color mapper
        cmapper = self._get_colormapper(element.vdims[0], element, ranges, style)

        # default CDS data mapping
        mapping = self.get_default_mapping(z, cmapper)
        if self.static_source:
            return {}, mapping, style

        # get raw values
        aggregate = element.gridded
        xvals = aggregate.dimension_values(x)
        yvals = aggregate.dimension_values(y)
        zvals = aggregate.dimension_values(2, flat=True)

        # get orders
        order_seg = aggregate.dimension_values(x, expanded=False)
        order_ann = aggregate.dimension_values(y, expanded=False)

        # pretty print if necessary
        xvals = self._pprint(element, x, xvals)
        yvals = self._pprint(element, y, yvals)
        order_seg = self._pprint(element, x, order_seg)
        order_ann = self._pprint(element, y, order_ann)

        # annular wedges
        bins_ann = self._get_bins("radius", order_ann)
        inner_radius, outer_radius = self._get_bounds(bins_ann, yvals)

        bins_seg = self._get_bins("angle", order_seg, True)
        start_angle, end_angle = self._get_bounds(bins_seg, xvals)

        # create ColumnDataSources
        data_annular = {"start_angle": start_angle,
                        "end_angle": end_angle,
                        "inner_radius": inner_radius,
                        "outer_radius": outer_radius,
                        z: zvals, x: xvals, y: yvals}

        if any(isinstance(t, HoverTool) for t in self.state.tools):
            for vdim in element.vdims:
                sanitized = dimension_sanitizer(vdim.name)
                values = ['-' if is_nan(v) else vdim.pprint_value(v)
                          for v in aggregate.dimension_values(vdim)]
                data_annular[sanitized] = values

        data_text_seg_labels = self._get_seg_labels_data(order_seg, bins_seg)
        data_text_ann_labels = self._get_ann_labels_data(order_ann, bins_ann)

        data_multi_line = self._get_multiline_data(order_seg, bins_seg)

        data = {'annular_wedge_1': data_annular,
                'text_1': data_text_seg_labels,
                'text_2': data_text_ann_labels,
                'multi_line_1': data_multi_line}


        return data, mapping, style


class QuadMeshPlot(ColorbarPlot):

    show_legend = param.Boolean(default=False, doc="""
        Whether to show legend for the plot.""")

    _plot_methods = dict(single='quad')
    style_opts = ['cmap', 'color'] + line_properties + fill_properties

    def get_data(self, element, ranges, style):
        x, y, z = element.dimensions()[:3]
        if self.invert_axes: x, y = y, x
        cmapper = self._get_colormapper(z, element, ranges, style)
        cmapper = {'field': z.name, 'transform': cmapper}

        irregular = element.interface.irregular(element, x)
        if irregular:
            mapping = dict(xs='xs', ys='ys', fill_color=cmapper)
        else:
            mapping = {'left': 'left', 'right': 'right',
                       'fill_color': cmapper,
                       'top': 'top', 'bottom': 'bottom'}

        if self.static_source:
            return {}, mapping, style

        zdata = element.dimension_values(z, flat=False)
        if irregular:
            dims = element.kdims
            if self.invert_axes: dims = dims[::-1]
            X, Y = [element.interface.coords(element, d, expanded=True, edges=True)
                    for d in dims]
            X, Y = colormesh(X, Y)
            zvals = zdata.T.flatten() if self.invert_axes else zdata.flatten()
            XS, YS = [], []
            for x, y, zval in zip(X, Y, zvals):
                if np.isfinite(zval):
                    XS.append(list(x[:-1]))
                    YS.append(list(y[:-1]))
            data = {'xs': XS, 'ys': YS, z.name: zvals[np.isfinite(zvals)]}
        else:
            xc, yc = (element.interface.coords(element, x, edges=True),
                      element.interface.coords(element, y, edges=True))
            x0, y0 = cartesian_product([xc[:-1], yc[:-1]], copy=True)
            x1, y1 = cartesian_product([xc[1:], yc[1:]], copy=True)
            zvals = zdata.flatten() if self.invert_axes else zdata.T.flatten()
            data = {'left': x0, 'right': x1, dimension_sanitizer(z.name): zvals,
                    'bottom': y0, 'top': y1}
            if any(isinstance(t, HoverTool) for t in self.state.tools) and not self.static_source:
                data[dimension_sanitizer(x.name)] = element.dimension_values(x)
                data[dimension_sanitizer(y.name)] = element.dimension_values(y)
        return data, mapping, style


    def _init_glyph(self, plot, mapping, properties):
        """
        Returns a Bokeh glyph object.
        """
        properties = mpl_to_bokeh(properties)
        properties = dict(properties, **mapping)
        if 'xs' in mapping:
            renderer = plot.patches(**properties)
        else:
            renderer = plot.quad(**properties)
        if self.colorbar and 'color_mapper' in self.handles:
            self._draw_colorbar(plot, self.handles['color_mapper'])
        return renderer, renderer.glyph
