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
        Dictionary to specify colors for clipped values, allows
        setting color for NaN values and for values above and below
        the min and max value. The min, max or NaN color may specify
        an RGB(A) color as a color hex string of the form #FFFFFF or
        #FFFFFFFF or a length 3 or length 4 tuple specifying values in
        the range 0-1 or a named HTML color.""")

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
        Define starting angle of the first annulars. By default, beings at 
        12 o clock.""")

    padding_inner = param.Number(default=0.1, bounds=(0, 0.5), doc="""
        Define the radius fraction of inner, empty space.""")

    padding_outer = param.Number(default=0.05, bounds=(0, 1), doc="""
        Define the radius fraction of outer space including the labels.""")

    show_nth_label = param.Number(default=1, doc="""
        Define every nth label to be plotted. By default, every label is
        shown.""")

    separate_nth_segment = param.Number(default=0, doc="""
        Add separation lines between segments for better readability. By
        default, does not show any separation lines.""")

    # Force x and y ranges to be numerical
    _y_range_type = Range1d
    _x_range_type = Range1d

    # Map each glyph to a style group
    _style_groups = {'annular_wedge': 'annular',
                     'text': 'label',
                     'multi_line': 'separator'}

    _draw_order = ["annular_wedge", "multi_line", "text"]

    style_opts = (['separator_' + p for p in line_properties] + \
                  ['annular_' + p for p in fill_properties + line_properties] + \
                  ['label_' + p for p in text_properties] + ['width', 'cmap'])


    @staticmethod
    def _extract_implicit_order(array):
        """Iterate given `array` and extract unique values in
        existing order.

        """

        order = []
        contained = set()

        for element in array:
            if element not in contained:
                order.append(element)
                contained.update([element])

        return order


    @staticmethod
    def _map_order_to_bins(start, end, order, reverse=False):
        """Map elements from given `order` array to bins ranging from `start`
        to `end`.
        """

        size = len(order)
        bounds = np.linspace(start, end, size + 1)
        bins = np.vstack([bounds[:-1], bounds[1:]]).T

        if reverse:
            bins = bins[::-1]

        mapping = dict(zip(order, bins))

        return mapping


    @staticmethod
    def _get_bounds(mapper, values):
        """Extract first and second value from tuples of mapped bins.

        """

        array = np.array([mapper.get(x) for x in values])

        return array[:, 0], array[:, 1]


    @staticmethod
    def _extract_labels(mapper):
        """Extracts text label and radiant for segment texts.

        """

        values = [(text, ((end - start) / 2) + start)
                  for text, (start, end) in mapper.items()]

        text, radiants = zip(*values)

        return text, np.array(radiants)


    @staticmethod
    def _compute_separations(inner, outer, angles):
        """Compute x and y positions for separation lines for given angles.

        """

        y_start = np.sin(angles) * inner + 0.5
        y_end = np.sin(angles) * outer + 0.5

        x_start = np.cos(angles) * inner + 0.5
        x_end = np.cos(angles) * outer + 0.5

        return zip(x_start, x_end), zip(y_start, y_end)


    def _postprocess_hover(self, renderer, source):
        """Limit hover tool to annular wedges only.

        """

        if isinstance(renderer.glyph, AnnularWedge):
            super(RadialHeatMapPlot, self)._postprocess_hover(renderer, source)


    def get_extents(self, view, ranges):
        lower = -self.padding_outer
        upper = 1 + self.padding_outer
        return (lower, lower, upper, upper)


    def get_data(self, element, ranges, style):

        # dimension labels
        dim_labels = element.dimensions(label=True)[:3]
        x, y, z = [dimension_sanitizer(d) for d in dim_labels]

        if self.invert_axes: x, y = y, x
        cmapper = self._get_colormapper(element.vdims[0], element, ranges, style)

        # define default CDS data mappings
        map_annular = dict(x=0.5, y=0.5,
                           inner_radius="inner_radius",
                           outer_radius="outer_radius",
                           start_angle="start_angle",
                           end_angle="end_angle",
                           fill_color={'field': z, 'transform': cmapper})

        map_text = dict(x="x", y="y", text="text",
                        angle="angle", text_align="center")

        map_multi_line = dict(xs="xs", ys="ys")

        mapping = {'annular_wedge_1': map_annular,
                   'text_1': map_text,
                   'multi_line_1': map_multi_line}

        if self.static_source:
            return {}, mapping, style

        # get raw values
        aggregate = element.gridded
        xvals = aggregate.dimension_values(x)
        yvals = aggregate.dimension_values(y)
        zvals = aggregate.dimension_values(2, flat=True)

        # pretty print x and y dimension values if necessary
        def _pprint(dim_label, vals):
            if vals.dtype.kind not in 'SU':
                dim = aggregate.get_dimension(dim_label)
                return [dim.pprint_value(v) for v in vals]

            return vals

        xvals = _pprint(x, xvals)
        yvals = _pprint(y, yvals)


        # get orders
        order_segment = self._extract_implicit_order(xvals)
        order_annular = self._extract_implicit_order(yvals)

        # annular wedges
        radius_max = 0.5

        bins_annular = self._map_order_to_bins(radius_max * self.padding_inner,
                                               radius_max,
                                               order_annular)

        bins_segment = self._map_order_to_bins(self.start_angle,
                                               self.start_angle + 2 * np.pi,
                                               order_segment, True)

        start_angle, end_angle = self._get_bounds(bins_segment, xvals)
        inner_radius, outer_radius = self._get_bounds(bins_annular, yvals)

        data_annular = {"start_angle":start_angle,
                        "end_angle":end_angle,
                        "inner_radius":inner_radius,
                        "outer_radius":outer_radius,
                        z:zvals, x:xvals, y: yvals}

        # text for labels
        text_nth = order_segment[::self.show_nth_label]
        text_mapping = {x: bins_segment[x] for x in text_nth}
        text_labels, text_radiant = self._extract_labels(text_mapping)
        text_y_coord = np.sin(text_radiant) * radius_max * 1.01 + 0.5
        text_x_coord = np.cos(text_radiant) * radius_max * 1.01 + 0.5

        data_text = dict(x=text_x_coord,
                         y=text_y_coord,
                         text=text_labels,
                         angle=1.5 * np.pi + text_radiant)

        # multi_lines for separation
        if self.separate_nth_segment > 1:
            separate_nth = order_segment[::self.separate_nth_segment]
            angles = np.array([bins_segment[x][1] for x in separate_nth])
            xs, ys = self._compute_separations(radius_max * self.padding_inner,
                                               radius_max,
                                               angles)
        else:
            xs, ys = [], []

        data_multi_line = dict(xs=list(xs), ys=list(ys))

        # create data dict
        data = {'annular_wedge_1': data_annular,
                'text_1': data_text,
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
