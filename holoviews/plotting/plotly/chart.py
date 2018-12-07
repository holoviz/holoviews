from __future__ import absolute_import, division, unicode_literals

import param
import numpy as np

from ...core.data import Dataset
from ...core import util
from ...element import Bars
from ...operation import interpolate_curve
from ..util import get_axis_padding
from .element import ElementPlot, ColorbarPlot


class ChartPlot(ElementPlot):

    trace_kwargs = {'type': 'scatter'}

    def get_data(self, element, ranges, style):
        return [dict(x=element.dimension_values(0),
                     y=element.dimension_values(1))]


class ScatterPlot(ChartPlot, ColorbarPlot):

    color_index = param.ClassSelector(default=None, class_=(util.basestring, int),
                                      allow_None=True, doc="""
      Index of the dimension from which the color will the drawn""")

    style_opts = ['symbol', 'color', 'cmap', 'alpha', 'size', 'sizemin']

    _nonvectorized_styles = ['cmap', 'opacity', 'sizemin']

    trace_kwargs = {'type': 'scatter', 'mode': 'markers'}

    _style_key = 'marker'

    def graph_options(self, element, ranges, style):
        opts = super(ScatterPlot, self).graph_options(element, ranges, style)
        cdim = element.get_dimension(self.color_index)
        if cdim:
            copts = self.get_color_opts(cdim, element, ranges, style)
            copts['color'] = element.dimension_values(cdim)
            opts['marker'].update(opts)
        return opts


class CurvePlot(ChartPlot):

    interpolation = param.ObjectSelector(objects=['linear', 'steps-mid',
                                                  'steps-pre', 'steps-post'],
                                         default='linear', doc="""
        Defines how the samples of the Curve are interpolated,
        default is 'linear', other options include 'steps-mid',
        'steps-pre' and 'steps-post'.""")

    trace_kwargs = {'type': 'scatter', 'mode': 'lines'}

    style_opts = ['color', 'dash', 'line_width']

    _nonvectorized_styles = style_opts

    _style_key = 'line'

    def get_data(self, element, ranges, style):
        if 'steps' in self.interpolation:
            element = interpolate_curve(element, interpolation=self.interpolation)
        return [dict(x=element.dimension_values(0),
                     y=element.dimension_values(1))]


class AreaPlot(ChartPlot):

    style_opts = ['color', 'dash', 'line_width']

    trace_kwargs = {'type': 'scatter', 'mode': 'lines'}

    _style_key = 'line'

    def get_extents(self, element, ranges, range_type='combined'):
        vdims = element.vdims[:2]
        vdim = vdims[0].name
        if len(vdims) > 1:
            new_range = {}
            for r in ranges[vdim]:
                new_range[r] = util.max_range([ranges[vd.name][r] for vd in vdims])
            ranges[vdim] = new_range
        else:
            s0, s1 = ranges[vdim]['soft']
            s0 = min(s0, 0) if util.isfinite(s0) else 0
            s1 = max(s1, 0) if util.isfinite(s1) else 0
            ranges[vdim]['soft'] = (s0, s1)
        return super(AreaPlot, self).get_extents(element, ranges, range_type)

    def get_data(self, element, ranges, style):
        if len(element.vdims) == 1:
            kwargs = super(AreaPlot, self).get_data(element, ranges, style)[0]
            kwargs['fill'] = 'tozeroy'
            return [kwargs]
        xs = element.dimension_values(0)
        ys = element.dimension_values(1)
        bottom = element.dimension_values(2)
        return [dict(x=xs, y=bottom, fill=None),
                dict(x=xs, y=ys, fill='tonexty')]


class SpreadPlot(ChartPlot):

    style_opts = ['color', 'dash', 'line_width']

    trace_kwargs = {'type': 'scatter', 'mode': 'lines'}

    _style_key = 'line'

    def get_data(self, element, ranges, style):
        xs = element.dimension_values(0)
        mean = element.dimension_values(1)
        neg_error = element.dimension_values(2)
        pos_idx = 3 if len(element.dimensions()) > 3 else 2
        pos_error = element.dimension_values(pos_idx)
        lower = mean - neg_error
        upper = mean + pos_error
        return [dict(x=xs, y=lower, fill=None),
                dict(x=xs, y=upper, fill='tonexty')]
    

class ErrorBarsPlot(ChartPlot, ColorbarPlot):

    trace_kwargs = {'type': 'scatter', 'mode': 'lines', 'line': {'width': 0}}

    style_opts = ['color', 'dash', 'line_width', 'thickness']

    _nonvectorized_styles = style_opts

    _style_key = 'error_y'

    def get_data(self, element, ranges, style):
        neg_error = element.dimension_values(2)
        pos_idx = 3 if len(element.dimensions()) > 3 else 2
        pos_error = element.dimension_values(pos_idx)
        error_y = dict(type='data', array=pos_error, arrayminus=neg_error)
        return [dict(x=element.dimension_values(0),
                     y=element.dimension_values(1),
                     error_y=error_y)]



class BarPlot(ElementPlot):

    stacked = param.Boolean(default=False, doc="""
       Whether the bars should be stacked or grouped.""")

    # Deprecated parameters

    group_index = param.Integer(default=1, doc="""
       Index of the dimension in the supplied Bars
       Element, which will be laid out into groups.""")

    category_index = param.Integer(default=None, doc="""
       Index of the dimension in the supplied Bars
       Element, which will be laid out into categories.""")

    stack_index = param.Integer(default=None, doc="""
       Index of the dimension in the supplied Bars
       Element, which will stacked.""")

    stacked = param.Boolean(default=False)

    trace_kwargs = {'type': 'bar'}

    def get_extents(self, element, ranges, range_type='combined'):
        """
        Make adjustments to plot extents by computing
        stacked bar heights, adjusting the bar baseline
        and forcing the x-axis to be categorical.
        """
        if self.batched:
            overlay = self.current_frame
            element = Bars(overlay.table(), kdims=element.kdims+overlay.kdims,
                           vdims=element.vdims)
            for kd in overlay.kdims:
                ranges[kd.name]['combined'] = overlay.range(kd)

        extents = super(BarPlot, self).get_extents(element, ranges, range_type)
        xdim = element.kdims[0]
        ydim = element.vdims[0]

        # Compute stack heights
        if self.stacked or self.stack_index:
            ds = Dataset(element)
            pos_range = ds.select(**{ydim.name: (0, None)}).aggregate(xdim, function=np.sum).range(ydim)
            neg_range = ds.select(**{ydim.name: (None, 0)}).aggregate(xdim, function=np.sum).range(ydim)
            y0, y1 = util.max_range([pos_range, neg_range])
        else:
            y0, y1 = ranges[ydim.name]['combined']

        padding = 0 if self.overlaid else self.padding
        _, ypad, _ = get_axis_padding(padding)
        y0, y1 = util.range_pad(y0, y1, ypad, self.logy)

        # Set y-baseline
        if y0 < 0:
            y1 = max([y1, 0])
        elif self.logy:
            y0 = (ydim.range[0] or (10**(np.log10(y1)-2)) if y1 else 0.01)
        else:
            y0 = 0

        # Ensure x-axis is picked up as categorical
        nx = len(element.dimension_values(0, False))
        return (-0.5, y0, nx-0.5, y1)

    def _get_factors(self, element):
        """
        Get factors for categorical axes.
        """
        gdim = None
        sdim = None
        if element.ndims == 1:
            pass
        elif not (self.stacked or self.stack_index):
            gdim = element.get_dimension(1)
        else:
            sdim = element.get_dimension(1)

        xdim, ydim = element.dimensions()[:2]
        xvals = element.dimension_values(0, False)
        xvals = [x if xvals.dtype.kind in 'SU' else xdim.pprint_value(x)
                 for x in xvals]
        if gdim and not sdim:
            gvals = element.dimension_values(gdim, False)
            xvals = sorted([(x, g) for x in xvals for g in gvals])
            is_str = gvals.dtype.kind in 'SU'
            xvals = [(x, g if is_str else gdim.pprint_value(g)) for (x, g) in xvals]
        coords = xvals, []
        if self.invert_axes: coords = coords[::-1]
        return coords


    def _get_axis_dims(self, element):
        if element.ndims > 1 and not (self.stacked or self.stack_index):
            xdims = element.kdims
        else:
            xdims = element.kdims[0]
        return (xdims, element.vdims[0])


    def generate_plot(self, key, ranges):
        if self.stack_index is not None:
            self.warning('Bars stack_index plot option is deprecated '
                         'and will be ignored, set stacked=True/False '
                         'instead.')
        if self.category_index is not None:
            self.warning('Bars category_index plot option is deprecated '
                         'and will be ignored, set stacked=True/False '
                         'instead.')
        if self.group_index not in (None, 1):
            self.warning('Bars group_index plot option is deprecated '
                         'and will be ignored, set stacked=True/False '
                         'instead.')

        element = self._get_frame(key)
        ranges = self.compute_ranges(self.hmap, key, ranges)
        ranges = util.match_spec(element, ranges)

        # Get x, y, group, stack and color dimensions
        xdim = element.kdims[0]
        vdim = element.vdims[0]
        group_dim, stack_dim = None, None
        if element.ndims == 1:
            grouping = None
        elif self.stacked or self.stack_index:
            grouping = 'stacked'
            stack_dim = element.get_dimension(1)
        else:
            grouping = 'grouped'
            group_dim = element.get_dimension(1)

        layout = self.init_layout(key, element, ranges)
        if element.ndims == 1:
            bars = [dict(
                type='bar',
                x=[xdim.pprint_value(v) for v in element.dimension_values(xdim)],
                y=element.dimension_values(vdim))]
        else:
            group_dim = group_dim or stack_dim
            els = element.groupby(group_dim)
            bars = []
            for k, el in els.items():
                bars.append(dict(
                    type='bar',
                    x=[xdim.pprint_value(v) for v in el.dimension_values(xdim)],
                    y=el.dimension_values(vdim), name=k))
            layout['barmode'] = 'stack' if stack_dim else 'group'

        self.handles['layout'] = layout
        fig = dict(data=bars, layout=layout)
        self.handles['fig'] = fig
        return fig
