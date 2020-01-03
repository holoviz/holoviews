from __future__ import absolute_import, division, unicode_literals

import param
import numpy as np

from .selection import PlotlyOverlaySelectionDisplay
from ...core.data import Dataset
from ...core import util
from ...element import Bars
from ...operation import interpolate_curve
from ..mixin import AreaMixin
from ..util import get_axis_padding
from .element import ElementPlot, ColorbarPlot


class ChartPlot(ElementPlot):

    trace_kwargs = {'type': 'scatter'}

    def get_data(self, element, ranges, style):
        x, y = ('y', 'x') if self.invert_axes else ('x', 'y')
        return [{x: element.dimension_values(0),
                 y: element.dimension_values(1)}]


class ScatterPlot(ChartPlot, ColorbarPlot):

    color_index = param.ClassSelector(default=None, class_=(util.basestring, int),
                                      allow_None=True, doc="""
      Index of the dimension from which the color will the drawn""")

    style_opts = [
        'visible',
        'marker',
        'color',
        'cmap',
        'alpha',
        'size',
        'sizemin',
        'selectedpoints',
    ]

    _nonvectorized_styles = ['visible', 'cmap', 'alpha', 'sizemin', 'selectedpoints']

    trace_kwargs = {'type': 'scatter', 'mode': 'markers'}

    _style_key = 'marker'

    selection_display = PlotlyOverlaySelectionDisplay()

    def graph_options(self, element, ranges, style):
        opts = super(ScatterPlot, self).graph_options(element, ranges, style)
        cdim = element.get_dimension(self.color_index)
        if cdim:
            copts = self.get_color_opts(cdim, element, ranges, style)
            copts['color'] = element.dimension_values(cdim)
            opts['marker'].update(copts)

        # If cmap was present and applicable, it was processed by get_color_opts above.
        # Remove it now to avoid plotly validation error
        opts.get('marker', {}).pop('cmap', None)
        return opts


class CurvePlot(ChartPlot, ColorbarPlot):

    interpolation = param.ObjectSelector(objects=['linear', 'steps-mid',
                                                  'steps-pre', 'steps-post'],
                                         default='linear', doc="""
        Defines how the samples of the Curve are interpolated,
        default is 'linear', other options include 'steps-mid',
        'steps-pre' and 'steps-post'.""")

    trace_kwargs = {'type': 'scatter', 'mode': 'lines'}

    style_opts = ['visible', 'color', 'dash', 'line_width']

    _nonvectorized_styles = style_opts

    _style_key = 'line'

    def get_data(self, element, ranges, style):
        if 'steps' in self.interpolation:
            element = interpolate_curve(element, interpolation=self.interpolation)
        return super(CurvePlot, self).get_data(element, ranges, style)


class AreaPlot(AreaMixin, ChartPlot):

    style_opts = ['visible', 'color', 'dash', 'line_width']

    trace_kwargs = {'type': 'scatter', 'mode': 'lines'}

    _style_key = 'line'

    def get_data(self, element, ranges, style):
        x, y = ('y', 'x') if self.invert_axes else ('x', 'y')
        if len(element.vdims) == 1:
            kwargs = super(AreaPlot, self).get_data(element, ranges, style)[0]
            kwargs['fill'] = 'tozero'+y
            return [kwargs]
        xs = element.dimension_values(0)
        ys = element.dimension_values(1)
        bottom = element.dimension_values(2)
        return [{x: xs, y: bottom, 'fill': None},
                {x: xs, y: ys, 'fill': 'tonext'+y}]


class SpreadPlot(ChartPlot):

    style_opts = ['visible', 'color', 'dash', 'line_width']

    trace_kwargs = {'type': 'scatter', 'mode': 'lines'}

    _style_key = 'line'

    def get_data(self, element, ranges, style):
        x, y = ('y', 'x') if self.invert_axes else ('x', 'y')
        xs = element.dimension_values(0)
        mean = element.dimension_values(1)
        neg_error = element.dimension_values(2)
        pos_idx = 3 if len(element.dimensions()) > 3 else 2
        pos_error = element.dimension_values(pos_idx)
        lower = mean - neg_error
        upper = mean + pos_error
        return [{x: xs, y: lower, 'fill': None},
                {x: xs, y: upper, 'fill': 'tonext'+y}]


class ErrorBarsPlot(ChartPlot, ColorbarPlot):

    trace_kwargs = {'type': 'scatter', 'mode': 'lines', 'line': {'width': 0}}

    style_opts = ['visible', 'color', 'dash', 'line_width', 'thickness']

    _nonvectorized_styles = style_opts

    _style_key = 'error_y'

    selection_display = PlotlyOverlaySelectionDisplay()

    def get_data(self, element, ranges, style):
        x, y = ('y', 'x') if self.invert_axes else ('x', 'y')
        error_k = 'error_' + x if element.horizontal else 'error_' + y
        neg_error = element.dimension_values(2)
        pos_idx = 3 if len(element.dimensions()) > 3 else 2
        pos_error = element.dimension_values(pos_idx)
        error_v = dict(type='data', array=pos_error, arrayminus=neg_error)
        return [{x: element.dimension_values(0),
                 y: element.dimension_values(1),
                 error_k: error_v}]


class BarPlot(ElementPlot):

    stacked = param.Boolean(default=False, doc="""
       Whether the bars should be stacked or grouped.""")

    show_legend = param.Boolean(default=True, doc="""
        Whether to show legend for the plot.""")

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

    style_opts = ['visible']

    trace_kwargs = {'type': 'bar'}

    selection_display = PlotlyOverlaySelectionDisplay()

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

    def _get_axis_dims(self, element):
        if element.ndims > 1 and not (self.stacked or self.stack_index):
            xdims = element.kdims
        else:
            xdims = element.kdims[0]
        return (xdims, element.vdims[0])

    def get_data(self, element, ranges, style):
        if self.stack_index is not None:
            self.param.warning(
                'Bars stack_index plot option is deprecated and will '
                'be ignored, set stacked=True/False instead.')
        if self.category_index is not None:
            self.param.warning(
                'Bars category_index plot option is deprecated and '
                'will be ignored, set stacked=True/False instead.')
        if self.group_index not in (None, 1):
            self.param.warning(
                'Bars group_index plot option is deprecated and will '
                'be ignored, set stacked=True/False instead.')

        # Get x, y, group, stack and color dimensions
        xdim = element.kdims[0]
        vdim = element.vdims[0]
        group_dim, stack_dim = None, None
        if element.ndims == 1:
            pass
        elif self.stacked or self.stack_index:
            stack_dim = element.get_dimension(1)
        else:
            group_dim = element.get_dimension(1)

        if self.invert_axes:
            x, y = ('y', 'x')
            orientation = 'h'
        else:
            x, y = ('x', 'y')
            orientation = 'v'

        if element.ndims == 1:
            bars = [{
                'orientation': orientation, 'showlegend': False,
                x: [xdim.pprint_value(v) for v in element.dimension_values(xdim)],
                y: element.dimension_values(vdim)}]
        else:
            group_dim = group_dim or stack_dim
            els = element.groupby(group_dim)
            bars = []
            for k, el in els.items():
                bars.append({
                    'orientation': orientation, 'name': group_dim.pprint_value(k),
                    x: [xdim.pprint_value(v) for v in el.dimension_values(xdim)],
                    y: el.dimension_values(vdim)})
        return bars

    def init_layout(self, key, element, ranges):
        layout = super(BarPlot, self).init_layout(key, element, ranges)
        stack_dim = None
        if element.ndims > 1 and (self.stacked or self.stack_index):
            stack_dim = element.get_dimension(1)
        layout['barmode'] = 'stack' if stack_dim else 'group'
        return layout


class HistogramPlot(ElementPlot):

    trace_kwargs = {'type': 'bar'}

    style_opts = [
        'visible', 'color', 'line_color', 'line_width', 'opacity', 'selectedpoints'
    ]

    _style_key = 'marker'

    selection_display = PlotlyOverlaySelectionDisplay()

    def get_data(self, element, ranges, style):
        xdim = element.kdims[0]
        ydim = element.vdims[0]
        values = element.interface.coords(element, ydim)
        edges = element.interface.coords(element, xdim)
        binwidth = edges[1] - edges[0]

        if self.invert_axes:
            ys = edges
            xs = values
            orientation = 'h'
        else:
            xs = edges
            ys = values
            orientation = 'v'
        return [{'x': xs, 'y': ys, 'width': binwidth, 'orientation': orientation}]

    def init_layout(self, key, element, ranges):
        layout = super(HistogramPlot, self).init_layout(key, element, ranges)
        layout['barmode'] = 'overlay'
        return layout
