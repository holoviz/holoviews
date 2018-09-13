import param
import plotly.graph_objs as go

from ...core import util
from ...operation import interpolate_curve
from .element import ElementPlot, ColorbarPlot


class ScatterPlot(ColorbarPlot):

    color_index = param.ClassSelector(default=None, class_=(util.basestring, int),
                                      allow_None=True, doc="""
      Index of the dimension from which the color will the drawn""")

    style_opts = ['symbol', 'color', 'cmap', 'fillcolor', 'opacity', 'fill', 'marker', 'size']

    graph_obj = go.Scatter

    def graph_options(self, element, ranges):
        opts = super(ScatterPlot, self).graph_options(element, ranges)
        opts['mode'] = 'markers'
        style = self.style[self.cyclic_index]
        cdim = element.get_dimension(self.color_index)
        if cdim:
            copts = self.get_color_opts(cdim, element, ranges, style)
            copts['color'] = element.dimension_values(cdim)
            opts['marker'] = copts
        else:
            opts['marker'] = style
        return opts


class PointPlot(ScatterPlot):

    def get_data(self, element, ranges):
        data = dict(x=element.dimension_values(0),
                    y=element.dimension_values(1))
        return (), data


class CurvePlot(ElementPlot):

    interpolation = param.ObjectSelector(objects=['linear', 'steps-mid',
                                                  'steps-pre', 'steps-post'],
                                         default='linear', doc="""
        Defines how the samples of the Curve are interpolated,
        default is 'linear', other options include 'steps-mid',
        'steps-pre' and 'steps-post'.""")

    graph_obj = go.Scatter

    style_opts = ['color', 'dash', 'width', 'line_width']

    def graph_options(self, element, ranges):
        if 'steps' in self.interpolation:
            element = interpolate_curve(element, interpolation=self.interpolation)
        opts = super(CurvePlot, self).graph_options(element, ranges)
        opts['mode'] = 'lines'
        style = self.style[self.cyclic_index]
        if 'line_width' in style:
            style['width'] = style.pop('line_width')
        opts['line'] = style
        return opts

    def get_data(self, element, ranges):
        return (), dict(x=element.dimension_values(0),
                        y=element.dimension_values(1))


class ErrorBarsPlot(ElementPlot):

    graph_obj = go.Scatter

    style_opts = ['color', 'dash', 'width', 'opacity', 'thickness']

    def graph_options(self, element, ranges):
        opts = super(ErrorBarsPlot, self).graph_options(element, ranges)
        opts['mode'] = 'lines'
        opts['line'] = {'width': 0}
        return opts

    def get_data(self, element, ranges):
        neg_error = element.dimension_values(2)
        pos_idx = 3 if len(element.dimensions()) > 3 else 2
        pos_error = element.dimension_values(pos_idx)

        style = self.style[self.cyclic_index]
        if 'line_width' in style:
            style['width'] = style.pop('line_width')
        error_y = dict(type='data', array=pos_error,
                       arrayminus=neg_error, visible=True, **style)
        return (), dict(x=element.dimension_values(0),
                        y=element.dimension_values(1),
                        error_y=error_y)


class BivariatePlot(ColorbarPlot):

    ncontours = param.Integer(default=None)

    graph_obj = go.Histogram2dcontour

    style_opts = ['cmap']

    def graph_options(self, element, ranges):
        opts = super(BivariatePlot, self).graph_options(element, ranges)
        if self.ncontours:
            opts['autocontour'] = False
            opts['ncontours'] = self.ncontours
        style = self.style[self.cyclic_index]
        copts = self.get_color_opts(None, element, ranges, style)
        return dict(opts, **copts)

    def get_data(self, element, ranges):
        return (), dict(x=element.dimension_values(0),
                        y=element.dimension_values(1))


class DistributionPlot(ElementPlot):

    graph_obj = go.Histogram

    def get_data(self, element, ranges):
        return (), dict(x=element.dimension_values(0))


class BarPlot(ElementPlot):

    group_index = param.Integer(default=0, doc="""
       Index of the dimension in the supplied Bars
       Element, which will be laid out into groups.""")

    category_index = param.Integer(default=1, doc="""
       Index of the dimension in the supplied Bars
       Element, which will be laid out into categories.""")

    stack_index = param.Integer(default=2, doc="""
       Index of the dimension in the supplied Bars
       Element, which will stacked.""")

    graph_obj = go.Bar

    def generate_plot(self, key, ranges):
        element = self._get_frame(key)
        ranges = self.compute_ranges(self.hmap, key, ranges)
        ranges = util.match_spec(element, ranges)

        cat_dim = element.get_dimension(self.category_index)
        stack_dim = element.get_dimension(self.stack_index)
        x_dim = element.get_dimension(self.group_index)
        vdim = element.get_dimension(element.ndims)
        if cat_dim and stack_dim:
            self.warning("Plotly does not support stacking and categories "
                         "on a bar chart at the same time.")

        if element.ndims == 1:
            bars = [go.Bar(x=element.dimension_values(x_dim),
                          y=element.dimension_values(vdim))]
        else:
            group_dim = cat_dim if cat_dim else stack_dim
            els = element.groupby(group_dim)
            bars = []
            for k, el in els.items():
                bars.append(go.Bar(x=el.dimension_values(x_dim),
                                   y=el.dimension_values(vdim), name=k))
        layout = self.init_layout(key, element, ranges, x_dim, vdim)
        self.handles['layout'] = layout
        layout['barmode'] = 'group' if cat_dim else 'stacked'

        fig = go.Figure(data=bars, layout=layout)
        self.handles['fig'] = fig
        return fig


class BoxWhiskerPlot(ElementPlot):

    boxpoints = param.ObjectSelector(objects=["all", "outliers",
                                              "suspectedoutliers", False],
                                     default='outliers', doc="""
        Which points to show, valid options are 'all', 'outliers',
        'suspectedoutliers' and False""")

    jitter = param.Number(default=0, doc="""
        Sets the amount of jitter in the sample points drawn. If "0",
        the sample points align along the distribution axis. If "1",
        the sample points are drawn in a random jitter of width equal
        to the width of the box(es).""")

    mean = param.ObjectSelector(default=False, objects=[True, False, 'sd'],
                                doc="""
        If "True", the mean of the box(es)' underlying distribution
        is drawn as a dashed line inside the box(es). If "sd" the
        standard deviation is also drawn.""")

    style_opts = ['color', 'opacity', 'outliercolor', 'symbol',
                  'size']

    def generate_plot(self, key, ranges):
        element = self._get_frame(key)
        ranges = self.compute_ranges(self.hmap, key, ranges)
        ranges = util.match_spec(element, ranges)

        style = self.style[self.cyclic_index]
        orientation = 'h' if self.invert_axes else 'v'
        axis = 'x' if self.invert_axes else 'y'
        box_opts = dict(boxmean=self.mean, jitter=self.jitter,
                        marker=style, orientation=orientation)
        groups = element.groupby(element.kdims)
        groups = groups.data.items() if element.kdims else [(element.label, element)]
        plots = []
        for key, group in groups:
            if element.kdims:
                label = ','.join([d.pprint_value(v) for d, v in zip(element.kdims, key)])
            else:
                label = key
            data = {axis: group.dimension_values(group.vdims[0])}
            plots.append(go.Box(name=label, **dict(box_opts, **data)))
        layout = self.init_layout(key, element, ranges, element.kdims, element.vdims)
        self.handles['layout'] = layout
        fig = go.Figure(data=plots, layout=layout)
        self.handles['fig'] = fig
        return fig

    def get_extents(self, element, ranges, range_type='combined'):
        return (None, None, None, None)
