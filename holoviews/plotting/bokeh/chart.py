import numpy as np

import param

from ...core import Dimension
from ...core.util import max_range
from ...element import Chart
from .element import ElementPlot, line_properties, fill_properties
from .path import PathPlot


class PointPlot(ElementPlot):

    style_opts = ['marker', 'color'] + line_properties + fill_properties

    def get_data(self, element, ranges=None):
        return dict(x=element.data[:, 0], y=element.data[:, 1])
    
    def init_glyph(self, element, plot, source, ranges):
        plot.scatter(x='x', y='y', source=source, legend=element.label, **self.style)


class CurvePlot(ElementPlot):

    style_opts = ['color'] + line_properties
    
    def get_data(self, element, ranges=None):
        return dict(x=element.data[:, 0], y=element.data[:, 1])
    
    def init_glyph(self, element, plot, source, ranges):
        plot.line(x='x', y='y', source=source, legend=element.label, **self.style)


class SpreadPlot(ElementPlot):

    style_opts = ['color'] + line_properties + fill_properties

    def __init__(self, *args, **kwargs):
        super(SpreadPlot, self).__init__(*args, **kwargs)
        self._extent = None

    def get_data(self, element, ranges=None):
        lower = element.data[:, 1] - element.data[:, 2]
        upper = element.data[:, 1] + element.data[:, 3]
        band_x = np.append(element.data[:, 0], element.data[::-1, 0])
        band_y = np.append(lower, upper[::-1])
        return dict(xs=[band_x], ys=[band_y])
        
    def init_glyph(self, element, plot, source, ranges):
        self.handles['patches'] = plot.patches(xs='xs', ys='ys', source=source, 
                                               legend=element.label, **self.style)

    def get_extents(self, view, ranges):
        x0, y0, x1, y1 = super(SpreadPlot, self).get_extents(view, ranges)
        normopts = self.lookup_options(view, 'norm')
        if normopts.options.get('framewise', False):
            y0 = view.data[:, 1] - view.data[:, 2]
            y1 = view.data[:, 1] + view.data[:, 3]
        else:
            if not self._extent:
                max_spread = lambda x: (np.min(x.data[:, 1] - x.data[:, 2]),
                                        np.max(x.data[:,1] + x.data[:, 3]))
                y0, y1 = max_range(self.map.traverse(max_spread, (type(view),)))
                self._extent = (y0, y1)
            else:
                y0, y1 = self._extent
        return x0, y0, x1, y1


class HistogramPlot(ElementPlot):

    style_opts = ['color'] + line_properties + fill_properties

    def get_data(self, element, ranges=None):
        return dict(top=element.values, left=element.edges[:-1],
                    right=element.edges[1:])

    def init_glyph(self, element, plot, source, ranges):
        self.handles['lines'] = plot.quad(top='top', bottom=0, left='left',
                                          right='right', source=source,
                                          legend=element.label, **self.style)


class ErrorPlot(PathPlot):

    horizontal = param.Boolean(default=False)

    style_opts = ['color'] + line_properties
    
    def get_data(self, element, ranges=None):
        data = element.data
        err_xs = []
        err_ys = []
        for x, y, neg, pos in data:
            if self.horizontal:
                err_xs.append((x - neg, x + pos))
                err_ys.append((y, y))
            else:
                err_xs.append((x, x))
                err_ys.append((y - neg, y + pos))
        return dict(xs=err_xs, ys=err_ys)



class LinkedScatter(Chart):
    
    group = param.String(default='LinkedScatter')

    kdims = param.List(default=[Dimension('x'), Dimension('y')])

    
class LinkedScatterPlot(ElementPlot):
    
    cols = param.Integer(default=3)

    tools = param.String(default="pan,wheel_zoom,box_zoom,reset,resize,box_select,lasso_select")
    
    style_opts = ['color', 'marker', 'size'] + fill_properties + line_properties
    
    def get_data(self, element, ranges=None):
        return {d: element.dimension_values(d) for d in element.dimensions(label=True)}

    def permutations(self, element):
        dims = element.dimensions(label=True)
        return [(d1, d2) for (d1, d2) in combinations(dims, 2) if d1 != d2]

    def initialize_plot(self, ranges=None, plot=None):
        element = self.map.last
        key = self.keys[-1]
        ranges = self.compute_ranges(self.map, key, ranges)
        ranges = match_spec(element, ranges)

        plots = OrderedDict()
        passed_plots = []
        for d1, d2 in self.permutations(element):
            sub_ranges = {'x_range': ranges[d1], 'y_range': ranges[d2]}
            plots[d1, d2] = self._init_plot(key, plots=passed_plots, ranges=sub_ranges,
                                            xlabel=str(element.get_dimension(d1)),
                                            ylabel=str(element.get_dimension(d2)))
            passed_plots.append(plots[d1, d2])
        i = 0
        grid_plots = []
        plot_list = plots.values()
        while i<len(plot_list):
            grid_plots.append(plot_list[i:i+3])
            i+=3
        self.handles['plot'] = gridplot(grid_plots)
        
        source = self._init_datasource(element)
        self.handles['source'] = source
        self.init_glyph(element, plots, source)
        
        return plot
    
    def init_glyph(self, element, plot, source, ranges):
        for idx, (d1, d2) in enumerate(self.permutations(element)):
            plot[d1, d2].scatter(x=d1, y=d2, source=source, legend=element.label, **self.style[idx])

            
    def update_frame(self, key, ranges=None, plot=None):
        if plot is None:
            plot = self.handles['plot']
        element = self._get_frame(key)
        source = self.handles['source']
        self._update_datasource(source, element)
