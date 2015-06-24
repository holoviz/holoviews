from itertools import combinations
import numpy as np
import param

from bokeh.plotting import figure, gridplot
from bokeh.models import ColumnDataSource
from bokeh.models import GlyphRenderer
from bokeh import mpl

from ...core import OrderedDict, Dimension, Store
from ...core.util import match_spec
from ...element import Chart, Image, HeatMap, RGB, Raster
from ..plot import GenericElementPlot, GenericOverlayPlot

from .plot import BokehPlot


class ElementPlot(GenericElementPlot, BokehPlot):
    
    aspect = param.Parameter(default=1)

    bgcolor = param.Parameter(default='white')
    
    show_legend = param.Boolean(default=False, doc="""
        Whether to show legend for the plot.""")

    xlog = param.Boolean(default=False)
    
    ylog = param.Boolean(default=False)
    
    width = param.Integer(default=300)
    
    height = param.Integer(default=300)

    title_color = param.Parameter(default=None)

    title_font = param.String(default=None)

    title_size = param.String(default=None)

    tools = param.String(default="pan,wheel_zoom,box_zoom,reset,resize")

    select = param.Boolean(default=True)
    
    def __init__(self, element, plot=None, subplot=False, **params):
        self.subplot = subplot
        super(ElementPlot, self).__init__(element, **params)
        self.handles = {} if plot is None else self.handles['plot']

    
    def _init_plot(self, key, plots, title=None, ranges=None, xlabel=None, ylabel=None, zlabel=None):
        y_axis_type = 'log' if self.ylog else 'linear'
        x_axis_type = 'log' if self.xlog else 'linear'
        
        view = self._get_frame(key)
        subplots = list(self.subplots.values()) if self.subplots else []

        plot_kwargs = {}
        title = self._format_title(key)
        xlabel, ylabel, zlabel = self._axis_labels(view, subplots, xlabel, ylabel, zlabel)


        # Try finding shared ranges in other plots in the same Layout
        if plots:
            for plot in plots:
                if plot.xaxis[0].axis_label == xlabel:
                    plot_kwargs['x_range'] = plot.x_range
                if plot.xaxis[0].axis_label == ylabel:
                    plot_kwargs['y_range'] = plot.x_range
                if plot.yaxis[0].axis_label == ylabel:
                    plot_kwargs['y_range'] = plot.y_range
                if plot.yaxis[0].axis_label == xlabel:
                    plot_kwargs['x_range'] = plot.y_range

        if not 'x_range' in plot_kwargs:
            if 'x_range' in ranges:
                plot_kwargs['x_range'] = ranges['x_range']
            else:
                l, _, r, _ = self.get_extents(view, ranges)
                if all(x is not None for x in (l, r)):
                    plot_kwargs['x_range'] = [l, r]
                
        if not 'y_range' in plot_kwargs:
            if 'y_range' in ranges:
                plot_kwargs['y_range'] = ranges['y_range']
            else:
                _, b, _, t = self.get_extents(view, ranges)
                if all(y is not None for y in (b, t)):
                    plot_kwargs['y_range'] = [b, t]

        plot = figure(x_axis_type=x_axis_type, x_axis_label=xlabel,
                      y_axis_type=y_axis_type, y_axis_label=ylabel, tools=self.tools,
                      title=title, width=self.width, height=self.height, **plot_kwargs)
        return plot


    def _update_plot(self, key, plot):
        plot.title = self._format_title(key)
        plot.background_fill = self.bgcolor
        if self.title_color:
            plot.title_text_color = self.title_color
        if self.title_font:
            plot.title_text_font = self.title_font
        if self.title_size:
            plot.title_text_font_size = self.title_size


    def _init_datasource(self, element):
        return ColumnDataSource(data=self.get_data(element))


    def _update_datasource(self, source, element):
        for k, v in self.get_data(element).items():
            source.data[k] = v

    
    def initialize_plot(self, ranges=None, plot=None, plots=None):
        element = self.map.last
        key = self.keys[-1]

        ranges = self.compute_ranges(self.map, key, ranges)
        ranges = match_spec(element, ranges)
        if plot is None:
            plot = self._init_plot(key, ranges=ranges, plots=plots)
        source = self._init_datasource(element)
        style = self.style[self.cyclic_index]
        self.handles['plot'] = plot
        self.handles['source'] = source
        self.init_glyph(element, plot, source, style)
        
        return plot

        
    def update(self, key):
        """
        Update the internal state of the Plot to represent the given
        key tuple (where integers represent frames). Returns this
        state.
        """
        if not self.drawn:
            plot = self.initialize_plot()
        self.update_frame(key, plot=plot)
        return self.state


    def update_frame(self, key, ranges=None, plot=None):
        if plot is None:
            plot = self.handles['plot']
        element = self._get_frame(key)
        source = self.handles['source']
        self._update_datasource(source, element)
        self._update_plot(key, plot)


class BokehMPLWrapper(ElementPlot):

    def __init__(self, element, plot=None, subplot=False, **params):
        self.subplot = subplot
        super(ElementPlot, self).__init__(element, **params)
        plot = Store.registry['matplotlib'][type(element)]
        self.mplplot = plot(element, **self.lookup_options(element, 'plot').options)


    def initialize_plot(self, ranges=None, plot=None, plots=None):
        self.mplplot.initialize_plot(ranges)
        plot = mpl.to_bokeh(self.mplplot.state)
        self.handles['plot'] = plot
        return plot


    def update_frame(self, key, ranges=None, plot=None):
        self.mplplot.update_frame(key, ranges)
        self.handles['plot'] = mpl.to_bokeh(self.mplplot.state)


class OverlayPlot(GenericOverlayPlot, ElementPlot):
    

    def initialize_plot(self, ranges=None, plot=None, plots=None):
        """
        Plot all the views contained in the AdjointLayout Object using axes
        appropriate to the layout configuration. All the axes are
        supplied by LayoutPlot - the purpose of the call is to
        invoke subplots with correct options and styles and hide any
        empty axes as necessary.
        """
        key = self.keys[-1]
        ranges = self.compute_ranges(self.map, key, ranges)
        if plot is None:
            plot = self._init_plot(key, ranges=ranges, plots=plots)
        self.handles['plot'] = plot
        
        for subplot in self.subplots.values():
            subplot.initialize_plot(ranges, plot, plots)

        return plot

        
    def update_frame(self, key, ranges=None, plot=None):
        """
        Update the internal state of the Plot to represent the given
        key tuple (where integers represent frames). Returns this
        state.
        """
        for subplot in self.subplots.values():
            subplot.update_frame(key, ranges, plot)


line_properties = ['line_width', 'line_color', 'line_alpha',
                   'line_join', 'line_cap', 'line_dash']

fill_properties = ['fill_color', 'fill_alpha']

text_properties = ['text_font', 'text_font_size', 'text_font_style', 'text_color',
                   'text_alpha', 'text_align', 'text_baseline']

class PointPlot(ElementPlot):

    style_opts = ['marker', 'color'] + line_properties + fill_properties

    def get_data(self, element):
        return dict(x=element.data[:, 0], y=element.data[:, 1])
    
    def init_glyph(self, element, plot, source, style):
        plot.scatter(x='x', y='y', source=source, legend=element.label, **style)


class CurvePlot(ElementPlot):

    style_opts = ['color'] + line_properties
    
    def get_data(self, element):
        return dict(x=element.data[:, 0], y=element.data[:, 1])
    
    def init_glyph(self, element, plot, source, style):
        plot.line(x='x', y='y', source=source, legend=element.label, **style)



class RasterPlot(ElementPlot):
    
    style_opts = ['palette']

    def get_data(self, element):
        if isinstance(element, Image):
            l, b, r, t = element.bounds.lbrt()
        else:
            l, b, r, t = element.extents
            if type(element) == Raster:
                b, t = t, b
        img = element.data
        if img.ndim == 3:
            if img.shape[2] == 3: # alpha channel not included
                img = np.dstack([img, np.ones(img.shape[:2])])
            img = (img * 255).view(dtype=np.uint32)[:, :, 0]
        return dict(image=[img], x=[l], y=[b], dw=[r-l], dh=[t-b])
    
    def init_glyph(self, element, plot, source, style):
        kwargs = dict(style, image='image', x='x', y='y', dw='dw',
                      dh='dh', source=source, legend=element.label, **style)
        if isinstance(element, RGB):
            self.handles['img'] = plot.image_rgba(**kwargs)
        else:
            self.handles['img'] = plot.image(**kwargs)

    
class PathPlot(ElementPlot):

    style_opts = ['color'] + line_properties

    def get_data(self, element):
        xs = [path[:, 0] for path in element.data]
        ys = [path[:, 1] for path in element.data]
        return dict(xs=xs, ys=ys)

    def init_glyph(self, element, plot, source, style):
        self.handles['lines'] = plot.multi_line(xs='xs', ys='ys', source=source, 
                                                legend=element.label, **style)


class HistogramPlot(ElementPlot):

    style_opts = ['color'] + line_properties + fill_properties

    def get_data(self, element):
        return dict(top=element.values, left=element.edges[:-1],
                    right=element.edges[1:])

    def init_glyph(self, element, plot, source, style):
        self.handles['lines'] = plot.quad(top='top', bottom=0, left='left',
                                          right='right', source=source,
                                          legend=element.label, **style)


class ErrorPlot(PathPlot):

    horizontal = param.Boolean(default=False)

    style_opts = ['color'] + line_properties
    
    def get_data(self, element):
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
    

class PolygonPlot(PathPlot):

    style_opts = ['color'] + line_properties + fill_properties
        
    def init_glyph(self, element, plot, source, style):
        self.handles['patches'] = plot.patches(xs='xs', ys='ys', source=source, 
                                               legend=element.label, **style)

class TextPlot(ElementPlot):

    style_opts = text_properties

    def get_data(self, element):
        return dict(x=[element.x], y=[element.y], text=[element.text])

    def init_glyph(self, element, plot, source, style):
        self.handles['text'] = plot.text(x='x', y='y', text='text', source=source, **style)

    def get_extents(self, element, ranges=None):
        return None, None, None, None


class LinkedScatter(Chart):
    
    group = param.String(default='LinkedScatter')

    kdims = param.List(default=[Dimension('x'), Dimension('y')])

    
class LinkedScatterPlot(ElementPlot):
    
    cols = param.Integer(default=3)

    tools = param.String(default="pan,wheel_zoom,box_zoom,reset,resize,box_select,lasso_select")
    
    style_opts = ['color', 'marker', 'size'] + fill_properties + line_properties
    
    def get_data(self, element):
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
        style = self.style
        self.handles['source'] = source
        self.init_glyph(element, plots, source, style)
        
        return plot
    
    def init_glyph(self, element, plot, source, style):
        for idx, (d1, d2) in enumerate(self.permutations(element)):
            plot[d1, d2].scatter(x=d1, y=d2, source=source, legend=element.label, **style[idx])

            
    def update_frame(self, key, ranges=None, plot=None):
        if plot is None:
            plot = self.handles['plot']
        element = self._get_frame(key)
        source = self.handles['source']
        self._update_datasource(source, element)
