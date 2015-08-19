from itertools import combinations
import numpy as np
import param

from bokeh.plotting import figure, gridplot
from bokeh.models import ColumnDataSource
from bokeh.models import GlyphRenderer
from bokeh import mpl

from ...core import OrderedDict, Store, HoloMap
from ...core.util import match_spec, max_range
from ..plot import GenericElementPlot, GenericOverlayPlot

from .plot import BokehPlot


class ElementPlot(GenericElementPlot, BokehPlot):
    
    aspect = param.Parameter(default=1)

    bgcolor = param.Parameter(default='white')
    
    shared_axes = param.Boolean(default=False, doc="""
        Whether to invert the share axes across plots
        for linked panning and zooming.""")

    invert_xaxis = param.Boolean(default=False, doc="""
        Whether to invert the plot x-axis.""")

    invert_yaxis = param.Boolean(default=False, doc="""
        Whether to invert the plot y-axis.""")

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

    xaxis = param.ObjectSelector(default='bottom',
                                 objects=['top', 'bottom', 'bare', 'top-bare',
                                          'bottom-bare', None], doc="""
        Whether and where to display the xaxis, bare options allow suppressing
        all axis labels including ticks and xlabel.""")

    yaxis = param.ObjectSelector(default='left',
                                      objects=['left', 'right', 'bare', 'left-bare',
                                               'right-bare', None], doc="""
        Whether and where to display the yaxis, bare options allow suppressing
        all axis labels including ticks and ylabel.""")
    
    def __init__(self, element, plot=None, subplot=False, **params):
        self.subplot = subplot
        super(ElementPlot, self).__init__(element, **params)
        self.style = self.style[self.cyclic_index]
        self.handles = {} if plot is None else self.handles['plot']

    
    def _init_plot(self, key, plots, title=None, ranges=None, xlabel=None, ylabel=None, zlabel=None):
        y_axis_type = 'log' if self.ylog else 'linear'
        x_axis_type = 'log' if self.xlog else 'linear'
        
        view = self._get_frame(key)
        subplots = list(self.subplots.values()) if self.subplots else []

        plot_kwargs = {}
        title = self._format_title(key) if self.show_title else None
        xlabel, ylabel, zlabel = self._axis_labels(view, subplots, xlabel, ylabel, zlabel)

        # Try finding shared ranges in other plots in the same Layout
        if plots and self.shared_axes:
            for plot in plots:
                if plot is None or not hasattr(plot, 'xaxis'): continue
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
        if self.invert_xaxis:
            plot_kwargs['x_ranges'] = plot_kwargs['x_ranges'][::-1]
                
        if not 'y_range' in plot_kwargs:
            if 'y_range' in ranges:
                plot_kwargs['y_range'] = ranges['y_range']
            else:
                _, b, _, t = self.get_extents(view, ranges)
                if all(y is not None for y in (b, t)):
                    plot_kwargs['y_range'] = [b, t]
        if self.invert_yaxis:
            plot_kwargs['y_range'] = plot_kwargs['y_range'][::-1]

        plot = figure(x_axis_type=x_axis_type, x_axis_label=xlabel, min_border=2,
                      y_axis_type=y_axis_type, y_axis_label=ylabel, tools=self.tools,
                      title=title, width=self.width, height=self.height, **plot_kwargs)
        return plot


    def _update_plot(self, key, plot):
        plot.title = self._format_title(key) if self.show_title else None
        plot.background_fill = self.bgcolor
        if self.title_color:
            plot.title_text_color = self.title_color
        if self.title_font:
            plot.title_text_font = self.title_font
        if self.title_size:
            plot.title_text_font_size = self.title_size
        if self.xaxis in ['bottom-bare' or None]:
            xaxis = plot.xaxis[0]
            xaxis.axis_label = ''
            xaxis.major_label_text_font_size = '0pt'
            xaxis.major_tick_line_color = None
            xaxis.ticker.num_minor_ticks = 0
        if self.yaxis in ['left-bare' or None]:
            yaxis = plot.yaxis[0]
            yaxis.axis_label = ''
            yaxis.major_label_text_font_size = '0pt'
            yaxis.major_tick_line_color = None
            yaxis.ticker.num_minor_ticks = 0


    def _init_datasource(self, element, ranges=None):
        return ColumnDataSource(data=self.get_data(element, ranges))


    def _update_datasource(self, source, element, ranges):
        for k, v in self.get_data(element, ranges).items():
            source.data[k] = v

    
    def initialize_plot(self, ranges=None, plot=None, plots=None, source=None):
        element = self.map.last
        key = self.keys[-1]

        ranges = self.compute_ranges(self.map, key, ranges)
        ranges = match_spec(element, ranges)
        if plot is None:
            plot = self._init_plot(key, ranges=ranges, plots=plots)
        if source is None:
            source = self._init_datasource(element, ranges)
        self.handles['plot'] = plot
        self.handles['source'] = source
        self.init_glyph(element, plot, source, ranges)
        self._update_plot(key, plot)

        return plot


    def update_frame(self, key, ranges=None, plot=None):
        if plot is None:
            plot = self.handles['plot']
        element = self._get_frame(key)
        if element:
            source = self.handles['source']
            self._update_datasource(source, element, ranges)
            self._update_plot(key, plot)


class BokehMPLWrapper(ElementPlot):

    def __init__(self, element, plot=None, subplot=False, **params):
        self.subplot = subplot
        super(ElementPlot, self).__init__(element, **params)
        if isinstance(element, HoloMap):
            etype = element.type
        else:
            etype = type(element)
        plot = Store.registry['matplotlib'][etype]
        self.mplplot = plot(element, **self.lookup_options(element, 'plot').options)


    def initialize_plot(self, ranges=None, plot=None, plots=None):
        self.mplplot.initialize_plot(ranges)
        plot = mpl.to_bokeh(self.mplplot.state)
        self.handles['plot'] = plot
        return plot


    def update_frame(self, key, ranges=None, plot=None):
        if key in self.map:
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
