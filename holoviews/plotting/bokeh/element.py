import param

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh import mpl

from ...core import Store, HoloMap
from ...core.util import match_spec
from ..plot import GenericElementPlot, GenericOverlayPlot

from .plot import BokehPlot


class ElementPlot(BokehPlot, GenericElementPlot):
    
    bgcolor = param.Parameter(default='white', doc="""
        Background color of the plot.""")

    invert_xaxis = param.Boolean(default=False, doc="""
        Whether to invert the plot x-axis.""")

    invert_yaxis = param.Boolean(default=False, doc="""
        Whether to invert the plot y-axis.""")

    show_legend = param.Boolean(default=False, doc="""
        Whether to show legend for the plot.""")

    shared_axes = param.Boolean(default=True, doc="""
        Whether to invert the share axes across plots
        for linked panning and zooming.""")

    title_color = param.Parameter(default='black', doc="""
        Color of the title defined as recognized color string,
        hex RGB value tuple.""")

    title_font = param.String(default=None, doc="""
        Title font to apply to the plot.""")

    title_size = param.String(default='12pt', doc="""
        Title font size to apply to the plot.""")

    xlog = param.Boolean(default=False, doc="""
        Whether the x-axis of the plot will be a log axis.""")
    
    ylog = param.Boolean(default=False, doc="""
        Whether the x-axis of the plot will be a log axis.""")
    
    tools = param.List(default=['pan', 'wheel_zoom', 'box_zoom',
                                'reset', 'resize'], doc="""
        A list of plugin tools to use on the plot.""")

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

        tools = ','.join(self.tools)
        plot = figure(x_axis_type=x_axis_type, x_axis_label=xlabel, min_border=2,
                      y_axis_type=y_axis_type, y_axis_label=ylabel, tools=tools,
                      title=title, width=self.width, height=self.height,
                      **plot_kwargs)
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
