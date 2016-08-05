from __future__ import absolute_import

import numpy as np
from matplotlib import pyplot as plt

import param

from ...core.options import Store
from ...interface.pandas import DFrame, DataFrameView, pd
from .element import ElementPlot


class DFrameViewPlot(ElementPlot):
    """
    DFramePlot provides a wrapper around Pandas dataframe plots.  It
    takes a single DataFrameView or DFrameMap as input and plots it
    using the plotting command selected via the plot_type.

    The plot_options specifies the valid options to be supplied to the
    selected plot_type via options.style_opts.
    """

    aspect = param.Parameter(default='square', doc="""
        Aspect ratio defaults to square, 'equal' or numeric values
        are also supported.""")

    show_grid = param.Boolean(default=False, doc="""
        Whether to show a Cartesian grid on the plot.""")

    plot_type = param.ObjectSelector(default='scatter_matrix',
                                     objects=['plot', 'boxplot',
                                              'hist', 'scatter_matrix',
                                              'autocorrelation_plot'], doc="""
       Selects which Pandas plot type to use, valid options include: 'plot',
       'boxplot', 'hist', 'scatter_matrix' and 'autocorrelation_plot'.""")

    dframe_options = {'plot': ['kind', 'stacked', 'xerr',
                               'yerr', 'share_x', 'share_y',
                               'table', 'style', 'x', 'y',
                               'secondary_y', 'legend',
                               'logx', 'logy', 'position',
                               'colormap', 'mark_right'],
                      'hist': ['column', 'by', 'grid',
                               'xlabelsize', 'xrot',
                               'ylabelsize', 'yrot',
                               'sharex', 'sharey', 'hist',
                               'layout', 'bins'],
                      'boxplot': ['column', 'by', 'fontsize',
                                  'layout', 'grid', 'rot'],
                      'scatter_matrix': ['alpha', 'grid', 'diagonal',
                                         'marker', 'range_padding',
                                         'hist_kwds', 'density_kwds'],
                      'autocorrelation': ['kwds']}

    xticks = param.Number(default=None, doc="""
        By default we don't mess with Pandas based tickmarks""")

    yticks = param.Number(default=None, doc="""
        By default we don't mess with Pandas based tickmarks""")

    style_opts = list({opt for opts in dframe_options.values() for opt in opts})

    def __init__(self, view, **params):
        super(DFrameViewPlot, self).__init__(view, **params)
        if self.hmap.last.plot_type and 'plot_type' not in params:
            self.plot_type = self.hmap.last.plot_type


    def initialize_plot(self, ranges=None):
        element = self.hmap.last
        self._validate(element)

        style = self._process_style(self.style[self.cyclic_index])
        axis = self.handles['axis']
        self._update_plot(axis, element, style)
        if 'fig' in self.handles and self.handles['fig'] != plt.gcf():
            self.handles['fig'] = plt.gcf()

        return self._finalize_axis(self.keys[-1], **self.get_axis_kwargs(element))


    def _process_style(self, style):
        style_keys = style.keys()
        for k in style_keys:
            if k not in self.dframe_options[self.plot_type]:
                self.warning('Plot option %s does not apply to %s plot type.' % (k, self.plot_type))
                style.pop(k)
        if self.plot_type not in ['autocorrelation_plot']:
            style['figsize'] = self.fig_size

        # Legacy fix for Pandas, can be removed for Pandas >0.14
        if self.plot_type == 'boxplot':
            style['return_type'] = 'axes'
        return style


    def get_extents(self, view, ranges):
        x0, y0, x1, y1 = (np.NaN,) * 4
        if ranges:
            if view.x:
                x0, x1 = ranges[view.x]
            if view.x2:
                y0, y1 = ranges[view.x2]
            elif view.y:
                y0, y1 = ranges[view.y]
        return (x0, y0, x1, y1)


    def get_axis_kwargs(self, element):
        if element.x:
            xlabel = str(element.get_dimension(element.x))
        if element.x2:
            ylabel = str(element.get_dimension(element.x2))
        elif element.y:
            ylabel = str(element.get_dimension(element.y))
        return dict(xlabel=xlabel, ylabel=ylabel)


    def _validate(self, dfview):
        composed = self.handles['axis'] is not None

        if composed and dfview.ndims > 1 and self.plot_type in ['hist']:
            raise Exception("Multiple %s plots cannot be composed." % self.plot_type)


    def _update_plot(self, axis, view, style):
        if self.plot_type == 'scatter_matrix':
            pd.scatter_matrix(view.data, ax=axis, **style)
        elif self.plot_type == 'autocorrelation_plot':
            pd.tools.plotting.autocorrelation_plot(view.data, ax=axis, **style)
        elif self.plot_type == 'plot':
            opts = dict({'x': view.x, 'y': view.y}, **style)
            view.data.plot(ax=self.handles['axis'], **opts)
        else:
            getattr(view.data, self.plot_type)(ax=axis, **style)


    def update_handles(self, key, axis, view, ranges, style):
        """
        Update the plot for an animation.
        """
        if not self.plot_type in ['hist', 'scatter_matrix']:
            if self.zorder == 0 and axis:
                axis.cla()
        self._update_plot(axis, view, style)
        return self.get_axis_kwargs(view)


Store.register({DataFrameView: DFrameViewPlot,
                DFrame: DFrameViewPlot}, 'matplotlib')
