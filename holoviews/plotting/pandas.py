from __future__ import absolute_import

from matplotlib import pyplot as plt

import param

from ..core.options import Store
from ..interface.pandas import DFrame, DataFrameView, pd
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
                                              'autocorrelation_plot'],
                                     doc="""Selects which Pandas plot type to use.""")

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

    xticks = param.Number(default=0, doc="""
        By default we don't mess with Pandas based tickmarks""")

    yticks = param.Number(default=0, doc="""
        By default we don't mess with Pandas based tickmarks""")

    style_opts = list({opt for opts in dframe_options.values() for opt in opts})

    apply_databounds = False

    def __init__(self, view, **params):
        super(DFrameViewPlot, self).__init__(view, **params)
        if self.map.last.plot_type and 'plot_type' not in params:
            self.plot_type = self.map.last.plot_type


    def __call__(self, ranges=None):
        dfview = self.map.last
        self._validate(dfview)

        self._update_plot(dfview)
        if 'fig' in self.handles and self.handles['fig'] != plt.gcf():
            self.handles['fig'] = plt.gcf()

        return self._finalize_axis(self.keys[-1])


    def _process_style(self, style):
        style_keys = style.keys()
        for k in style_keys:
            if k not in self.dframe_options[self.plot_type]:
                self.warning('Plot option %s does not apply to %s plot type.' % (k, self.plot_type))
                style.pop(k)
        if self.plot_type not in ['autocorrelation_plot']:
            style['figsize'] = self.figure_size

        # Legacy fix for Pandas, can be removed for Pandas >0.14
        if self.plot_type == 'boxplot':
            style['return_type'] = 'axes'
        return style


    def _validate(self, dfview):
        composed = self.handles['axis'] is not None

        if composed and dfview.ndims > 1 and self.plot_type in ['hist']:
            raise Exception("Multiple %s plots cannot be composed." % self.plot_type)


    def _update_plot(self, axis, view):
        style = self._process_style(self.style[self.cyclic_index])
        if self.plot_type == 'scatter_matrix':
            pd.scatter_matrix(view.data, ax=axis, **style)
        elif self.plot_type == 'autocorrelation_plot':
            pd.tools.plotting.autocorrelation_plot(view.data, ax=axis, **style)
        elif self.plot_type == 'plot':
            opts = dict({'x': view.x, 'y': view.y}, **style)
            view.data.plot(ax=self.handles['axis'], **opts)
        else:
            getattr(view.data, self.plot_type)(ax=axis, **style)


    def update_handles(self, axis, view, key, ranges=None):
        """
        Update the plot for an animation.
        """
        if not self.plot_type in ['hist', 'scatter_matrix']:
            if self.zorder == 0 and axis:
                axis.cla()
        self._update_plot(axis, view)


Store.registry.update({DataFrameView: DFrameViewPlot,
                       DFrame: DFrameViewPlot})
