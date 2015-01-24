from __future__ import absolute_import

from matplotlib import pyplot as plt

import param

from ..core import Element
from ..interface.pandas import DFrame, DataFrameView, pd
from .plot import Plot


class DFrameViewPlot(Plot):
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

    show_grid = param.Boolean(default=True, doc="""
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

    style_opts = list({opt for opts in dframe_options.values() for opt in opts})

    apply_databounds = False

    def __init__(self, view, **params):
        super(DFrameViewPlot, self).__init__(view, **params)
        if self._map.last.plot_type and 'plot_type' not in params:
            self.plot_type = self._map.last.plot_type


    def __call__(self, axis=None, lbrt=None):
        dfview = self._map.last
        self._validate(dfview, axis)

        self.ax = self._init_axis(axis)
        style = self.settings.closest(dfview, 'style')[self.cyclic_index]
        self.style = self._process_style(style)

        self._update_plot(dfview)
        if 'fig' in self.handles and self.handles['fig'] != plt.gcf():
            self.handles['fig'] = plt.gcf()

        return self._finalize_axis(self._keys[-1], lbrt=lbrt)


    def _process_style(self, styles):
        style_keys = styles.keys()
        for k in style_keys:
            if k not in self.dframe_options[self.plot_type]:
                self.warning('Plot option %s does not apply to %s plot type.' % (k, self.plot_type))
                styles.pop(k)
        if self.plot_type not in ['autocorrelation_plot']:
            styles['figsize'] = self.size

        # Legacy fix for Pandas, can be removed for Pandas >0.14
        if self.plot_type == 'boxplot':
            styles['return_type'] = 'axes'
        return styles


    def _validate(self, dfview, axis):
        composed = axis is not None

        if composed and dfview.ndims > 1 and self.plot_type in ['hist']:
            raise Exception("Multiple %s plots cannot be composed." % self.plot_type)


    def _update_plot(self, view):
        if self.plot_type == 'scatter_matrix':
            pd.scatter_matrix(view.data, ax=self.ax, **self.style)
        elif self.plot_type == 'autocorrelation_plot':
            pd.tools.plotting.autocorrelation_plot(view.data, ax=self.ax, **self.style)
        elif self.plot_type == 'plot':
            opts = dict({'x': view.x, 'y': view.y}, **self.style)
            view.data.plot(ax=self.ax, **opts)
        else:
            getattr(view.data, self.plot_type)(ax=self.ax, **self.style)


    def update_handles(self, view, key, lbrt=None):
        """
        Update the plot for an animation.
        """
        if not self.plot_type in ['hist', 'scatter_matrix']:
            if self.zorder == 0 and self.ax: self.ax.cla()
        self._update_plot(view)


Plot.defaults.update({DataFrameView: DFrameViewPlot,
                      DFrame: DFrameViewPlot})
