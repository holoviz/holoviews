from __future__ import absolute_import

from matplotlib import pyplot as plt

import param

from .. import View
from ..interface.pandas import DFrame, DFrameStack, DFrameOverlay, DataFrameView, pd
from .viewplots import Plot, OverlayPlot



class DFramePlot(OverlayPlot):
    """
    A high-level plot, which will plot any DataFrameView or DFrameStack type
    including DataOverlays.

    A generic plot that visualizes DFrameStacks containing DFrameOverlay or
    DataFrameView objects.
    """

    _stack_type = DFrameStack
    _view_type = DFrameOverlay

    style_opts = param.List(default=[], constant=True, doc="""
     DataPlot renders overlay layers which individually have style
     options but DataPlot itself does not.""")

    def __call__(self, axis=None, lbrt=None, **kwargs):

        ax = self._init_axis(axis)

        stacks = self._stack.split_overlays()

        for zorder, stack in enumerate(stacks):
            plotopts = View.options.plotting(stack).opts

            plotype = Plot.defaults[stack.type]
            plot = plotype(stack, size=self.size,
                           show_xaxis=self.show_xaxis, show_yaxis=self.show_yaxis,
                           show_legend=self.show_legend, show_title=self.show_title,
                           show_grid=self.show_grid, zorder=zorder,
                           **dict(plotopts, **kwargs))
            plot.aspect = self.aspect

            plot(ax)
            self.plots.append(plot)

        if axis is None: plt.close(self.handles['fig'])
        return ax if axis else self.handles['fig']


    def update_frame(self, n, lbrt=None):
        n = n if n < len(self) else len(self) - 1
        for zorder, plot in enumerate(self.plots):
            plot.update_frame(n)



class DFrameViewPlot(Plot):
    """
    DFramePlot provides a wrapper around Pandas dataframe plots.
    It takes a single DataFrameView or DFrameStack as input and plots it using
    the plotting command selected via the plot_type.

    The plot_options specifies the valid options to be supplied
    to the selected plot_type via options.style_opts.
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

    _stack_type = DFrameStack
    _view_type = DataFrameView


    def __init__(self, view, **params):
        super(DFrameViewPlot, self).__init__(view, **params)
        if self._stack.last.plot_type and 'plot_type' not in params:
            self.plot_type = self._stack.last.plot_type


    def __call__(self, axis=None, cyclic_index=0, lbrt=None):
        dfview = self._stack.last

        self._validate(dfview, axis)

        self.ax = self._init_axis(axis)
        self.style = self._process_style(View.options.style(dfview)[cyclic_index])

        self._update_plot(dfview)
        if 'fig' in self.handles and self.handles['fig'] != plt.gcf():
            self.handles['fig'] = plt.gcf()

        return self._finalize_axis(-1, lbrt=lbrt)


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

        if composed and self.plot_type == 'scatter_matrix':
            raise Exception("Scatter Matrix plots can't be composed.")
        elif composed and len(dfview.dimensions) > 1 and self.plot_type in ['hist']:
            raise Exception("Multiple %s plots cannot be composed." % self.plot_type)


    def _update_plot(self, dfview):
        if self.plot_type == 'scatter_matrix':
            pd.scatter_matrix(dfview.data, ax=self.ax, **self.style)
        elif self.plot_type == 'autocorrelation_plot':
            pd.tools.plotting.autocorrelation_plot(dfview.data, ax=self.ax, **self.style)
        elif self.plot_type == 'plot':
            opts = dict({'x': dfview.x, 'y': dfview.y}, **self.style)
            dfview.data.plot(ax=self.ax, **opts)
        else:
            getattr(dfview.data, self.plot_type)(ax=self.ax, **self.style)


    def update_frame(self, n, lbrt=None):
        """
        Update the plot for an animation.
        """
        n = n if n < len(self) else len(self) - 1
        dfview = list(self._stack.values())[n]
        if not self.plot_type in ['hist', 'scatter_matrix']:
            if self.zorder == 0: self.ax.cla()
        self._update_plot(dfview)
        self._finalize_axis(n, lbrt=lbrt)
        plt.draw()


Plot.defaults.update({DataFrameView: DFrameViewPlot,
                      DFrame: DFrameViewPlot,
                      DFrameOverlay: DFramePlot})
